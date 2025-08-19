# dqn_solver.py - The Outer Loop (DQN Implementation)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque
import config
from state import DialogueState
from pragmatics import PragmaticRewardCalculator

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Replay Buffer ---
# Stores transitions for training
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- State Featurization ---
def featurize_state(state: DialogueState, meaning_map: dict, utterance_map: dict) -> torch.Tensor:
    """
    Converts a DialogueState object into a fixed-size tensor for the DQN.
    """
    # 1. Turn index (normalized)
    turn_feature = [state.turn_index / config.HORIZON]

    # 2. Dialogue history (e.g., last 3 utterances, padded, one-hot)
    history_len = 3
    history_feature = [0] * (history_len * len(utterance_map))
    for i, utterance in enumerate(list(state.dialogue_history)[-history_len:]):
        if utterance in utterance_map:
            history_feature[i * len(utterance_map) + utterance_map[utterance]] = 1

    # 3. Speaker's private meaning (one-hot encoded)
    meaning_feature = [0] * len(meaning_map)
    if state.speaker_private_meaning in meaning_map:
        meaning_feature[meaning_map[state.speaker_private_meaning]] = 1
    
    # 4. Speaker's belief about the listener (vector of probabilities)
    belief_dict = state.get_belief_dict()
    belief_feature = [belief_dict.get(m, 0) for m in sorted(meaning_map.keys())]
    
    # Concatenate all features into a single vector
    features = turn_feature + history_feature + meaning_feature + belief_feature
    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)


class DQNSolver:
    """
    Manages the DQN training process, including the agent's policy and learning steps.
    """
    def __init__(self, metrics: 'MetricsTracker'):
        self.metrics = metrics
        
        # Create mappings for discrete features
        self.meaning_map = {m: i for i, m in enumerate(sorted(config.ALL_MEANINGS))}
        self.utterance_map = {u: i for i, u in enumerate(sorted(config.ALL_UTTERANCES))}
        
        state_dim = 1 + (3 * len(self.utterance_map)) + len(self.meaning_map) + len(self.meaning_map)
        action_dim = len(self.utterance_map)
        
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.DQN_LEARNING_RATE)
        self.memory = ReplayBuffer(config.DQN_MEMORY_SIZE)
        
        self.pragmatic_calculator = PragmaticRewardCalculator()

    def select_action(self, state: DialogueState, steps_done: int) -> (str, torch.Tensor):
        """
        Selects an action using an epsilon-greedy policy.
        """
        sample = random.random()
        eps_threshold = config.DQN_EPS_END + (config.DQN_EPS_START - config.DQN_EPS_END) * \
            np.exp(-1. * steps_done / config.DQN_EPS_DECAY)
        
        utterance_list = sorted(self.utterance_map.keys())
        
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = featurize_state(state, self.meaning_map, self.utterance_map)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].view(1, 1)
                return utterance_list[action_idx.item()], action_idx
        else:
            action_idx = random.randrange(len(utterance_list))
            return utterance_list[action_idx], torch.tensor([[action_idx]], device=device, dtype=torch.long)

    def optimize_model(self):
        """
        Performs one step of optimization on the policy network.
        """
        if len(self.memory) < config.DQN_BATCH_SIZE:
            return

        transitions = self.memory.sample(config.DQN_BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(config.DQN_BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
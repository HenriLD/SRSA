"""
solvers.py - The Outer Loop

Its job is to solve the main recursive Bellman Equation to determine the long-term, strategic value.
"""
import itertools
import numpy as np
from tqdm import tqdm
import config
from state import DialogueState
from pragmatics import PragmaticRewardCalculator
from collections import deque

class TabularBellmanSolver:
    """
    Implements a solver for the strategic component using backward induction.
    It solves for the Q-values for all state-action pairs, starting
    from the final turn and working backwards. This is an "offline" process.
    """
    def __init__(self, reward_calculator: PragmaticRewardCalculator, metrics: 'MetricsTracker'):
        self.reward_calculator = reward_calculator
        self.metrics = metrics
        self.q_table = {}  # Q(s,a) lookup: (turn, state) -> {action: value}
        self.v_table = {}  # V(s) lookup: (turn, state) -> value
        self.pragmatic_cache = {} # (turn, state) -> (rewards, listener_model)

    def _get_next_state(self, s_t: DialogueState, u_t: str, listener_model: dict) -> DialogueState:
        """Calculates the deterministic next state given the current state and an action."""
        # Determine next speaker/listener
        next_speaker_id = s_t.listener_id
        next_listener_id = s_t.speaker_id

        # The new belief is the listener's posterior from the pragmatic model
        posterior = listener_model[u_t]

        # Apply belief decay
        num_meanings = len(config.ALL_MEANINGS)
        decayed_belief = {
            m: (1 - config.BELIEF_DECAY_DELTA) * prob + config.BELIEF_DECAY_DELTA / num_meanings
            for m, prob in posterior.items()
        }

        # The current listener's private meaning is the next speaker's private meaning
        next_speaker_private_meaning = config.AGENT_PRIVATE_MEANINGS[next_speaker_id]

        s_t_plus_1 = DialogueState(
            turn_index=s_t.turn_index + 1,
            dialogue_history=s_t.dialogue_history + (u_t,),
            speaker_id=next_speaker_id,
            listener_id=next_listener_id,
            speaker_private_meaning=next_speaker_private_meaning,
            speaker_belief_of_listener=frozenset(decayed_belief.items())
        )
        return s_t_plus_1

    def solve_for_policy(self):
        """
        Pre-computes the Q-function for all reachable states using a two-phase process:
        1. A forward pass to discover all states reachable from the initial state.
        2. A backward induction pass over only those reachable states to solve for values.
        """
        print("Phase 1: Discovering reachable state space...")
        initial_belief = frozenset(
            {m: 1.0 / len(config.ALL_MEANINGS) for m in config.ALL_MEANINGS}.items()
        )
        initial_speaker_id = 'A'
        initial_listener_id = 'B'

        initial_state = DialogueState(
            turn_index=0,
            dialogue_history=(),
            speaker_id=initial_speaker_id,
            listener_id=initial_listener_id,
            speaker_private_meaning=config.AGENT_PRIVATE_MEANINGS[initial_speaker_id],
            speaker_belief_of_listener=initial_belief
        )

        reachable_states = {t: set() for t in range(config.HORIZON + 1)}
        queue = deque([initial_state])
        visited_states = {initial_state}

        pbar = tqdm(desc="Discovering States")
        while queue:
            s_t = queue.popleft()
            pbar.update(1)

            reachable_states[s_t.turn_index].add(s_t)

            if s_t.turn_index >= config.HORIZON:
                continue

            # We need the listener model to find the next state's belief
            _, listener_model = self.reward_calculator.calculate_rewards_and_listener_model(s_t)

            speaker_utterances = config.AGENT_UTTERANCES.get(s_t.speaker_id, config.ALL_UTTERANCES)
            for u_t in speaker_utterances:
                s_t_plus_1 = self._get_next_state(s_t, u_t, listener_model)
                if s_t_plus_1 not in visited_states:
                    visited_states.add(s_t_plus_1)
                    queue.append(s_t_plus_1)
        pbar.close()

        print("Phase 2: Backward induction over reachable states...")
        # Initialize V for the terminal states at the horizon (value is 0)
        for s_H in reachable_states.get(config.HORIZON, []):
            belief_dict = s_H.get_belief_dict()
            
            # The speaker in s_H was the listener in the final communicative turn (H-1).
            # We care about their final belief state.
            # The 'listener_id' in s_H refers to the speaker of turn H-1, who had the goal.
            previous_speaker_id = s_H.listener_id
            previous_speaker_goal = config.AGENT_PRIVATE_MEANINGS[previous_speaker_id]
            
            # The 'speaker_id' in s_H was the listener of turn H-1. We need their true state.
            final_listener_id = s_H.speaker_id
            final_listener_true_state = config.AGENT_PRIVATE_MEANINGS[final_listener_id]

            goal_mappings = config.GOAL_ACHIEVEMENT_MAPPINGS.get(previous_speaker_goal)
            
            terminal_reward = 0.0
            if goal_mappings and final_listener_true_state in goal_mappings:
                # The single correct meaning given the listener's true, private state
                correct_terminal_meaning = goal_mappings[final_listener_true_state]
                
                # The reward is the log of the listener's final belief in that correct meaning.
                belief_in_correct_meaning = belief_dict.get(correct_terminal_meaning, 1e-9) # Use epsilon for log(0)
                terminal_reward = np.log(belief_in_correct_meaning)
            else:
                # Fallback to original terminal reward for non-goal-oriented scenarios
                true_meaning_of_previous_speaker = config.AGENT_PRIVATE_MEANINGS[s_H.listener_id]
                belief_in_correct_meaning = belief_dict.get(true_meaning_of_previous_speaker, 1e-9)
                terminal_reward = np.log(belief_in_correct_meaning)

            self.v_table[(config.HORIZON, s_H)] = terminal_reward * config.FINAL_REWARD_SCALAR

        for t in tqdm(range(config.HORIZON - 1, -1, -1), desc="Solving Turns"):
            for s_t in reachable_states[t]:
                # Get immediate rewards R_t and cache them along with the listener model
                rewards, listener_model = self.reward_calculator.calculate_rewards_and_listener_model(s_t)
                self.pragmatic_cache[(t, s_t)] = (rewards, listener_model)

                q_values_s_t = {}
                speaker_utterances = config.AGENT_UTTERANCES.get(s_t.speaker_id, config.ALL_UTTERANCES)
                for u_t in speaker_utterances:
                    R_t = rewards[u_t]

                    # Find the deterministic next state s_{t+1}
                    s_t_plus_1 = self._get_next_state(s_t, u_t, listener_model)

                    # V_{t+1} is known because we are iterating backwards
                    V_t_plus_1 = self.v_table.get((t + 1, s_t_plus_1), 0.0)

                    # Apply the Bellman Equation: Q_t = R_t + gamma * V_{t+1}
                    q_values_s_t[u_t] = R_t + config.GAMMA * V_t_plus_1

                self.q_table[(t, s_t)] = q_values_s_t
                self.metrics.log_q_values(t, s_t, q_values_s_t)

                # Calculate V_t(s_t) from Q_t(s_t, u) using the log-sum-exp formulation
                q_vals = np.array(list(q_values_s_t.values()))
                if len(q_vals) == 0:
                    self.v_table[(t, s_t)] = 0.0
                else:
                    # V_t(s_t) = alpha * log sum_u exp(Q_t(s_t, u) / alpha)
                    scaled_q_vals = q_vals / config.ALPHA
                    max_q = np.max(scaled_q_vals)
                    log_sum_exp = max_q + np.log(np.sum(np.exp(scaled_q_vals - max_q)))
                    self.v_table[(t, s_t)] = config.ALPHA * log_sum_exp

        print("Offline solving complete.")
        return self.q_table, self.pragmatic_cache
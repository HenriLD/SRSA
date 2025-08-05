"""
solvers.py - The Outer Loop

This module implements the "Outer Loop" strategist. Its job is to solve the main
recursive Bellman Equation to determine the long-term, strategic value.
"""
import itertools
import numpy as np
from tqdm import tqdm
import config
from state import DialogueState
from pragmatics import PragmaticRewardCalculator

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
        # Cache for pragmatic computations to be used by the DialogueManager
        self.pragmatic_cache = {} # (turn, state) -> (rewards, listener_model)

    def _generate_all_possible_states(self, turn: int):
        """Generates all reachable states for a given turn."""
        # This is a simplified generator. A full implementation would need to
        # consider the constraints of history and belief evolution.
        # For now, we iterate over all combinations of core state components.
        possible_states = []
        agent_ids = list(config.AGENT_PRIVATE_MEANINGS.keys())
        # Placeholder for belief states. In a real scenario, these would be
        # reachable beliefs, but here we simplify to a few canonical ones.
        # Let's create uniform and one-hot beliefs.
        uniform_belief = {m: 1.0/len(config.ALL_MEANINGS) for m in config.ALL_MEANINGS}
        one_hot_beliefs = []
        for m_true in config.ALL_MEANINGS:
            b = {m: 0.0 for m in config.ALL_MEANINGS}
            b[m_true] = 1.0
            one_hot_beliefs.append(b)
        
        possible_beliefs = [uniform_belief] + one_hot_beliefs

        for speaker_id, listener_id in itertools.permutations(agent_ids, 2):
            speaker_meaning = config.AGENT_PRIVATE_MEANINGS[speaker_id]
            for belief_dict in possible_beliefs:
                # State must be hashable
                belief_frozenset = frozenset(belief_dict.items())
                # History is simplified to be empty for this offline solver
                state = DialogueState(
                    turn_index=turn,
                    dialogue_history=(),
                    speaker_id=speaker_id,
                    listener_id=listener_id,
                    speaker_private_meaning=speaker_meaning,
                    speaker_belief_of_listener=belief_frozenset
                )
                possible_states.append(state)
        return possible_states


    def solve_for_policy(self):
        """
        Pre-computes and stores the entire Q-function using backward induction.
        """
        print("Starting offline solving phase (Backward Induction)...")
        # Iterate backwards from the turn before the last (H-1) down to 0
        for t in tqdm(range(config.HORIZON - 1, -1, -1), desc="Solving Turns"):
            # Generate all possible states for the current turn `t`
            possible_states = self._generate_all_possible_states(t)

            for s_t in possible_states:
                # 1. Get immediate rewards R_t(s_t, u_t)
                rewards, listener_model = self.reward_calculator.calculate_rewards_and_listener_model(s_t)
                self.pragmatic_cache[(t, s_t)] = (rewards, listener_model)

                q_values_s_t = {}
                for u_t in config.ALL_UTTERANCES:
                    R_t = rewards[u_t]
                    
                    # 2. Calculate expected future value E[V_{t+1}]
                    # If we are at the last turn, future value is 0.
                    if t == config.HORIZON - 1:
                        E_V_t_plus_1 = 0.0
                    else:
                        # This part is complex as it requires summing over all possible next states s_{t+1}
                        # For this implementation, we simplify by assuming a single deterministic next state
                        # transition for the purpose of the Bellman update.
                        # A full implementation requires a transition model P(s_{t+1}|s_t, u_t).
                        # Let's find the V-value for a plausible next state.
                        
                        # The next state is from the listener's perspective
                        next_speaker_id = s_t.listener_id
                        next_listener_id = s_t.speaker_id
                        
                        # The listener's new belief is the posterior from the pragmatic model
                        posterior = listener_model[u_t]
                        
                        # Apply belief decay
                        decayed_belief = {m: (1 - config.BELIEF_DECAY_DELTA) * posterior[m] +
                                             config.BELIEF_DECAY_DELTA / len(config.ALL_MEANINGS)
                                          for m in config.ALL_MEANINGS}
                        
                        s_t_plus_1 = DialogueState(
                            turn_index=t + 1,
                            dialogue_history=(), # Simplified
                            speaker_id=next_speaker_id,
                            listener_id=next_listener_id,
                            speaker_private_meaning=config.AGENT_PRIVATE_MEANINGS[next_speaker_id],
                            speaker_belief_of_listener=frozenset(decayed_belief.items())
                        )
                        
                        # V_{t+1} is known because we are iterating backwards
                        # We use .get(..., 0) in case the exact state was not in our generated set.
                        V_t_plus_1 = self.v_table.get((t + 1, s_t_plus_1), 0.0)
                        E_V_t_plus_1 = V_t_plus_1
                        
                    # 3. Apply the Bellman Equation: Q_t = R_t + gamma * E[V_{t+1}]
                    q_values_s_t[u_t] = R_t + config.GAMMA * E_V_t_plus_1
                
                self.q_table[(t, s_t)] = q_values_s_t
                self.metrics.log_q_values(t, s_t, q_values_s_t)

                # 4. Calculate V_t(s_t) from Q_t(s_t, u)
                # V_t(s_t) = alpha * log sum_u exp(Q_t(s_t, u) / alpha)
                q_vals = np.array(list(q_values_s_t.values()))
                log_sum_exp = np.log(np.sum(np.exp(q_vals / config.ALPHA)))
                self.v_table[(t, s_t)] = config.ALPHA * log_sum_exp
        
        print("Offline solving complete.")
        return self.q_table, self.pragmatic_cache
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
        print("Starting offline solving phase...")

        # --- Phase 1: Forward pass to discover all reachable states ---
        print("Phase 1: Discovering reachable state space...")
        initial_belief = frozenset(
            {m: 1.0 / len(config.ALL_MEANINGS) for m in config.ALL_MEANINGS}.items()
        )
        # The DialogueManager starts with a specific agent, so we match that.
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
        reachable_states[0] = {initial_state}

        for t in tqdm(range(config.HORIZON), desc="Discovering States"):
            for s_t in reachable_states[t]:
                # We need the listener model to find the next state's belief
                _, listener_model = self.reward_calculator.calculate_rewards_and_listener_model(s_t)
                for u_t in config.ALL_UTTERANCES:
                    s_t_plus_1 = self._get_next_state(s_t, u_t, listener_model)
                    if s_t_plus_1.turn_index <= config.HORIZON:
                        reachable_states[s_t_plus_1.turn_index].add(s_t_plus_1)

        # --- Phase 2: Backward induction over the discovered reachable states ---
        print("Phase 2: Backward induction over reachable states...")
        # Initialize V for the terminal states at the horizon (value is 0)
        for s_H in reachable_states.get(config.HORIZON, []):
            self.v_table[(config.HORIZON, s_H)] = 0.0

        for t in tqdm(range(config.HORIZON - 1, -1, -1), desc="Solving Turns"):
            for s_t in reachable_states[t]:
                # 1. Get immediate rewards R_t and cache them along with the listener model
                rewards, listener_model = self.reward_calculator.calculate_rewards_and_listener_model(s_t)
                self.pragmatic_cache[(t, s_t)] = (rewards, listener_model)

                q_values_s_t = {}
                for u_t in config.ALL_UTTERANCES:
                    R_t = rewards[u_t]

                    # 2. Find the deterministic next state s_{t+1}
                    s_t_plus_1 = self._get_next_state(s_t, u_t, listener_model)

                    # 3. V_{t+1} is known because we are iterating backwards
                    V_t_plus_1 = self.v_table.get((t + 1, s_t_plus_1), 0.0)

                    # 4. Apply the Bellman Equation: Q_t = R_t + gamma * V_{t+1}
                    q_values_s_t[u_t] = R_t + config.GAMMA * V_t_plus_1

                self.q_table[(t, s_t)] = q_values_s_t
                self.metrics.log_q_values(t, s_t, q_values_s_t)

                # 5. Calculate V_t(s_t) from Q_t(s_t, u) using the log-sum-exp formulation
                q_vals = np.array(list(q_values_s_t.values()))
                if len(q_vals) == 0:
                    self.v_table[(t, s_t)] = 0.0
                else:
                    # V_t(s_t) = alpha * log sum_u exp(Q_t(s_t, u) / alpha)
                    log_sum_exp = np.log(np.sum(np.exp(q_vals / config.ALPHA)))
                    self.v_table[(t, s_t)] = config.ALPHA * log_sum_exp

        print("Offline solving complete.")
        return self.q_table, self.pragmatic_cache
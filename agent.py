"""
agent.py - The Strategic Agent

This class defines the dialogue agent itself. It encapsulates the agent's private state
and its behavioral logic for both speaking (acting) and listening.
"""
import numpy as np
import config
from state import DialogueState, Utterance, BeliefDistribution

class StrategicAgent:
    """
    Represents a rational, forward-looking dialogue participant. It uses the
    pre-computed Q-table to act strategically and updates its beliefs as a listener.
    """
    def __init__(self, agent_id: str, private_meaning: str, initial_belief: BeliefDistribution, q_table: dict, metrics: 'MetricsTracker'):
        self.id = agent_id
        self.private_meaning = private_meaning
        self.belief_of_other_agent = initial_belief
        self.q_table = q_table
        self.metrics = metrics

    def act(self, state: DialogueState) -> Utterance:
        """
        Selects an optimal communicative action (utterance) when speaking.
        This method is the runtime execution of the optimal speaker policy S*_t.
        """
        # 1. Retrieve pre-computed Q-values for the current state.
        # Use .get() to handle cases where a state might not be in the table due to generation simplification.
        q_values = self.q_table.get((state.turn_index, state), {})
        if not q_values:
            print(f"Warning: State not found in Q-table: {state}. Defaulting to uniform policy.")
            # Fallback to a uniform random policy if state is not found
            return np.random.choice(list(config.ALL_UTTERANCES))

        # 2. Form the optimal strategic policy (softmax over Q-values)
        # S*_t(u_t|s_t) propto exp(Q_t(s_t, u_t) / alpha)
        utterances = list(q_values.keys())
        q_vals = np.array([q_values[u] for u in utterances])
        
        # The rationality parameter ALPHA is the inverse of RL temperature
        policy_probs = np.exp(q_vals / config.ALPHA)
        policy_probs /= np.sum(policy_probs)
        
        policy = {u: p for u, p in zip(utterances, policy_probs)}
        self.metrics.log_policy(state.turn_index, state, policy)

        # 3. Sample a single utterance from this policy distribution
        chosen_utterance = np.random.choice(utterances, p=policy_probs)
        return chosen_utterance

    def listen(self, utterance: Utterance, pragmatic_listener_model: dict, turn: int, speaker_private_meaning: str):
        """
        Interprets an utterance and updates internal belief state about the other agent.
        """
        # 1. Use the provided converged listener model L*_t for Bayesian inference
        # B'_{new}(m|u) = L*_t(m_S | u_t)
        posterior_belief = pragmatic_listener_model[utterance]
        pre_decay_belief = posterior_belief.copy()

        # 2. Apply the belief decay mechanism from Equation 18
        num_meanings = len(config.ALL_MEANINGS)
        final_belief = {
            m: (1 - config.BELIEF_DECAY_DELTA) * prob + config.BELIEF_DECAY_DELTA / num_meanings
            for m, prob in posterior_belief.items()
        }
        
        self.metrics.log_belief_update(
        turn=turn,
        listener_id=self.id,
        pre_decay=pre_decay_belief,
        post_decay=final_belief,
        speaker_private_meaning=speaker_private_meaning
    )

        # 3. Update internal belief state
        self.belief_of_other_agent = final_belief
"""
metrics.py - Metrics and Analysis

To provide a centralized and robust mechanism for logging all relevant values during
a simulation. An instance of this class is a dependency for all major components, enabling
detailed, post-hoc analysis of the agent's internal workings.
"""
import json
from state import DialogueState, AgentID

class MetricsTracker:
    """
    A centralized logger for all simulation data. Its role is to create a
    detailed transcript of not just the conversation, but the entire
    computational process behind it.
    """
    def __init__(self):
        self.log_entries = []

    def _add_entry(self, entry_type: str, data: dict):
        self.log_entries.append({'type': entry_type, **data})

    def log_q_values(self, turn: int, state: DialogueState, q_values: dict):
        """
        Purpose: To log the strategic Q-values, Q_t(s_t, u_t), which represent
        the total expected future value of each action.
        Research Question: How does the agent trade off immediate communicative
        success (high R_t) with long-term strategic advantage (high V_{t+1})?
        """
        self._add_entry('q_values', {
            'turn': turn,
            'state': str(state),
            'q_values': q_values
        })

    def log_policy(self, turn: int, state: DialogueState, policy: dict):
        """
        Purpose: To record the agent's final policy, S*_t(u_t|s_t), a softmax
        distribution over actions.
        Research Question: Under what conditions does the agent develop a policy
        that is intentionally ambiguous (i.e., has high entropy)?
        """
        self._add_entry('policy', {
            'turn': turn,
            'state': str(state),
            'policy': policy
        })

    def log_belief_update(self, turn: int, listener_id: AgentID, pre_decay: dict, post_decay: dict):
        """
        Purpose: To track the listener's belief evolution, specifically
        capturing the state of belief before and after the decay mechanism is applied.
        Research Question: How effectively does belief decay prevent the
        propagation of errors?
        """
        self._add_entry('belief_update', {
            'turn': turn,
            'listener_id': listener_id,
            'pre_decay_belief': pre_decay,
            'post_decay_belief': post_decay
        })

    def log_turn(self, turn: int, speaker_id: AgentID, utterance: str, listener_id: AgentID):
        """Logs the high-level events of a turn."""
        self._add_entry('turn_event', {
            'turn': turn,
            'speaker': speaker_id,
            'utterance': utterance,
            'listener': listener_id
        })


    def save(self, file_path: str):
        """Saves all logged data to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
        print(f"Metrics saved to {file_path}")
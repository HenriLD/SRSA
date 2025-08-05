"""
state.py - Core Data Structures

To define simple, immutable data containers for representing the core concepts of
the model. This ensures data consistency and makes the state hashable, a requirement for
using it as a key in the Q-table.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, FrozenSet

# Type aliases for enhanced readability.
Meaning = Any
Utterance = Any
AgentID = Any
BeliefDistribution = Dict[Meaning, float]

@dataclass(frozen=True)
class DialogueState:
    """
    Represents the complete state s_t from the perspective of the current speaker.

    This object contains all information relevant for the speaker to make an
    optimal decision, as defined in Section 2.3 and 3.1 of the paper. It is
    "frozen" to make it hashable, allowing it to be used as a key in lookup
    tables for Q-values and V-values.

    Attributes:
        turn_index: The current turn t.
        dialogue_history: The public history w_t = (u_1, ..., u_{t-1})
        speaker_id: The identifier of the agent currently acting as speaker.
        listener_id: The identifier of the agent currently listening.
        speaker_private_meaning: The speaker's private information, m_{S_t}.
        speaker_belief_of_listener: The speaker's belief about the listener's
                                    private meaning, B_{S,t}(m_{L_t})
    """
    turn_index: int
    dialogue_history: Tuple[Utterance, ...]
    speaker_id: AgentID
    listener_id: AgentID
    speaker_private_meaning: Meaning
    # Use a frozenset of items for the belief so it's hashable.
    speaker_belief_of_listener: FrozenSet[Tuple[Meaning, float]]

    def get_belief_dict(self) -> BeliefDistribution:
        """Returns the belief distribution as a mutable dictionary."""
        return dict(self.speaker_belief_of_listener)
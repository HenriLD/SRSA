"""
dialogue_manager.py - Simulation Orchestrator

This class acts as the environment or "game master," managing the dialogue flow,
turn-taking, and state transitions.
"""
import config
from state import DialogueState
from agent import StrategicAgent

class DialogueManager:
    """
    Orchestrates the dialogue simulation. It holds the ground truth, manages agent roles,
    and triggers state transitions.
    """
    def __init__(self, agent_A: StrategicAgent, agent_B: StrategicAgent, pragmatic_cache: dict, metrics: 'MetricsTracker'):
        self.agents = {'A': agent_A, 'B': agent_B}
        self.pragmatic_cache = pragmatic_cache
        self.metrics = metrics
        self.dialogue_history = []
        # Agent 'A' starts the conversation by default
        self.turn_order = ['A', 'B'] * (config.HORIZON // 2 + 1)

    def run_dialogue(self):
        """
        Executes the main simulation loop from t=0 to the horizon.
        """
        print("\nStarting online simulation phase...")
        for t in range(config.HORIZON):
            # 1. Determine speaker and listener
            speaker_id = self.turn_order[t]
            listener_id = 'B' if speaker_id == 'A' else 'A'
            speaker = self.agents[speaker_id]
            listener = self.agents[listener_id]

            print(f"\n--- Turn {t}: Speaker={speaker_id}, Listener={listener_id} ---")

            # 2. Construct the DialogueState for the speaker
            current_state = DialogueState(
                turn_index=t,
                dialogue_history=tuple(self.dialogue_history),
                speaker_id=speaker_id,
                listener_id=listener_id,
                speaker_private_meaning=speaker.private_meaning,
                speaker_belief_of_listener=frozenset(speaker.belief_of_other_agent.items())
            )
            print(f"State: {current_state}")
            print(f"Speaker's belief about listener: {speaker.belief_of_other_agent}")

            # 3. Call the speaker's act() method
            utterance = speaker.act(current_state)
            self.metrics.log_turn(t, speaker_id, utterance, listener_id)
            print(f"Speaker '{speaker_id}' says: '{utterance}'")
            
            # 4. Record the utterance in public history
            self.dialogue_history.append(utterance)

            # 5. Retrieve the pre-computed pragmatic listener model
            _, listener_model = self.pragmatic_cache.get(
                (t, current_state),
                (None, None) # Default value
            )
            
            if not listener_model:
                print(f"Warning: Pragmatic model not found for state {current_state}. Listener cannot update beliefs.")
                continue

            # 6. Call the listener's listen() method
            listener.listen(utterance, listener_model, t, speaker.private_meaning)
            print(f"Listener '{listener_id}' updated beliefs to: {listener.belief_of_other_agent}")

        print("\n--- Dialogue Finished ---")
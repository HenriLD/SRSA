"""
metrics.py - Metrics and Analysis

To provide a centralized and robust mechanism for logging all relevant values during
a simulation. An instance of this class is a dependency for all major components, enabling
detailed, post-hoc analysis of the agent's internal workings.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from state import DialogueState, AgentID
import config

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

    def log_belief_update(self, turn: int, listener_id: AgentID, pre_decay: dict, post_decay: dict, speaker_private_meaning: str):
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
            'post_decay_belief': post_decay,
            'speaker_private_meaning': speaker_private_meaning
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
        """Saves all logged data to the specified JSON file path."""
        with open(file_path, 'w') as f:
            json.dump(self.log_entries, f, indent=2)
        print(f"Metrics saved to {file_path}")

    def generate_belief_matrices_from_log(self, log_file_path: str, agent_meanings_config: dict, output_dir: str):
        """
        Parses the log file and generates a single PNG with two side-by-side
        heatmaps, a central utterance list, and highlighted correct meanings.
        """
        with open(log_file_path, 'r') as f:
            logs = json.load(f)

        belief_updates = [log for log in logs if log['type'] == 'belief_update']
        turn_events = {log['turn']: log for log in logs if log['type'] == 'turn_event'}
        
        if not belief_updates:
            print("No belief updates found in the log. Skipping graph generation.")
            return

        print("\n--- Generating Belief Evolution Graph ---")

        # 1. Aggregate data
        agent_ids = list(agent_meanings_config.keys())
        all_meanings = sorted(list(config.ALL_MEANINGS))
        max_turn = max(bu['turn'] for bu in belief_updates)

        belief_history = {agent_id: {} for agent_id in agent_ids}
        for bu in belief_updates:
            belief_history[bu['listener_id']][bu['turn']] = bu['post_decay_belief']

        # 2. Create matrices
        matrices = {agent_id: np.full((max_turn + 1, len(all_meanings)), np.nan) for agent_id in agent_ids}
        for agent_id in agent_ids:
            for turn, beliefs in belief_history[agent_id].items():
                for i, meaning in enumerate(all_meanings):
                    matrices[agent_id][turn, i] = beliefs.get(meaning, 0)

        # 3. Plot matrices with a central column for utterances
        fig, axes = plt.subplots(1, 3, figsize=(len(all_meanings) * 5 + 2, max_turn + 3), 
                                 gridspec_kw={'width_ratios': [len(all_meanings), 1, len(all_meanings)]})

        fig.suptitle('Belief Evolution of Agents Over Turns', fontsize=16)
        
        plot_axes = [axes[0], axes[2]] # Axes for the heatmaps
        utterance_ax = axes[1] # Axis for the central text
        cax = None # To hold the mappable object for the colorbar

        for i, agent_id in enumerate(agent_ids):
            ax = plot_axes[i]
            matrix = matrices[agent_id]
            
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='grey', alpha=0.4)
            cax = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    if not np.isnan(matrix[r, c]):
                        text_color = "black" if matrix[r, c] > 0.6 else "white"
                        ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", color=text_color, fontsize=9)

            ax.set_title(f"Beliefs of Agent {agent_id}")
            ax.set_xticks(np.arange(len(all_meanings)))
            ax.set_xticklabels(all_meanings, rotation=45, ha="right")
            ax.set_yticks(np.arange(max_turn + 1))

            correct_meaning = agent_meanings_config[agent_id]
            for label in ax.get_xticklabels():
                if label.get_text() == correct_meaning:
                    label.set_color('green')
                    label.set_weight('bold')
            
            if i == 0:
                ax.set_ylabel("Turn Number")
            else:
                ax.set_yticklabels([])

        # Configure and populate the central utterance axis
        utterance_ax.set_ylim(max_turn + 0.5, -0.5) # Align with heatmap rows
        utterance_ax.axis('off') # Hide the box, ticks, and labels
        utterance_ax.set_title("Utterance", y=0.97)
        for turn in range(max_turn + 1):
            utterance = turn_events.get(turn, {}).get('utterance', '')
            utterance_ax.text(0.5, turn, f"'{utterance}'", ha='center', va='center', fontsize=10, wrap=True)

        fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2, wspace=0.1)
        cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.7])
        if cax:
            fig.colorbar(cax, cax=cbar_ax, label='Belief Probability')

        output_path = os.path.join(output_dir, "belief_evolution.png")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"Belief evolution graph saved to: {output_path}")
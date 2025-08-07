"""
pragmatics.py - The Inner Loop

This module implements the "Inner Loop" of the hybrid model. Its sole
responsibility is to compute the immediate reward R_t for taking an action u_t in state s_t.
"""
import numpy as np
import config
from state import DialogueState

class PragmaticRewardCalculator:
    """
    This class is responsible for the pragmatic reasoning part of the model.
    It simulates a recursive reasoning process between a speaker and a listener
    for a single turn to determine the communicative value of an utterance.
    """
    def __init__(self):
        self.literal_listener = self._initialize_literal_listener()

    def _initialize_literal_listener(self) -> dict:
        """
        Builds the L_0 listener model programmatically based on substring matching.
        An utterance u refers to any meaning m if u is a substring of m.
        """
        listener = {}
        num_meanings = len(config.ALL_MEANINGS)

        for u in config.ALL_UTTERANCES:
            listener[u] = {m: 0.0 for m in config.ALL_MEANINGS}
            
            # Find all meanings that contain the utterance as a substring
            matching_meanings = [m for m in config.ALL_MEANINGS if u in m]
            
            if matching_meanings:
                # Assign uniform probability over the matching meanings
                prob = 1.0 / len(matching_meanings)
                for m in matching_meanings:
                    listener[u][m] = prob
            else:
                # If no meaning contains the utterance, it's ambiguous across all meanings
                for m in config.ALL_MEANINGS:
                    listener[u][m] = 1.0 / num_meanings
        return listener

    def calculate_rewards_and_listener_model(self, state: DialogueState) -> (dict, dict):
        """
        Primary public method. Orchestrates the pragmatic calculation for a given state.

        Process:
        1. Runs the Alternating Maximization (AM) algorithm to find the
           converged, optimal per-turn speaker (S*_t) and listener (L*_t) models.
        2. Uses the optimal listener model L*_t to calculate the immediate
           reward for every possible utterance according to Equation 11.

        Returns:
            A tuple containing:
            - A dictionary of rewards for all utterances.
            - The converged listener model L*_t.
        """
        # The speaker's belief about the listener's meaning is used to weight the optimization
        listener_meaning_belief = state.get_belief_dict()

        # --- Alternating Maximization (AM) Algorithm ---
        # Initialize listener model (L_t) with the literal listener (L_0)
        listener_model = self.literal_listener

        for _ in range(config.MAX_AM_ITERATIONS):
            # Speaker update (S_t) based on current listener (L_t)
            speaker_model = {}
            for m_s in config.ALL_MEANINGS:
                # Numerator: exp(log(L_t(m_s|u)) - C(u)) = L_t(m_s|u) / exp(C(u))
                utilities = {u: listener_model[u][m_s] / np.exp(config.UTTERANCE_COSTS[u]) for u in config.ALL_UTTERANCES}
                # Normalization
                total_utility = sum(utilities.values())
                speaker_model[m_s] = {u: utilities[u] / total_utility if total_utility > 0 else 0 for u in utilities}

            # Listener update (L_{t+1}) based on new speaker (S_t)
            next_listener_model = {}
            prev_model_for_norm = np.array(list(listener_model[u][m] for u in config.ALL_UTTERANCES for m in config.ALL_MEANINGS))

            for u in config.ALL_UTTERANCES:
                # Numerator: S_t(u|m) * P(m) where P(m) is speaker's belief about listener's meaning
                posteriors = {m: speaker_model[m][u] * listener_meaning_belief[m] for m in config.ALL_MEANINGS}
                # Normalization (Bayes' rule)
                total_prob = sum(posteriors.values())
                next_listener_model[u] = {m: posteriors[m] / total_prob if total_prob > 0 else 0 for m in posteriors}

            # Check for convergence
            next_model_for_norm = np.array(list(next_listener_model[u][m] for u in config.ALL_UTTERANCES for m in config.ALL_MEANINGS))
            if np.linalg.norm(next_model_for_norm - prev_model_for_norm) < config.CONVERGENCE_THRESHOLD:
                break
            listener_model = next_listener_model

        # This is the converged listener model L*_t
        optimal_listener_model = listener_model

        # --- Calculate Immediate Rewards ---
        # R_t(s_t, u_t) = log L*_t(m_{S_t}|u_t) - C(u_t)
        rewards = {}
        speaker_true_meaning = state.speaker_private_meaning
        for u in config.ALL_UTTERANCES:
            prob_correct_interpretation = optimal_listener_model[u][speaker_true_meaning]
            # Add a small epsilon to avoid log(0)
            log_prob = np.log(prob_correct_interpretation + 1e-9)
            cost = config.UTTERANCE_COSTS[u]
            rewards[u] = log_prob - cost

        return rewards, optimal_listener_model
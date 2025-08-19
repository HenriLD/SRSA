"""
config.py - Simulation Configuration

To externalize all constants and hyperparameters, allowing for easy configuration of
different experimental setups without changing the source code. This file acts as the control
panel for the entire simulation.
"""

# --- Model Hyperparameters ---

# The discount factor (gamma) from Equation 9. It determines the agent's foresight.
# A value close to 1 makes the agent highly value future rewards. Setting
# GAMMA=0 recovers the myopic CRSA model as a special case.
GAMMA = 0.8

# The rationality parameter (alpha) from Equation 6, which controls the
# stochasticity of the agent's policy. It is the inverse of the RL temperature
# (alpha_RSA = 1/alpha_RL). A high alpha leads to a more deterministic, "optimal" policy,
# while a low alpha leads to more random exploration.
ALPHA = 5

# The belief decay factor (delta) from Equation 18. It controls the rate at which
# an agent's confidence in its beliefs erodes, mixing the posterior with a
# uniform distribution to mitigate error propagation and promote robustness.
BELIEF_DECAY_DELTA = 0.05

# The dialogue horizon (H), the maximum number of turns. For the tabular
# backward induction solver, this must be a finite number.
HORIZON = 6

# The final states reward scalar, which emphasizes the importance of reaching
# a high-value final state in the dialogue.
FINAL_REWARD_SCALAR = 10.0

# --- Inner Loop (Pragmatics) Parameters ---

# Maximum number of iterations for the Alternating Maximization (AM) algorithm
# used to solve the per-turn CRSA optimization in the Inner Loop.
MAX_AM_ITERATIONS = 20

# The change threshold below which the AM algorithm is considered converged.
CONVERGENCE_THRESHOLD = 1e-3

# --- Environment Definition ---

# For a tractable tabular model, the state and action spaces must be finite and discrete.
ALL_MEANINGS = ('red_square', 'large_blue_square', 'small_red_circle', 'dull_green_circle', 'green_triangle', 'shiny_red_square', 'dull_blue_circle', 'small_shiny_red_square', 'small_dull_blue_circle')
ALL_UTTERANCES = ('red', 'blue', 'square', 'circle', 'green', 'triangle', 'shiny', 'dull', 'large', 'small')

# A simple cost function for utterances. Here we assume no cost.
UTTERANCE_COSTS = {u: 0.0 for u in ALL_UTTERANCES}

# --- Simulation Settings ---

# The true, private meanings for the agents in the simulation.
# AGENT_PRIVATE_MEANINGS = {agent_id: meaning}
AGENT_PRIVATE_MEANINGS = {
    'A': 'small_shiny_red_square',
    'B': 'small_dull_blue_circle',
}

# Options: 'TABULAR' for the original Bellman solver, 'DQN' for the neural network approximator.
SOLVER_TYPE = 'TABULAR'

# --- DQN Hyperparameters ---
DQN_BATCH_SIZE = 128
DQN_MEMORY_SIZE = 10000
DQN_LEARNING_RATE = 1e-4
DQN_EPS_START = 0.9
DQN_EPS_END = 0.05
DQN_EPS_DECAY = 1000
DQN_TARGET_UPDATE_FREQUENCY = 10
DQN_NUM_EPISODES = 500
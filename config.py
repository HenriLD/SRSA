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
GAMMA = 0.9

# The rationality parameter (alpha) from Equation 6, which controls the
# stochasticity of the agent's policy. It is the inverse of the RL temperature
# (alpha_RSA = 1/alpha_RL). A high alpha leads to a more deterministic, "optimal" policy,
# while a low alpha leads to more random exploration.
ALPHA = 1.0

# The belief decay factor (delta) from Equation 18. It controls the rate at which
# an agent's confidence in its beliefs erodes, mixing the posterior with a
# uniform distribution to mitigate error propagation and promote robustness.
BELIEF_DECAY_DELTA = 0.05

# The dialogue horizon (H), the maximum number of turns. For the tabular
# backward induction solver, this must be a finite number.
HORIZON = 5

# --- Inner Loop (Pragmatics) Parameters ---

# Maximum number of iterations for the Alternating Maximization (AM) algorithm
# used to solve the per-turn CRSA optimization in the Inner Loop.
MAX_AM_ITERATIONS = 100

# The change threshold below which the AM algorithm is considered converged.
CONVERGENCE_THRESHOLD = 1e-4

# --- Environment Definition ---

# For a tractable tabular model, the state and action spaces must be finite and discrete.
ALL_MEANINGS = ('m1', 'm2', 'ms3')
ALL_UTTERANCES = ('u1', 'u2', 'u3', 'u4')

# An utterance can be mapped to a single meaning. Utterances not in this
# dictionary are considered semantically ambiguous (i.e., having a uniform
# literal interpretation).
LITERAL_MEANING = {
    'u1': 'm1',
    'u2': 'm2',
    'u3': 'm3',
    # 'u4' is intentionally omitted to be ambiguous.
}

# A simple cost function for utterances. Here we assume no cost.
UTTERANCE_COSTS = {u: 0.0 for u in ALL_UTTERANCES}

# --- Simulation Settings ---

# The true, private meanings for the agents in the simulation.
# AGENT_PRIVATE_MEANINGS = {agent_id: meaning}
AGENT_PRIVATE_MEANINGS = {
    'A': 'm1',
    'B': 'm2',
}
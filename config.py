# --- Scenario Selection ---
# Options: 'ORIGINAL', 'DIRECTIONS'
SCENARIO = 'DIRECTIONS'

# --- Model Hyperparameters ---
GAMMA = 0.95  # Increased to encourage long-term planning
ALPHA = 5
BELIEF_DECAY_DELTA = 0.05
HORIZON = 4 # Shorter horizon for a simpler scenario
FINAL_REWARD_SCALAR = 20.0 # Increased to emphasize goal completion

# --- Inner Loop (Pragmatics) Parameters ---
MAX_AM_ITERATIONS = 20
CONVERGENCE_THRESHOLD = 1e-3

# --- Scenario-Specific Configurations ---

if SCENARIO == 'ORIGINAL':
    # --- Environment Definition (Original) ---
    ALL_MEANINGS = ('red_square', 'large_blue_square', 'small_red_circle', 'dull_green_circle', 'green_triangle', 'shiny_red_square', 'dull_blue_circle', 'small_shiny_red_square', 'small_dull_blue_circle')
    ALL_UTTERANCES = ('red', 'blue', 'square', 'circle', 'green', 'triangle', 'shiny', 'dull', 'large', 'small')
    UTTERANCE_COSTS = {u: 0.0 for u in ALL_UTTERANCES}
    AGENT_PRIVATE_MEANINGS = {
        'A': 'small_shiny_red_square',
        'B': 'small_dull_blue_circle',
    }
    LITERAL_LISTENER_MAPPINGS = {}

elif SCENARIO == 'DIRECTIONS':
    # --- Environment Definition (Giving Directions) ---
    # Meta-Meaning: 'get_to_train_station'

    # Sub-Meanings (what can be communicated)
    ALL_MEANINGS = ('directions_for_local', 'directions_for_tourist', 'ask_familiarity', 'is_local', 'is_tourist')

    # Utterances
    ALL_UTTERANCES = ('turn_left_at_courthouse', 'turn_left_on_maple', 'are_you_familiar_with_landmarks', 'yes', 'no')

    # Utterance Costs
    UTTERANCE_COSTS = {
        'turn_left_at_courthouse': 1.0,
        'turn_left_on_maple': 0.5, # More complex utterance
        'are_you_familiar_with_landmarks': 0.9, # Low cost question
        'yes': 0.0,
        'no': 0.2
    }

    # Agent Private Meanings (Listener's internal state)
    # Speaker (A) has a goal, Listener (B) has a state

    LITERAL_LISTENER_MAPPINGS = {
        'turn_left_at_courthouse': ['directions_for_local'],
        'turn_left_on_maple': ['directions_for_tourist'],
        'are_you_familiar_with_landmarks': ['ask_familiarity'],
        'yes': ['is_local'],
        'no': ['is_tourist']
    }

    AGENT_PRIVATE_MEANINGS = {
        'A': 'is_tourist',  # Speaker's meta-meaning
        'B': 'get_to_train_station',       # Listener's true internal state
    }
    # Defines which sub-meaning achieves the goal for a given listener state
    GOAL_ACHIEVEMENT_MAPPINGS = {
        'get_to_train_station': {
            'is_local': 'directions_for_local',
            'is_tourist': 'directions_for_tourist'
        }
    }



SOLVER_TYPE = 'TABULAR'  # Options: 'TABULAR', 'DQN'

# --- DQN Hyperparameters ---
DQN_BATCH_SIZE = 128
DQN_MEMORY_SIZE = 10000
DQN_LEARNING_RATE = 1e-4
DQN_EPS_START = 0.9
DQN_EPS_END = 0.05
DQN_EPS_DECAY = 1000
DQN_TARGET_UPDATE_FREQUENCY = 10
DQN_NUM_EPISODES = 500
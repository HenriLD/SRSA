# main.py

import config
from metrics import MetricsTracker
from pragmatics import PragmaticRewardCalculator
from solvers import TabularBellmanSolver
from agent import StrategicAgent
from dialogue_manager import DialogueManager
import os
import datetime
import torch
from tqdm import tqdm

def run_tabular_simulation(results_dir: str):
    """
    Runs the original simulation using the TabularBellmanSolver.
    """
    metrics_tracker = MetricsTracker()
    pragmatic_calculator = PragmaticRewardCalculator()
    bellman_solver = TabularBellmanSolver(pragmatic_calculator, metrics_tracker)

    print("--- Running Tabular Simulation ---")
    # Offline Solving Phase
    q_table, pragmatic_cache = bellman_solver.solve_for_policy()

    # Online Simulation Phase
    initial_uniform_belief = {m: 1.0 / len(config.ALL_MEANINGS) for m in config.ALL_MEANINGS}
    agent_A = StrategicAgent('A', config.AGENT_PRIVATE_MEANINGS['A'], initial_uniform_belief.copy(), q_table, metrics_tracker)
    agent_B = StrategicAgent('B', config.AGENT_PRIVATE_MEANINGS['B'], initial_uniform_belief.copy(), q_table, metrics_tracker)
    
    dialogue_manager = DialogueManager(agent_A, agent_B, pragmatic_cache, metrics_tracker)
    dialogue_manager.run_dialogue()

    # Analysis
    log_file_path = os.path.join(results_dir, "tabular_simulation_log.json")
    metrics_tracker.save(log_file_path)
    metrics_tracker.generate_belief_matrices_from_log(log_file_path, config.AGENT_PRIVATE_MEANINGS, results_dir)

def run_dqn_training(results_dir: str):
    """
    Runs the training loop for the DQN-based agent.
    """
    # Import DQN classes locally to avoid dependency if not used
    from dqn_solver import DQNSolver, featurize_state
    from state import DialogueState
    
    print("--- Running DQN Training ---")
    metrics_tracker = MetricsTracker()
    solver = DQNSolver(metrics_tracker)
    
    steps_done = 0
    
    for i_episode in tqdm(range(config.DQN_NUM_EPISODES), desc="Training Episodes"):
        # Initialize the environment for each episode
        initial_belief = frozenset({m: 1.0 / len(config.ALL_MEANINGS) for m in config.ALL_MEANINGS}.items())
        state = DialogueState(0, (), 'A', 'B', config.AGENT_PRIVATE_MEANINGS['A'], initial_belief)

        for t in range(config.HORIZON):
            state_tensor = featurize_state(state, solver.meaning_map, solver.utterance_map)
            utterance, action_tensor = solver.select_action(state, steps_done)
            steps_done += 1

            rewards, listener_model = solver.pragmatic_calculator.calculate_rewards_and_listener_model(state)
            immediate_reward = rewards[utterance]
            reward_tensor = torch.tensor([immediate_reward], device=solver.device)
            
            # Determine next state
            next_state = None
            if t < config.HORIZON - 1:
                next_speaker_id = state.listener_id
                posterior = listener_model[utterance]
                next_belief = frozenset({
                    m: (1 - config.BELIEF_DECAY_DELTA) * p + config.BELIEF_DECAY_DELTA / len(config.ALL_MEANINGS) 
                    for m, p in posterior.items()
                }.items())
                
                next_state = DialogueState(
                    turn_index=t + 1,
                    dialogue_history=state.dialogue_history + (utterance,),
                    speaker_id=next_speaker_id,
                    listener_id=state.speaker_id,
                    speaker_private_meaning=config.AGENT_PRIVATE_MEANINGS[next_speaker_id],
                    speaker_belief_of_listener=next_belief
                )
                next_state_tensor = featurize_state(next_state, solver.meaning_map, solver.utterance_map)
            else:
                next_state_tensor = None

            solver.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
            
            state = next_state
            if state is None:
                break
            
            solver.optimize_model()

        if i_episode % config.DQN_TARGET_UPDATE_FREQUENCY == 0:
            solver.update_target_net()
    
    print("DQN Training Complete.")
    model_path = os.path.join(results_dir, "dqn_policy_net.pth")
    torch.save(solver.policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main():
    """
    Main executable for the project.
    """
    print("--- Configuration ---")
    print(f"SOLVER_TYPE: {config.SOLVER_TYPE}")
    print(f"GAMMA: {config.GAMMA}, ALPHA: {config.ALPHA}, HORIZON: {config.HORIZON}")
    print("-" * 21)

    # Setup results directory
    if not os.path.exists('results'):
        os.makedirs('results')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join('results', f"{timestamp}_{config.SOLVER_TYPE}")
    os.makedirs(results_dir)
    print(f"Results will be saved in: {results_dir}")

    # --- Route to the selected solver ---
    if config.SOLVER_TYPE == 'TABULAR':
        run_tabular_simulation(results_dir)
    elif config.SOLVER_TYPE == 'DQN':
        run_dqn_training(results_dir)
    else:
        print(f"Error: Unknown SOLVER_TYPE '{config.SOLVER_TYPE}' in config.py")

if __name__ == "__main__":
    main()
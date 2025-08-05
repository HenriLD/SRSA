"""
main.py - Main Entry Point

This is the top-level script that wires all the other components together and
executes a simulation run, delineating the offline "solving" phase and the
online "simulation" phase.
"""
import config
from metrics import MetricsTracker
from pragmatics import PragmaticRewardCalculator
from solvers import TabularBellmanSolver
from agent import StrategicAgent
from dialogue_manager import DialogueManager

def main():
    """
    Serves as the main executable for the project.
    """
    # 1. Setup: Initialize metrics and load parameters from config
    metrics_tracker = MetricsTracker()
    print("--- Configuration ---")
    print(f"GAMMA: {config.GAMMA}, ALPHA: {config.ALPHA}, HORIZON: {config.HORIZON}")
    print(f"AGENT_PRIVATE_MEANINGS: {config.AGENT_PRIVATE_MEANINGS}")
    print("-" * 21)

    # 2. Instantiation: Create computational modules
    pragmatic_calculator = PragmaticRewardCalculator()
    bellman_solver = TabularBellmanSolver(pragmatic_calculator, metrics_tracker)

    # 3. Offline Solving Phase: Pre-calculate the Q-function
    q_table, pragmatic_cache = bellman_solver.solve_for_policy()

    # 4. Online Simulation Phase
    # a. Initialize agents with their private info and the shared policy (Q-table)
    initial_uniform_belief = {m: 1.0 / len(config.ALL_MEANINGS) for m in config.ALL_MEANINGS}

    agent_A = StrategicAgent(
        agent_id='A',
        private_meaning=config.AGENT_PRIVATE_MEANINGS['A'],
        initial_belief=initial_uniform_belief.copy(),
        q_table=q_table,
        metrics=metrics_tracker
    )
    agent_B = StrategicAgent(
        agent_id='B',
        private_meaning=config.AGENT_PRIVATE_MEANINGS['B'],
        initial_belief=initial_uniform_belief.copy(),
        q_table=q_table,
        metrics=metrics_tracker
    )
    
    # b. Initialize the Dialogue Manager
    dialogue_manager = DialogueManager(agent_A, agent_B, pragmatic_cache, metrics_tracker)

    # c. Run the turn-by-turn simulation
    dialogue_manager.run_dialogue()

    # 5. Analysis: Save the comprehensive log file
    metrics_tracker.save("simulation_log.json")

if __name__ == "__main__":
    main()
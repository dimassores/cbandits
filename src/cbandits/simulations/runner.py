# src/simulations/runner.py

import numpy as np
import pandas as pd
import time
import os
import datetime

# Import algorithms
from ..algorithms import UCB_B1, UCB_M1, UCB_B2, UCB_B2C
# Import environment
from ..environments import GeneralCostRewardEnvironment
# Import configuration
from config.simulation_config import (
    NUM_RUNS, BUDGETS, ARM_CONFIGS, ALGORITHM_PARAMS, MIN_EXPECTED_COST
)

def run_simulations():
    """
    Main function to run the bandit simulations.
    It iterates through defined algorithms, budgets, and simulation runs,
    collecting reward and regret data.
    """
    print("Starting budget-constrained bandit simulations...")
    print(f"Number of simulation runs per algorithm/budget: {NUM_RUNS}")
    print(f"Budgets to simulate: {BUDGETS}")
    print(f"Number of arms: {len(ARM_CONFIGS)}")

    results = [] # To store all simulation data

    # Define algorithms to run
    # Map algorithm names to their classes
    algorithms_to_run = {
        "UCB-B1": UCB_B1,
        "UCB-M1": UCB_M1,
        "UCB-B2": UCB_B2,
        "UCB-B2C": UCB_B2C,
    }

    num_arms = len(ARM_CONFIGS)

    for algo_name, AlgoClass in algorithms_to_run.items():
        print(f"\n--- Running Algorithm: {algo_name} ---")
        
        # Instantiate the environment (can be done once per algorithm type if its state is reset)
        # We pass a seed to the environment for reproducibility of the arm samples across runs
        # if a fixed seed is desired for the entire simulation.
        # However, for Monte Carlo runs, usually the *algorithm's* internal state is reset,
        # but the environment can generate new random samples for each pull.
        # For full run-to-run reproducibility of Monte Carlo, pass an outer seed to rng.
        
        # We will use a unique seed for each simulation run to ensure independent trials.
        # The environment constructor uses np.random.default_rng(seed).
        
        for budget in BUDGETS:
            print(f"  Simulating with Budget (B) = {budget}")
            
            cumulative_rewards_per_run = []
            cumulative_regrets_per_run = []

            for run_idx in range(NUM_RUNS):
                # Initialize environment for each run to ensure fresh starts and unique random seeds
                # for true randomness across Monte Carlo runs.
                # If you want the *entire* sequence of runs to be reproducible, manage outer seeds.
                # For now, let's let RNG manage its own internal state across runs if no seed is passed
                # This leads to different random sequences each time runner.py is executed,
                # but within a single execution, each of the NUM_RUNS will be independent.
                env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=ARM_CONFIGS)
                
                # The optimal static policy pulls k* until the budget is depleted. # 
                # Its total reward is roughly r* * B + O(1). # 
                # The total number of pulls for the optimal static policy is N_pi*(B) = inf {n : S_n > B}. # 
                # The expected total reward for optimal static policy is E[REW_pi*(B)] ~ r* * B.
                optimal_static_reward_expected = env.get_optimal_reward_rate() * budget

                # Initialize algorithm for each run
                # The algorithm class handles its own internal state reset
                algorithm = AlgoClass(num_arms=num_arms, arm_configs=ARM_CONFIGS, algorithm_params=ALGORITHM_PARAMS.get(algo_name, {}))
                algorithm.reset() # Ensure algorithm state is clean

                current_total_cost = 0.0
                current_total_reward = 0.0
                epoch = 0 # Represents 'n' in the paper

                # Continue pulling arms until the budget is depleted
                # We assume the reward corresponding to the final epoch during which
                # the budget is depleted is gathered by the controller. # 
                while current_total_cost <= budget:
                    epoch += 1
                    
                    # Select an arm
                    chosen_arm = algorithm.select_arm(current_total_cost, epoch)
                    
                    # Pull the arm from the environment
                    cost, reward = env.pull_arm(chosen_arm)

                    # Update algorithm's state with observed cost and reward
                    algorithm.update_state(chosen_arm, cost, reward)

                    current_total_cost += cost
                    current_total_reward += reward
                
                # After the loop, the budget is exceeded. The last reward is included.
                # Record total reward for this run
                cumulative_rewards_per_run.append(current_total_reward)
                
                # Calculate regret for this run.
                # Regret_pi(B) = E[REW_opt(B)] - E[REW_pi(B)]. # 
                # We approximate E[REW_opt(B)] with the reward of the optimal static policy.
                # The optimality gap for pi* is O(1). # 
                regret_for_run = optimal_static_reward_expected - current_total_reward
                cumulative_regrets_per_run.append(regret_for_run)

            # Calculate average and standard deviation over NUM_RUNS for this algorithm and budget
            avg_reward = np.mean(cumulative_rewards_per_run)
            std_reward = np.std(cumulative_rewards_per_run)
            avg_regret = np.mean(cumulative_regrets_per_run)
            std_regret = np.std(cumulative_regrets_per_run)

            results.append({
                "algorithm": algo_name,
                "budget": budget,
                "avg_reward": avg_reward,
                "std_reward": std_reward,
                "avg_regret": avg_regret,
                "std_regret": std_regret,
                "optimal_static_reward_expected": optimal_static_reward_expected,
            })
            print(f"    Avg Reward: {avg_reward:.2f}, Avg Regret: {avg_regret:.2f}")

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"simulation_results_{timestamp}.csv")
    
    results_df.to_csv(output_filename, index=False)
    print(f"\nSimulations completed. Results saved to {output_filename}")

if __name__ == "__main__":
    run_simulations()
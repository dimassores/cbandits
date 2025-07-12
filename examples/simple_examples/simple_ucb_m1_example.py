#!/usr/bin/env python3
"""
Simple example demonstrating how to use UCB-M1 algorithm
for budget-constrained multi-armed bandits with heavy-tailed distributions.
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import from src
sys.path.append('.')

from src.algorithms import UCB_M1
from src.environments import GeneralCostRewardEnvironment

def simple_ucb_m1_test():
    """
    A simple test of UCB-M1 algorithm with 3 arms.
    UCB-M1 is designed for heavy-tailed cost and reward distributions
    using median-based estimators for robust performance.
    """
    print("=== Simple UCB-M1 Test ===\n")
    
    # Step 1: Define the arms (bandit machines)
    # Each arm has a cost and reward distribution
    # Note: UCB-M1 assumes known second-order moments and handles heavy-tailed distributions
    arm_configs = [
        {
            "name": "Arm 1 (Best)",
            "type": "heavy_tailed",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 3.0,    # Expected reward = 3.0
                "alpha_pareto_X": 3.0,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.5,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 1.0,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.5, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.1,       # Correlation between cost and reward
                "var_X": 0.1,     # Cost variance (for algorithm)
                "var_R": 0.2,     # Reward variance (for algorithm)
                "cov_XR": 0.05,   # Cost-reward covariance (for algorithm)
                "M_X": 10.0,      # Maximum possible cost (for bounds)
                "M_R": 10.0,      # Maximum possible reward (for bounds)
            }
        },
        {
            "name": "Arm 2 (Medium)",
            "type": "heavy_tailed", 
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 2.0,    # Expected reward = 2.0 (lower than Arm 1)
                "alpha_pareto_X": 2.5,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.6,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.7,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.4, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.05,      # Correlation between cost and reward
                "var_X": 0.1,     # Cost variance (for algorithm)
                "var_R": 0.2,     # Reward variance (for algorithm)
                "cov_XR": 0.03,   # Cost-reward covariance (for algorithm)
                "M_X": 10.0,      # Maximum possible cost
                "M_R": 10.0,      # Maximum possible reward
            }
        },
        {
            "name": "Arm 3 (Worst)",
            "type": "heavy_tailed",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 1.0,    # Expected reward = 1.0 (lowest)
                "alpha_pareto_X": 2.0,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.7,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.0,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.3, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.0,       # Correlation between cost and reward
                "var_X": 0.1,     # Cost variance (for algorithm)
                "var_R": 0.2,     # Reward variance (for algorithm)
                "cov_XR": 0.02,   # Cost-reward covariance (for algorithm)
                "M_X": 10.0,      # Maximum possible cost
                "M_R": 10.0,      # Maximum possible reward
            }
        }
    ]
    
    # Step 2: Define UCB-M1 algorithm parameters
    algorithm_params = {
        "alpha": 2.1,        # Confidence bound parameter (controls exploration)
        "b_min_cost": 0.01,  # Small constant for numerical stability
    }
    
    # Step 3: Set up the environment and algorithm
    num_arms = len(arm_configs)
    budget = 1000  # Total budget to spend
    
    # Create environment with a fixed seed for reproducibility
    env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=42)
    
    # Create UCB-M1 algorithm
    algorithm = UCB_M1(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)
    
    # Step 4: Run the simulation
    print(f"Running UCB-M1 with budget = {budget}")
    print(f"Number of arms: {num_arms}")
    print(f"Expected reward rates: {[arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]}")
    print(f"Algorithm parameters: alpha={algorithm_params['alpha']}, b_min_cost={algorithm_params['b_min_cost']}")
    print()
    
    current_total_cost = 0.0
    current_total_reward = 0.0
    epoch = 0
    arm_pulls = [0] * num_arms
    
    print("Epoch | Arm | Cost | Reward | Total Cost | Total Reward | Arm Pulls")
    print("-" * 70)
    
    while current_total_cost <= budget:
        epoch += 1
        
        # Select arm using UCB-M1
        chosen_arm = algorithm.select_arm(current_total_cost, epoch)
        
        # Pull the arm
        cost, reward = env.pull_arm(chosen_arm)
        
        # Update algorithm state
        algorithm.update_state(chosen_arm, cost, reward)
        
        # Update tracking variables
        current_total_cost += cost
        current_total_reward += reward
        arm_pulls[chosen_arm] += 1
        
        # Print progress every 50 epochs
        if epoch % 50 == 0:
            print(f"{epoch:5d} | {chosen_arm:3d} | {cost:4.2f} | {reward:6.2f} | {current_total_cost:10.2f} | {current_total_reward:12.2f} | {arm_pulls}")
    
    # Step 5: Results
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print(f"Total epochs: {epoch}")
    print(f"Final total cost: {current_total_cost:.2f}")
    print(f"Final total reward: {current_total_reward:.2f}")
    print(f"Arm pull counts: {arm_pulls}")
    
    # Calculate optimal reward rate
    optimal_rate = env.get_optimal_reward_rate()
    optimal_reward = optimal_rate * budget
    regret = optimal_reward - current_total_reward
    
    print(f"Optimal reward rate: {optimal_rate:.3f}")
    print(f"Optimal total reward: {optimal_reward:.2f}")
    print(f"Regret: {regret:.2f}")
    
    # Show which arm is optimal
    reward_rates = [arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]
    optimal_arm = np.argmax(reward_rates)
    print(f"Optimal arm: {optimal_arm} (reward rate: {reward_rates[optimal_arm]:.3f})")
    print(f"Arm {optimal_arm} was pulled {arm_pulls[optimal_arm]} times out of {epoch} total pulls")
    
    # Show arm pull percentages
    print(f"Arm pull percentages:")
    for i, pulls in enumerate(arm_pulls):
        percentage = (pulls / epoch) * 100
        print(f"  Arm {i}: {pulls} pulls ({percentage:.1f}%) - Expected rate: {reward_rates[i]:.3f}")

if __name__ == "__main__":
    simple_ucb_m1_test() 
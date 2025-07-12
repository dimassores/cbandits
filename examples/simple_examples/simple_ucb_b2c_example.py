#!/usr/bin/env python3
"""
Simple example demonstrating how to use UCB-B2C algorithm
for budget-constrained multi-armed bandits with unknown variances and correlated cost-reward.
"""

import numpy as np
import sys
import os

# Add the current directory to the path so we can import from src
sys.path.append('.')

from src.algorithms import UCB_B2C
from src.environments import GeneralCostRewardEnvironment

def simple_ucb_b2c_test():
    """
    A simple test of UCB-B2C algorithm with 3 arms.
    UCB-B2C is designed for correlated cost and reward distributions
    with unknown second-order moments.
    """
    print("=== Simple UCB-B2C Test ===\n")
    
    # Step 1: Define the arms (bandit machines)
    # Each arm has a cost and reward distribution
    # Note: UCB-B2C assumes correlated cost and reward (cov_XR != 0.0)
    arm_configs = [
        {
            "name": "Arm 1 (Best)",
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.5, "max_X": 1.5,  # Cost bounds
                "min_R": 1.0, "max_R": 3.0,  # Reward bounds
                "correlation": 0.5,           # Positive correlation
                "mean_X": 1.0,                # Expected cost = 1.0
                "mean_R": 3.0,                # Expected reward = 3.0
                "var_X": 0.083,               # Cost variance (calculated from uniform)
                "var_R": 0.333,               # Reward variance (calculated from uniform)
                "cov_XR": 0.13,               # Cost-reward covariance (positive)
                "M_X": 1.5,                   # Maximum possible cost
                "M_R": 3.0,                   # Maximum possible reward
            }
        },
        {
            "name": "Arm 2 (Medium)",
            "type": "bounded_uniform", 
            "params": {
                "min_X": 0.5, "max_X": 1.5,  # Cost bounds
                "min_R": 0.5, "max_R": 2.5,  # Reward bounds (lower than Arm 1)
                "correlation": 0.3,           # Moderate correlation
                "mean_X": 1.0,                # Expected cost = 1.0
                "mean_R": 2.0,                # Expected reward = 2.0 (lower than Arm 1)
                "var_X": 0.083,               # Cost variance
                "var_R": 0.333,               # Reward variance
                "cov_XR": 0.08,               # Cost-reward covariance (moderate)
                "M_X": 1.5,                   # Maximum possible cost
                "M_R": 2.5,                   # Maximum possible reward
            }
        },
        {
            "name": "Arm 3 (Worst)",
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.5, "max_X": 1.5,  # Cost bounds
                "min_R": 0.0, "max_R": 1.0,  # Reward bounds (lowest)
                "correlation": -0.2,          # Negative correlation
                "mean_X": 1.0,                # Expected cost = 1.0
                "mean_R": 1.0,                # Expected reward = 1.0 (lowest)
                "var_X": 0.083,               # Cost variance
                "var_R": 0.083,               # Reward variance
                "cov_XR": -0.01,              # Cost-reward covariance (negative)
                "M_X": 1.5,                   # Maximum possible cost
                "M_R": 1.0,                   # Maximum possible reward
            }
        }
    ]
    
    # Step 2: Define UCB-B2C algorithm parameters
    algorithm_params = {
        "alpha": 2.1,        # Confidence bound parameter (controls exploration)
        "b_min_cost": 0.01,  # Small constant for numerical stability
        "omega_bar": 2.0,    # Upper bound on correlation parameter
    }
    
    # Step 3: Set up the environment and algorithm
    num_arms = len(arm_configs)
    budget = 1000  # Total budget to spend
    
    # Create environment with a fixed seed for reproducibility
    env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=42)
    
    # Create UCB-B2C algorithm
    algorithm = UCB_B2C(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)
    
    # Step 4: Run the simulation
    print(f"Running UCB-B2C with budget = {budget}")
    print(f"Number of arms: {num_arms}")
    print(f"Expected reward rates: {[arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]}")
    print(f"Algorithm parameters: alpha={algorithm_params['alpha']}, b_min_cost={algorithm_params['b_min_cost']}, omega_bar={algorithm_params['omega_bar']}")
    print()
    
    current_total_cost = 0.0
    current_total_reward = 0.0
    epoch = 0
    arm_pulls = [0] * num_arms
    
    print("Epoch | Arm | Cost | Reward | Total Cost | Total Reward | Arm Pulls")
    print("-" * 70)
    
    while current_total_cost <= budget:
        epoch += 1
        
        # Select arm using UCB-B2C
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
    simple_ucb_b2c_test() 
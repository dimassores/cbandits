#!/usr/bin/env python3
"""
Simple example demonstrating how to use UCB-B1 algorithm
for budget-constrained multi-armed bandits.
"""

import numpy as np

# Import from the cbandits package
from cbandits import UCB_B1, GeneralCostRewardEnvironment

def simple_ucb_b1_test():
    """
    A simple test of UCB-B1 algorithm with 3 arms.
    """
    print("=== Simple UCB-B1 Test ===\n")
    
    # Step 1: Define the arms (bandit machines)
    # Each arm has a cost and reward distribution
    arm_configs = [
        {
            "name": "Arm 1 (Best)",
            "type": "gaussian",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 3.0,    # Expected reward = 3.0
                "var_X": 0.1,     # Cost variance
                "var_R": 0.2,     # Reward variance  
                "cov_XR": 0.05,   # Cost-reward covariance
                "M_X": 5.0,       # Max cost bound
                "M_R": 5.0,       # Max reward bound
            }
        },
        {
            "name": "Arm 2 (Medium)",
            "type": "gaussian", 
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 2.0,    # Expected reward = 2.0 (lower than Arm 1)
                "var_X": 0.1,
                "var_R": 0.2,
                "cov_XR": 0.03,
                "M_X": 5.0,
                "M_R": 5.0,
            }
        },
        {
            "name": "Arm 3 (Worst)",
            "type": "gaussian",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 1.0,    # Expected reward = 1.0 (lowest)
                "var_X": 0.1,
                "var_R": 0.2,
                "cov_XR": 0.02,
                "M_X": 5.0,
                "M_R": 5.0,
            }
        }
    ]
    
    # Step 2: Define UCB-B1 algorithm parameters
    algorithm_params = {
        "alpha": 2.1,        # Confidence bound parameter
        "L": 2,              # Confidence bound parameter  
        "b_min_cost": 0.01,  # Small constant for numerical stability
        "M_X": 5.0,          # Max cost bound
        "M_R": 5.0,          # Max reward bound
    }
    
    # Step 3: Set up the environment and algorithm
    num_arms = len(arm_configs)
    budget = 1000  # Total budget to spend
    
    # Create environment with a fixed seed for reproducibility
    env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=42)
    
    # Create UCB-B1 algorithm
    algorithm = UCB_B1(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)
    
    # Step 4: Run the simulation
    print(f"Running UCB-B1 with budget = {budget}")
    print(f"Number of arms: {num_arms}")
    print(f"Expected reward rates: {[arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]}")
    print()
    
    current_total_cost = 0.0
    current_total_reward = 0.0
    epoch = 0
    arm_pulls = [0] * num_arms
    
    print("Epoch | Arm | Cost | Reward | Total Cost | Total Reward | Arm Pulls")
    print("-" * 70)
    
    while current_total_cost <= budget:
        epoch += 1
        
        # Select arm using UCB-B1
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

def main():
    """Main function for running the UCB-B1 example."""
    simple_ucb_b1_test()

if __name__ == "__main__":
    main() 
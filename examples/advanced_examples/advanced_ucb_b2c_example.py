#!/usr/bin/env python3
"""
Advanced example demonstrating UCB-B2C algorithm for budget-constrained multi-armed bandits.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to the path so we can import from src
sys.path.append('.')

# Import from the cbandits package
from cbandits import UCB_B2C, GeneralCostRewardEnvironment

def run_ucb_b2c_experiment(arm_configs, algorithm_params, budget, num_runs=10, seed=42):
    """
    Run UCB-B2C experiment with given parameters and return results.
    """
    num_arms = len(arm_configs)
    results = []
    
    for run in range(num_runs):
        # Create environment with different seed for each run
        env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=seed+run)
        
        # Create algorithm
        algorithm = UCB_B2C(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)
        algorithm.reset()
        
        current_total_cost = 0.0
        current_total_reward = 0.0
        epoch = 0
        arm_pulls = [0] * num_arms
        
        while current_total_cost <= budget:
            epoch += 1
            chosen_arm = algorithm.select_arm(current_total_cost, epoch)
            cost, reward = env.pull_arm(chosen_arm)
            algorithm.update_state(chosen_arm, cost, reward)
            
            current_total_cost += cost
            current_total_reward += reward
            arm_pulls[chosen_arm] += 1
        
        # Calculate optimal reward for regret computation
        optimal_rate = env.get_optimal_reward_rate()
        optimal_reward = optimal_rate * budget
        regret = optimal_reward - current_total_reward
        
        results.append({
            'run': run,
            'total_reward': current_total_reward,
            'total_cost': current_total_cost,
            'regret': regret,
            'epochs': epoch,
            'arm_pulls': arm_pulls.copy(),
            'optimal_rate': optimal_rate
        })
    
    return results

def compare_parameters():
    """
    Compare UCB-B2C performance with different parameter settings.
    """
    print("=== UCB-B2C Parameter Comparison ===\n")
    
    # Define a challenging environment with 4 arms
    # Note: UCB-B2C uses correlated distributions (cov_XR != 0.0)
    arm_configs = [
        {
            "name": "High Reward, High Cost",
            "type": "bounded_uniform",
            "params": {
                "min_X": 1.5, "max_X": 2.5,  # High cost range
                "min_R": 5.0, "max_R": 7.0,  # High reward range (rate ≈ 3.0)
                "correlation": 0.5,          # Positive correlation
                "mean_X": 2.0,              # Expected cost = 2.0
                "mean_R": 6.0,              # Expected reward = 6.0
                "var_X": 0.083,             # Cost variance
                "var_R": 0.333,             # Reward variance
                "cov_XR": 0.13,             # Cost-reward covariance (positive)
                "M_X": 2.5,                 # Max cost bound
                "M_R": 7.0,                 # Max reward bound
            }
        },
        {
            "name": "Medium Reward, Low Cost", 
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.4, "max_X": 0.6,  # Low cost range
                "min_R": 1.2, "max_R": 1.6,  # Medium reward range (rate ≈ 2.8)
                "correlation": 0.3,          # Moderate correlation
                "mean_X": 0.5,              # Expected cost = 0.5
                "mean_R": 1.4,              # Expected reward = 1.4
                "var_X": 0.003,             # Cost variance
                "var_R": 0.013,             # Reward variance
                "cov_XR": 0.08,             # Cost-reward covariance (moderate)
                "M_X": 0.6,                 # Max cost bound
                "M_R": 1.6,                 # Max reward bound
            }
        },
        {
            "name": "Low Reward, Very Low Cost",
            "type": "bounded_uniform", 
            "params": {
                "min_X": 0.15, "max_X": 0.25,  # Very low cost range
                "min_R": 0.4, "max_R": 0.6,    # Low reward range (rate ≈ 2.5)
                "correlation": 0.1,            # Weak correlation
                "mean_X": 0.2,                # Expected cost = 0.2
                "mean_R": 0.5,                # Expected reward = 0.5
                "var_X": 0.0008,              # Cost variance
                "var_R": 0.003,               # Reward variance
                "cov_XR": 0.01,               # Cost-reward covariance (weak)
                "M_X": 0.25,                  # Max cost bound
                "M_R": 0.6,                   # Max reward bound
            }
        },
        {
            "name": "Trap Arm (High Cost, Low Reward)",
            "type": "bounded_uniform",
            "params": {
                "min_X": 2.5, "max_X": 3.5,  # High cost range
                "min_R": 1.5, "max_R": 2.5,  # Low reward range (rate ≈ 0.67)
                "correlation": -0.2,         # Negative correlation
                "mean_X": 3.0,              # Expected cost = 3.0
                "mean_R": 2.0,              # Expected reward = 2.0
                "var_X": 0.083,             # Cost variance
                "var_R": 0.083,             # Reward variance
                "cov_XR": -0.01,            # Cost-reward covariance (negative)
                "M_X": 3.5,                 # Max cost bound
                "M_R": 2.5,                 # Max reward bound
            }
        }
    ]
    
    # Calculate true reward rates for reference
    reward_rates = [arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]
    optimal_arm = np.argmax(reward_rates)
    
    print(f"Arm configurations:")
    for i, (arm, rate) in enumerate(zip(arm_configs, reward_rates)):
        print(f"  Arm {i}: {arm['name']} (reward rate: {rate:.2f})")
    print(f"Optimal arm: {optimal_arm} with reward rate: {reward_rates[optimal_arm]:.2f}")
    print(f"Note: All arms use correlated cost-reward distributions (cov_XR != 0.0)\n")
    
    # Test different parameter combinations
    parameter_sets = [
        {
            "name": "Conservative (Low Exploration)",
            "params": {"alpha": 1.5, "b_min_cost": 0.01, "omega_bar": 2.0}
        },
        {
            "name": "Default (Balanced)",
            "params": {"alpha": 2.1, "b_min_cost": 0.01, "omega_bar": 2.0}
        },
        {
            "name": "Aggressive (High Exploration)", 
            "params": {"alpha": 3.0, "b_min_cost": 0.01, "omega_bar": 2.0}
        },
        {
            "name": "Very Aggressive",
            "params": {"alpha": 4.0, "b_min_cost": 0.01, "omega_bar": 2.0}
        },
        {
            "name": "Conservative with High Stability",
            "params": {"alpha": 1.5, "b_min_cost": 0.1, "omega_bar": 2.0}
        },
        {
            "name": "Aggressive with Low Stability",
            "params": {"alpha": 3.0, "b_min_cost": 0.001, "omega_bar": 2.0}
        }
    ]
    
    budget = 2000
    num_runs = 20
    
    all_results = []
    
    for param_set in parameter_sets:
        print(f"Testing: {param_set['name']}")
        print(f"  Parameters: alpha={param_set['params']['alpha']}, b_min_cost={param_set['params']['b_min_cost']}, omega_bar={param_set['params']['omega_bar']}")
        
        results = run_ucb_b2c_experiment(
            arm_configs=arm_configs,
            algorithm_params=param_set['params'],
            budget=budget,
            num_runs=num_runs,
            seed=42
        )
        
        # Aggregate results
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_regret = np.mean([r['regret'] for r in results])
        avg_epochs = np.mean([r['epochs'] for r in results])
        
        # Calculate average arm pull distribution
        avg_arm_pulls = np.mean([r['arm_pulls'] for r in results], axis=0)
        
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average regret: {avg_regret:.2f}")
        print(f"  Average epochs: {avg_epochs:.1f}")
        print(f"  Average arm pulls: {avg_arm_pulls}")
        print(f"  Optimal arm pull ratio: {avg_arm_pulls[optimal_arm]/np.sum(avg_arm_pulls):.3f}")
        print()
        
        all_results.append({
            'name': param_set['name'],
            'alpha': param_set['params']['alpha'],
            'b_min_cost': param_set['params']['b_min_cost'],
            'omega_bar': param_set['params']['omega_bar'],
            'avg_reward': avg_reward,
            'avg_regret': avg_regret,
            'avg_epochs': avg_epochs,
            'avg_arm_pulls': avg_arm_pulls,
            'optimal_arm_ratio': avg_arm_pulls[optimal_arm]/np.sum(avg_arm_pulls)
        })
    
    # Create summary table
    print("=== SUMMARY TABLE ===")
    print(f"{'Parameter Set':<30} {'Alpha':<6} {'b_min':<6} {'omega_bar':<10} {'Avg Reward':<12} {'Avg Regret':<12} {'Opt Arm %':<10}")
    print("-" * 90)
    
    for result in all_results:
        print(f"{result['name']:<30} {result['alpha']:<6} {result['b_min_cost']:<6} {result['omega_bar']:<10} {result['avg_reward']:<12.2f} "
              f"{result['avg_regret']:<12.2f} {result['optimal_arm_ratio']*100:<10.1f}%")
    
    # Find best performing parameter set
    best_result = min(all_results, key=lambda x: x['avg_regret'])
    print(f"\nBest performing parameter set: {best_result['name']}")
    print(f"Lowest regret: {best_result['avg_regret']:.2f}")
    
    return all_results

if __name__ == "__main__":
    # Run parameter comparison
    results = compare_parameters() 
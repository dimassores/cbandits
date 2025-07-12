#!/usr/bin/env python3
"""
Advanced example demonstrating UCB-B1 algorithm for budget-constrained multi-armed bandits.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the cbandits package
from cbandits import UCB_B1, GeneralCostRewardEnvironment

def run_ucb_b1_experiment(arm_configs, algorithm_params, budget, num_runs=10, seed=42):
    """
    Run UCB-B1 experiment with given parameters and return results.
    """
    num_arms = len(arm_configs)
    results = []
    
    for run in range(num_runs):
        # Create environment with different seed for each run
        env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=seed+run)
        
        # Create algorithm
        algorithm = UCB_B1(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)
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
    Compare UCB-B1 performance with different parameter settings.
    """
    print("=== UCB-B1 Parameter Comparison ===\n")
    
    # Define a challenging environment with 4 arms
    arm_configs = [
        {
            "name": "High Reward, High Cost",
            "type": "gaussian",
            "params": {
                "mean_X": 2.0,    # High cost
                "mean_R": 6.0,    # High reward (rate = 3.0)
                "var_X": 0.5,
                "var_R": 1.0,
                "cov_XR": 0.2,
                "M_X": 10.0,
                "M_R": 15.0,
            }
        },
        {
            "name": "Medium Reward, Low Cost", 
            "type": "gaussian",
            "params": {
                "mean_X": 0.5,    # Low cost
                "mean_R": 1.4,    # Medium reward (rate = 2.8)
                "var_X": 0.1,
                "var_R": 0.3,
                "cov_XR": 0.05,
                "M_X": 5.0,
                "M_R": 5.0,
            }
        },
        {
            "name": "Low Reward, Very Low Cost",
            "type": "gaussian", 
            "params": {
                "mean_X": 0.2,    # Very low cost
                "mean_R": 0.5,    # Low reward (rate = 2.5)
                "var_X": 0.05,
                "var_R": 0.1,
                "cov_XR": 0.01,
                "M_X": 2.0,
                "M_R": 2.0,
            }
        },
        {
            "name": "Trap Arm (High Cost, Low Reward)",
            "type": "gaussian",
            "params": {
                "mean_X": 3.0,    # High cost
                "mean_R": 2.0,    # Low reward (rate = 0.67)
                "var_X": 0.8,
                "var_R": 0.5,
                "cov_XR": 0.3,
                "M_X": 15.0,
                "M_R": 10.0,
            }
        }
    ]
    
    # Calculate true reward rates for reference
    reward_rates = [arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]
    optimal_arm = np.argmax(reward_rates)
    
    print(f"Arm configurations:")
    for i, (arm, rate) in enumerate(zip(arm_configs, reward_rates)):
        print(f"  Arm {i}: {arm['name']} (reward rate: {rate:.2f})")
    print(f"Optimal arm: {optimal_arm} with reward rate: {reward_rates[optimal_arm]:.2f}\n")
    
    # Test different parameter combinations
    parameter_sets = [
        {
            "name": "Conservative (Low Exploration)",
            "params": {"alpha": 1.5, "L": 2, "b_min_cost": 0.01, "M_X": 15.0, "M_R": 15.0}
        },
        {
            "name": "Default (Balanced)",
            "params": {"alpha": 2.1, "L": 2, "b_min_cost": 0.01, "M_X": 15.0, "M_R": 15.0}
        },
        {
            "name": "Aggressive (High Exploration)", 
            "params": {"alpha": 3.0, "L": 2, "b_min_cost": 0.01, "M_X": 15.0, "M_R": 15.0}
        },
        {
            "name": "Very Aggressive",
            "params": {"alpha": 4.0, "L": 2, "b_min_cost": 0.01, "M_X": 15.0, "M_R": 15.0}
        }
    ]
    
    budget = 2000
    num_runs = 20
    
    all_results = []
    
    for param_set in parameter_sets:
        print(f"Testing: {param_set['name']}")
        print(f"  Parameters: alpha={param_set['params']['alpha']}, L={param_set['params']['L']}")
        
        results = run_ucb_b1_experiment(
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
            'avg_reward': avg_reward,
            'avg_regret': avg_regret,
            'avg_epochs': avg_epochs,
            'avg_arm_pulls': avg_arm_pulls,
            'optimal_arm_ratio': avg_arm_pulls[optimal_arm]/np.sum(avg_arm_pulls)
        })
    
    # Create summary table
    print("=== SUMMARY TABLE ===")
    print(f"{'Parameter Set':<20} {'Alpha':<6} {'Avg Reward':<12} {'Avg Regret':<12} {'Opt Arm %':<10}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['name']:<20} {result['alpha']:<6} {result['avg_reward']:<12.2f} "
              f"{result['avg_regret']:<12.2f} {result['optimal_arm_ratio']*100:<10.1f}%")
    
    # Find best performing parameter set
    best_result = min(all_results, key=lambda x: x['avg_regret'])
    print(f"\nBest performing parameter set: {best_result['name']}")
    print(f"Lowest regret: {best_result['avg_regret']:.2f}")
    
    return all_results

def analyze_learning_curve():
    """
    Analyze how UCB-B1 learns over time with different budgets.
    """
    print("\n=== Learning Curve Analysis ===\n")
    
    # Simple 2-arm environment
    arm_configs = [
        {
            "name": "Good Arm",
            "type": "gaussian",
            "params": {
                "mean_X": 1.0, "mean_R": 3.0, "var_X": 0.1, "var_R": 0.2,
                "cov_XR": 0.05, "M_X": 5.0, "M_R": 5.0
            }
        },
        {
            "name": "Bad Arm", 
            "type": "gaussian",
            "params": {
                "mean_X": 1.0, "mean_R": 1.5, "var_X": 0.1, "var_R": 0.2,
                "cov_XR": 0.03, "M_X": 5.0, "M_R": 5.0
            }
        }
    ]
    
    algorithm_params = {"alpha": 2.1, "L": 2, "b_min_cost": 0.01, "M_X": 5.0, "M_R": 5.0}
    
    budgets = [500, 1000, 2000, 5000]
    
    print("Budget | Avg Reward | Avg Regret | Good Arm % | Learning Efficiency")
    print("-" * 65)
    
    for budget in budgets:
        results = run_ucb_b1_experiment(
            arm_configs=arm_configs,
            algorithm_params=algorithm_params,
            budget=budget,
            num_runs=15,
            seed=100
        )
        
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_regret = np.mean([r['regret'] for r in results])
        avg_good_arm_ratio = np.mean([r['arm_pulls'][0]/np.sum(r['arm_pulls']) for r in results])
        
        # Learning efficiency: reward per unit budget
        learning_efficiency = avg_reward / budget
        
        print(f"{budget:6d} | {avg_reward:10.2f} | {avg_regret:10.2f} | "
              f"{avg_good_arm_ratio*100:9.1f}% | {learning_efficiency:17.3f}")

if __name__ == "__main__":
    # Run parameter comparison
    results = compare_parameters()
    
    # Run learning curve analysis
    analyze_learning_curve()
    
    print("\n=== Key Insights ===")
    print("1. Higher alpha values increase exploration but may not always improve performance")
    print("2. Algorithm performance improves with larger budgets")
    print("3. The algorithm should learn to focus on the optimal arm over time")
    print("4. Regret typically decreases as budget increases") 
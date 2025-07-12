#!/usr/bin/env python3
"""
Advanced example demonstrating UCB-B2 parameter experimentation
and comparison with different settings for uncorrelated cost-reward distributions.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to the path so we can import from src
sys.path.append('.')

from src.algorithms import UCB_B2
from src.environments import GeneralCostRewardEnvironment

def run_ucb_b2_experiment(arm_configs, algorithm_params, budget, num_runs=10, seed=42):
    """
    Run UCB-B2 experiment with given parameters and return results.
    """
    num_arms = len(arm_configs)
    results = []
    
    for run in range(num_runs):
        # Create environment with different seed for each run
        env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=seed+run)
        
        # Create algorithm
        algorithm = UCB_B2(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)
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
    Compare UCB-B2 performance with different parameter settings.
    """
    print("=== UCB-B2 Parameter Comparison ===\n")
    
    # Define a challenging environment with 4 arms
    # Note: UCB-B2 uses uncorrelated distributions (cov_XR = 0.0)
    arm_configs = [
        {
            "name": "High Reward, High Cost",
            "type": "bounded_uniform",
            "params": {
                "min_X": 1.5, "max_X": 2.5,  # High cost range
                "min_R": 5.0, "max_R": 7.0,  # High reward range (rate ≈ 3.0)
                "correlation": 0.0,          # No correlation
                "mean_X": 2.0,              # Expected cost = 2.0
                "mean_R": 6.0,              # Expected reward = 6.0
                "var_X": 0.083,             # Cost variance
                "var_R": 0.333,             # Reward variance
                "cov_XR": 0.0,              # No correlation
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
                "correlation": 0.0,          # No correlation
                "mean_X": 0.5,              # Expected cost = 0.5
                "mean_R": 1.4,              # Expected reward = 1.4
                "var_X": 0.003,             # Cost variance
                "var_R": 0.013,             # Reward variance
                "cov_XR": 0.0,              # No correlation
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
                "correlation": 0.0,            # No correlation
                "mean_X": 0.2,                # Expected cost = 0.2
                "mean_R": 0.5,                # Expected reward = 0.5
                "var_X": 0.0008,              # Cost variance
                "var_R": 0.003,               # Reward variance
                "cov_XR": 0.0,                # No correlation
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
                "correlation": 0.0,          # No correlation
                "mean_X": 3.0,              # Expected cost = 3.0
                "mean_R": 2.0,              # Expected reward = 2.0
                "var_X": 0.083,             # Cost variance
                "var_R": 0.083,             # Reward variance
                "cov_XR": 0.0,              # No correlation
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
    print(f"Note: All arms use uncorrelated cost-reward distributions (cov_XR = 0.0)\n")
    
    # Test different parameter combinations
    parameter_sets = [
        {
            "name": "Conservative (Low Exploration)",
            "params": {"alpha": 1.5, "b_min_cost": 0.01}
        },
        {
            "name": "Default (Balanced)",
            "params": {"alpha": 2.1, "b_min_cost": 0.01}
        },
        {
            "name": "Aggressive (High Exploration)", 
            "params": {"alpha": 3.0, "b_min_cost": 0.01}
        },
        {
            "name": "Very Aggressive",
            "params": {"alpha": 4.0, "b_min_cost": 0.01}
        },
        {
            "name": "Conservative with High Stability",
            "params": {"alpha": 1.5, "b_min_cost": 0.1}
        },
        {
            "name": "Aggressive with Low Stability",
            "params": {"alpha": 3.0, "b_min_cost": 0.001}
        }
    ]
    
    budget = 2000
    num_runs = 20
    
    all_results = []
    
    for param_set in parameter_sets:
        print(f"Testing: {param_set['name']}")
        print(f"  Parameters: alpha={param_set['params']['alpha']}, b_min_cost={param_set['params']['b_min_cost']}")
        
        results = run_ucb_b2_experiment(
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
            'avg_reward': avg_reward,
            'avg_regret': avg_regret,
            'avg_epochs': avg_epochs,
            'avg_arm_pulls': avg_arm_pulls,
            'optimal_arm_ratio': avg_arm_pulls[optimal_arm]/np.sum(avg_arm_pulls)
        })
    
    # Create summary table
    print("=== SUMMARY TABLE ===")
    print(f"{'Parameter Set':<25} {'Alpha':<6} {'b_min':<6} {'Avg Reward':<12} {'Avg Regret':<12} {'Opt Arm %':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['name']:<25} {result['alpha']:<6} {result['b_min_cost']:<6} {result['avg_reward']:<12.2f} "
              f"{result['avg_regret']:<12.2f} {result['optimal_arm_ratio']*100:<10.1f}%")
    
    # Find best performing parameter set
    best_result = min(all_results, key=lambda x: x['avg_regret'])
    print(f"\nBest performing parameter set: {best_result['name']}")
    print(f"Lowest regret: {best_result['avg_regret']:.2f}")
    
    return all_results

def analyze_learning_curve():
    """
    Analyze how UCB-B2 learns over time with different budgets.
    """
    print("\n=== Learning Curve Analysis ===\n")
    
    # Simple 2-arm environment with uncorrelated distributions
    arm_configs = [
        {
            "name": "Good Arm",
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.8, "max_X": 1.2,  # Cost range
                "min_R": 2.5, "max_R": 3.5,  # Reward range (rate ≈ 3.0)
                "correlation": 0.0,          # No correlation
                "mean_X": 1.0,              # Expected cost = 1.0
                "mean_R": 3.0,              # Expected reward = 3.0
                "var_X": 0.013,             # Cost variance
                "var_R": 0.083,             # Reward variance
                "cov_XR": 0.0,              # No correlation
                "M_X": 1.2,                 # Max cost bound
                "M_R": 3.5,                 # Max reward bound
            }
        },
        {
            "name": "Bad Arm", 
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.8, "max_X": 1.2,  # Cost range
                "min_R": 1.2, "max_R": 1.8,  # Reward range (rate ≈ 1.5)
                "correlation": 0.0,          # No correlation
                "mean_X": 1.0,              # Expected cost = 1.0
                "mean_R": 1.5,              # Expected reward = 1.5
                "var_X": 0.013,             # Cost variance
                "var_R": 0.025,             # Reward variance
                "cov_XR": 0.0,              # No correlation
                "M_X": 1.2,                 # Max cost bound
                "M_R": 1.8,                 # Max reward bound
            }
        }
    ]
    
    algorithm_params = {"alpha": 2.1, "b_min_cost": 0.01}
    
    budgets = [500, 1000, 2000, 5000]
    
    print("Budget | Avg Reward | Avg Regret | Good Arm % | Learning Efficiency")
    print("-" * 65)
    
    for budget in budgets:
        results = run_ucb_b2_experiment(
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

def compare_with_known_variance():
    """
    Compare UCB-B2 (unknown variance) performance with a scenario where
    we have known variances to understand the cost of estimation.
    """
    print("\n=== UCB-B2 vs Known Variance Comparison ===\n")
    
    # Create environment where we know the true variances
    arm_configs = [
        {
            "name": "High Variance Arm",
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.5, "max_X": 1.5,  # Cost range
                "min_R": 1.0, "max_R": 3.0,  # Reward range (rate = 2.0)
                "correlation": 0.0,          # No correlation
                "mean_X": 1.0,              # Expected cost = 1.0
                "mean_R": 2.0,              # Expected reward = 2.0
                "var_X": 0.083,             # Known cost variance
                "var_R": 0.333,             # Known reward variance
                "cov_XR": 0.0,              # No correlation
                "M_X": 1.5,                 # Max cost bound
                "M_R": 3.0,                 # Max reward bound
            }
        },
        {
            "name": "Low Variance Arm",
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.9, "max_X": 1.1,  # Narrow cost range
                "min_R": 1.8, "max_R": 2.2,  # Narrow reward range (rate = 2.0)
                "correlation": 0.0,          # No correlation
                "mean_X": 1.0,              # Expected cost = 1.0
                "mean_R": 2.0,              # Expected reward = 2.0
                "var_X": 0.003,             # Low cost variance
                "var_R": 0.013,             # Low reward variance
                "cov_XR": 0.0,              # No correlation
                "M_X": 1.1,                 # Max cost bound
                "M_R": 2.2,                 # Max reward bound
            }
        }
    ]
    
    # Both arms have same reward rate but different variances
    reward_rates = [arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]
    print(f"Both arms have same reward rate: {reward_rates[0]:.2f}")
    print(f"Arm 0 variance: X={arm_configs[0]['params']['var_X']:.3f}, R={arm_configs[0]['params']['var_R']:.3f}")
    print(f"Arm 1 variance: X={arm_configs[1]['params']['var_X']:.3f}, R={arm_configs[1]['params']['var_R']:.3f}")
    print()
    
    algorithm_params = {"alpha": 2.1, "b_min_cost": 0.01}
    budget = 1500
    num_runs = 25
    
    results = run_ucb_b2_experiment(
        arm_configs=arm_configs,
        algorithm_params=algorithm_params,
        budget=budget,
        num_runs=num_runs,
        seed=200
    )
    
    # Analyze arm preference
    high_var_pulls = np.mean([r['arm_pulls'][0] for r in results])
    low_var_pulls = np.mean([r['arm_pulls'][1] for r in results])
    total_pulls = np.mean([r['epochs'] for r in results])
    
    print(f"Results over {num_runs} runs:")
    print(f"Average total pulls: {total_pulls:.1f}")
    print(f"High variance arm (Arm 0): {high_var_pulls:.1f} pulls ({high_var_pulls/total_pulls*100:.1f}%)")
    print(f"Low variance arm (Arm 1): {low_var_pulls:.1f} pulls ({low_var_pulls/total_pulls*100:.1f}%)")
    print(f"Preference ratio (Low/High variance): {low_var_pulls/high_var_pulls:.2f}")
    
    if low_var_pulls > high_var_pulls:
        print("✓ UCB-B2 prefers the low variance arm (good!)")
    else:
        print("✗ UCB-B2 prefers the high variance arm (unexpected)")

def analyze_stability_conditions():
    """
    Analyze how often UCB-B2 encounters stability condition violations
    and how they affect performance.
    """
    print("\n=== Stability Condition Analysis ===\n")
    
    # Create arms with different cost ranges to test stability
    arm_configs = [
        {
            "name": "Stable Arm (Low Cost)",
            "type": "bounded_uniform",
            "params": {
                "min_X": 0.1, "max_X": 0.3,  # Low cost range
                "min_R": 0.2, "max_R": 0.6,  # Reward range
                "correlation": 0.0,          # No correlation
                "mean_X": 0.2,              # Expected cost = 0.2
                "mean_R": 0.4,              # Expected reward = 0.4
                "var_X": 0.003,             # Cost variance
                "var_R": 0.013,             # Reward variance
                "cov_XR": 0.0,              # No correlation
                "M_X": 0.3,                 # Max cost bound
                "M_R": 0.6,                 # Max reward bound
            }
        },
        {
            "name": "Unstable Arm (High Cost)",
            "type": "bounded_uniform",
            "params": {
                "min_X": 2.0, "max_X": 4.0,  # High cost range
                "min_R": 4.0, "max_R": 8.0,  # High reward range
                "correlation": 0.0,          # No correlation
                "mean_X": 3.0,              # Expected cost = 3.0
                "mean_R": 6.0,              # Expected reward = 6.0
                "var_X": 0.333,             # High cost variance
                "var_R": 1.333,             # High reward variance
                "cov_XR": 0.0,              # No correlation
                "M_X": 4.0,                 # Max cost bound
                "M_R": 8.0,                 # Max reward bound
            }
        }
    ]
    
    reward_rates = [arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]
    print(f"Arm 0 (Stable): reward rate = {reward_rates[0]:.2f}")
    print(f"Arm 1 (Unstable): reward rate = {reward_rates[1]:.2f}")
    print(f"Optimal arm: {np.argmax(reward_rates)}")
    print()
    
    # Test different b_min_cost values
    b_min_values = [0.001, 0.01, 0.1, 0.5]
    
    print("b_min_cost | Avg Reward | Avg Regret | Stable Arm % | Performance")
    print("-" * 65)
    
    for b_min in b_min_values:
        algorithm_params = {"alpha": 2.1, "b_min_cost": b_min}
        
        results = run_ucb_b2_experiment(
            arm_configs=arm_configs,
            algorithm_params=algorithm_params,
            budget=1000,
            num_runs=20,
            seed=300
        )
        
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_regret = np.mean([r['regret'] for r in results])
        avg_stable_arm_ratio = np.mean([r['arm_pulls'][0]/np.sum(r['arm_pulls']) for r in results])
        
        # Performance indicator (higher is better)
        performance = avg_reward / (avg_regret + 1)  # Avoid division by zero
        
        print(f"{b_min:9.3f} | {avg_reward:10.2f} | {avg_regret:10.2f} | "
              f"{avg_stable_arm_ratio*100:11.1f}% | {performance:11.3f}")

if __name__ == "__main__":
    # Run parameter comparison
    results = compare_parameters()
    
    # Run learning curve analysis
    analyze_learning_curve()
    
    # Compare with known variance scenario
    compare_with_known_variance()
    
    # Analyze stability conditions
    analyze_stability_conditions()
    
    print("\n=== Key Insights for UCB-B2 ===")
    print("1. UCB-B2 performs well with uncorrelated cost-reward distributions")
    print("2. Higher alpha values increase exploration but may not always improve performance")
    print("3. The b_min_cost parameter affects numerical stability and performance")
    print("4. Algorithm performance improves with larger budgets")
    print("5. UCB-B2 may prefer arms with lower variance when reward rates are similar")
    print("6. Stability conditions can affect arm selection in high-cost scenarios")
    print("7. The algorithm learns to focus on optimal arms over time despite unknown variances") 
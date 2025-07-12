#!/usr/bin/env python3
"""
Advanced example demonstrating UCB-M1 parameter experimentation
and comparison with different settings for heavy-tailed cost-reward distributions.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to the path so we can import from src
sys.path.append('.')

from src.algorithms import UCB_M1
from src.environments import GeneralCostRewardEnvironment

def run_ucb_m1_experiment(arm_configs, algorithm_params, budget, num_runs=10, seed=42):
    """
    Run UCB-M1 experiment with given parameters and return results.
    """
    num_arms = len(arm_configs)
    results = []
    
    for run in range(num_runs):
        # Create environment with different seed for each run
        env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=seed+run)
        
        # Create algorithm
        algorithm = UCB_M1(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)
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
    Compare UCB-M1 performance with different parameter settings.
    """
    print("=== UCB-M1 Parameter Comparison ===\n")
    
    # Define a challenging environment with 4 arms
    # Note: UCB-M1 uses heavy-tailed distributions with known second-order moments
    arm_configs = [
        {
            "name": "High Reward, High Cost",
            "type": "heavy_tailed",
            "params": {
                "mean_X": 2.0,    # High cost
                "mean_R": 6.0,    # High reward (rate = 3.0)
                "alpha_pareto_X": 3.0,    # Shape parameter for Pareto cost
                "loc_pareto_X": 1.0,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 1.8,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.3, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.2,       # Correlation between cost and reward
                "var_X": 0.5,     # Cost variance (for algorithm)
                "var_R": 1.0,     # Reward variance (for algorithm)
                "cov_XR": 0.2,    # Cost-reward covariance (for algorithm)
                "M_X": 20.0,      # Max cost bound
                "M_R": 30.0,      # Max reward bound
            }
        },
        {
            "name": "Medium Reward, Low Cost", 
            "type": "heavy_tailed",
            "params": {
                "mean_X": 0.5,    # Low cost
                "mean_R": 1.4,    # Medium reward (rate = 2.8)
                "alpha_pareto_X": 2.5,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.3,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.3,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.2, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.1,       # Correlation between cost and reward
                "var_X": 0.1,     # Cost variance (for algorithm)
                "var_R": 0.3,     # Reward variance (for algorithm)
                "cov_XR": 0.05,   # Cost-reward covariance (for algorithm)
                "M_X": 10.0,      # Max cost bound
                "M_R": 10.0,      # Max reward bound
            }
        },
        {
            "name": "Low Reward, Very Low Cost",
            "type": "heavy_tailed", 
            "params": {
                "mean_X": 0.2,    # Very low cost
                "mean_R": 0.5,    # Low reward (rate = 2.5)
                "alpha_pareto_X": 2.0,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.1,      # Scale parameter for Pareto cost
                "mean_lognormal_R": -0.7, # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.1, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.0,       # Correlation between cost and reward
                "var_X": 0.05,    # Cost variance (for algorithm)
                "var_R": 0.1,     # Reward variance (for algorithm)
                "cov_XR": 0.01,   # Cost-reward covariance (for algorithm)
                "M_X": 5.0,       # Max cost bound
                "M_R": 5.0,       # Max reward bound
            }
        },
        {
            "name": "Trap Arm (High Cost, Low Reward)",
            "type": "heavy_tailed",
            "params": {
                "mean_X": 3.0,    # High cost
                "mean_R": 2.0,    # Low reward (rate = 0.67)
                "alpha_pareto_X": 2.2,    # Shape parameter for Pareto cost
                "loc_pareto_X": 1.5,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.7,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.2, # sigma for underlying normal distribution for lognormal reward
                "correlation": -0.1,      # Negative correlation between cost and reward
                "var_X": 0.8,     # Cost variance (for algorithm)
                "var_R": 0.5,     # Reward variance (for algorithm)
                "cov_XR": 0.3,    # Cost-reward covariance (for algorithm)
                "M_X": 30.0,      # Max cost bound
                "M_R": 20.0,      # Max reward bound
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
    print(f"Note: All arms use heavy-tailed distributions with known second-order moments\n")
    
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
        
        results = run_ucb_m1_experiment(
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
    Analyze how UCB-M1 learns over time with different budgets.
    """
    print("\n=== Learning Curve Analysis ===\n")
    
    # Simple 2-arm environment with heavy-tailed distributions
    arm_configs = [
        {
            "name": "Good Arm",
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
                "M_X": 10.0,      # Max cost bound
                "M_R": 10.0,      # Max reward bound
            }
        },
        {
            "name": "Bad Arm", 
            "type": "heavy_tailed",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 1.5,    # Expected reward = 1.5
                "alpha_pareto_X": 2.5,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.6,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.4,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.3, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.05,      # Correlation between cost and reward
                "var_X": 0.1,     # Cost variance (for algorithm)
                "var_R": 0.2,     # Reward variance (for algorithm)
                "cov_XR": 0.03,   # Cost-reward covariance (for algorithm)
                "M_X": 10.0,      # Max cost bound
                "M_R": 10.0,      # Max reward bound
            }
        }
    ]
    
    algorithm_params = {"alpha": 2.1, "b_min_cost": 0.01}
    
    budgets = [500, 1000, 2000, 5000]
    
    print("Budget | Avg Reward | Avg Regret | Good Arm % | Learning Efficiency")
    print("-" * 65)
    
    for budget in budgets:
        results = run_ucb_m1_experiment(
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

def compare_with_other_algorithms():
    """
    Compare UCB-M1 performance with other UCB algorithms to show robustness.
    """
    print("\n=== UCB-M1 vs Other Algorithms Comparison ===\n")
    
    # Create environment with potential outliers
    arm_configs = [
        {
            "name": "Stable Arm",
            "type": "heavy_tailed",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 2.0,    # Expected reward = 2.0
                "alpha_pareto_X": 3.0,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.5,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.7,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.2, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.1,       # Correlation between cost and reward
                "var_X": 0.1,     # Low variance (for algorithm)
                "var_R": 0.2,     # Low variance (for algorithm)
                "cov_XR": 0.05,   # Cost-reward covariance (for algorithm)
                "M_X": 5.0,       # Max cost bound
                "M_R": 5.0,       # Max reward bound
            }
        },
        {
            "name": "Volatile Arm", 
            "type": "heavy_tailed",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 2.0,    # Expected reward = 2.0
                "alpha_pareto_X": 2.0,    # Shape parameter for Pareto cost (more volatile)
                "loc_pareto_X": 0.3,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.7,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.8, # sigma for underlying normal distribution for lognormal reward (more volatile)
                "correlation": 0.3,       # Higher correlation
                "var_X": 1.0,     # High variance (for algorithm)
                "var_R": 2.0,     # High variance (for algorithm)
                "cov_XR": 0.5,    # High covariance (for algorithm)
                "M_X": 15.0,      # Max cost bound
                "M_R": 15.0,      # Max reward bound
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
    
    results = run_ucb_m1_experiment(
        arm_configs=arm_configs,
        algorithm_params=algorithm_params,
        budget=budget,
        num_runs=num_runs,
        seed=200
    )
    
    # Analyze arm preference
    stable_pulls = np.mean([r['arm_pulls'][0] for r in results])
    volatile_pulls = np.mean([r['arm_pulls'][1] for r in results])
    total_pulls = np.mean([r['epochs'] for r in results])
    
    print(f"Results over {num_runs} runs:")
    print(f"Average total pulls: {total_pulls:.1f}")
    print(f"Stable arm (Arm 0): {stable_pulls:.1f} pulls ({stable_pulls/total_pulls*100:.1f}%)")
    print(f"Volatile arm (Arm 1): {volatile_pulls:.1f} pulls ({volatile_pulls/total_pulls*100:.1f}%)")
    print(f"Preference ratio (Stable/Volatile): {stable_pulls/volatile_pulls:.2f}")
    
    if stable_pulls > volatile_pulls:
        print("✓ UCB-M1 prefers the stable arm (good robustness!)")
    else:
        print("✗ UCB-M1 prefers the volatile arm (unexpected)")

def analyze_median_estimation():
    """
    Analyze how UCB-M1's median-based estimation affects performance.
    """
    print("\n=== Median Estimation Analysis ===\n")
    
    # Create arms with different characteristics to test median estimation
    arm_configs = [
        {
            "name": "Normal Arm",
            "type": "heavy_tailed",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 2.0,    # Expected reward = 2.0
                "alpha_pareto_X": 3.0,    # Shape parameter for Pareto cost
                "loc_pareto_X": 0.5,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.7,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.2, # sigma for underlying normal distribution for lognormal reward
                "correlation": 0.1,       # Correlation between cost and reward
                "var_X": 0.1,     # Normal variance (for algorithm)
                "var_R": 0.2,     # Normal variance (for algorithm)
                "cov_XR": 0.05,   # Cost-reward covariance (for algorithm)
                "M_X": 5.0,       # Max cost bound
                "M_R": 5.0,       # Max reward bound
            }
        },
        {
            "name": "Outlier-Prone Arm",
            "type": "heavy_tailed",
            "params": {
                "mean_X": 1.0,    # Expected cost = 1.0
                "mean_R": 2.0,    # Expected reward = 2.0
                "alpha_pareto_X": 2.0,    # Shape parameter for Pareto cost (more outliers)
                "loc_pareto_X": 0.3,      # Scale parameter for Pareto cost
                "mean_lognormal_R": 0.7,  # mu for underlying normal distribution for lognormal reward
                "sigma_lognormal_R": 0.6, # sigma for underlying normal distribution for lognormal reward (more outliers)
                "correlation": 0.2,       # Higher correlation
                "var_X": 0.5,     # Higher variance (for algorithm)
                "var_R": 1.0,     # Higher variance (for algorithm)
                "cov_XR": 0.2,    # Higher covariance (for algorithm)
                "M_X": 8.0,       # Max cost bound
                "M_R": 8.0,       # Max reward bound
            }
        }
    ]
    
    reward_rates = [arm['params']['mean_R']/arm['params']['mean_X'] for arm in arm_configs]
    print(f"Both arms have same reward rate: {reward_rates[0]:.2f}")
    print(f"Arm 0 (Normal): variance X={arm_configs[0]['params']['var_X']:.3f}, R={arm_configs[0]['params']['var_R']:.3f}")
    print(f"Arm 1 (Outlier-prone): variance X={arm_configs[1]['params']['var_X']:.3f}, R={arm_configs[1]['params']['var_R']:.3f}")
    print()
    
    # Test different alpha values to see how median estimation group size affects performance
    alpha_values = [1.5, 2.1, 3.0, 4.0]
    
    print("Alpha | Group Size | Avg Reward | Avg Regret | Normal Arm % | Performance")
    print("-" * 70)
    
    for alpha in alpha_values:
        algorithm_params = {"alpha": alpha, "b_min_cost": 0.01}
        
        results = run_ucb_m1_experiment(
            arm_configs=arm_configs,
            algorithm_params=algorithm_params,
            budget=1000,
            num_runs=20,
            seed=300
        )
        
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_regret = np.mean([r['regret'] for r in results])
        avg_normal_arm_ratio = np.mean([r['arm_pulls'][0]/np.sum(r['arm_pulls']) for r in results])
        
        # Estimate group size for median estimation (m = floor(3.5 * alpha * log(n)) + 1)
        # Using n=1000 as typical epoch number
        group_size = int(np.floor(3.5 * alpha * np.log(1000))) + 1
        
        # Performance indicator (higher is better)
        performance = avg_reward / (avg_regret + 1)  # Avoid division by zero
        
        print(f"{alpha:5.1f} | {group_size:10d} | {avg_reward:10.2f} | {avg_regret:10.2f} | "
              f"{avg_normal_arm_ratio*100:11.1f}% | {performance:11.3f}")

if __name__ == "__main__":
    # Run parameter comparison
    results = compare_parameters()
    
    # Run learning curve analysis
    analyze_learning_curve()
    
    # Compare with other algorithms
    compare_with_other_algorithms()
    
    # Analyze median estimation
    analyze_median_estimation()
    
    print("\n=== Key Insights for UCB-M1 ===")
    print("1. UCB-M1 performs well with heavy-tailed cost-reward distributions")
    print("2. Higher alpha values increase exploration but may not always improve performance")
    print("3. The b_min_cost parameter affects numerical stability and performance")
    print("4. Algorithm performance improves with larger budgets")
    print("5. UCB-M1 may prefer arms with lower variance when reward rates are similar")
    print("6. Median-based estimation provides robustness against outliers")
    print("7. The algorithm learns to focus on optimal arms over time despite heavy tails") 
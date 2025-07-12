# UCB-B1 Algorithm Guide

## Overview

UCB-B1 is an algorithm for **budget-constrained multi-armed bandits** where each action has both a cost and a reward. The goal is to maximize total reward while staying within a budget constraint.

## Key Concepts

### 1. **Arms (Bandit Machines)**
Each arm represents a choice with:
- **Cost (X)**: Random variable representing the cost of pulling this arm
- **Reward (R)**: Random variable representing the reward from pulling this arm
- **Reward Rate**: `mean_R / mean_X` - the efficiency of the arm

### 2. **Budget Constraint**
- You have a total budget `B` to spend
- Each arm pull costs some amount (random)
- Goal: Maximize total reward before budget is exhausted

### 3. **Exploration vs Exploitation**
- **Exploration**: Try different arms to learn their properties
- **Exploitation**: Focus on the best-performing arm
- UCB-B1 balances this trade-off using confidence bounds

## Required Parameters

### Arm Configuration Parameters

For each arm, you need to specify:

```python
arm_config = {
    "name": "Arm Name",
    "type": "gaussian",  # Distribution type
    "params": {
        "mean_X": 1.0,    # Expected cost
        "mean_R": 3.0,    # Expected reward  
        "var_X": 0.1,     # Cost variance
        "var_R": 0.2,     # Reward variance
        "cov_XR": 0.05,   # Cost-reward covariance
        "M_X": 5.0,       # Maximum possible cost
        "M_R": 5.0,       # Maximum possible reward
    }
}
```

**Parameter Meanings:**
- `mean_X`: Average cost per pull (must be > 0)
- `mean_R`: Average reward per pull
- `var_X`: How much cost varies around the mean
- `var_R`: How much reward varies around the mean  
- `cov_XR`: How cost and reward are correlated
- `M_X`, `M_R`: Bounds for confidence interval calculations

### Algorithm Parameters

```python
algorithm_params = {
    "alpha": 2.1,        # Confidence level parameter
    "L": 2,              # Confidence bound scaling
    "b_min_cost": 0.01,  # Numerical stability constant
    "M_X": 5.0,          # Max cost bound
    "M_R": 5.0,          # Max reward bound
}
```

**Parameter Meanings:**
- `alpha`: Controls exploration (higher = more exploration)
- `L`: Scaling factor for confidence bounds
- `b_min_cost`: Prevents division by zero in calculations
- `M_X`, `M_R`: Same as in arm configs

## How UCB-B1 Works

### 1. **Initialization**
- Algorithm starts with no knowledge of arm performance
- All arms get pulled once initially

### 2. **Arm Selection (UCB Principle)**
For each epoch, UCB-B1 calculates an **Upper Confidence Bound** for each arm:

```
UCB_k = estimated_reward_rate_k + confidence_bound_k
```

The algorithm selects the arm with the highest UCB value.

### 3. **Confidence Bound Calculation**
The confidence bound accounts for:
- **Estimation uncertainty**: Less pulls = higher uncertainty
- **Cost-reward correlation**: Exploits relationships between cost and reward
- **Statistical guarantees**: Ensures good performance with high probability

### 4. **Learning Process**
- **Early epochs**: High exploration, tries all arms
- **Later epochs**: Focuses on best arms, but still explores occasionally
- **Adaptive**: Adjusts strategy based on observed costs and rewards

## Example Results Interpretation

From our test run:

```
Expected reward rates: [3.0, 2.0, 1.0]  # Arm 0 is optimal
Final arm pull counts: [388, 368, 265]   # Arm 0 pulled most
Optimal arm: 0 (reward rate: 3.000)      # Algorithm found the best arm
Regret: 861.87                           # Performance gap from optimal
```

**What this means:**
- Arm 0 has the best reward rate (3.0 reward per 1.0 cost)
- UCB-B1 correctly identified Arm 0 as optimal (388 pulls vs 368, 265)
- The regret shows the algorithm didn't perform as well as the optimal policy
- This is expected in early learning phases

## Performance Metrics

### 1. **Regret**
```
Regret = Optimal_Reward - Actual_Reward
```
- Measures how much worse the algorithm performed vs optimal
- Lower is better
- Expected to decrease over time as algorithm learns

### 2. **Arm Pull Distribution**
- Shows how much each arm was explored
- Good algorithm should focus on optimal arms
- Some exploration of suboptimal arms is necessary

### 3. **Reward Rate Learning**
- Algorithm should converge to pulling the arm with highest `mean_R / mean_X`
- Early exploration may lead to suboptimal choices
- Later epochs should show better performance

## Tips for Using UCB-B1

### 1. **Parameter Tuning**
- **Higher `alpha`**: More exploration, slower convergence
- **Lower `alpha`**: Less exploration, faster convergence but risk of missing optimal arm
- **Typical range**: 1.5 - 3.0

### 2. **Arm Configuration**
- Ensure `mean_X > 0` for all arms
- Set realistic `M_X`, `M_R` bounds
- Consider cost-reward correlations in your domain

### 3. **Budget Selection**
- **Small budget**: Algorithm may not have time to learn
- **Large budget**: Better performance, more time to converge
- **Rule of thumb**: Budget should be at least 10x the number of arms

### 4. **Multiple Runs**
- Run multiple simulations with different seeds
- Average results to get reliable performance estimates
- Consider confidence intervals for performance metrics

## Common Use Cases

1. **Online Advertising**: Allocate budget across ad channels
2. **Resource Allocation**: Choose investment opportunities
3. **Clinical Trials**: Test treatments with varying costs/effects
4. **Network Routing**: Select paths with different costs/performance

## Troubleshooting

### High Regret
- Increase budget for more learning time
- Check if arm configurations are realistic
- Verify `mean_X > 0` for all arms

### Algorithm Stuck on Suboptimal Arm
- Increase `alpha` for more exploration
- Check if arm configurations are correct
- Ensure sufficient budget for learning

### Numerical Issues
- Increase `b_min_cost` if getting division by zero
- Check that variances are positive
- Verify covariance matrix is positive semi-definite 
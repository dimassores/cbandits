# UCB-B2 Algorithm Guide

## Overview

UCB-B2 is an algorithm for **budget-constrained multi-armed bandits** designed for bounded and uncorrelated cost and reward distributions. Unlike UCB-B1, UCB-B2 does not assume known second-order moments and must estimate them from samples, making it more practical for real-world applications where distribution parameters are unknown.

## Key Concepts

### 1. **Arms (Bandit Machines)**
Each arm represents a choice with:
- **Cost (X)**: Random variable representing the cost of pulling this arm
- **Reward (R)**: Random variable representing the reward from pulling this arm
- **Reward Rate**: `mean_R / mean_X` - the efficiency of the arm
- **Uncorrelated**: Cost and reward are assumed to be independent

### 2. **Budget Constraint**
- You have a total budget `B` to spend
- Each arm pull costs some amount (random)
- Goal: Maximize total reward before budget is exhausted

### 3. **Exploration vs Exploitation**
- **Exploration**: Try different arms to learn their properties
- **Exploitation**: Focus on the best-performing arm
- UCB-B2 balances this trade-off using confidence bounds

### 4. **Key Difference from UCB-B1**
- **UCB-B1**: Assumes known second-order moments (variances, covariances)
- **UCB-B2**: Estimates all moments from samples (more practical)

## Required Parameters

### Arm Configuration Parameters

For each arm, you need to specify:

```python
arm_config = {
    "name": "Arm Name",
    "type": "bounded_uniform",  # Distribution type
    "params": {
        "min_X": 0.5, "max_X": 1.5,  # Cost bounds
        "min_R": 1.0, "max_R": 3.0,  # Reward bounds
        "correlation": 0.0,          # No correlation for UCB-B2
        "mean_X": 1.0,              # Expected cost
        "mean_R": 2.0,              # Expected reward
        "var_X": 0.083,             # Cost variance
        "var_R": 0.333,             # Reward variance
        "cov_XR": 0.0,              # Cost-reward covariance (0 for uncorrelated)
        "M_X": 1.5,                 # Maximum possible cost
        "M_R": 3.0,                 # Maximum possible reward
    }
}
```

**Parameter Meanings:**
- `min_X`, `max_X`: Bounds for cost distribution
- `min_R`, `max_R`: Bounds for reward distribution
- `correlation`: Should be 0.0 for UCB-B2 (uncorrelated assumption)
- `mean_X`: Average cost per pull (must be > 0)
- `mean_R`: Average reward per pull
- `var_X`: How much cost varies around the mean
- `var_R`: How much reward varies around the mean
- `cov_XR`: Cost-reward covariance (0 for uncorrelated case)
- `M_X`, `M_R`: Bounds for confidence interval calculations

### Algorithm Parameters

```python
algorithm_params = {
    "alpha": 2.1,        # Confidence level parameter
    "b_min_cost": 0.01,  # Numerical stability constant
}
```

**Parameter Meanings:**
- `alpha`: Controls exploration (higher = more exploration)
- `b_min_cost`: Prevents division by zero in calculations

## How UCB-B2 Works

### 1. **Initialization**
- Algorithm starts with no knowledge of arm performance
- All arms get pulled once initially
- No prior knowledge of variances or covariances needed

### 2. **Arm Selection (UCB Principle)**
For each epoch, UCB-B2 calculates an **Upper Confidence Bound** for each arm:

```
UCB_k = estimated_reward_rate_k + confidence_bound_k
```

The algorithm selects the arm with the highest UCB value.

### 3. **Confidence Bound Calculation**
The confidence bound accounts for:
- **Estimation uncertainty**: Less pulls = higher uncertainty
- **Empirical variance**: Uses sample variance instead of known variance
- **Bounded distributions**: Leverages known bounds M_X, M_R
- **Uncorrelated assumption**: Simpler confidence bounds than UCB-B1

### 4. **Learning Process**
- **Early epochs**: High exploration, tries all arms
- **Later epochs**: Focuses on best arms, but still explores occasionally
- **Adaptive**: Adjusts strategy based on observed costs and rewards
- **Empirical estimation**: Continuously updates variance estimates

## Example Results Interpretation

From a typical test run:

```
Expected reward rates: [3.0, 2.0, 1.0]  # Arm 0 is optimal
Final arm pull counts: [375, 380, 265]   # Arm 0 and 1 similar
Optimal arm: 0 (reward rate: 3.000)      # Algorithm found the best arm
Regret: 925.45                           # Performance gap from optimal
```

**What this means:**
- Arm 0 has the best reward rate (3.0 reward per 1.0 cost)
- UCB-B2 correctly identified Arm 0 as optimal (375 pulls vs 380, 265)
- The regret shows the algorithm didn't perform as well as the optimal policy
- This is expected in early learning phases, especially with unknown variances

## Performance Metrics

### 1. **Regret**
```
Regret = Optimal_Reward - Actual_Reward
```
- Measures how much worse the algorithm performed vs optimal
- Lower is better
- Expected to decrease over time as algorithm learns
- May be higher than UCB-B1 due to variance estimation overhead

### 2. **Arm Pull Distribution**
- Shows how much each arm was explored
- Good algorithm should focus on optimal arms
- Some exploration of suboptimal arms is necessary
- May show more exploration than UCB-B1 due to uncertainty in variance estimates

### 3. **Reward Rate Learning**
- Algorithm should converge to pulling the arm with highest `mean_R / mean_X`
- Early exploration may lead to suboptimal choices
- Later epochs should show better performance
- Convergence may be slower than UCB-B1 due to variance estimation

## Tips for Using UCB-B2

### 1. **Parameter Tuning**
- **Higher `alpha`**: More exploration, slower convergence
- **Lower `alpha`**: Less exploration, faster convergence but risk of missing optimal arm
- **Typical range**: 1.5 - 3.0
- **Default**: 2.1 (good balance for most scenarios)

### 2. **Arm Configuration**
- Ensure `mean_X > 0` for all arms
- Set realistic `M_X`, `M_R` bounds based on your domain
- Keep `correlation = 0.0` for uncorrelated assumption
- Use bounded distributions (uniform, truncated normal, etc.)

### 3. **Budget Selection**
- **Small budget**: Algorithm may not have time to learn variances
- **Large budget**: Better performance, more time to converge
- **Rule of thumb**: Budget should be at least 15x the number of arms (more than UCB-B1)

### 4. **Multiple Runs**
- Run multiple simulations with different seeds
- Average results to get reliable performance estimates
- Consider confidence intervals for performance metrics
- Variance in results may be higher than UCB-B1

## Common Use Cases

1. **Online Advertising**: Allocate budget across ad channels with unknown performance variance
2. **Resource Allocation**: Choose investment opportunities with bounded but unknown risks
3. **Clinical Trials**: Test treatments with varying costs/effects and unknown variance
4. **Network Routing**: Select paths with different costs/performance and unknown variability
5. **A/B Testing**: Compare website variants with unknown conversion rate variance

## When to Use UCB-B2 vs Other Algorithms

### **Use UCB-B2 when:**
- ✅ Cost and reward distributions are bounded
- ✅ Cost and reward are uncorrelated (or correlation is negligible)
- ✅ Second-order moments (variances) are unknown
- ✅ You want practical implementation without strong distribution assumptions
- ✅ You have reasonable bounds on costs and rewards

### **Use UCB-B1 when:**
- ✅ You know the variances and covariances of your distributions
- ✅ Cost and reward may be correlated
- ✅ You want potentially better performance with known moments

### **Use UCB-B2C when:**
- ✅ Cost and reward are correlated
- ✅ Second-order moments are unknown
- ✅ You want to exploit correlation structure

### **Use UCB-M1 when:**
- ✅ Distributions are heavy-tailed
- ✅ You have weaker moment assumptions
- ✅ You want robust performance with outliers

## Troubleshooting

### High Regret
- Increase budget for more learning time
- Check if arm configurations are realistic
- Verify `mean_X > 0` for all arms
- Consider increasing `alpha` for more exploration

### Algorithm Stuck on Suboptimal Arm
- Increase `alpha` for more exploration
- Check if arm configurations are correct
- Ensure sufficient budget for learning
- Verify that bounds M_X, M_R are appropriate

### Numerical Issues
- Increase `b_min_cost` if getting division by zero
- Check that variances are positive
- Ensure M_X, M_R bounds are realistic
- Verify that cost and reward samples are within bounds

### Slow Convergence
- This is expected with UCB-B2 due to variance estimation
- Consider using UCB-B1 if you know the variances
- Increase budget for more learning time
- Check if the uncorrelated assumption holds 
# UCB-B2C Algorithm Guide

## Overview

UCB-B2C is an algorithm for **budget-constrained multi-armed bandits** designed for bounded and correlated cost and reward distributions. Unlike UCB-B2, UCB-B2C exploits the correlation between cost and reward to achieve better performance, while still estimating all second-order moments from samples rather than assuming they are known.

## Key Concepts

### 1. **Arms (Bandit Machines)**
Each arm represents a choice with:
- **Cost (X)**: Random variable representing the cost of pulling this arm
- **Reward (R)**: Random variable representing the reward from pulling this arm
- **Reward Rate**: `mean_R / mean_X` - the efficiency of the arm
- **Correlated**: Cost and reward may be dependent on each other

### 2. **Budget Constraint**
- You have a total budget `B` to spend
- Each arm pull costs some amount (random)
- Goal: Maximize total reward before budget is exhausted

### 3. **Exploration vs Exploitation**
- **Exploration**: Try different arms to learn their properties
- **Exploitation**: Focus on the best-performing arm
- UCB-B2C balances this trade-off using confidence bounds

### 4. **Key Difference from UCB-B2**
- **UCB-B2**: Assumes uncorrelated cost and reward
- **UCB-B2C**: Exploits correlation between cost and reward for better performance
- **Both**: Estimate second-order moments from samples

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
        "correlation": 0.3,          # Correlation between cost and reward
        "mean_X": 1.0,              # Expected cost
        "mean_R": 2.0,              # Expected reward
        "var_X": 0.083,             # Cost variance
        "var_R": 0.333,             # Reward variance
        "cov_XR": 0.025,            # Cost-reward covariance
        "M_X": 1.5,                 # Maximum possible cost
        "M_R": 3.0,                 # Maximum possible reward
    }
}
```

**Parameter Meanings:**
- `min_X`, `max_X`: Bounds for cost distribution
- `min_R`, `max_R`: Bounds for reward distribution
- `correlation`: Correlation coefficient between cost and reward (-1 to 1)
- `mean_X`: Average cost per pull (must be > 0)
- `mean_R`: Average reward per pull
- `var_X`: How much cost varies around the mean
- `var_R`: How much reward varies around the mean
- `cov_XR`: Cost-reward covariance (can be positive or negative)
- `M_X`, `M_R`: Bounds for confidence interval calculations

### Algorithm Parameters

```python
algorithm_params = {
    "alpha": 2.1,        # Confidence level parameter
    "b_min_cost": 0.01,  # Numerical stability constant
    "omega_bar": 2.0,    # Upper bound on correlation parameter
}
```

**Parameter Meanings:**
- `alpha`: Controls exploration (higher = more exploration)
- `b_min_cost`: Prevents division by zero in calculations
- `omega_bar`: Upper bound on the correlation parameter omega_k

## How UCB-B2C Works

### 1. **Initialization**
- Algorithm starts with no knowledge of arm performance
- All arms get pulled once initially
- No prior knowledge of variances or covariances needed
- Stores all samples for empirical estimation

### 2. **Arm Selection (UCB Principle)**
For each epoch, UCB-B2C calculates an **Upper Confidence Bound** for each arm:

```
UCB_k = estimated_reward_rate_k + confidence_bound_k
```

The algorithm selects the arm with the highest UCB value.

### 3. **Confidence Bound Calculation**
The confidence bound accounts for:
- **Estimation uncertainty**: Less pulls = higher uncertainty
- **Empirical variance**: Uses sample variance instead of known variance
- **Correlation exploitation**: Uses LMMSE estimation to learn correlation
- **Bounded distributions**: Leverages known bounds M_X, M_R
- **Reduced variance**: Exploits correlation to reduce uncertainty

### 4. **Learning Process**
- **Early epochs**: High exploration, tries all arms
- **Later epochs**: Focuses on best arms, but still explores occasionally
- **Adaptive**: Adjusts strategy based on observed costs and rewards
- **Correlation learning**: Continuously estimates correlation structure
- **Sample storage**: Keeps all samples for empirical estimation

## Example Results Interpretation

From a typical test run:

```
Expected reward rates: [3.0, 2.0, 1.0]  # Arm 0 is optimal
Final arm pull counts: [390, 365, 265]   # Arm 0 pulled most
Optimal arm: 0 (reward rate: 3.000)      # Algorithm found the best arm
Regret: 845.23                           # Performance gap from optimal
```

**What this means:**
- Arm 0 has the best reward rate (3.0 reward per 1.0 cost)
- UCB-B2C correctly identified Arm 0 as optimal (390 pulls vs 365, 265)
- The regret shows the algorithm didn't perform as well as the optimal policy
- Performance may be better than UCB-B2 due to correlation exploitation

## Performance Metrics

### 1. **Regret**
```
Regret = Optimal_Reward - Actual_Reward
```
- Measures how much worse the algorithm performed vs optimal
- Lower is better
- Expected to decrease over time as algorithm learns
- May be lower than UCB-B2 due to correlation exploitation

### 2. **Arm Pull Distribution**
- Shows how much each arm was explored
- Good algorithm should focus on optimal arms
- Some exploration of suboptimal arms is necessary
- May show less exploration than UCB-B2 due to better correlation estimates

### 3. **Reward Rate Learning**
- Algorithm should converge to pulling the arm with highest `mean_R / mean_X`
- Early exploration may lead to suboptimal choices
- Later epochs should show better performance
- Convergence may be faster than UCB-B2 due to correlation exploitation

## Tips for Using UCB-B2C

### 1. **Parameter Tuning**
- **Higher `alpha`**: More exploration, slower convergence
- **Lower `alpha`**: Less exploration, faster convergence but risk of missing optimal arm
- **Typical range**: 1.5 - 3.0
- **Default**: 2.1 (good balance for most scenarios)
- **`omega_bar`**: Set based on expected correlation strength in your domain

### 2. **Arm Configuration**
- Ensure `mean_X > 0` for all arms
- Set realistic `M_X`, `M_R` bounds based on your domain
- Use appropriate correlation values for your scenario
- Use bounded distributions (uniform, truncated normal, etc.)

### 3. **Budget Selection**
- **Small budget**: Algorithm may not have time to learn correlations
- **Large budget**: Better performance, more time to converge
- **Rule of thumb**: Budget should be at least 15x the number of arms

### 4. **Multiple Runs**
- Run multiple simulations with different seeds
- Average results to get reliable performance estimates
- Consider confidence intervals for performance metrics
- Results may be more stable than UCB-B2 due to correlation exploitation

## Common Use Cases

1. **Online Advertising**: Allocate budget across ad channels where cost and conversion are correlated
2. **Resource Allocation**: Choose investment opportunities where risk and return are related
3. **Clinical Trials**: Test treatments where cost and effectiveness may be correlated
4. **Network Routing**: Select paths where latency and bandwidth are related
5. **A/B Testing**: Compare website variants where cost and conversion rate are correlated

## When to Use UCB-B2C vs Other Algorithms

### **Use UCB-B2C when:**
- ✅ Cost and reward distributions are bounded
- ✅ Cost and reward are correlated (or you suspect they might be)
- ✅ Second-order moments (variances, covariances) are unknown
- ✅ You want to exploit correlation for better performance
- ✅ You have reasonable bounds on costs and rewards

### **Use UCB-B1 when:**
- ✅ You know the variances and covariances of your distributions
- ✅ Cost and reward may be correlated
- ✅ You want potentially better performance with known moments

### **Use UCB-B2 when:**
- ✅ Cost and reward are uncorrelated (or correlation is negligible)
- ✅ Second-order moments are unknown
- ✅ You want simpler implementation without correlation handling

### **Use UCB-M1 when:**
- ✅ Distributions are heavy-tailed
- ✅ You have weaker moment assumptions
- ✅ You want robust performance with outliers

## Understanding Correlation in UCB-B2C

### **Positive Correlation**
- Higher costs tend to lead to higher rewards
- Example: More expensive ads may have higher conversion rates
- UCB-B2C can exploit this to make better decisions

### **Negative Correlation**
- Higher costs tend to lead to lower rewards
- Example: More expensive treatments may not always be more effective
- UCB-B2C can learn to avoid overpriced options

### **No Correlation**
- Cost and reward are independent
- UCB-B2C reduces to UCB-B2 behavior
- No performance penalty for using UCB-B2C

## Troubleshooting

### High Regret
- Increase budget for more learning time
- Check if arm configurations are realistic
- Verify `mean_X > 0` for all arms
- Consider increasing `alpha` for more exploration
- Check if correlation values are appropriate

### Algorithm Stuck on Suboptimal Arm
- Increase `alpha` for more exploration
- Check if arm configurations are correct
- Ensure sufficient budget for learning
- Verify that bounds M_X, M_R are appropriate
- Check if correlation structure is correctly specified

### Numerical Issues
- Increase `b_min_cost` if getting division by zero
- Check that variances are positive
- Ensure M_X, M_R bounds are realistic
- Verify that cost and reward samples are within bounds
- Check if `omega_bar` is set appropriately

### Slow Convergence
- This may happen if correlation structure is complex
- Consider using UCB-B1 if you know the correlations
- Increase budget for more learning time
- Check if the correlation assumption holds
- Verify that `omega_bar` is not too restrictive

### Poor Performance vs UCB-B2
- Check if correlation values are realistic
- Verify that `omega_bar` is set appropriately
- Consider if your scenario actually has correlation
- Run with UCB-B2 to compare performance
- Check if the correlation structure is beneficial 
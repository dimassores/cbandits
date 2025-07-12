# UCB-M1 Algorithm Guide

## Overview

UCB-M1 is an algorithm for **budget-constrained multi-armed bandits** designed for heavy-tailed cost and reward distributions. Unlike the other UCB algorithms, UCB-M1 uses median-based estimators to achieve robust performance even when distributions have outliers or heavy tails, requiring only weaker moment assumptions.

## Key Concepts

### 1. **Arms (Bandit Machines)**
Each arm represents a choice with:
- **Cost (X)**: Random variable representing the cost of pulling this arm
- **Reward (R)**: Random variable representing the reward from pulling this arm
- **Reward Rate**: `mean_R / mean_X` - the efficiency of the arm
- **Heavy-tailed**: Distributions may have outliers or extreme values

### 2. **Budget Constraint**
- You have a total budget `B` to spend
- Each arm pull costs some amount (random)
- Goal: Maximize total reward before budget is exhausted

### 3. **Exploration vs Exploitation**
- **Exploration**: Try different arms to learn their properties
- **Exploitation**: Focus on the best-performing arm
- UCB-M1 balances this trade-off using confidence bounds

### 4. **Key Difference from Other UCB Algorithms**
- **UCB-B1/B2/B2C**: Use mean-based estimators, assume bounded or sub-Gaussian distributions
- **UCB-M1**: Uses median-based estimators, handles heavy-tailed distributions
- **Robustness**: Less sensitive to outliers and extreme values

## Required Parameters

### Arm Configuration Parameters

For each arm, you need to specify:

```python
arm_config = {
    "name": "Arm Name",
    "type": "heavy_tailed",  # Distribution type
    "params": {
        "mean_X": 1.0,    # Expected cost
        "mean_R": 3.0,    # Expected reward  
        "var_X": 0.1,     # Cost variance
        "var_R": 0.2,     # Reward variance
        "cov_XR": 0.05,   # Cost-reward covariance
        "M_X": 5.0,       # Maximum possible cost (for bounds)
        "M_R": 5.0,       # Maximum possible reward (for bounds)
    }
}
```

**Parameter Meanings:**
- `mean_X`: Average cost per pull (must be > 0)
- `mean_R`: Average reward per pull
- `var_X`: How much cost varies around the mean
- `var_R`: How much reward varies around the mean  
- `cov_XR`: How cost and reward are correlated
- `M_X`, `M_R`: Bounds for confidence interval calculations (less critical for UCB-M1)

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

## How UCB-M1 Works

### 1. **Initialization**
- Algorithm starts with no knowledge of arm performance
- All arms get pulled once initially
- Stores all samples for median-based estimation
- Uses known second-order moments for confidence bounds

### 2. **Arm Selection (UCB Principle)**
For each epoch, UCB-M1 calculates an **Upper Confidence Bound** for each arm:

```
UCB_k = median_reward_rate_k + confidence_bound_k
```

The algorithm selects the arm with the highest UCB value.

### 3. **Median-Based Estimation**
UCB-M1 uses a sophisticated median-based approach:
- **Grouping**: Divides samples into groups based on epoch number
- **Group rates**: Calculates reward rate for each group
- **Median rate**: Uses median of group rates as the main estimator
- **Robustness**: Less sensitive to outliers than mean-based methods

### 4. **Confidence Bound Calculation**
The confidence bound accounts for:
- **Estimation uncertainty**: Less pulls = higher uncertainty
- **Known second-order moments**: Uses known variances and covariances
- **Heavy-tailed robustness**: Designed for distributions with outliers
- **Statistical guarantees**: Ensures good performance with high probability

### 5. **Learning Process**
- **Early epochs**: High exploration, tries all arms
- **Later epochs**: Focuses on best arms, but still explores occasionally
- **Adaptive**: Adjusts strategy based on observed costs and rewards
- **Robust learning**: Median-based approach handles outliers gracefully

## Example Results Interpretation

From a typical test run:

```
Expected reward rates: [3.0, 2.0, 1.0]  # Arm 0 is optimal
Final arm pull counts: [385, 370, 265]   # Arm 0 pulled most
Optimal arm: 0 (reward rate: 3.000)      # Algorithm found the best arm
Regret: 875.32                           # Performance gap from optimal
```

**What this means:**
- Arm 0 has the best reward rate (3.0 reward per 1.0 cost)
- UCB-M1 correctly identified Arm 0 as optimal (385 pulls vs 370, 265)
- The regret shows the algorithm didn't perform as well as the optimal policy
- Performance is robust even with potential outliers in the data

## Performance Metrics

### 1. **Regret**
```
Regret = Optimal_Reward - Actual_Reward
```
- Measures how much worse the algorithm performed vs optimal
- Lower is better
- Expected to decrease over time as algorithm learns
- May be more stable than other algorithms due to robustness

### 2. **Arm Pull Distribution**
- Shows how much each arm was explored
- Good algorithm should focus on optimal arms
- Some exploration of suboptimal arms is necessary
- May show more stable exploration patterns due to median-based estimation

### 3. **Reward Rate Learning**
- Algorithm should converge to pulling the arm with highest `mean_R / mean_X`
- Early exploration may lead to suboptimal choices
- Later epochs should show better performance
- Convergence may be more stable due to outlier resistance

## Tips for Using UCB-M1

### 1. **Parameter Tuning**
- **Higher `alpha`**: More exploration, slower convergence
- **Lower `alpha`**: Less exploration, faster convergence but risk of missing optimal arm
- **Typical range**: 1.5 - 3.0
- **Default**: 2.1 (good balance for most scenarios)

### 2. **Arm Configuration**
- Ensure `mean_X > 0` for all arms
- Set realistic variances and covariances
- UCB-M1 is less sensitive to extreme M_X, M_R bounds
- Use heavy-tailed distributions (Pareto, Student's t, etc.)

### 3. **Budget Selection**
- **Small budget**: Algorithm may not have time to learn
- **Large budget**: Better performance, more time to converge
- **Rule of thumb**: Budget should be at least 10x the number of arms

### 4. **Multiple Runs**
- Run multiple simulations with different seeds
- Average results to get reliable performance estimates
- Consider confidence intervals for performance metrics
- Results may be more stable than other algorithms due to robustness

## Common Use Cases

1. **Financial Trading**: Allocate budget across trading strategies with heavy-tailed returns
2. **Risk Management**: Choose insurance policies with rare but extreme events
3. **Clinical Trials**: Test treatments where outcomes may have outliers
4. **Network Security**: Select security measures with rare but severe threats
5. **Supply Chain**: Choose suppliers where costs may have extreme variations

## When to Use UCB-M1 vs Other Algorithms

### **Use UCB-M1 when:**
- ✅ Cost and reward distributions are heavy-tailed
- ✅ You expect outliers or extreme values
- ✅ You have weaker moment assumptions
- ✅ You want robust performance regardless of distribution shape
- ✅ You know second-order moments but want outlier resistance

### **Use UCB-B1 when:**
- ✅ Distributions are sub-Gaussian or bounded
- ✅ You know the variances and covariances
- ✅ You want potentially better performance with well-behaved distributions
- ✅ Cost and reward may be correlated

### **Use UCB-B2 when:**
- ✅ Distributions are bounded but not heavy-tailed
- ✅ Cost and reward are uncorrelated
- ✅ Second-order moments are unknown
- ✅ You want simpler implementation

### **Use UCB-B2C when:**
- ✅ Distributions are bounded but not heavy-tailed
- ✅ Cost and reward are correlated
- ✅ Second-order moments are unknown
- ✅ You want to exploit correlation

## Understanding Heavy-Tailed Distributions

### **What are Heavy-Tailed Distributions?**
- Distributions with more extreme values than normal distributions
- Examples: Pareto, Student's t, Cauchy, log-normal
- Common in finance, insurance, and natural phenomena

### **Why Median-Based Estimation Helps**
- **Mean**: Sensitive to outliers, can be pulled far from typical values
- **Median**: Resistant to outliers, represents typical values better
- **UCB-M1**: Uses median of group means, combining robustness with efficiency

### **When Heavy-Tailed Distributions Occur**
- **Financial returns**: Stock prices, currency exchange rates
- **Insurance claims**: Rare but expensive events
- **Network traffic**: Burst patterns and congestion
- **Natural disasters**: Earthquakes, floods, pandemics

## Troubleshooting

### High Regret
- Increase budget for more learning time
- Check if arm configurations are realistic
- Verify `mean_X > 0` for all arms
- Consider increasing `alpha` for more exploration
- Check if distributions are truly heavy-tailed

### Algorithm Stuck on Suboptimal Arm
- Increase `alpha` for more exploration
- Check if arm configurations are correct
- Ensure sufficient budget for learning
- Verify that variances and covariances are appropriate
- Check if the heavy-tailed assumption is valid

### Numerical Issues
- Increase `b_min_cost` if getting division by zero
- Check that variances are positive
- Ensure covariance matrix is positive semi-definite
- Verify that cost and reward samples are reasonable

### Slow Convergence
- This may happen with very heavy-tailed distributions
- Consider using UCB-B1 if distributions are well-behaved
- Increase budget for more learning time
- Check if the heavy-tailed assumption is appropriate
- Verify that second-order moments are correctly specified

### Poor Performance vs Other Algorithms
- Check if your distributions are actually heavy-tailed
- Verify that variances and covariances are realistic
- Consider if the median-based approach is beneficial for your scenario
- Run with other algorithms to compare performance
- Check if outliers are actually present in your data

## Advanced Considerations

### **Group Size Calculation**
UCB-M1 uses `m = floor(3.5 * alpha * log(n)) + 1` groups:
- More groups = more robust but slower convergence
- Fewer groups = faster convergence but less robust
- Adjust `alpha` to control the trade-off

### **Median vs Mean Trade-offs**
- **Median**: Robust to outliers, may be less efficient with normal data
- **Mean**: Efficient with normal data, sensitive to outliers
- **UCB-M1**: Balances both through group-based median estimation

### **Heavy-Tailed vs Bounded Distributions**
- **Heavy-tailed**: Infinite support, extreme values possible
- **Bounded**: Finite support, values within known bounds
- **UCB-M1**: Designed for heavy-tailed, but works with bounded too 
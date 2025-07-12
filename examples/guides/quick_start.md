# Quick Start Guide: Testing UCB-B1 Algorithm

# 🚀 Getting Started in 5 Minutes

### 1. **Basic Test** (Run the simple example)
```bash
python simple_ucb_b1_example.py
```

### 2. **Advanced Testing** (Parameter comparison)
```bash
python advanced_ucb_b1_example.py
```

### 3. **Full Simulation Suite** (All algorithms)
```bash
python src/simulations/runner.py
```

## 📋 Required Parameters Summary

### **Arm Configuration** (for each arm)
```python
arm_config = {
    "name": "Arm Name",
    "type": "gaussian",  # or "heavy_tailed", "bounded_uniform"
    "params": {
        "mean_X": 1.0,    # Expected cost (MUST be > 0)
        "mean_R": 3.0,    # Expected reward
        "var_X": 0.1,     # Cost variance
        "var_R": 0.2,     # Reward variance
        "cov_XR": 0.05,   # Cost-reward covariance
        "M_X": 5.0,       # Max cost bound
        "M_R": 5.0,       # Max reward bound
    }
}
```

### **Algorithm Parameters**
```python
algorithm_params = {
    "alpha": 2.1,        # Exploration parameter (1.5-4.0)
    "L": 2,              # Confidence bound scaling
    "b_min_cost": 0.01,  # Numerical stability
    "M_X": 5.0,          # Max cost bound
    "M_R": 5.0,          # Max reward bound
}
```

## 🎯 Key Concepts

### **What UCB-B1 Does**
- **Input**: Multiple arms, each with random cost and reward
- **Goal**: Maximize total reward within a budget constraint
- **Method**: Balances exploration (trying different arms) vs exploitation (focusing on best arms)

### **How to Interpret Results**
- **Reward Rate**: `mean_R / mean_X` (higher is better)
- **Regret**: Difference from optimal performance (lower is better)
- **Arm Pulls**: Distribution showing which arms were chosen
- **Optimal Arm**: The arm with highest reward rate

## 📊 Example Results Interpretation

From our test run:
```
Expected reward rates: [3.0, 2.0, 1.0]  # Arm 0 is best
Final arm pulls: [388, 368, 265]         # Arm 0 pulled most
Regret: 861.87                           # Performance gap
```

**What this means:**
- ✅ Algorithm correctly identified best arm (Arm 0)
- ✅ Focused on optimal arm (388 pulls vs others)
- ⚠️ Some regret due to early exploration phase

## 🔧 Parameter Tuning Guide

### **Alpha (Exploration Parameter)**
- **1.5**: Conservative, less exploration
- **2.1**: Default, balanced approach  
- **3.0**: Aggressive, more exploration
- **4.0**: Very aggressive, maximum exploration

### **Budget Selection**
- **Small (< 1000)**: May not have time to learn
- **Medium (1000-5000)**: Good for testing
- **Large (> 5000)**: Better performance, more learning time

## 🧪 Testing Scenarios

### **1. Simple 3-Arm Test**
```python
# Use simple_ucb_b1_example.py
# Tests basic functionality with clear optimal arm
```

### **2. Challenging 4-Arm Test** 
```python
# Use advanced_ucb_b1_example.py
# Tests parameter sensitivity with similar reward rates
```

### **3. Heavy-Tailed Distributions**
```python
# Modify arm_configs to use "heavy_tailed" type
# Tests robustness with non-Gaussian distributions
```

## 📈 Performance Metrics

### **Primary Metrics**
- **Total Reward**: Sum of all rewards collected
- **Regret**: `Optimal_Reward - Actual_Reward`
- **Arm Pull Distribution**: How much each arm was explored

### **Secondary Metrics**
- **Learning Efficiency**: `Total_Reward / Budget`
- **Optimal Arm Ratio**: Percentage of pulls on best arm
- **Convergence Speed**: How quickly algorithm focuses on optimal arm

## 🚨 Common Issues & Solutions

### **High Regret**
- ✅ Increase budget for more learning time
- ✅ Check if `mean_X > 0` for all arms
- ✅ Verify arm configurations are realistic

### **Algorithm Stuck on Suboptimal Arm**
- ✅ Increase `alpha` for more exploration
- ✅ Check arm configurations are correct
- ✅ Ensure sufficient budget

### **Numerical Errors**
- ✅ Increase `b_min_cost` if division by zero
- ✅ Check variances are positive
- ✅ Verify covariance matrix is valid

## 🎯 Best Practices

### **1. Start Simple**
- Begin with 2-3 arms and Gaussian distributions
- Use moderate budget (1000-2000)
- Use default parameters (alpha=2.1)

### **2. Validate Results**
- Run multiple times with different seeds
- Check that optimal arm is identified
- Verify regret decreases with larger budgets

### **3. Experiment Gradually**
- Change one parameter at a time
- Document parameter effects
- Use consistent evaluation metrics

### **4. Scale Up Carefully**
- Add more arms gradually
- Test with different distribution types
- Increase budget systematically

## 📚 Next Steps

1. **Run the examples** to understand basic usage
2. **Modify parameters** to see their effects
3. **Create your own scenarios** with different arm configurations
4. **Compare with other algorithms** (UCB-M1, UCB-B2, UCB-B2C)
5. **Analyze results** using the provided metrics


## 💡 Pro Tips

1. **Use fixed seeds** for reproducible results during development
2. **Start with balanced arms** (similar costs) to understand the algorithm
3. **Monitor arm pull distribution** to see learning progress
4. **Test with edge cases** (very different reward rates, high variance)
5. **Document your experiments** for future reference 
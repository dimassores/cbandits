# Examples

This folder contains practical examples and guides for using the budget-constrained bandit algorithms.

## 🎯 Available Algorithms

The library implements four budget-constrained bandit algorithms:

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **UCB-B1** | Upper Confidence Bound with Budget constraint | General budget-constrained scenarios |
| **UCB-B2** | Enhanced UCB with improved budget handling | Scenarios with varying cost distributions |
| **UCB-B2C** | UCB-B2 with cost-aware exploration | High-cost, high-reward environments |
| **UCB-M1** | Modified UCB for budget constraints | Scenarios requiring conservative exploration |

## 📁 Folder Structure

```
examples/
├── README.md                    # This file - main overview
├── guides/                      # Documentation and guides
│   ├── quick_start.md          # Generic quick start guide (5 min)
│   ├── ucb_b1_guide.md         # Detailed UCB-B1 guide (15 min)
│   ├── ucb_b2_guide.md         # Detailed UCB-B2 guide (15 min)
│   ├── ucb_b2c_guide.md        # Detailed UCB-B2C guide (15 min)
│   └── ucb_m1_guide.md         # Detailed UCB-M1 guide (15 min)
├── simple_examples/            # Basic examples for each algorithm
│   ├── simple_ucb_b1_example.py
│   ├── simple_ucb_b2_example.py
│   ├── simple_ucb_b2c_example.py
│   └── simple_ucb_m1_example.py
└── advanced_examples/          # Parameter experimentation
    ├── advanced_ucb_b1_example.py
    ├── advanced_ucb_b2_example.py
    ├── advanced_ucb_b2c_example.py
    └── advanced_ucb_m1_example.py
```

## 🚀 Quick Start

### 1. **Read the Quick Start Guide**
```bash
# Start with the generic guide
cat guides/quick_start.md
```

### 2. **Run a Simple Example**
```bash
# Choose your algorithm and run the simple example
python simple_examples/simple_ucb_b1_example.py
```

### 3. **Experiment with Parameters**
```bash
# Run advanced examples for parameter tuning
python advanced_examples/advanced_ucb_b1_example.py
```

### 4. **Read Algorithm-Specific Guides**
```bash
# Deep dive into specific algorithms
cat guides/ucb_b1_guide.md
```

## 📖 What Each Folder Contains

### `guides/` - Documentation
- **`quick_start.md`**: Generic 5-minute guide for all algorithms
- **`ucb_*_guide.md`**: Detailed 15-minute guides for each algorithm
- **Purpose**: Learn theory, parameters, and best practices
- **Best for**: Understanding concepts and troubleshooting

### `simple_examples/` - Basic Demonstrations
- **Purpose**: Basic algorithm demonstration with 3-4 arms
- **What they show**: How to set up arms, run algorithms, interpret results
- **Best for**: First-time users, understanding basic concepts
- **Runtime**: ~30 seconds each

### `advanced_examples/` - Parameter Experimentation
- **Purpose**: Parameter comparison and performance analysis
- **What they show**: How different parameters affect algorithm performance
- **Best for**: Understanding parameter tuning, comparing strategies
- **Runtime**: ~2-3 minutes each

## 🎯 Learning Path

### **Beginner Path**
1. Read `guides/quick_start.md` (5 minutes)
2. Run `simple_examples/simple_ucb_b1_example.py`
3. Read `guides/ucb_b1_guide.md` for deep understanding

### **Intermediate Path**
1. Complete beginner path
2. Run `advanced_examples/advanced_ucb_b1_example.py`
3. Experiment with different parameters
4. Try other algorithms (UCB-B2, UCB-B2C, UCB-M1)

### **Advanced Path**
1. Complete intermediate path
2. Compare algorithms on same scenarios
3. Create custom arm configurations
4. Analyze performance across different environments

## 🔧 Example Scenarios

### **Simple Examples** (3-4 Arms)
- Clear optimal arm identification
- Basic algorithm behavior demonstration
- Quick validation of implementation

### **Advanced Examples** (4+ Arms)
- Challenging scenarios with similar reward rates
- Parameter sensitivity testing
- Performance comparison across settings

### **Common Test Configurations**
```python
# Basic 3-arm test (clear optimal)
reward_rates = [3.0, 2.0, 1.0]  # Arm 0 is clearly best

# Challenging 4-arm test (similar rates)
reward_rates = [3.0, 2.8, 2.5, 0.67]  # Harder to distinguish

# Parameter comparison
alpha_values = [1.5, 2.1, 3.0, 4.0]  # Exploration vs exploitation
```

## 📊 Expected Results

### **Simple Example Output**
```
Expected reward rates: [3.0, 2.0, 1.0]
Final arm pulls: [388, 368, 265]
Optimal arm: 0 (reward rate: 3.000)
Regret: 861.87
```

### **Advanced Example Output**
```
Parameter Set        Alpha  Avg Reward   Avg Regret   Opt Arm % 
----------------------------------------------------------------------
Conservative        1.5    5697.40      302.60       13.1%
Default             2.1    5832.35      167.65       26.9%
Aggressive          3.0    5920.97      79.03        51.8%
Very Aggressive     4.0    5992.00      8.00         99.7%
```

## 💡 Best Practices

### **Getting Started**
1. **Use fixed seeds** for reproducible results during development
2. **Start with small budgets** (1000-2000) for quick testing
3. **Monitor arm pull distribution** to see learning progress
4. **Run multiple times** to understand variability

### **Parameter Tuning**
1. **Modify parameters gradually** to see their effects
2. **Document your experiments** for future reference
3. **Compare algorithms** on the same scenarios
4. **Test with edge cases** (very different reward rates, high variance)

### **Performance Analysis**
1. **Focus on regret** as the primary metric
2. **Consider arm pull distribution** for learning efficiency
3. **Analyze convergence speed** to optimal arm
4. **Test with different budget sizes** to understand scaling

## 🔗 Related Files

### **Core Implementation**
- `src/algorithms/` - Algorithm implementations
- `src/environments/general_cost_reward_env.py` - Environment simulation
- `config/simulation_config.py` - Default configuration

### **Full Simulation Suite**
- `src/simulations/runner.py` - Complete simulation framework
- `tests/` - Unit tests and validation

## 🚧 Current Status

- ✅ **UCB-B1**: Complete examples and guides
- 🔄 **UCB-B2**: Examples and guides in development
- 🔄 **UCB-B2C**: Examples and guides in development  
- 🔄 **UCB-M1**: Examples and guides in development

## 🤝 Contributing

When adding examples for new algorithms:
1. Follow the existing naming convention
2. Use similar structure to existing examples
3. Update this README.md with new algorithm information
4. Create corresponding guide in `guides/` folder 
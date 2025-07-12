# Installation Guide

This guide explains how to install and use the `cbandits` library.

## Quick Installation

### From Source (Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dimassores/cbandits.git
   cd cbandits
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   ```

   Or with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### From PyPI (When Published)

```bash
pip install cbandits
```

## Development Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

4. **Run tests:**
   ```bash
   pytest
   ```

## Usage Examples

### Basic Usage

```python
from cbandits import UCB_B1, GeneralCostRewardEnvironment
import numpy as np

# Define your bandit arms
arm_configs = [
    {
        "name": "Arm 1",
        "type": "gaussian",
        "params": {
            "mean_X": 1.0,    # Expected cost
            "mean_R": 3.0,    # Expected reward
            "var_X": 0.1,     # Cost variance
            "var_R": 0.2,     # Reward variance
            "cov_XR": 0.05,   # Cost-reward covariance
            "M_X": 5.0,       # Max cost bound
            "M_R": 5.0,       # Max reward bound
        }
    },
    # ... more arms
]

# Set up algorithm parameters
algorithm_params = {
    "alpha": 2.1,
    "L": 2,
    "b_min_cost": 0.01,
    "M_X": 5.0,
    "M_R": 5.0,
}

# Create environment and algorithm
num_arms = len(arm_configs)
budget = 1000

env = GeneralCostRewardEnvironment(num_arms=num_arms, arm_configs=arm_configs, seed=42)
algorithm = UCB_B1(num_arms=num_arms, arm_configs=arm_configs, algorithm_params=algorithm_params)

# Run simulation
current_total_cost = 0.0
current_total_reward = 0.0
epoch = 0

while current_total_cost <= budget:
    epoch += 1
    
    # Select arm
    chosen_arm = algorithm.select_arm(current_total_cost, epoch)
    
    # Pull arm
    cost, reward = env.pull_arm(chosen_arm)
    
    # Update algorithm
    algorithm.update_state(chosen_arm, cost, reward)
    
    # Update totals
    current_total_cost += cost
    current_total_reward += reward

print(f"Final reward: {current_total_reward:.2f}")
print(f"Regret: {env.get_optimal_reward_rate() * budget - current_total_reward:.2f}")
```

### Running Examples

The library includes several example scripts:

```bash
# Run the simple UCB-B1 example
python examples/simple_examples/simple_ucb_b1_example.py

# Or use the console script (if installed)
cbandits-demo
```

### Available Algorithms

- **UCB-B1**: For sub-Gaussian cases with known second-order moments
- **UCB-B2**: For bounded and uncorrelated cost/reward distributions
- **UCB-B2C**: For bounded and correlated cost/reward distributions
- **UCB-M1**: For heavy-tailed cost and reward distributions

## Building and Distributing

### Build the Package

```bash
# Build source distribution
python setup.py sdist

# Build wheel
python setup.py bdist_wheel

# Or using modern tools
pip install build
python -m build
```

### Upload to PyPI (When Ready)

```bash
pip install twine
twine upload dist/*
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've installed the package with `pip install -e .`
2. **Missing dependencies**: Install with `pip install -e ".[dev]"` for all dependencies
3. **Version conflicts**: Check your Python version (requires >=3.8)

### Getting Help

- Check the [examples](examples/) directory for usage examples
- Read the [algorithm guides](examples/guides/) for detailed explanations
- Run the test suite to verify your installation: `pytest`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
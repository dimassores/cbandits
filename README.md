# Budget-Constrained Bandits with General Cost and Reward Distributions

[![codecov](https://codecov.io/gh/dimassores/cbandits/branch/main/graph/badge.svg)](https://codecov.io/gh/dimassores/cbandits)

This repository provides implementations and simulation frameworks for various algorithms addressing the budget-constrained multi-armed bandit problem, focusing on scenarios with random, potentially correlated, and heavy-tailed cost and reward distributions. The algorithms are based on the research presented in "Budget-Constrained Bandits over General Cost and Reward Distributions" by Caycі, Eryilmaz, and Srikant (AISTATS 2020).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Running Simulations](#running-simulations)
- [Analyzing Results](#analyzing-results)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Reference](#reference)

## Introduction

The multi-armed bandit problem is a classic model for the exploration-exploitation dilemma. This project extends the classical setting to a more general and realistic scenario where each action (arm pull) incurs a random cost and yields a random reward, both of which can be correlated and have heavy-tailed distributions. The objective is to maximize the total expected reward under a finite budget constraint on the total cost.

The repository includes implementations of:
* **UCB-B1**: An algorithm for sub-Gaussian cases with known second-order moments, exploiting cost-reward correlation.
* **UCB-M1**: Designed for heavy-tailed cost and reward distributions, achieving $O(\log B)$ regret with fewer moment assumptions by using median-based estimators.
* **UCB-B2**: For bounded and uncorrelated cost/reward with unknown second-order moments.
* **UCB-B2C**: Handles bounded and correlated cost/reward with unknown second-order moments, learning the correlation via LMMSE estimation.

The implemented algorithms achieve tight problem-dependent regret bounds and are optimal up to a universal constant factor for jointly Gaussian cost and reward pairs. For more details, see the reference paper below.

## Features

* Implementation of UCB-B1, UCB-M1, UCB-B2, and UCB-B2C algorithms.
* Flexible environment simulation for general cost and reward distributions (including correlated and heavy-tailed).
* Modular design for easy extension and testing of new algorithms or environments.
* Utilities for statistical estimation (empirical means, variances, LMMSE, median-based).
* Simulation runner for reproducible experimental results.
* Jupyter notebook template for data analysis and visualization.

## Project Structure

```text
├── src/cbandits/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── ucb_b1.py
│   │   ├── ucb_m1.py
│   │   ├── ucb_b2.py
│   │   ├── ucb_b2c.py
│   │   └── base_bandit_algorithm.py
│   │
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── bandit_environment.py
│   │   └── general_cost_reward_env.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_structures.py
│   │   ├── estimators.py
│   │   └── plot_utils.py
│   │
│   └── simulations/
│       ├── __init__.py
│       └── runner.py
│
├── docs/
│   ├── README.md              # Documentation index
│   ├── installation.md        # Installation guide
│   ├── development.md         # Development guide
│   └── automated-testing.md   # Automated testing setup
│
├── examples/
│   ├── README.md
│   ├── simple_examples/
│   │   ├── simple_ucb_b1_example.py
│   │   ├── simple_ucb_b2_example.py
│   │   ├── simple_ucb_b2c_example.py
│   │   └── simple_ucb_m1_example.py
│   ├── advanced_examples/
│   │   ├── advanced_ucb_b1_example.py
│   │   ├── advanced_ucb_b2_example.py
│   │   ├── advanced_ucb_b2c_example.py
│   │   └── advanced_ucb_m1_example.py
│   └── guides/
│       ├── quick_start.md
│       ├── ucb_b1_guide.md
│       ├── ucb_b2_guide.md
│       ├── ucb_b2c_guide.md
│       └── ucb_m1_guide.md
│
├── config/
│   ├── __init__.py
│   └── simulation_config.py
│
├── notebooks/
│   └── analysis.ipynb  # (template, currently empty)
│
├── tests/
│   ├── __init__.py
│   ├── test_estimators.py
│   ├── test_algorithms.py
│   └── test_environments.py
│
├── data/
│   ├── raw/        # (for raw simulation data, if generated)
│   └── processed/  # (for processed results, if generated)
│
├── .github/workflows/
│   ├── ci.yml              # Main CI pipeline
│   └── pr-checks.yml       # Pull request checks
│
├── .gitignore
├── README.md               # Project overview
├── pyproject.toml          # Package configuration
├── Makefile                # Development commands
├── .pre-commit-config.yaml # Pre-commit hooks
└── LICENSE
```

## Installation

### Quick Start

Install the library directly from the repository:

```bash
# Clone and install in development mode
git clone https://github.com/dimassores/cbandits.git
cd cbandits
pip install -e .
```

### From PyPI (When Published)

```bash
pip install cbandits
```

### Development Setup

For detailed installation instructions, see [docs/installation.md](docs/installation.md).

1. Clone the repository:
   ```bash
   git clone https://github.com/dimassores/cbandits.git
   cd cbandits
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   Or with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Quick Start

🚀 **Get started in 5 minutes!**

1. **Run a basic example**:
   ```bash
   python examples/simple_examples/simple_ucb_b1_example.py
   ```

2. **Experiment with parameters**:
   ```bash
   python examples/advanced_examples/advanced_ucb_b1_example.py
   ```

3. **Read the guides**:
   - [Quick Start Guide](examples/Quick_Start_Guide.md) - Essential parameters and tips
   - [UCB-B1 Guide](examples/UCB_B1_Guide.md) - Comprehensive algorithm explanation

4. **Run the full simulation suite**:
   ```bash
   python src/simulations/runner.py
   ```

### What You'll See

The examples demonstrate:
- **3-arm bandit problem** with clear optimal arm
- **Parameter comparison** showing exploration vs exploitation trade-offs
- **Learning curves** showing how performance improves with budget
- **Regret analysis** measuring algorithm performance

### Example Output
```
Expected reward rates: [3.0, 2.0, 1.0]  # Arm 0 is optimal
Final arm pulls: [388, 368, 265]         # Algorithm focuses on best arm
Regret: 861.87                           # Performance gap from optimal
```

## Usage

### Basic Usage

1. Set up the environment as described in the Installation section.
2. Run the simulation runner to generate experimental results:
   ```bash
   python src/simulations/runner.py
   ```

### Advanced Usage

For more detailed examples and experimentation:

- **Start with examples**: See the [examples/](examples/) folder for practical demonstrations
- **Parameter tuning**: Use `examples/advanced_ucb_b1_example.py` to experiment with different settings
- **Custom scenarios**: Modify the examples to create your own bandit problems
- **Performance analysis**: Use the provided metrics to evaluate algorithm performance

## Configuration

Simulation parameters can be configured in the `config/simulation_config.py` file.

## Running Simulations

The main entry point for running simulations is `src/simulations/runner.py`.

## Analyzing Results

A Jupyter notebook template is provided in the `notebooks/` directory for analyzing results. (Currently, `analysis.ipynb` is empty; you can use it as a starting point for your own analysis.)

## Documentation

For comprehensive documentation, see the [docs/](docs/) directory:

- **[Installation Guide](docs/installation.md)** - Complete setup instructions
- **[Development Guide](docs/development.md)** - Development workflow and best practices
- **[Automated Testing](docs/automated-testing.md)** - CI/CD setup and usage
- **[Documentation Index](docs/README.md)** - Complete documentation overview

## Data Directory

The `data/raw/` and `data/processed/` directories are provided for storing simulation data and processed results. These directories are empty by default and will be populated if you modify the code to save outputs.

## Testing

Run the test suite using:
```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License.

## Reference

This project is based on the following academic paper:

* Caycі, S., Eryilmaz, A., & Srikant, R. (2020). **Budget-Constrained Bandits over General Cost and Reward Distributions**. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS) 2020*, Palermo, Italy. PMLR: Volume 108.




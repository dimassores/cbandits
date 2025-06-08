# Budget-Constrained Bandits with General Cost and Reward Distributions

This repository provides implementations and simulation frameworks for various algorithms addressing the budget-constrained multi-armed bandit problem, focusing on scenarios with random, potentially correlated, and heavy-tailed cost and reward distributions. The algorithms are based on the research presented in "Budget-Constrained Bandits over General Cost and Reward Distributions" by Caycі, Eryilmaz, and Srikant (AISTATS 2020).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
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
├── src/
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
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

## Installation

To install the project, follow these steps:

1. Clone the repository (replace the URL with your actual repository if different):
   ```bash
   git clone <your-repo-url> cbandits
   cd cbandits
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set up the environment as described in the Installation section.
2. Run the simulation runner to generate experimental results:
   ```bash
   python src/simulations/runner.py
   ```

## Configuration

Simulation parameters can be configured in the `config/simulation_config.py` file.

## Running Simulations

The main entry point for running simulations is `src/simulations/runner.py`.

## Analyzing Results

A Jupyter notebook template is provided in the `notebooks/` directory for analyzing results. (Currently, `analysis.ipynb` is empty; you can use it as a starting point for your own analysis.)

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




# Budget-Constrained Bandits with General Cost and Reward Distributions

This repository provides implementations and simulation frameworks for various algorithms addressing the budget-constrained multi-armed bandit problem, particularly focusing on scenarios with random, potentially correlated, and heavy-tailed cost and reward distributions. The algorithms are based on the research presented in "Budget-Constrained Bandits over General Cost and Reward Distributions" by Caycі, Eryilmaz, and Srikant (AISTATS 2020).

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
- [Acknowledgments](#references)

## Introduction

The multi-armed bandit problem is a classic model for the exploration-exploitation dilemma. This project extends the classical setting to a more general and realistic scenario where each action (arm pull) incurs a random cost and yields a random reward, both of which can be correlated and have heavy-tailed distributions. The objective is to maximize the total expected reward under a finite budget constraint on the total cost[cite: 1, 2, 3].

The repository includes implementations of:
* **UCB-B1**: An algorithm for sub-Gaussian cases with known second-order moments, exploiting cost-reward correlation[cite: 72, 75, 76].
* **UCB-M1**: Designed for heavy-tailed cost and reward distributions, achieving $O(\text{log } B)$ regret with fewer moment assumptions by using median-based estimators[cite: 98, 103, 108, 112].
* **UCB-B2**: For bounded and uncorrelated cost/reward with unknown second-order moments[cite: 132, 133].
* **UCB-B2C**: Handles bounded and correlated cost/reward with unknown second-order moments, learning the correlation via LMMSE estimation[cite: 142, 143, 144, 145].

The theoretical foundations show that an $O(\text{log } B)$ regret is achievable if moments of order $(2+\gamma)$ for some $\gamma>0$ exist for all cost-reward pairs[cite: 4, 25]. The implemented algorithms are shown to achieve tight problem-dependent regret bounds, being optimal up to a universal constant factor for jointly Gaussian cost and reward pairs[cite: 5, 28, 127].

## Features

* Implementation of UCB-B1, UCB-M1, UCB-B2, and UCB-B2C algorithms.
* Flexible environment simulation for general cost and reward distributions (including correlated and heavy-tailed).
* Modular design for easy extension and testing of new algorithms or environments.
* Utilities for statistical estimation (empirical means, variances, LMMSE, median-based).
* Simulation runner for reproducible experimental results.
* Jupyter notebooks for data analysis and visualization.

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
│   ├── exploration.ipynb
│   └── analysis.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_estimators.py
│   ├── test_algorithms.py
│   └── test_environments.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

To install the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/dimassores/budget-constrained-bandits.git
   cd budget-constrained-bandits
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

To use the project, follow these steps:

1. Set up the environment as described in the Installation section.
2. Run the simulation runner to generate experimental results:
   ```bash
   python src/simulations/runner.py
   ```

## Configuration

The configuration for the simulation can be found in the `config/simulation_config.py` file.

## Running Simulations

The simulation runner is located in the `src/simulations/runner.py` file.

## Analyzing Results

The results can be analyzed using the Jupyter notebooks in the `notebooks/` directory.

## Testing

The test suite can be run using:
```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License.

## Reference

This project is built upon the theoretical framework and algorithms proposed in the following academic paper:

* Caycі, S., Eryilmaz, A., & Srikant, R. (2020). **Budget-Constrained Bandits over General Cost and Reward Distributions**. *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS) 2020*, Palermo, Italy. PMLR: Volume 108.




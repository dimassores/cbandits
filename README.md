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
- [Acknowledgments](#acknowledgments)

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

├── src/
│   ├── algorithms/
│   │   ├── init.py
│   │   ├── ucb_b1.py
│   │   ├── ucb_m1.py
│   │   ├── ucb_b2.py
│   │   ├── ucb_b2c.py
│   │   └── base_bandit_algorithm.py
│   │
│   ├── environments/
│   │   ├── init.py
│   │   ├── bandit_environment.py
│   │   └── general_cost_reward_env.py
│   │
│   ├── utils/
│   │   ├── init.py
│   │   ├── data_structures.py
│   │   ├── estimators.py
│   │   └── plot_utils.py
│   │
│   └── simulations/
│       ├── init.py
│       └── runner.py
│
├── config/
│   ├── init.py
│   └── simulation_config.py
│
├── notebooks/
│   ├── exploration.ipynb
│   └── analysis.ipynb
│
├── tests/
│   ├── init.py
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


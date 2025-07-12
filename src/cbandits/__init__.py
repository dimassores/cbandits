"""
Budget-Constrained Bandits with General Cost and Reward Distributions

A Python library implementing various algorithms for budget-constrained 
multi-armed bandit problems, focusing on scenarios with random, potentially 
correlated, and heavy-tailed cost and reward distributions.

This library is based on the research presented in "Budget-Constrained Bandits 
over General Cost and Reward Distributions" by Caycı, Eryilmaz, and Srikant (AISTATS 2020).

Available Algorithms:
- UCB-B1: For sub-Gaussian cases with known second-order moments
- UCB-B2: For bounded and uncorrelated cost/reward with unknown second-order moments  
- UCB-B2C: For bounded and correlated cost/reward with unknown second-order moments
- UCB-M1: For heavy-tailed cost and reward distributions

Example Usage:
    >>> from cbandits import UCB_B1, GeneralCostRewardEnvironment
    >>> # Set up your bandit problem and run simulations
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
from .algorithms import (
    BaseBanditAlgorithm,
    UCB_B1,
    UCB_B2, 
    UCB_B2C,
    UCB_M1
)

from .environments import (
    BanditEnvironment,
    GeneralCostRewardEnvironment
)

from .utils import (
    calculate_empirical_mean,
    calculate_empirical_variance,
    calculate_lmmse_omega_empirical,
    calculate_lmmse_variance_reduction_empirical
)

# Define what gets imported with "from cbandits import *"
__all__ = [
    # Core algorithms
    "BaseBanditAlgorithm",
    "UCB_B1", 
    "UCB_B2",
    "UCB_B2C",
    "UCB_M1",
    
    # Environments
    "BanditEnvironment",
    "GeneralCostRewardEnvironment",
    
    # Utilities
    "calculate_empirical_mean",
    "calculate_empirical_variance", 
    "calculate_lmmse_omega_empirical",
    "calculate_lmmse_variance_reduction_empirical",
] 
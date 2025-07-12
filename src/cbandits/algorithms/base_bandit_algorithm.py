# src/algorithms/base_bandit_algorithm.py

from abc import ABC, abstractmethod
import numpy as np

class BaseBanditAlgorithm(ABC):
    """
    Abstract base class for all multi-armed bandit algorithms.
    This class defines the common interface that all specific bandit algorithm
    implementations must adhere to.
    """

    def __init__(self, num_arms: int, arm_configs: list, algorithm_params: dict):
        """
        Initializes the base bandit algorithm.

        Args:
            num_arms (int): The total number of arms in the bandit problem.
            arm_configs (list): A list of dictionaries, each describing an arm's properties.
                                 This can include true means, variances, bounds, etc.,
                                 depending on what information the specific algorithm
                                 is assumed to know.
            algorithm_params (dict): A dictionary of algorithm-specific parameters
                                     (e.g., exploration rates, constants).
        """
        if not isinstance(num_arms, int) or num_arms <= 0:
            raise ValueError("num_arms must be a positive integer.")
        if not isinstance(arm_configs, list):
            raise ValueError("arm_configs must be a list.")
        if not all(isinstance(config, dict) for config in arm_configs):
            raise ValueError("Each item in arm_configs must be a dictionary.")
        if len(arm_configs) != num_arms:
            raise ValueError("Length of arm_configs must match num_arms.")
        if not isinstance(algorithm_params, dict):
            raise ValueError("algorithm_params must be a dictionary.")

        self.num_arms = num_arms
        self.arm_configs = arm_configs
        self.algorithm_params = algorithm_params

    @abstractmethod
    def select_arm(self, current_total_cost: float, current_epoch: int) -> int:
        """
        Abstract method to select an arm for the current epoch.
        This method must be implemented by concrete algorithm classes.

        Args:
            current_total_cost (float): The total cost accumulated so far by the policy.
            current_epoch (int): The current epoch number (e.g., the current time step 'n').

        Returns:
            int: The index of the chosen arm (0-indexed).
        """
        pass

    @abstractmethod
    def update_state(self, chosen_arm: int, cost: float, reward: float):
        """
        Abstract method to update the algorithm's internal state after an arm pull.
        This method must be implemented by concrete algorithm classes.

        Args:
            chosen_arm (int): The index of the arm that was pulled.
            cost (float): The cost observed from pulling the arm.
            reward (float): The reward observed from pulling the arm.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Abstract method to reset the algorithm's state for a new simulation run.
        This is crucial for running multiple independent simulations.
        """
        pass
# src/environments/bandit_environment.py

from abc import ABC, abstractmethod
import numpy as np

class BanditEnvironment(ABC):
    """
    Abstract base class for all multi-armed bandit environments.
    Defines the interface for interacting with the bandit arms.
    """

    def __init__(self, num_arms: int, arm_configs: list):
        """
        Initializes the bandit environment.

        Args:
            num_arms (int): The total number of arms in the bandit problem.
            arm_configs (list): A list of dictionaries, each describing an arm's
                                 underlying statistical properties (e.g., mean, variance,
                                 distribution type). This information is used by
                                 the environment to generate costs and rewards.
        """
        if not isinstance(num_arms, int) or num_arms <= 0:
            raise ValueError("num_arms must be a positive integer.")
        if not isinstance(arm_configs, list):
            raise ValueError("arm_configs must be a list.")
        if not all(isinstance(config, dict) for config in arm_configs):
            raise ValueError("Each item in arm_configs must be a dictionary.")
        if len(arm_configs) != num_arms:
            raise ValueError("Length of arm_configs must match num_arms.")

        self.num_arms = num_arms
        self.arm_configs = arm_configs

        # Calculate true optimal reward rate and best arm based on true distributions
        # This is for calculating regret later.
        self.true_reward_rates = np.zeros(num_arms)
        for k in range(num_arms):
            # Assuming 'mean_X' and 'mean_R' are available in arm_configs['params']
            mean_X = arm_configs[k]['params']['mean_X']
            mean_R = arm_configs[k]['params']['mean_R']
            if mean_X > 0: # Ensure positive expected cost for rate calculation
                self.true_reward_rates[k] = mean_R / mean_X
            else:
                self.true_reward_rates[k] = -np.inf # Or handle appropriately for non-positive expected costs

        self.optimal_reward_rate = np.max(self.true_reward_rates)
        # Find all arms that achieve the optimal rate. If multiple, any is fine for k* definition.
        self.optimal_arms = np.where(self.true_reward_rates == self.optimal_reward_rate)[0]
        self.optimal_arm_index = self.optimal_arms[0] # Pick one if multiple exist

        # Store the true expected cost of the optimal arm for regret calculation reference
        self.optimal_arm_expected_cost = self.arm_configs[self.optimal_arm_index]['params']['mean_X']

    @abstractmethod
    def pull_arm(self, arm_index: int) -> tuple[float, float]:
        """
        Abstract method to simulate pulling a specific arm.
        This method must be implemented by concrete environment classes.

        Args:
            arm_index (int): The index of the arm to pull (0-indexed).

        Returns:
            tuple[float, float]: A tuple containing (cost, reward) incurred/received.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Abstract method to reset the environment's state for a new simulation run.
        This might involve re-initializing random number generators or internal counters
        if the environment itself maintains state.
        """
        pass

    def get_optimal_reward_rate(self) -> float:
        """
        Returns the true optimal reward rate (reward per unit cost)
        among all arms, which is used for calculating regret.
        """
        return self.optimal_reward_rate

    def get_optimal_arm_expected_cost(self) -> float:
        """
        Returns the true expected cost of the optimal arm, used in regret calculation.
        """
        return self.optimal_arm_expected_cost
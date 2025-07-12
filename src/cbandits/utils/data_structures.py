# src/utils/data_structures.py

import numpy as np

class BanditHistory:
    """
    A class to manage and store historical data for each arm in a bandit simulation.
    This can be useful for algorithms that need to access past samples directly
    (e.g., UCB-M1 for median-based estimation, UCB-B2C for empirical LMMSE).
    """

    def __init__(self, num_arms: int):
        """
        Initializes the history storage for a given number of arms.

        Args:
            num_arms (int): The total number of arms in the bandit problem.
        """
        if not isinstance(num_arms, int) or num_arms <= 0:
            raise ValueError("num_arms must be a positive integer.")

        self.num_arms = num_arms
        # List of lists to store individual samples for each arm
        self._samples_X = [[] for _ in range(num_arms)]
        self._samples_R = [[] for _ in range(num_arms)]

        # Arrays to store aggregated statistics (useful for algorithms like UCB-B1, UCB-B2)
        self.arm_pulls = np.zeros(num_arms, dtype=int)
        self.total_costs = np.zeros(num_arms, dtype=float)
        self.total_rewards = np.zeros(num_arms, dtype=float)
        self.sum_sq_costs = np.zeros(num_arms, dtype=float)    # For variance calculation
        self.sum_sq_rewards = np.zeros(num_arms, dtype=float)  # For variance calculation
        self.sum_XR = np.zeros(num_arms, dtype=float)          # For covariance/LMMSE calculation

    def add_sample(self, arm_index: int, cost: float, reward: float):
        """
        Adds a new observed cost and reward sample for a specific arm.

        Args:
            arm_index (int): The index of the arm that was pulled.
            cost (float): The cost observed from pulling the arm.
            reward (float): The reward observed from pulling the arm.
        """
        if not (0 <= arm_index < self.num_arms):
            raise IndexError(f"Arm index {arm_index} out of bounds for {self.num_arms} arms.")

        self._samples_X[arm_index].append(cost)
        self._samples_R[arm_index].append(reward)

        self.arm_pulls[arm_index] += 1
        self.total_costs[arm_index] += cost
        self.total_rewards[arm_index] += reward
        self.sum_sq_costs[arm_index] += cost**2
        self.sum_sq_rewards[arm_index] += reward**2
        self.sum_XR[arm_index] += cost * reward

    def get_arm_samples(self, arm_index: int) -> tuple[list[float], list[float]]:
        """
        Returns all collected cost and reward samples for a specific arm.

        Args:
            arm_index (int): The index of the arm.

        Returns:
            tuple[list[float], list[float]]: A tuple containing lists of (costs, rewards)
                                             for the specified arm.
        """
        if not (0 <= arm_index < self.num_arms):
            raise IndexError(f"Arm index {arm_index} out of bounds for {self.num_arms} arms.")
        return self._samples_X[arm_index], self._samples_R[arm_index]

    def get_empirical_stats(self, arm_index: int) -> dict:
        """
        Returns aggregated empirical statistics for a specific arm.

        Args:
            arm_index (int): The index of the arm.

        Returns:
            dict: A dictionary containing 'pulls', 'total_cost', 'total_reward',
                  'sum_sq_cost', 'sum_sq_reward', 'sum_XR'.
        """
        if not (0 <= arm_index < self.num_arms):
            raise IndexError(f"Arm index {arm_index} out of bounds for {self.num_arms} arms.")
        return {
            "pulls": self.arm_pulls[arm_index],
            "total_cost": self.total_costs[arm_index],
            "total_reward": self.total_rewards[arm_index],
            "sum_sq_cost": self.sum_sq_costs[arm_index],
            "sum_sq_reward": self.sum_sq_rewards[arm_index],
            "sum_XR": self.sum_XR[arm_index]
        }

    def reset(self):
        """
        Resets all stored history and statistics for a new simulation run.
        """
        for k in range(self.num_arms):
            self._samples_X[k].clear()
            self._samples_R[k].clear()
        self.arm_pulls.fill(0)
        self.total_costs.fill(0.0)
        self.total_rewards.fill(0.0)
        self.sum_sq_costs.fill(0.0)
        self.sum_sq_rewards.fill(0.0)
        self.sum_XR.fill(0.0)
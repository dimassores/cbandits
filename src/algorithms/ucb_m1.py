# src/algorithms/ucb_m1.py

import numpy as np
from .base_bandit_algorithm import BaseBanditAlgorithm
from ..utils.estimators import calculate_empirical_mean, calculate_lmmse_omega, calculate_lmmse_variance_reduction

class UCB_M1(BaseBanditAlgorithm):
    """
    Implementation of the UCB-M1 algorithm for budget-constrained bandits.
    This algorithm is designed for heavy-tailed cost and reward distributions,
    using median-based estimators to achieve O(log B) regret with weaker
    moment assumptions than UCB-B1.
    """

    def __init__(self, num_arms, arm_configs, algorithm_params):
        """
        Initializes the UCB-M1 algorithm.

        Args:
            num_arms (int): The total number of arms in the bandit problem.
            arm_configs (list): A list of dictionaries, each describing an arm's properties.
                                 Expected to contain 'params' with 'mean_X', 'mean_R',
                                 'var_X', 'var_R', 'cov_XR' (for computing omega_k and V_XR).
            algorithm_params (dict): Dictionary of algorithm-specific parameters
                                     (e.g., 'alpha', 'b_min_cost').
        """
        super().__init__(num_arms, arm_configs, algorithm_params)

        self.alpha = algorithm_params.get('alpha', 2.1) # Default alpha if not provided
        self.b_min_cost = algorithm_params.get('b_min_cost', 0.1) # Small positive constant for denominator stability

        # Initialize arm-specific statistics to store *all* observed samples
        # for median-based estimation
        self.arm_samples_X = [[] for _ in range(num_arms)]
        self.arm_samples_R = [[] for _ in range(num_arms)]
        self.arm_pulls = np.zeros(num_arms) # T_k(n)

        # Store known second-order moments for omega_k and V_XR
        self.var_X = np.array([config['params']['var_X'] for config in arm_configs])
        self.var_R = np.array([config['params']['var_R'] for config in arm_configs])
        self.cov_XR = np.array([config['params']['cov_XR'] for config in arm_configs])
        
        # Calculate omega_k and V(X, R) (Eq. 8, 9)
        self.omega_k = np.zeros(num_arms)
        self.V_XR = np.zeros(num_arms) # V(X_1,k, R_1,k)

        for k in range(num_arms):
            if self.var_X[k] > 1e-9: # Avoid division by zero if variance is zero
                self.omega_k[k] = self.cov_XR[k] / self.var_X[k]
            else:
                self.omega_k[k] = 0.0
            
            self.V_XR[k] = self.var_R[k] - (self.omega_k[k]**2 * self.var_X[k]) if self.var_X[k] > 1e-9 else self.var_R[k]
            self.V_XR[k] = max(0, self.V_XR[k]) # Ensure non-negative

    def _get_median_rate_estimator(self, k, current_epoch):
        """
        Calculates the median-based rate estimator for arm k. [cite: 104]
        """
        T_k = self.arm_pulls[k]
        if T_k == 0:
            return 0.0 # Or some default value if no pulls

        # m = floor(3.5 * alpha * log(n)) + 1 [cite: 104]
        m = int(np.floor(3.5 * self.alpha * np.log(current_epoch))) + 1
        
        # If T_k is very small, we might not have enough samples for 'm' groups.
        # Handle this by making 'm' equal to T_k if T_k is smaller than calculated 'm'.
        # This ensures each group has at least 1 sample, or 'm' is effectively 1 if T_k < 1.
        m = max(1, min(m, int(T_k))) 

        group_size = int(T_k // m)
        if group_size == 0: # Not enough samples to form 'm' groups of size >= 1
            # In this case, just use the empirical mean directly, or handle as needed
            # For robustness, we can just use the overall empirical mean if m > T_k
            # and still return a single rate.
            emp_X = np.mean(self.arm_samples_X[k])
            emp_R = np.mean(self.arm_samples_R[k])
            return max(0, emp_R) / max(self.b_min_cost, emp_X)

        rates_from_groups = []
        for j in range(m):
            start_idx = j * group_size
            end_idx = start_idx + group_size
            
            group_X = self.arm_samples_X[k][start_idx:end_idx]
            group_R = self.arm_samples_R[k][start_idx:end_idx]

            # Calculate empirical mean for each group [cite: 105]
            emp_X_group = np.mean(group_X)
            emp_R_group = np.mean(group_R)

            # Calculate rate for the group [cite: 105]
            # max(0, emp_R_group) and max(self.b_min_cost, emp_X_group) for stability
            group_rate = max(0, emp_R_group) / max(self.b_min_cost, emp_X_group)
            rates_from_groups.append(group_rate)
        
        # Return the median of the group rates [cite: 105]
        return np.median(rates_from_groups)

    def _get_median_empirical_X_estimator(self, k, current_epoch):
        """
        Calculates the median of empirical mean of X for arm k, used in the denominator. [cite: 105]
        """
        T_k = self.arm_pulls[k]
        if T_k == 0:
            return self.b_min_cost # Return a stable value if no pulls

        m = int(np.floor(3.5 * self.alpha * np.log(current_epoch))) + 1
        m = max(1, min(m, int(T_k))) 

        group_size = int(T_k // m)
        if group_size == 0:
            return max(self.b_min_cost, np.mean(self.arm_samples_X[k]))

        mean_X_from_groups = []
        for j in range(m):
            start_idx = j * group_size
            end_idx = start_idx + group_size
            
            group_X = self.arm_samples_X[k][start_idx:end_idx]
            mean_X_from_groups.append(np.mean(group_X))
        
        return np.median(mean_X_from_groups)


    def select_arm(self, current_total_cost, current_epoch):
        """
        Selects an arm based on the UCB-M1 strategy.

        Args:
            current_total_cost (float): The total cost accumulated so far.
            current_epoch (int): The current epoch number (n).

        Returns:
            int: The index of the selected arm.
        """
        # Ensure all arms have been pulled at least once to get initial estimates
        for k in range(self.num_arms):
            if self.arm_pulls[k] == 0:
                return k

        ucb_values = np.zeros(self.num_arms)
        log_n = np.log(current_epoch)

        for k in range(self.num_arms):
            T_k = self.arm_pulls[k]
            
            # Median-based rate estimator [cite: 105]
            r_bar_k = self._get_median_rate_estimator(k, current_epoch)
            
            # Median-based empirical mean for X in the denominator [cite: 105]
            median_emp_X_k = self._get_median_empirical_X_estimator(k, current_epoch)

            # Deviations in cost and reward [cite: 104]
            epsilon_k_n_M = 11 * np.sqrt(self.alpha * self.V_XR[k] * log_n / T_k)
            eta_k_n_M = 11 * np.sqrt(self.alpha * self.var_X[k] * log_n / T_k)

            # Stability condition check (from Proposition 2, lambda=1.28) [cite: 106]
            stability_condition_met = True
            lambda_val = 1.28
            # The denominator is (median_emp_X_k)^+ [cite: 105]
            effective_theta1 = max(self.b_min_cost, median_emp_X_k) 

            if eta_k_n_M >= effective_theta1 * (lambda_val - 1) / lambda_val:
                stability_condition_met = False
            
            if not stability_condition_met:
                c_k_n_M = np.inf # Set confidence bound to infinity if stability condition not met
            else:
                # Calculate the confidence bound term c_k,n^H [cite: 105]
                # Note: (r_bar_k - omega_k) in the numerator
                c_k_n_M_numerator = epsilon_k_n_M + (r_bar_k - self.omega_k[k]) * eta_k_n_M
                c_k_n_M = (2 * np.sqrt(2) * c_k_n_M_numerator) / effective_theta1

            ucb_values[k] = r_bar_k + c_k_n_M

        # Select the arm with the maximum UCB value
        selected_arm = np.argmax(ucb_values)
        return selected_arm

    def update_state(self, chosen_arm, cost, reward):
        """
        Updates the algorithm's internal state after an arm pull.

        Args:
            chosen_arm (int): The index of the arm that was pulled.
            cost (float): The cost incurred by pulling the arm.
            reward (float): The reward received from pulling the arm.
        """
        self.arm_pulls[chosen_arm] += 1
        self.arm_samples_X[chosen_arm].append(cost)
        self.arm_samples_R[chosen_arm].append(reward)

    def reset(self):
        """
        Resets the algorithm's state for a new simulation run.
        """
        self.arm_pulls.fill(0)
        for k in range(self.num_arms):
            self.arm_samples_X[k].clear()
            self.arm_samples_R[k].clear()
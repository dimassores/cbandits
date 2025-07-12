# src/algorithms/ucb_b2c.py

import numpy as np
from .base_bandit_algorithm import BaseBanditAlgorithm
from ..utils.estimators import calculate_empirical_mean, calculate_empirical_variance, calculate_lmmse_omega_empirical, calculate_lmmse_variance_reduction_empirical

class UCB_B2C(BaseBanditAlgorithm):
    """
    Implementation of the UCB-B2C algorithm for budget-constrained bandits.
    This algorithm is designed for bounded and correlated cost and reward distributions,
    where second-order moments (including correlation) are unknown and must be estimated
    from samples.
    """

    def __init__(self, num_arms, arm_configs, algorithm_params):
        """
        Initializes the UCB-B2C algorithm.

        Args:
            num_arms (int): The total number of arms in the bandit problem.
            arm_configs (list): A list of dictionaries, each describing an arm's properties.
                                 Expected to contain 'params' with 'M_X' and 'M_R' for boundedness.
            algorithm_params (dict): Dictionary of algorithm-specific parameters
                                     (e.g., 'alpha', 'b_min_cost', 'omega_bar').
        """
        super().__init__(num_arms, arm_configs, algorithm_params)
        
        self.alpha = algorithm_params.get('alpha', 2.1) # Default alpha if not provided
        self.b_min_cost = algorithm_params.get('b_min_cost', 0.1) # Small positive constant for denominator stability
        self.omega_bar = algorithm_params.get('omega_bar', 2.0) # Given parameter, max_k omega_k for M_Z calculation #

        # Initialize arm-specific statistics to store all observed samples
        self.arm_samples_X = [[] for _ in range(num_arms)]
        self.arm_samples_R = [[] for _ in range(num_arms)]
        self.arm_pulls = np.zeros(num_arms) # T_k(n)

        # Max bounds for costs and rewards (M_X, M_R) - assumed known for UCB-B2C #
        self.M_X = np.array([config['params']['M_X'] for config in arm_configs])
        self.M_R = np.array([config['params']['M_R'] for config in arm_configs])
        
        # M_Z = M_R + omega_bar * M_X #
        # This M_Z is used in the bias terms epsilon_k,n^B2C and nu_k,n(L_k) #
        self.M_Z = self.M_R + self.omega_bar * self.M_X # Element-wise sum

    def select_arm(self, current_total_cost, current_epoch):
        """
        Selects an arm based on the UCB-B2C strategy.

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
        
        # We use current_epoch as 'n' for log(n) term as described in the paper (e.g., log(n^alpha) -> alpha * log(n)) #
        log_n_alpha = self.alpha * np.log(current_epoch) 

        for k in range(self.num_arms):
            T_k = self.arm_pulls[k]
            
            # Empirical means
            emp_mean_X = calculate_empirical_mean(np.sum(self.arm_samples_X[k]), T_k)
            emp_mean_R = calculate_empirical_mean(np.sum(self.arm_samples_R[k]), T_k)
            
            # Rate estimator
            # max(0, emp_mean_R) to ensure non-negative reward part
            # max(self.b_min_cost, emp_mean_X) to ensure positive denominator and stability
            r_hat_k = max(0, emp_mean_R) / max(self.b_min_cost, emp_mean_X)

            # Estimate omega_k and the reduced variance L_k empirically
            # Need mean_X, mean_R, E[X^2], E[R^2], E[XR] for these
            # For simplicity, we calculate these sums for empirical estimates
            sum_X = np.sum(self.arm_samples_X[k])
            sum_R = np.sum(self.arm_samples_R[k])
            sum_X_sq = np.sum(np.array(self.arm_samples_X[k])**2)
            sum_R_sq = np.sum(np.array(self.arm_samples_R[k])**2)
            sum_XR = np.sum(np.array(self.arm_samples_X[k]) * np.array(self.arm_samples_R[k]))
            
            hat_omega_k_n = calculate_lmmse_omega_empirical(
                sum_X, sum_R, sum_X_sq, sum_R_sq, sum_XR, T_k
            )
            
            # Estimate L_k,n(hat_omega_k,n) which is hat_L_k,n(hat_omega_k,n) from the paper #
            # This is the empirical variance of (R - omega*X)
            hat_L_k_n_omega_k_n = calculate_lmmse_variance_reduction_empirical(
                sum_X, sum_R, sum_X_sq, sum_R_sq, sum_XR, T_k, hat_omega_k_n
            )
            
            # Bias terms epsilon_k,n^B2C and eta_k,n^B2C as per UCB-B2C definition (Section 6.2) #
            # epsilon_k,n^B2C = sqrt(2*hat_L_k,n(hat_omega_k,n)*log(n^alpha)/T_k) + 3*M_Z*log(n^alpha)/T_k
            epsilon_k_n_b2c = np.sqrt(2 * hat_L_k_n_omega_k_n * log_n_alpha / T_k) + \
                              (3 * self.M_Z[k] * log_n_alpha / T_k)
            
            # eta_k,n^B2C = sqrt(2*hat_V_k,n(X_k)*log(n^alpha)/T_k) + 3*M_X*log(n^alpha)/T_k
            # hat_V_k,n(X_k) is empirical variance of X
            emp_var_X = calculate_empirical_variance(sum_X_sq, sum_X, T_k)
            eta_k_n_b2c = np.sqrt(2 * emp_var_X * log_n_alpha / T_k) + \
                          (3 * self.M_X[k] * log_n_alpha / T_k)

            # Stability condition check (Proposition 2, lambda=1.28) #
            stability_condition_met = True
            lambda_val = 1.28
            
            effective_theta1 = max(self.b_min_cost, emp_mean_X) # (E_n[X_k])^+ #

            if eta_k_n_b2c >= effective_theta1 * (lambda_val - 1) / lambda_val:
                stability_condition_met = False
            
            if not stability_condition_met:
                c_k_n_b2c = np.inf # Set confidence bound to infinity if stability condition not met
            else:
                # Calculate the confidence bound term c_k,n^B2C
                # c_k,n^B2C = 1.4 * (epsilon_k,n^B2C + (r_hat_k - hat_omega_k,n)^+ * eta_k,n^B2C) / (E_n[X_k])^+ #
                # Note: (r_hat_k - hat_omega_k_n)^+ implies max(0, r_hat_k - hat_omega_k_n)
                c_k_n_b2c_numerator = epsilon_k_n_b2c + max(0, r_hat_k - hat_omega_k_n) * eta_k_n_b2c
                c_k_n_b2c = 1.4 * c_k_n_b2c_numerator / effective_theta1

            ucb_values[k] = r_hat_k + c_k_n_b2c

        # Select the arm with the maximum UCB value #
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
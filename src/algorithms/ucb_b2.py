# src/algorithms/ucb_b2.py

import numpy as np
from .base_bandit_algorithm import BaseBanditAlgorithm
from ..utils.estimators import calculate_empirical_mean, calculate_empirical_variance

class UCB_B2(BaseBanditAlgorithm):
    """
    Implementation of the UCB-B2 algorithm for budget-constrained bandits.
    This algorithm is designed for bounded and uncorrelated cost and reward distributions,
    where second-order moments are unknown and must be estimated from samples.
    """

    def __init__(self, num_arms, arm_configs, algorithm_params):
        """
        Initializes the UCB-B2 algorithm.

        Args:
            num_arms (int): The total number of arms in the bandit problem.
            arm_configs (list): A list of dictionaries, each describing an arm's properties.
                                 Expected to contain 'params' with 'M_X' and 'M_R' for boundedness.
            algorithm_params (dict): Dictionary of algorithm-specific parameters
                                     (e.g., 'alpha', 'b_min_cost').
        """
        super().__init__(num_arms, arm_configs, algorithm_params)
        
        self.alpha = algorithm_params.get('alpha', 2.1) # Default alpha if not provided
        self.b_min_cost = algorithm_params.get('b_min_cost', 0.1) # Small positive constant for denominator stability

        # Initialize arm-specific statistics
        self.arm_pulls = np.zeros(num_arms)             # T_k(n)
        self.total_costs = np.zeros(num_arms)           # Sum of X_i,k
        self.total_rewards = np.zeros(num_arms)         # Sum of R_i,k
        self.sum_sq_costs = np.zeros(num_arms)          # Sum of (X_i,k)^2 for empirical variance
        self.sum_sq_rewards = np.zeros(num_arms)        # Sum of (R_i,k)^2 for empirical variance

        # Max bounds for costs and rewards (M_X, M_R) - assumed known for UCB-B2
        self.M_X = np.array([config['params']['M_X'] for config in arm_configs])
        self.M_R = np.array([config['params']['M_R'] for config in arm_configs])

    def select_arm(self, current_total_cost, current_epoch):
        """
        Selects an arm based on the UCB-B2 strategy.

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
        
        # We use current_epoch as 'n' for log(n) term as described in the paper (e.g., log(n^alpha) -> alpha * log(n))
        log_n_alpha = self.alpha * np.log(current_epoch) 

        for k in range(self.num_arms):
            T_k = self.arm_pulls[k]
            
            # Empirical estimates for mu_X and mu_R
            emp_mean_X = calculate_empirical_mean(self.total_costs[k], T_k)
            emp_mean_R = calculate_empirical_mean(self.total_rewards[k], T_k)
            
            # Empirical variance estimates
            # For uncorrelated case, Var(R_1,k - omega*X_1,k) simplifies to Var(R_1,k) if omega=0
            # Since correlation is not exploited here, we use empirical variances of R and X separately.
            emp_var_R = calculate_empirical_variance(self.sum_sq_rewards[k], self.total_rewards[k], T_k)
            emp_var_X = calculate_empirical_variance(self.sum_sq_costs[k], self.total_costs[k], T_k)
            
            # Rate estimator
            # max(0, emp_mean_R) to ensure non-negative reward part
            # max(self.b_min_cost, emp_mean_X) to ensure positive denominator and stability
            r_hat_k = max(0, emp_mean_R) / max(self.b_min_cost, emp_mean_X)

            # Bias terms epsilon_k,n^B2 and eta_k,n^B2 as per UCB-B2 definition (Section 6.1)
            # sqrt(2*V_hat*log(n^alpha)/T_k) + 3*M*log(n^alpha)/T_k
            epsilon_k_n_b2 = np.sqrt(2 * emp_var_R * log_n_alpha / T_k) + \
                             (3 * self.M_R[k] * log_n_alpha / T_k)
            
            eta_k_n_b2 = np.sqrt(2 * emp_var_X * log_n_alpha / T_k) + \
                         (3 * self.M_X[k] * log_n_alpha / T_k)

            # Stability condition check (Proposition 2, lambda=1.28)
            stability_condition_met = True
            lambda_val = 1.28
            
            effective_theta1 = max(self.b_min_cost, emp_mean_X) # (E_n[X_k])^+

            if eta_k_n_b2 >= effective_theta1 * (lambda_val - 1) / lambda_val:
                stability_condition_met = False
            
            if not stability_condition_met:
                c_k_n_b2 = np.inf # Set confidence bound to infinity if stability condition not met
            else:
                # Calculate the confidence bound term c_k,n^B2 (Eq. 17)
                # For uncorrelated case, (r_hat_k - omega_k) becomes approximately r_hat_k
                # as omega_k would be 0 if genuinely uncorrelated.
                # The paper states: c_k,n^B2 = 1.4 * (epsilon_k,n^B2 + r_hat_k * eta_k,n^B2) / (E_n[X_k])^+
                c_k_n_b2_numerator = epsilon_k_n_b2 + r_hat_k * eta_k_n_b2
                c_k_n_b2 = 1.4 * c_k_n_b2_numerator / effective_theta1

            ucb_values[k] = r_hat_k + c_k_n_b2

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
        self.total_costs[chosen_arm] += cost
        self.total_rewards[chosen_arm] += reward
        self.sum_sq_costs[chosen_arm] += cost**2
        self.sum_sq_rewards[chosen_arm] += reward**2

    def reset(self):
        """
        Resets the algorithm's state for a new simulation run.
        """
        self.arm_pulls.fill(0)
        self.total_costs.fill(0)
        self.total_rewards.fill(0)
        self.sum_sq_costs.fill(0)
        self.sum_sq_rewards.fill(0)
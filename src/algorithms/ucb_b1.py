# src/algorithms/ucb_b1.py

import numpy as np
from .base_bandit_algorithm import BaseBanditAlgorithm
from ..utils.estimators import calculate_empirical_mean, calculate_lmmse_omega_empirical, calculate_lmmse_variance_reduction_empirical

class UCB_B1(BaseBanditAlgorithm):
    """
    Implementation of the UCB-B1 algorithm for budget-constrained bandits.
    This algorithm assumes known second-order moments and exploits cost-reward correlation.
    It is designed for sub-Gaussian cost and reward distributions.
    """

    def __init__(self, num_arms, arm_configs, algorithm_params):
        """
        Initializes the UCB-B1 algorithm.

        Args:
            num_arms (int): The total number of arms in the bandit problem.
            arm_configs (list): A list of dictionaries, each describing an arm's properties.
                                 Expected to contain 'params' with 'mean_X', 'mean_R',
                                 'var_X', 'var_R', 'cov_XR' for second-order moments.
            algorithm_params (dict): Dictionary of algorithm-specific parameters
                                     (e.g., 'alpha', 'L', 'b_min_cost').
        """
        super().__init__(num_arms, arm_configs, algorithm_params)
        
        self.alpha = algorithm_params.get('alpha', 2.1) # Default alpha if not provided
        self.L = algorithm_params.get('L', 2)           # Default L if not provided
        self.b_min_cost = algorithm_params.get('b_min_cost', 0.1) # Small positive constant for denominator stability # 

        # Initialize arm-specific statistics
        self.arm_pulls = np.zeros(num_arms)             # T_k(n) # 
        self.total_costs = np.zeros(num_arms)           # Sum of X_i,k
        self.total_rewards = np.zeros(num_arms)         # Sum of R_i,k

        # Store known second-order moments # 
        self.var_X = np.array([config['params']['var_X'] for config in arm_configs])
        self.var_R = np.array([config['params']['var_R'] for config in arm_configs])
        self.cov_XR = np.array([config['params']['cov_XR'] for config in arm_configs])
        
        # Calculate omega_k and V(X, R) (Eq. 8, 9)
        self.omega_k = np.zeros(num_arms)
        self.V_XR = np.zeros(num_arms) # V(X_1,k, R_1,k)

        for k in range(num_arms):
            if self.var_X[k] > 1e-9: # Avoid division by zero if variance is zero
                self.omega_k[k] = self.cov_XR[k] / self.var_X[k] # 
            else:
                self.omega_k[k] = 0.0 # If Var(X)=0, omega is undefined, can be treated as 0 for practical purposes
            
            # V(X_1,k, R_1,k) = Var(R_1,k) - omega_k^2 * Var(X_1,k) # 
            # If Var(X_1,k)=0, then V(X_1,k, R_1,k) = Var(R_1,k) # 
            self.V_XR[k] = self.var_R[k] - (self.omega_k[k]**2 * self.var_X[k]) if self.var_X[k] > 1e-9 else self.var_R[k] # 
            # Ensure V_XR is non-negative due to potential floating point inaccuracies
            self.V_XR[k] = max(0, self.V_XR[k])

        # M_X and M_R for bounded case (if applicable, from Theorem 1.1) # 
        # For jointly Gaussian, M_X=M_R=0 is stated, but for practical UCB bounds,
        # we still need to consider some "effective" bound if we were to use the first part of Theorem 1
        # For now, let's assume M_X and M_R are provided in algorithm_params for UCB-B1 if relevant
        # or we might infer them from the distribution if it's truly bounded.
        # As per the paper, for jointly Gaussian, M_X=M_R=0 # 
        self.M_X = algorithm_params.get('M_X', 0.0) # Used in specific UCB-B1 bound, set to 0 for Gaussian # 
        self.M_R = algorithm_params.get('M_R', 0.0) # Used in specific UCB-B1 bound, set to 0 for Gaussian # 


    def select_arm(self, current_total_cost, current_epoch):
        """
        Selects an arm based on the UCB-B1 strategy.

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
            
            # Empirical estimates for mu_X and mu_R # 
            # Using max(self.b_min_cost, ...) to prevent division by zero for the denominator
            emp_mean_X = self.total_costs[k] / T_k
            emp_mean_R = self.total_rewards[k] / T_k
            
            # Rate estimator # 
            # max(0, emp_mean_R) to ensure non-negative reward part
            # max(self.b_min_cost, emp_mean_X) to ensure positive denominator and stability # 
            r_hat_k = max(0, emp_mean_R) / max(self.b_min_cost, emp_mean_X) # 

            # Calculate epsilon_k,n^g and eta_k,n^g for the confidence bound # 
            # M_R and M_X are considered 0 for the Jointly Gaussian case in Theorem 1.2 # 
            epsilon_k_n_g = (2 * self.alpha * self.M_R * log_n / (3 * T_k)) + \
                            np.sqrt(self.L * self.alpha * self.V_XR[k] * log_n / T_k) # 
            
            eta_k_n_g = (2 * self.alpha * self.M_X * log_n / (3 * T_k)) + \
                        np.sqrt(self.L * self.alpha * self.var_X[k] * log_n / T_k) # 

            # Stability condition check: eta < theta_1 * (lambda - 1) / lambda # 
            # Here, theta_1 is emp_mean_X and lambda = 1.28 # 
            # Check if emp_mean_X is sufficiently positive for stability
            stability_condition_met = True
            if emp_mean_X < self.b_min_cost: # If emp_mean_X is too close to zero
                stability_condition_met = False
            else:
                # The condition is η < θ₁ for λ > 1.
                # The term (lambda-1)/lambda is close to 1 for lambda=1.28.
                # A simpler interpretation of the stability condition (Remark 1) is η < θ₁ # 
                # Let's use the explicit condition from Proposition 2 and the context around it.
                # Proposition 2 condition: eta_k_n_g < emp_mean_X * (lambda - 1) / lambda with lambda = 1.28 # 
                lambda_val = 1.28
                if eta_k_n_g >= emp_mean_X * (lambda_val - 1) / lambda_val:
                    stability_condition_met = False
            
            if not stability_condition_met:
                c_k_n_b1 = np.inf # Set confidence bound to infinity if stability condition not met # 
            else:
                # Calculate the confidence bound term c_k,n^b1 # 
                # Note: (r_hat_k - omega_k) for the epsilon term in the numerator # 
                # Ensure the denominator (empirical_mean_X) is positive and stable
                c_k_n_b1_numerator = epsilon_k_n_g + (r_hat_k - self.omega_k[k]) * eta_k_n_g # 
                c_k_n_b1_denominator = max(self.b_min_cost, emp_mean_X) # (E_n[X_k])^+ # 
                c_k_n_b1 = 1.4 * c_k_n_b1_numerator / c_k_n_b1_denominator # 

            ucb_values[k] = r_hat_k + c_k_n_b1 # 

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
        self.arm_pulls[chosen_arm] += 1 # 
        self.total_costs[chosen_arm] += cost # 
        self.total_rewards[chosen_arm] += reward # 

    def reset(self):
        """
        Resets the algorithm's state for a new simulation run.
        """
        self.arm_pulls.fill(0)
        self.total_costs.fill(0)
        self.total_rewards.fill(0)
# src/utils/estimators.py

import numpy as np

def calculate_empirical_mean(total_sum: float, num_pulls: int) -> float:
    """
    Calculates the empirical mean of observations.

    Args:
        total_sum (float): The sum of all observed values.
        num_pulls (int): The number of observations.

    Returns:
        float: The empirical mean. Returns 0.0 if num_pulls is 0 to avoid division by zero.
    """
    if num_pulls == 0:
        return 0.0
    return total_sum / num_pulls

def calculate_empirical_variance(sum_sq_values: float, total_sum: float, num_pulls: int) -> float:
    """
    Calculates the empirical variance of observations.
    Uses the formula: Var(X) = E[X^2] - (E[X])^2.
    For sample variance, uses: (sum_sq_values / N) - (total_sum / N)^2.

    Args:
        sum_sq_values (float): The sum of squared observed values.
        total_sum (float): The sum of observed values.
        num_pulls (int): The number of observations.

    Returns:
        float: The empirical variance. Returns 0.0 if num_pulls is 0 or 1.
               Ensures non-negative result due to floating point inaccuracies.
    """
    if num_pulls < 2: # Variance requires at least 2 samples, or is 0 for 1 sample
        return 0.0
    
    mean = total_sum / num_pulls
    mean_sq = sum_sq_values / num_pulls
    variance = mean_sq - (mean ** 2)
    
    return max(0.0, variance) # Ensure non-negative variance

def calculate_lmmse_omega_empirical(sum_X: float, sum_R: float, sum_X_sq: float, sum_R_sq: float, sum_XR: float, num_pulls: int) -> float:
    """
    Calculates the empirical Linear Minimum Mean Square Error (LMMSE) estimator omega_k.
    omega_k = Cov(X, R) / Var(X).
    Empirical Cov(X,R) = E[XR] - E[X]E[R]
    Empirical Var(X) = E[X^2] - (E[X])^2

    Args:
        sum_X (float): Sum of cost values.
        sum_R (float): Sum of reward values.
        sum_X_sq (float): Sum of squared cost values.
        sum_R_sq (float): Sum of squared reward values.
        sum_XR (float): Sum of (cost * reward) products.
        num_pulls (int): Number of observations.

    Returns:
        float: The empirical LMMSE omega_k. Returns 0.0 if num_pulls is 0 or if empirical variance of X is zero.
    """
    if num_pulls < 2: # Need at least 2 samples to estimate covariance/variance reliably
        return 0.0

    emp_mean_X = calculate_empirical_mean(sum_X, num_pulls)
    emp_mean_R = calculate_empirical_mean(sum_R, num_pulls)

    emp_var_X = calculate_empirical_variance(sum_X_sq, sum_X, num_pulls)

    # If empirical variance of X is too small (near zero), omega is ill-defined
    if emp_var_X < 1e-9: # A small threshold to prevent division by near-zero
        return 0.0

    emp_mean_XR = calculate_empirical_mean(sum_XR, num_pulls)
    emp_cov_XR = emp_mean_XR - (emp_mean_X * emp_mean_R)

    return emp_cov_XR / emp_var_X

def calculate_lmmse_variance_reduction_empirical(sum_X: float, sum_R: float, sum_X_sq: float, sum_R_sq: float, sum_XR: float, num_pulls: int, omega: float) -> float:
    """
    Calculates the empirical minimum variance V(X,R) = Var(R - omega*X).
    This is also L_hat_k,n(hat_omega_k,n) from the paper.
    Var(R - omega*X) = Var(R) + omega^2 * Var(X) - 2 * omega * Cov(X,R)

    Args:
        sum_X (float): Sum of cost values.
        sum_R (float): Sum of reward values.
        sum_X_sq (float): Sum of squared cost values.
        sum_R_sq (float): Sum of squared reward values.
        sum_XR (float): Sum of (cost * reward) products.
        num_pulls (int): Number of observations.
        omega (float): The LMMSE omega value (can be the empirical omega).

    Returns:
        float: The empirical reduced variance. Returns 0.0 if num_pulls is 0 or 1.
               Ensures non-negative result.
    """
    if num_pulls < 2:
        return 0.0
    
    emp_var_R = calculate_empirical_variance(sum_R_sq, sum_R, num_pulls)
    emp_var_X = calculate_empirical_variance(sum_X_sq, sum_X, num_pulls)

    emp_mean_X = calculate_empirical_mean(sum_X, num_pulls)
    emp_mean_R = calculate_empirical_mean(sum_R, num_pulls)
    emp_mean_XR = calculate_empirical_mean(sum_XR, num_pulls)
    emp_cov_XR = emp_mean_XR - (emp_mean_X * emp_mean_R)

    # V(X,R) = Var(R) + omega^2 * Var(X) - 2 * omega * Cov(X,R)
    # This is equivalent to min_omega' Var(R - omega'X)
    # which is Var(R) - (Cov(X,R)^2 / Var(X))
    # Or, the definition from the paper's section 4.2: Var(R_1,k) - omega_k^2 * Var(X_1,k)
    # We use the empirical versions of these values
    
    reduced_variance = emp_var_R - (omega**2 * emp_var_X) # Using the formula derived from orthogonality principle

    # Ensure non-negative value
    return max(0.0, reduced_variance)
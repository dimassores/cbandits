# config/simulation_config.py

import numpy as np

# --- General Simulation Settings ---
NUM_RUNS = 200 # Number of independent simulation runs for averaging results [cite: 1]
# Budgets (B) to test. These represent the total cost budget for the bandit problem.
# Varying these allows observation of regret scaling.
BUDGETS = [1000, 2000, 5000, 10000, 20000, 50000, 100000] # Example budgets [cite: 1]

# --- Arm Configurations ---
# Each dictionary in this list defines a single arm (bandit machine).
# 'name': A descriptive name for the arm.
# 'type': The distribution type for (Cost, Reward) pairs.
#         Supported types: 'gaussian', 'heavy_tailed', 'bounded_uniform'.
# 'params': A dictionary of parameters specific to the 'type' of distribution.
#           - For 'gaussian': 'mean_X', 'mean_R', 'var_X', 'var_R', 'cov_XR'
#           - For 'heavy_tailed': 'mean_X', 'mean_R', 'alpha_pareto_X', 'loc_pareto_X',
#                                 'mean_lognormal_R', 'sigma_lognormal_R', 'correlation'.
#                                 (Note: Correlation for heavy-tailed is simplified here;
#                                  exact correlation for arbitrary heavy-tailed is complex).
#           - For 'bounded_uniform': 'min_X', 'max_X', 'min_R', 'max_R', 'correlation'.
#                                    Also requires 'mean_X', 'mean_R', 'var_X', 'var_R', 'cov_XR'
#                                    (true values) for environment's optimal rate calculation.
#           - 'M_X', 'M_R': Maximum bounds for cost and reward, used by UCB-B2 and UCB-B2C
#                           (even if not directly sampled from a bounded distribution, these
#                           are needed for the algorithm's confidence bounds). For Gaussian,
#                           these can be set to 0 as per paper for B1, but for B2/B2C,
#                           they are essential.

ARM_CONFIGS = [
    {
        "name": "Arm A (Optimal Gaussian)",
        "type": "gaussian",
        "params": {
            "mean_X": 1.0,  # Expected Cost [cite: 1]
            "mean_R": 2.5,  # Expected Reward [cite: 1]
            "var_X": 0.1,   # Variance of Cost [cite: 1]
            "var_R": 0.3,   # Variance of Reward [cite: 1]
            "cov_XR": 0.05, # Covariance (X,R) [cite: 1]
            "M_X": 10.0,    # Max possible cost for bounded algorithm logic [cite: 1]
            "M_R": 10.0,    # Max possible reward for bounded algorithm logic [cite: 1]
        }
    },
    {
        "name": "Arm B (Suboptimal Gaussian)",
        "type": "gaussian",
        "params": {
            "mean_X": 1.2,
            "mean_R": 2.0, # Lower reward rate than Arm A (2.0/1.2 approx 1.67 vs 2.5/1.0 = 2.5) [cite: 1]
            "var_X": 0.2,
            "var_R": 0.4,
            "cov_XR": 0.02,
            "M_X": 10.0,
            "M_R": 10.0,
        }
    },
    {
        "name": "Arm C (Heavy-Tailed Cost)",
        "type": "heavy_tailed",
        "params": {
            "mean_X": 1.5, # True mean_X for regret calc. (actual Pareto mean is alpha*loc/(alpha-1)) [cite: 1]
            "mean_R": 3.5, # True mean_R for regret calc. (actual LogNormal mean is exp(mu + sigma^2/2)) [cite: 1]
            "alpha_pareto_X": 2.5, # Shape parameter for Pareto cost. (alpha > 2 for Var, >1 for Mean) [cite: 1]
            "loc_pareto_X": 0.8,   # Scale parameter (min value) for Pareto cost [cite: 1]
            "mean_lognormal_R": 1.2, # mu for underlying normal distribution for lognormal reward [cite: 1]
            "sigma_lognormal_R": 0.6, # sigma for underlying normal distribution for lognormal reward [cite: 1]
            "correlation": 0.1,    # Simplified correlation factor for heavy-tailed [cite: 1]
            # For heavy-tailed, you'd usually compute these from distribution properties if you want consistency
            # For simplicity, these are placeholders for environment's optimal rate calculation and algorithm's internal reference
            "var_X": 2.0, # Placeholder, depends on Pareto params [cite: 1]
            "var_R": 2.0, # Placeholder, depends on Lognormal params [cite: 1]
            "cov_XR": 0.15, # Placeholder, depends on correlation modeling [cite: 1]
            "M_X": 100.0, # Set a higher bound for heavy-tailed for algorithms needing it [cite: 1]
            "M_R": 100.0, # Set a higher bound for heavy-tailed for algorithms needing it [cite: 1]
        }
    },
    {
        "name": "Arm D (Bounded Uniform)",
        "type": "bounded_uniform",
        "params": {
            "min_X": 0.5, "max_X": 1.5, # Uniform cost range [cite: 1]
            "min_R": 1.0, "max_R": 3.0, # Uniform reward range [cite: 1]
            "correlation": 0.0,         # No correlation for this specific uniform arm [cite: 1]
            # True means and variances for uniform distributions
            "mean_X": (0.5 + 1.5) / 2, "mean_R": (1.0 + 3.0) / 2, # True means [cite: 1]
            "var_X": (1.5 - 0.5)**2 / 12, "var_R": (3.0 - 1.0)**2 / 12, # True variances [cite: 1]
            "cov_XR": 0.0, # True covariance [cite: 1]
            "M_X": 1.5,    # Max cost for bounded algorithm logic [cite: 1]
            "M_R": 3.0,    # Max reward for bounded algorithm logic [cite: 1]
        }
    }
]

# --- Algorithm-Specific Parameters ---
# These parameters are passed to the __init__ method of each algorithm.
ALGORITHM_PARAMS = {
    "UCB-B1": {
        "alpha": 2.1,  # Constant for log(n) term in confidence bound [cite: 1]
        "L": 2,        # Constant for sqrt(log(n)/T_k) term in confidence bound [cite: 1]
        "b_min_cost": 0.01, # Small constant to ensure denominator stability (epsilon_0 in some papers) [cite: 1]
        "M_X": 10.0,   # Used for bounded version of B1 (Theorem 1.1), can be ignored for pure Gaussian (Theorem 1.2 M_X=0) [cite: 1]
        "M_R": 10.0,   # Used for bounded version of B1 (Theorem 1.1) [cite: 1]
    },
    "UCB-M1": {
        "alpha": 2.1,      # Constant for log(n) term [cite: 1]
        "b_min_cost": 0.01, # Small constant for stability [cite: 1]
    },
    "UCB-B2": {
        "alpha": 2.1,      # Constant for log(n) term [cite: 1]
        "b_min_cost": 0.01, # Small constant for stability [cite: 1]
    },
    "UCB-B2C": {
        "alpha": 2.1,      # Constant for log(n) term [cite: 1]
        "b_min_cost": 0.01, # Small constant for stability [cite: 1]
        "omega_bar": 5.0,  # Upper bound for |omega_k| used in M_Z calculation [cite: 1]
    },
    # Add parameters for other algorithms as needed
}

# --- Derived Global Parameters (for regret calculation reference) ---
# The minimum expected cost across all arms (mu_star from the paper, Section 2) [cite: 1]
# This is used in the regret bounds (e.g., log(2B/mu_star))
MIN_EXPECTED_COST = min(arm["params"]["mean_X"] for arm in ARM_CONFIGS)
if MIN_EXPECTED_COST <= 0:
    # A warning or error is appropriate as the theory often assumes positive expected cost.
    raise ValueError("MIN_EXPECTED_COST must be greater than 0 for stable reward rate calculations.")
# src/environments/general_cost_reward_env.py

import numpy as np
from .bandit_environment import BanditEnvironment

class GeneralCostRewardEnvironment(BanditEnvironment):
    """
    Implements a multi-armed bandit environment with general cost and reward distributions.
    Supports jointly Gaussian, heavy-tailed (e.g., Pareto, Lognormal for cost),
    and potentially bounded distributions.
    """

    def __init__(self, num_arms: int, arm_configs: list, seed: int = None):
        """
        Initializes the General Cost-Reward Bandit Environment.

        Args:
            num_arms (int): The total number of arms.
            arm_configs (list): A list of dictionaries, each defining an arm's properties.
                                 Each config must include:
                                 - 'name' (str)
                                 - 'type' (str): 'gaussian', 'heavy_tailed', 'bounded_uniform' (example)
                                 - 'params' (dict):
                                     - For 'gaussian': 'mean_X', 'mean_R', 'var_X', 'var_R', 'cov_XR'
                                     - For 'heavy_tailed': 'mean_X', 'mean_R', 'alpha_pareto_X', 'loc_pareto_X',
                                                           'mean_lognormal_R', 'sigma_lognormal_R', 'correlation' (if any)
                                     - For 'bounded_uniform': 'min_X', 'max_X', 'min_R', 'max_R', 'correlation' (if any)
                                     - All should have 'mean_X', 'mean_R' for optimal rate calculation in base class.
                                     - 'var_X', 'var_R', 'cov_XR' (true values, even if unknown to algorithm)
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__(num_arms, arm_configs)
        self.rng = np.random.default_rng(seed) # Use NumPy's new Generator for reproducibility

        # Pre-process arm configurations for sampling
        self._arm_samplers = []
        for k, config in enumerate(self.arm_configs):
            arm_type = config['type']
            params = config['params']

            # Validate essential parameters for the base class's optimal rate calculation
            if 'mean_X' not in params or 'mean_R' not in params:
                raise ValueError(f"Arm {k} config missing 'mean_X' or 'mean_R' in params.")
            if params['mean_X'] <= 0:
                print(f"Warning: Arm {k} has non-positive expected cost ({params['mean_X']}). "
                      "This might lead to infinite reward rates and undefined behavior for some algorithms.")

            if arm_type == 'gaussian':
                # For jointly Gaussian, we need a covariance matrix
                cov_matrix = np.array([
                    [params['var_X'], params['cov_XR']],
                    [params['cov_XR'], params['var_R']]
                ])
                # Ensure covariance matrix is positive semi-definite
                try:
                    _ = np.linalg.cholesky(cov_matrix)
                except np.linalg.LinAlgError:
                    raise ValueError(f"Covariance matrix for arm {k} is not positive semi-definite: {cov_matrix}")
                
                self._arm_samplers.append({
                    'type': 'gaussian',
                    'mean': np.array([params['mean_X'], params['mean_R']]),
                    'cov': cov_matrix
                })
            elif arm_type == 'heavy_tailed':
                # Example: Pareto for cost, Lognormal for reward.
                # For simplicity, if correlation needs to be modeled, it needs to be
                # done carefully, e.g., by sampling a common latent variable.
                # Here, we'll assume a basic independent sampling for demonstration,
                # or a simple linear transformation if correlation is specified.
                
                # Check for required parameters for heavy-tailed
                if 'alpha_pareto_X' not in params or 'loc_pareto_X' not in params or \
                   'mean_lognormal_R' not in params or 'sigma_lognormal_R' not in params:
                   raise ValueError(f"Heavy-tailed arm {k} config missing required parameters.")

                # If correlation is specified, we will need a more complex sampling logic
                # For simplicity in this example, we'll draw independently and then
                # potentially adjust if a simple linear correlation model is assumed.
                # The paper's core models the (X,R) pair, suggesting joint sampling for correlation.
                
                # A more general approach for correlated heavy-tailed:
                # 1. Sample a latent normal variable Z.
                # 2. Transform Z to X and R using inverse CDFs, incorporating correlation.
                # For this basic implementation, we'll use a simplified approach assuming
                # that if correlation is given, it's about the (X,R) pair, but the sampling
                # might not perfectly match the *desired* correlation if distributions are very non-Gaussian.
                # A common approach to introduce correlation for arbitrary marginals is via copulas,
                # but that's beyond a simple environment implementation.
                
                self._arm_samplers.append({
                    'type': 'heavy_tailed',
                    'params_X': {'alpha': params['alpha_pareto_X'], 'loc': params['loc_pareto_X']},
                    'params_R': {'mean': params['mean_lognormal_R'], 'sigma': params['sigma_lognormal_R']},
                    'correlation': params.get('correlation', 0.0) # Simple correlation for illustration
                })
            elif arm_type == 'bounded_uniform':
                if 'min_X' not in params or 'max_X' not in params or \
                   'min_R' not in params or 'max_R' not in params:
                   raise ValueError(f"Bounded uniform arm {k} config missing required parameters.")
                self._arm_samplers.append({
                    'type': 'bounded_uniform',
                    'min_X': params['min_X'],
                    'max_X': params['max_X'],
                    'min_R': params['min_R'],
                    'max_R': params['max_R'],
                    'correlation': params.get('correlation', 0.0) # Simple correlation for illustration
                })
            else:
                raise ValueError(f"Unsupported arm type: {arm_type}")

    def pull_arm(self, arm_index: int) -> tuple[float, float]:
        """
        Simulates pulling a specific arm and returns a (cost, reward) pair.

        Args:
            arm_index (int): The index of the arm to pull.

        Returns:
            tuple[float, float]: A tuple containing the (cost, reward) sample.
        """
        if not (0 <= arm_index < self.num_arms):
            raise ValueError("Invalid arm_index.")

        sampler = self._arm_samplers[arm_index]
        arm_type = sampler['type']

        if arm_type == 'gaussian':
            # Samples a [cost, reward] pair from a multivariate normal distribution
            sample = self.rng.multivariate_normal(sampler['mean'], sampler['cov'])
            cost, reward = sample[0], sample[1]
        elif arm_type == 'heavy_tailed':
            # For heavy-tailed distributions, we can use Pareto for cost and Lognormal for reward.
            # Modeling correlation between non-Gaussian heavy-tailed distributions is complex
            # and often involves copulas. For a simple demo, we'll draw independently.
            # If `correlation` is > 0, we can introduce a simple linear dependency,
            # but it won't guarantee the exact correlation or marginals.
            
            # Pareto distribution for cost (a, m where m is scale, a is shape parameter)
            # numpy.random.pareto(a, size) + 1 to shift to be >= 1 or custom loc
            # Let's adjust for desired mean_X and loc
            # The numpy pareto distribution is actually Pareto Type II (Lomax)
            # For classical Pareto (Type I), X = (Z + 1) * m, where Z is Lomax(alpha).
            # The parameters in the config should correspond to how numpy generates.
            # For numpy.random.pareto(a), it draws from Pareto II shifted to start at 0.
            # So, if we want min_X = loc_pareto_X, we can do rng.pareto(alpha) + loc.
            
            # Simple Pareto: X_m * (1 - U)^(-1/alpha) where U~Uniform(0,1)
            # np.random.pareto(a) generates samples x such that P(x > x_m) = (x_m / x)^alpha
            # where x_m is implicitly 1 by default, and we can scale it.
            # If the config specifies 'loc_pareto_X' as the minimum value (x_m),
            # then we generate `rng.pareto(alpha) * loc_pareto_X + loc_pareto_X` is one way.
            # Or simpler: `rng.pareto(alpha)` gives values >=0.
            # The paper states: `rng.pareto(shape_param) + scale_param` for classical Pareto
            # Let's use `rng.pareto(a)` then scale/shift.
            
            alpha_pareto_X = sampler['params_X']['alpha']
            loc_pareto_X = sampler['params_X']['loc']
            
            # Generate Pareto samples starting from loc_pareto_X
            # (np.random.pareto(a) + 1) * xm generates samples >= xm (where xm is loc)
            # The numpy.random.pareto(a) function generates samples from a Pareto II (Lomax) distribution,
            # which is essentially a Pareto distribution shifted such that its minimum value is 0.
            # To obtain a classical Pareto distribution (Pareto I) with minimum value $x_m$,
            # one common way is to generate `(rng.pareto(alpha_pareto_X) + 1) * loc_pareto_X`.
            # However, the tutorialspoint link suggests `rng.pareto(a=shape_param, size=10) + 1` for xm=1.
            # Let's assume 'loc_pareto_X' is the `xm` (scale) parameter, and `alpha_pareto_X` is `a` (shape).
            # For a true Pareto I with minimum `x_m`, we often use `x_m / (U^(1/alpha))`.
            # Or using `numpy.random.pareto(alpha)` directly: `cost = (self.rng.pareto(alpha_pareto_X) + 1) * loc_pareto_X`
            # The exact definition of `loc_pareto_X` depends on the interpretation.
            # Let's assume for simplicity it is the scale parameter $x_m$.
            cost = (self.rng.pareto(alpha_pareto_X) + 1) * loc_pareto_X # Ensuring minimum value is not 0 or negative
            
            # Lognormal for reward (mean, sigma of the underlying normal distribution)
            mean_lognormal_R = sampler['params_R']['mean']
            sigma_lognormal_R = sampler['params_R']['sigma']
            reward = self.rng.lognormal(mean_lognormal_R, sigma_lognormal_R)

            # Simplified correlation: Mix with a common component (crude but illustrative)
            # This doesn't guarantee the exact correlation or marginals for non-Gaussian
            # heavy-tailed distributions, but can introduce some dependency.
            if sampler['correlation'] != 0:
                common_factor = self.rng.normal(0, 1) # A common latent variable
                cost = cost + sampler['correlation'] * common_factor
                reward = reward + sampler['correlation'] * common_factor

        elif arm_type == 'bounded_uniform':
            # Bounded Uniform for both cost and reward
            cost = self.rng.uniform(sampler['min_X'], sampler['max_X'])
            reward = self.rng.uniform(sampler['min_R'], sampler['max_R'])
            
            # Simple correlation (e.g., mixing for bounded distributions)
            if sampler['correlation'] != 0:
                common_factor = self.rng.uniform(0, 1)
                cost = cost + sampler['correlation'] * (common_factor - 0.5) * (sampler['max_X'] - sampler['min_X'])
                reward = reward + sampler['correlation'] * (common_factor - 0.5) * (sampler['max_R'] - sampler['min_R'])

        else:
            # Should not happen due to __init__ validation
            raise ValueError(f"Unknown arm type: {arm_type}")

        return cost, reward

    def reset(self):
        """
        Resets the environment's state for a new simulation run.
        For this environment, simply re-initialize the random number generator
        if a fixed seed was provided, or do nothing if using default non-seeded behavior
        (as default_rng handles its own state across calls).
        """
        # If a seed was passed to the constructor, reset rng to ensure identical runs
        # across multiple simulations. If not, rng.default_rng() manages its own state
        # but will not produce identical sequences across `reset` calls.
        if hasattr(self, '_initial_seed') and self._initial_seed is not None:
            self.rng = np.random.default_rng(self._initial_seed)
        # If no initial_seed was stored, we just let rng continue its sequence.
        # For full reproducibility of multiple runs *across sessions*, ensure `seed` is passed.
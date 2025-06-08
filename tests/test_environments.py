# tests/test_environments.py

import unittest
import numpy as np
import scipy.stats as stats

# Import environment classes
from src.environments import BanditEnvironment, GeneralCostRewardEnvironment

# Define common arm configurations for testing
# Ensure these match the types expected by GeneralCostRewardEnvironment
TEST_ARM_CONFIGS_GAUSSIAN = [
    {"name": "Arm 0", "type": "gaussian", "params": {"mean_X": 1.0, "mean_R": 2.0, "var_X": 0.1, "var_R": 0.2, "cov_XR": 0.05}},
    {"name": "Arm 1", "type": "gaussian", "params": {"mean_X": 1.1, "mean_R": 2.5, "var_X": 0.15, "var_R": 0.25, "cov_XR": 0.08}}, # Higher rate arm
]

TEST_ARM_CONFIGS_BOUNDED = [
    {"name": "Arm 0", "type": "bounded_uniform", "params": {"min_X": 0.5, "max_X": 1.5, "min_R": 1.0, "max_R": 3.0, "correlation": 0.0, "mean_X": 1.0, "mean_R": 2.0, "var_X": (1.5-0.5)**2/12, "var_R": (3.0-1.0)**2/12, "cov_XR": 0.0}},
    {"name": "Arm 1", "type": "bounded_uniform", "params": {"min_X": 0.8, "max_X": 1.8, "min_R": 1.2, "max_R": 2.5, "correlation": 0.5, "mean_X": 1.3, "mean_R": 1.85, "var_X": (1.8-0.8)**2/12, "var_R": (2.5-1.2)**2/12, "cov_XR": 0.0}}, # Adding some correlation
]

TEST_ARM_CONFIGS_HEAVY_TAILED = [
    {"name": "Arm 0", "type": "heavy_tailed", "params": {"mean_X": 1.5, "mean_R": 3.0, "alpha_pareto_X": 3.0, "loc_pareto_X": 1.0, "mean_lognormal_R": 1.0, "sigma_lognormal_R": 0.5, "correlation": 0.0, "var_X": 1.0, "var_R": 1.0, "cov_XR": 0.0}},
    {"name": "Arm 1", "type": "heavy_tailed", "params": {"mean_X": 1.2, "mean_R": 2.8, "alpha_pareto_X": 2.5, "loc_pareto_X": 0.8, "mean_lognormal_R": 1.2, "sigma_lognormal_R": 0.6, "correlation": 0.2, "var_X": 1.5, "var_R": 1.2, "cov_XR": 0.1}}, # Adding some correlation
]

class TestBanditEnvironment(unittest.TestCase):

    def test_base_bandit_environment_init(self):
        """Test BanditEnvironment initialization, including abstract class behavior."""
        with self.assertRaises(ValueError):
            BanditEnvironment(num_arms=0, arm_configs=[])
        
        # Test that trying to instantiate the abstract class directly raises a TypeError
        with self.assertRaises(TypeError):
            BanditEnvironment(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1])
        
        # The following assertions were for a successful instantiation,
        # but BanditEnvironment is abstract, so they are not reachable.
        # These checks are implicitly covered by tests for concrete subclasses like GeneralCostRewardEnvironment.
        # self.assertEqual(env.num_arms, 1)
        # self.assertAlmostEqual(env.get_optimal_reward_rate(), 2.0 / 1.0)
        # self.assertEqual(env.optimal_arm_index, 0)
        # self.assertAlmostEqual(env.get_optimal_arm_expected_cost(), 1.0)

        # Test BanditEnvironment's *internal logic* for optimal rate calculation
        # using a mock to bypass instantiation issue, or simply rely on concrete class tests
        # For simplicity, we'll rely on the GeneralCostRewardEnvironment tests for this.
        # However, for thoroughness of the base class logic itself, one could do:
        #
        # from unittest.mock import MagicMock
        # mock_env = MagicMock(spec=BanditEnvironment)
        # mock_env.num_arms = 2
        # mock_env.arm_configs = TEST_ARM_CONFIGS_GAUSSIAN
        # # Manually call the __init__ logic that calculates optimal rates
        # BanditEnvironment.__init__(mock_env, mock_env.num_arms, mock_env.arm_configs)
        # self.assertAlmostEqual(mock_env.get_optimal_reward_rate(), 2.5 / 1.1)
        # self.assertEqual(mock_env.optimal_arm_index, 1)
        #
        # But this is more complex and usually not strictly necessary if concrete classes fully cover it.
        # The test for GeneralCostRewardEnvironment's init will ensure the super().__init__ works.
        env_multi_concrete = GeneralCostRewardEnvironment(num_arms=2, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN)
        self.assertAlmostEqual(env_multi_concrete.get_optimal_reward_rate(), 2.5 / 1.1)
        self.assertEqual(env_multi_concrete.optimal_arm_index, 1)


class TestGeneralCostRewardEnvironment(unittest.TestCase):

    def test_init_gaussian_arms(self):
        """Test initialization with Gaussian arms."""
        env = GeneralCostRewardEnvironment(num_arms=2, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN, seed=42)
        self.assertEqual(env.num_arms, 2)
        self.assertEqual(env._arm_samplers[0]['type'], 'gaussian')
        self.assertTrue(np.allclose(env._arm_samplers[0]['mean'], [1.0, 2.0]))
        self.assertTrue(np.allclose(env._arm_samplers[0]['cov'], [[0.1, 0.05], [0.05, 0.2]]))

    def test_init_bounded_arms(self):
        """Test initialization with bounded uniform arms."""
        env = GeneralCostRewardEnvironment(num_arms=2, arm_configs=TEST_ARM_CONFIGS_BOUNDED, seed=42)
        self.assertEqual(env.num_arms, 2)
        self.assertEqual(env._arm_samplers[0]['type'], 'bounded_uniform')
        self.assertEqual(env._arm_samplers[0]['min_X'], 0.5)
        self.assertEqual(env._arm_samplers[0]['max_R'], 3.0)

    def test_init_heavy_tailed_arms(self):
        """Test initialization with heavy-tailed arms."""
        env = GeneralCostRewardEnvironment(num_arms=2, arm_configs=TEST_ARM_CONFIGS_HEAVY_TAILED, seed=42)
        self.assertEqual(env.num_arms, 2)
        self.assertEqual(env._arm_samplers[0]['type'], 'heavy_tailed')
        self.assertEqual(env._arm_samplers[0]['params_X']['alpha'], 3.0)
        self.assertEqual(env._arm_samplers[0]['params_R']['mean'], 1.0)


    def test_pull_arm_gaussian(self):
        """Test pulling an arm with Gaussian distribution."""
        env = GeneralCostRewardEnvironment(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], seed=42)
        cost, reward = env.pull_arm(0)
        self.assertIsInstance(cost, float)
        self.assertIsInstance(reward, float)
        
        # Pull multiple times to check if values are centered around mean
        costs = []
        rewards = []
        for _ in range(1000):
            c, r = env.pull_arm(0)
            costs.append(c)
            rewards.append(r)
        
        self.assertAlmostEqual(np.mean(costs), TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['mean_X'], places=1)
        self.assertAlmostEqual(np.mean(rewards), TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['mean_R'], places=1)

    def test_pull_arm_bounded_uniform(self):
        """Test pulling an arm with bounded uniform distribution."""
        env = GeneralCostRewardEnvironment(num_arms=1, arm_configs=TEST_ARM_CONFIGS_BOUNDED[:1], seed=42)
        cost, reward = env.pull_arm(0)
        self.assertIsInstance(cost, float)
        self.assertIsInstance(reward, float)
        
        # Check if values are within bounds
        self.assertGreaterEqual(cost, TEST_ARM_CONFIGS_BOUNDED[0]['params']['min_X'])
        self.assertLessEqual(cost, TEST_ARM_CONFIGS_BOUNDED[0]['params']['max_X'])
        self.assertGreaterEqual(reward, TEST_ARM_CONFIGS_BOUNDED[0]['params']['min_R'])
        self.assertLessEqual(reward, TEST_ARM_CONFIGS_BOUNDED[0]['params']['max_R'])

        # Pull multiple times to check distribution (mean for uniform)
        costs = []
        rewards = []
        for _ in range(1000):
            c, r = env.pull_arm(0)
            costs.append(c)
            rewards.append(r)
        
        expected_mean_X = (TEST_ARM_CONFIGS_BOUNDED[0]['params']['min_X'] + TEST_ARM_CONFIGS_BOUNDED[0]['params']['max_X']) / 2
        expected_mean_R = (TEST_ARM_CONFIGS_BOUNDED[0]['params']['min_R'] + TEST_ARM_CONFIGS_BOUNDED[0]['params']['max_R']) / 2
        
        self.assertAlmostEqual(np.mean(costs), expected_mean_X, places=1)
        self.assertAlmostEqual(np.mean(rewards), expected_mean_R, places=1)

    def test_pull_arm_heavy_tailed(self):
        """Test pulling an arm with heavy-tailed distribution (conceptual check)."""
        env = GeneralCostRewardEnvironment(num_arms=1, arm_configs=TEST_ARM_CONFIGS_HEAVY_TAILED[:1], seed=42)
        costs = []
        rewards = []
        for _ in range(1000):
            c, r = env.pull_arm(0)
            costs.append(c)
            rewards.append(r)
        
        self.assertIsInstance(costs[0], float)
        self.assertIsInstance(rewards[0], float)

        # For heavy-tailed, means can be less stable with limited samples.
        # Check for non-negativity and presence of large values (a characteristic of heavy-tails)
        self.assertTrue(all(c >= 0 for c in costs), "Costs should be non-negative")
        self.assertTrue(all(r >= 0 for r in rewards), "Rewards should be non-negative")
        
        # A more robust test for heavy-tails would involve statistical tests (e.g., QQ plots,
        # checking for very high outliers relative to a normal distribution with same mean/variance),
        # but for a basic unit test, verifying the type and range is sufficient.
        # Given Pareto distribution properties, we expect some larger values.
        self.assertTrue(np.max(costs) > TEST_ARM_CONFIGS_HEAVY_TAILED[0]['params']['loc_pareto_X'] * 5, "Expected some large cost values (heavy-tail)")
        self.assertTrue(np.max(rewards) > 10.0, "Expected some large reward values (heavy-tail)")


    def test_pull_arm_invalid_index(self):
        """Test pulling with an invalid arm index."""
        env = GeneralCostRewardEnvironment(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], seed=42)
        with self.assertRaises(ValueError):
            env.pull_arm(99)

    def test_environment_reset(self):
        """Test that the environment can be reset."""
        env = GeneralCostRewardEnvironment(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], seed=42)
        
        # Pull once
        cost1, reward1 = env.pull_arm(0)
        
        # Reset
        env.reset()
        
        # Pull again after reset with the same seed, should get same initial values
        cost2, reward2 = env.pull_arm(0)
        
        self.assertAlmostEqual(cost1, cost2)
        self.assertAlmostEqual(reward1, reward2)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
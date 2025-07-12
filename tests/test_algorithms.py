import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Import the algorithms
from cbandits import UCB_B1, UCB_M1, UCB_B2, UCB_B2C, BaseBanditAlgorithm, GeneralCostRewardEnvironment

# Define common arm configurations for testing
# These are simplified for testing purposes.
TEST_ARM_CONFIGS_GAUSSIAN = [
    {"name": "Arm 0", "type": "gaussian", "params": {"mean_X": 1.0, "mean_R": 2.0, "var_X": 0.1, "var_R": 0.2, "cov_XR": 0.05, "M_X": 5.0, "M_R": 5.0}},
    {"name": "Arm 1", "type": "gaussian", "params": {"mean_X": 1.1, "mean_R": 1.8, "var_X": 0.15, "var_R": 0.25, "cov_XR": 0.03, "M_X": 5.0, "M_R": 5.0}},
]

TEST_ARM_CONFIGS_BOUNDED = [
    {"name": "Arm 0", "type": "bounded_uniform", "params": {"min_X": 0.5, "max_X": 1.5, "min_R": 1.0, "max_R": 3.0, "correlation": 0.0, "mean_X": 1.0, "mean_R": 2.0, "var_X": 0.08, "var_R": 0.33, "cov_XR": 0.0, "M_X": 1.5, "M_R": 3.0}}, # ADDED M_X, M_R
    {"name": "Arm 1", "type": "bounded_uniform", "params": {"min_X": 0.8, "max_X": 1.8, "min_R": 1.2, "max_R": 2.5, "correlation": 0.0, "mean_X": 1.3, "mean_R": 1.85, "var_X": 0.08, "var_R": 0.13, "cov_XR": 0.0, "M_X": 1.8, "M_R": 2.5}}, # ADDED M_X, M_R
]

# Common algorithm parameters for testing
TEST_ALGO_PARAMS = {
    "alpha": 2.1,
    "L": 2,
    "b_min_cost": 0.1,
    "M_X": 5.0, # Example bounded M_X for B1/M1 if not strictly Gaussian
    "M_R": 5.0, # Example bounded M_R for B1/M1 if not strictly Gaussian
    "omega_bar": 2.0,
}

class TestAlgorithms(unittest.TestCase):

    def test_base_bandit_algorithm_init(self):
        """Test BaseBanditAlgorithm initialization."""
        # Test that trying to instantiate the abstract class directly raises a TypeError
        with self.assertRaises(TypeError):
            BaseBanditAlgorithm(num_arms=0, arm_configs=[], algorithm_params={})
        
        # Test that trying to instantiate with valid parameters also raises TypeError (abstract class)
        with self.assertRaises(TypeError):
            BaseBanditAlgorithm(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], algorithm_params=TEST_ALGO_PARAMS)

    def test_ucb_b1_init(self):
        """Test UCB-B1 initialization."""
        algo = UCB_B1(num_arms=2, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN, algorithm_params=TEST_ALGO_PARAMS)
        self.assertEqual(algo.num_arms, 2)
        self.assertEqual(len(algo.arm_pulls), 2)
        self.assertAlmostEqual(algo.omega_k[0], TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['cov_XR'] / TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['var_X'])
        self.assertAlmostEqual(algo.V_XR[0], TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['var_R'] - (algo.omega_k[0]**2 * TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['var_X']))

    def test_ucb_m1_init(self):
        """Test UCB-M1 initialization."""
        algo = UCB_M1(num_arms=2, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN, algorithm_params=TEST_ALGO_PARAMS)
        self.assertEqual(algo.num_arms, 2)
        self.assertEqual(len(algo.arm_samples_X), 2)
        self.assertAlmostEqual(algo.omega_k[0], TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['cov_XR'] / TEST_ARM_CONFIGS_GAUSSIAN[0]['params']['var_X'])

    def test_ucb_b2_init(self):
        """Test UCB-B2 initialization."""
        algo = UCB_B2(num_arms=2, arm_configs=TEST_ARM_CONFIGS_BOUNDED, algorithm_params=TEST_ALGO_PARAMS)
        self.assertEqual(algo.num_arms, 2)
        self.assertEqual(len(algo.arm_pulls), 2)
        self.assertAlmostEqual(algo.M_X[0], TEST_ARM_CONFIGS_BOUNDED[0]['params']['M_X'])

    def test_ucb_b2c_init(self):
        """Test UCB-B2C initialization."""
        algo = UCB_B2C(num_arms=2, arm_configs=TEST_ARM_CONFIGS_BOUNDED, algorithm_params=TEST_ALGO_PARAMS)
        self.assertEqual(algo.num_arms, 2)
        self.assertEqual(len(algo.arm_samples_X), 2)
        self.assertAlmostEqual(algo.M_Z[0], TEST_ARM_CONFIGS_BOUNDED[0]['params']['M_R'] + TEST_ALGO_PARAMS['omega_bar'] * TEST_ARM_CONFIGS_BOUNDED[0]['params']['M_X'])


    def test_ucb_b1_update_state(self):
        """Test UCB-B1 update_state method."""
        algo = UCB_B1(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 5.0, 10.0)
        self.assertEqual(algo.arm_pulls[0], 1)
        self.assertEqual(algo.total_costs[0], 5.0)
        self.assertEqual(algo.total_rewards[0], 10.0)

    def test_ucb_m1_update_state(self):
        """Test UCB-M1 update_state method."""
        algo = UCB_M1(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 5.0, 10.0)
        self.assertEqual(algo.arm_pulls[0], 1)
        self.assertEqual(algo.arm_samples_X[0][0], 5.0)
        self.assertEqual(algo.arm_samples_R[0][0], 10.0)

    def test_ucb_b2_update_state(self):
        """Test UCB-B2 update_state method."""
        algo = UCB_B2(num_arms=1, arm_configs=TEST_ARM_CONFIGS_BOUNDED[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 2.0, 4.0)
        self.assertEqual(algo.arm_pulls[0], 1)
        self.assertEqual(algo.total_costs[0], 2.0)
        self.assertEqual(algo.total_rewards[0], 4.0)
        self.assertEqual(algo.sum_sq_costs[0], 4.0)
        self.assertEqual(algo.sum_sq_rewards[0], 16.0)

    def test_ucb_b2c_update_state(self):
        """Test UCB-B2C update_state method."""
        algo = UCB_B2C(num_arms=1, arm_configs=TEST_ARM_CONFIGS_BOUNDED[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 2.0, 4.0)
        self.assertEqual(algo.arm_pulls[0], 1)
        self.assertEqual(algo.arm_samples_X[0][0], 2.0)
        self.assertEqual(algo.arm_samples_R[0][0], 4.0)

    def test_ucb_b1_reset(self):
        """Test UCB-B1 reset method."""
        algo = UCB_B1(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 5.0, 10.0)
        algo.reset()
        self.assertEqual(algo.arm_pulls[0], 0)
        self.assertEqual(algo.total_costs[0], 0.0)
        self.assertEqual(algo.total_rewards[0], 0.0)

    def test_ucb_m1_reset(self):
        """Test UCB-M1 reset method."""
        algo = UCB_M1(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 5.0, 10.0)
        algo.reset()
        self.assertEqual(algo.arm_pulls[0], 0)
        self.assertEqual(len(algo.arm_samples_X[0]), 0)
        self.assertEqual(len(algo.arm_samples_R[0]), 0)

    def test_ucb_b2_reset(self):
        """Test UCB-B2 reset method."""
        algo = UCB_B2(num_arms=1, arm_configs=TEST_ARM_CONFIGS_BOUNDED[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 2.0, 4.0)
        algo.reset()
        self.assertEqual(algo.arm_pulls[0], 0)
        self.assertEqual(algo.total_costs[0], 0.0)
        self.assertEqual(algo.total_rewards[0], 0.0)
        self.assertEqual(algo.sum_sq_costs[0], 0.0)
        self.assertEqual(algo.sum_sq_rewards[0], 0.0)

    def test_ucb_b2c_reset(self):
        """Test UCB-B2C reset method."""
        algo = UCB_B2C(num_arms=1, arm_configs=TEST_ARM_CONFIGS_BOUNDED[:1], algorithm_params=TEST_ALGO_PARAMS)
        algo.update_state(0, 2.0, 4.0)
        algo.reset()
        self.assertEqual(algo.arm_pulls[0], 0)
        self.assertEqual(len(algo.arm_samples_X[0]), 0)
        self.assertEqual(len(algo.arm_samples_R[0]), 0)

    # Basic arm selection test for cold start (all algorithms should pull arms sequentially)
    def test_cold_start_arm_selection(self):
        """Test that algorithms pull arms sequentially during cold start."""
        num_arms = 3
        # Use a generic config, actual values don't matter much for cold start
        generic_configs = [{"name": f"Arm {i}", "type": "gaussian", "params": {"mean_X": 1.0, "mean_R": 1.0, "var_X": 0.1, "var_R": 0.1, "cov_XR": 0.0, "M_X": 5.0, "M_R": 5.0}} for i in range(num_arms)]

        algorithms = [
            UCB_B1(num_arms, generic_configs, TEST_ALGO_PARAMS),
            UCB_M1(num_arms, generic_configs, TEST_ALGO_PARAMS),
            UCB_B2(num_arms, generic_configs, TEST_ALGO_PARAMS),
            UCB_B2C(num_arms, generic_configs, TEST_ALGO_PARAMS),
        ]

        for algo in algorithms:
            algo.reset()
            for i in range(num_arms):
                selected = algo.select_arm(0.0, i + 1)
                self.assertEqual(selected, i, f"Cold start failed for {type(algo).__name__}")
                # Simulate a pull so the arm is no longer unpulled
                algo.update_state(selected, 1.0, 1.0) # Dummy values


    @patch('numpy.random.default_rng')
    def test_ucb_m1_median_estimator_with_mock_data(self, mock_default_rng):
        """
        Test UCB-M1's median-based estimator with mocked data to control sampling.
        This is a conceptual test, full validation of estimator properties
        would be in `test_estimators.py`.
        """
        mock_rng_instance = MagicMock()
        mock_default_rng.return_value = mock_rng_instance

        num_arms = 1
        algo = UCB_M1(num_arms, TEST_ARM_CONFIGS_GAUSSIAN[:1], TEST_ALGO_PARAMS)

        # Force a large enough epoch for median estimator to attempt grouping
        with patch.object(algo, 'arm_samples_X', new=[[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 10.0]]), \
             patch.object(algo, 'arm_samples_R', new=[[2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 20.0]]), \
             patch.object(algo, 'arm_pulls', new=np.array([10])):
            
            # Epoch 10. T_k = 10. log(10) = 2.3025...
            # m = floor(3.5 * 2.1 * 2.3025) + 1 = floor(16.9) + 1 = 17
            # T_k (10) < m (17), so group_size = 0. It should fall back to overall empirical mean.
            # Overall emp_X = (1.0*3 + 2.0*3 + 3.0*3 + 10.0) / 10 = 28/10 = 2.8
            # Overall emp_R = (2.0*3 + 4.0*3 + 6.0*3 + 20.0) / 10 = 56/10 = 5.6
            # Expected rate = 5.6 / 2.8 = 2.0
            mock_epoch = 10
            rate_est = algo._get_median_rate_estimator(0, mock_epoch)
            self.assertAlmostEqual(rate_est, 2.0, places=5) # Corrected assertion here

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
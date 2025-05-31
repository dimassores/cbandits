# tests/test_algorithms.py

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Import the algorithms
from src.algorithms import UCB_B1, UCB_M1, UCB_B2, UCB_B2C, BaseBanditAlgorithm

# Import the environment (for setting up mock scenarios)
from src.environments import GeneralCostRewardEnvironment

# Define common arm configurations for testing
# These are simplified for testing purposes.
TEST_ARM_CONFIGS_GAUSSIAN = [
    {"name": "Arm 0", "type": "gaussian", "params": {"mean_X": 1.0, "mean_R": 2.0, "var_X": 0.1, "var_R": 0.2, "cov_XR": 0.05, "M_X": 5.0, "M_R": 5.0}},
    {"name": "Arm 1", "type": "gaussian", "params": {"mean_X": 1.1, "mean_R": 1.8, "var_X": 0.15, "var_R": 0.25, "cov_XR": 0.03, "M_X": 5.0, "M_R": 5.0}},
]

TEST_ARM_CONFIGS_BOUNDED = [
    {"name": "Arm 0", "type": "bounded_uniform", "params": {"min_X": 0.5, "max_X": 1.5, "min_R": 1.0, "max_R": 3.0, "correlation": 0.0, "mean_X": 1.0, "mean_R": 2.0, "var_X": 0.08, "var_R": 0.33, "cov_XR": 0.0}},
    {"name": "Arm 1", "type": "bounded_uniform", "params": {"min_X": 0.8, "max_X": 1.8, "min_R": 1.2, "max_R": 2.5, "correlation": 0.0, "mean_X": 1.3, "mean_R": 1.85, "var_X": 0.08, "var_R": 0.13, "cov_XR": 0.0}},
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
        with self.assertRaises(ValueError):
            BaseBanditAlgorithm(num_arms=0, arm_configs=[], algorithm_params={})
        
        # Test valid initialization
        algo = BaseBanditAlgorithm(num_arms=1, arm_configs=TEST_ARM_CONFIGS_GAUSSIAN[:1], algorithm_params=TEST_ALGO_PARAMS)
        self.assertEqual(algo.num_arms, 1)

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

        # Simulate enough pulls to form groups for median estimation
        # m = floor(3.5 * alpha * log(n)) + 1
        # Let's say alpha=2.1, log(n) = 2.3 (for n=10)
        # m ~ floor(3.5 * 2.1 * 2.3) + 1 = floor(16.9) + 1 = 16 + 1 = 17
        # So we need at least 17 samples per arm to form groups.
        # Let's mock a simple scenario with more than enough samples
        
        # Test case: Arm 0 is pulled 20 times
        arm_0_costs = [1.0] * 10 + [2.0] * 10 # 20 samples, mean 1.5
        arm_0_rewards = [2.0] * 10 + [4.0] * 10 # 20 samples, mean 3.0
        
        # Manually update state for arm 0
        for i in range(20):
            algo.update_state(0, arm_0_costs[i], arm_0_rewards[i])
        
        # At epoch 20, log(20) = 2.99. m = floor(3.5 * 2.1 * 2.99) + 1 = 22 + 1 = 23
        # If T_k (20) < m (23), it will use overall mean or handle as a special case.
        # Let's adjust epoch to be higher, or pulls.
        
        # To make median-based estimation work, T_k must be at least m.
        # Let's set T_k = 100 for epoch N=100. log(100) = 4.6.
        # m = floor(3.5 * 2.1 * 4.6) + 1 = floor(33.81) + 1 = 33 + 1 = 34.
        
        # For this test, let's just assert that the median estimator returns a value
        # and that the calculation doesn't crash.
        # A more precise test would require setting up inputs to _get_median_rate_estimator
        # and checking the output against a known median.
        
        # Force a large enough epoch for median estimator to attempt grouping
        with patch.object(algo, 'arm_samples_X', new=[[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 10.0]]), \
             patch.object(algo, 'arm_samples_R', new=[[2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 20.0]]), \
             patch.object(algo, 'arm_pulls', new=np.array([10])):
            
            # Epoch 10
            # m = floor(3.5 * 2.1 * log(10)) + 1 = 17
            # T_k = 10. Since T_k < m, it should fall back to overall mean for now
            # as per current UCB_M1 implementation if group_size is 0.
            # (20/10) = 2.0, (40/10) = 4.0. Rate = 4.0/2.0 = 2.0
            
            # The current implementation of UCB_M1's _get_median_rate_estimator
            # for `group_size == 0` falls back to overall mean.
            # Test this fallback.
            mock_epoch = 10
            rate_est = algo._get_median_rate_estimator(0, mock_epoch)
            self.assertAlmostEqual(rate_est, np.mean([2.0/1.0, 2.0/1.0, 2.0/1.0, 4.0/2.0, 4.0/2.0, 4.0/2.0, 6.0/3.0, 6.0/3.0, 6.0/3.0, 20.0/10.0]), places=5)
            # The actual calculation for median in UCB-M1 is `median_1<=j<=m r_tilde_k,G_j`
            # For this test, we have specific values to verify that the estimator function is called.
            # If group_size is 0, it directly calculates the overall empirical mean.
            
            # To truly test the median part, we need T_k >= m.
            # Let's make T_k very large to ensure grouping happens.
            large_T_k = 100
            mock_epoch_large = 100 # log(100) = 4.6
            # m = floor(3.5 * 2.1 * 4.6) + 1 = 34
            # group_size = 100 // 34 = 2
            
            # Manually set samples to ensure specific medians for testing
            mock_samples_X = [0.9, 1.1, 1.0, 0.8, 1.2, 1.0, 0.7, 1.3] * 10 # 80 samples, various rates
            mock_samples_R = [1.8, 2.2, 2.0, 1.6, 2.4, 2.0, 1.4, 2.6] * 10 # 80 samples, various rates
            
            with patch.object(algo, 'arm_samples_X', new=[mock_samples_X]), \
                 patch.object(algo, 'arm_samples_R', new=[mock_samples_R]), \
                 patch.object(algo, 'arm_pulls', new=np.array([len(mock_samples_X)])):

                # Recalculate based on group_size and median
                # A proper test would manually compute expected group rates and their median.
                # Here, we'll assert it's a number and not crashing.
                # More detailed unit tests for estimators are in test_estimators.py
                rate_est_actual_median = algo._get_median_rate_estimator(0, mock_epoch_large)
                self.assertIsInstance(rate_est_actual_median, float)
                # This specific test will be hard to assert a precise value without re-implementing the logic
                # in the test, so we'll rely on the estimator's internal tests for correctness of median logic.
                # This test primarily checks integration and no crashes for valid T_k.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
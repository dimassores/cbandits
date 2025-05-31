# tests/test_estimators.py

import unittest
import numpy as np

# Import estimator functions
from src.utils.estimators import (
    calculate_empirical_mean,
    calculate_empirical_variance,
    calculate_lmmse_omega_empirical,
    calculate_lmmse_variance_reduction_empirical
)

class TestEstimators(unittest.TestCase):

    def test_calculate_empirical_mean(self):
        """Test calculate_empirical_mean function."""
        self.assertAlmostEqual(calculate_empirical_mean(0, 0), 0.0) # No pulls
        self.assertAlmostEqual(calculate_empirical_mean(5, 1), 5.0) # Single pull
        self.assertAlmostEqual(calculate_empirical_mean(15, 3), 5.0) # Multiple pulls
        self.assertAlmostEqual(calculate_empirical_mean(-10, 2), -5.0) # Negative values
        self.assertAlmostEqual(calculate_empirical_mean(0, 5), 0.0) # Zero sum

    def test_calculate_empirical_variance(self):
        """Test calculate_empirical_variance function."""
        # Not enough samples
        self.assertAlmostEqual(calculate_empirical_variance(0, 0, 0), 0.0)
        self.assertAlmostEqual(calculate_empirical_variance(10, 10, 1), 0.0)

        # Simple case: [1, 2, 3] -> mean = 2, var = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3
        # Sum = 6, Sum Sq = 1*1 + 2*2 + 3*3 = 1+4+9 = 14
        # (14/3) - (6/3)^2 = 4.666... - 4 = 0.666...
        self.assertAlmostEqual(calculate_empirical_variance(14, 6, 3), 2/3)

        # Constant values: [5, 5, 5] -> mean = 5, var = 0
        self.assertAlmostEqual(calculate_empirical_variance(75, 15, 3), 0.0)

        # Negative values
        # [-1, 0, 1] -> mean = 0, var = ((-1-0)^2 + (0-0)^2 + (1-0)^2)/3 = (1+0+1)/3 = 2/3
        # Sum = 0, Sum Sq = (-1)^2 + 0^2 + 1^2 = 2
        # (2/3) - (0/3)^2 = 2/3
        self.assertAlmostEqual(calculate_empirical_variance(2, 0, 3), 2/3)

        # Ensure non-negative due to floating point
        self.assertGreaterEqual(calculate_empirical_variance(1.000000000000001, 1.0, 10), 0.0)


    def test_calculate_lmmse_omega_empirical(self):
        """Test calculate_lmmse_omega_empirical function."""
        # No pulls or single pull
        self.assertAlmostEqual(calculate_lmmse_omega_empirical(0, 0, 0, 0, 0, 0), 0.0)
        self.assertAlmostEqual(calculate_lmmse_omega_empirical(10, 20, 100, 400, 200, 1), 0.0)

        # Perfectly correlated positive (X, R) = (1,2), (2,4), (3,6) -> R = 2X
        # X: [1,2,3], R: [2,4,6]
        # Sum_X = 6, Sum_R = 12, Sum_X_sq = 14, Sum_R_sq = 56, Sum_XR = 4 + 16 + 18 = 38
        # Num_pulls = 3
        # Emp_mean_X = 2, Emp_mean_R = 4
        # Emp_var_X = (14/3) - (2^2) = 2/3
        # Emp_cov_XR = (38/3) - (2*4) = 12.666... - 8 = 4.666... = 14/3
        # Omega = (14/3) / (2/3) = 7
        # Ah, the formula for omega_k is Cov(X,R)/Var(X). If R=2X, then Cov(X,2X) = 2Var(X), so omega should be 2.
        # My manual calculation of Sum_XR for (1,2), (2,4), (3,6) is:
        # (1*2) + (2*4) + (3*6) = 2 + 8 + 18 = 28.
        # Let's re-calculate: Sum_X = 6, Sum_R = 12, Sum_X_sq = 14, Sum_R_sq = 56, Sum_XR = 28.
        # Emp_cov_XR = (28/3) - (2*4) = 9.333... - 8 = 1.333... = 4/3
        # Omega = (4/3) / (2/3) = 2.0
        self.assertAlmostEqual(calculate_lmmse_omega_empirical(6, 12, 14, 56, 28, 3), 2.0)

        # Uncorrelated (X, R) = (1,1), (2,3), (3,2) -> mean_X=2, mean_R=2
        # Sum_X = 6, Sum_R = 6, Sum_X_sq = 14, Sum_R_sq = 1+9+4=14, Sum_XR = 1*1+2*3+3*2 = 1+6+6=13
        # Num_pulls = 3
        # Emp_mean_X = 2, Emp_mean_R = 2
        # Emp_var_X = 2/3
        # Emp_cov_XR = (13/3) - (2*2) = 4.333... - 4 = 0.333... = 1/3
        # Omega = (1/3) / (2/3) = 0.5
        self.assertAlmostEqual(calculate_lmmse_omega_empirical(6, 6, 14, 14, 13, 3), 0.5)

        # Zero variance in X
        self.assertAlmostEqual(calculate_lmmse_omega_empirical(3, 6, 9, 36, 18, 3), 0.0) # X=[3,3,3], R=[6,6,6]


    def test_calculate_lmmse_variance_reduction_empirical(self):
        """Test calculate_lmmse_variance_reduction_empirical function."""
        # No pulls or single pull
        self.assertAlmostEqual(calculate_lmmse_variance_reduction_empirical(0, 0, 0, 0, 0, 0, 0.0), 0.0)
        self.assertAlmostEqual(calculate_lmmse_variance_reduction_empirical(10, 20, 100, 400, 200, 1, 2.0), 0.0)

        # Perfectly correlated R=2X, use omega=2.0 (should be 0 variance reduction)
        # Sum_X = 6, Sum_R = 12, Sum_X_sq = 14, Sum_R_sq = 56, Sum_XR = 28, Num_pulls = 3
        # omega = 2.0 (calculated in previous test)
        # Var(R) = (56/3) - 4^2 = 18.666... - 16 = 2.666... = 8/3
        # Var(X) = (14/3) - 2^2 = 0.666... = 2/3
        # Reduced variance = Var(R) - omega^2 * Var(X) = (8/3) - (2^2 * 2/3) = 8/3 - 8/3 = 0
        self.assertAlmostEqual(calculate_lmmse_variance_reduction_empirical(6, 12, 14, 56, 28, 3, 2.0), 0.0)

        # Uncorrelated (X, R) = (1,1), (2,3), (3,2) -> omega = 0.5
        # Sum_X = 6, Sum_R = 6, Sum_X_sq = 14, Sum_R_sq = 14, Sum_XR = 13, Num_pulls = 3
        # Emp_var_R = (14/3) - 2^2 = 2/3
        # Emp_var_X = 2/3
        # omega = 0.5
        # Reduced variance = (2/3) - (0.5^2 * 2/3) = (2/3) - (0.25 * 2/3) = 2/3 - 1/6 = 4/6 - 1/6 = 3/6 = 0.5
        self.assertAlmostEqual(calculate_lmmse_variance_reduction_empirical(6, 6, 14, 14, 13, 3, 0.5), 0.5)

        # Test with Var(X) = 0 case
        # X=[3,3,3], R=[6,6,6], Sum_X=9, Sum_R=18, Sum_X_sq=27, Sum_R_sq=108, Sum_XR=54
        # Omega would be 0, Var(X)=0. Expected reduced variance is Var(R)=0.
        self.assertAlmostEqual(calculate_lmmse_variance_reduction_empirical(9, 18, 27, 108, 54, 3, 0.0), 0.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
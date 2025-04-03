"""Tests for evaluation metrics."""

import unittest
import numpy as np
from src.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_smape,
    calculate_r2,
    calculate_all_metrics
)


class TestMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def setUp(self):
        """Set up test arrays."""
        # Simple test case
        self.actual = np.array([1, 2, 3, 4, 5])
        self.predicted = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        
        # Test case with zeros (for MAPE)
        self.actual_with_zeros = np.array([0, 1, 2, 3, 4])
        self.predicted_with_zeros = np.array([0.1, 1.1, 2.2, 2.9, 4.1])
        
        # Perfect prediction
        self.perfect_actual = np.array([1, 2, 3, 4, 5])
        self.perfect_predicted = np.array([1, 2, 3, 4, 5])
    
    def test_mae(self):
        """Test Mean Absolute Error calculation."""
        # Normal case
        mae = calculate_mae(self.actual, self.predicted)
        expected_mae = 0.18
        self.assertAlmostEqual(mae, expected_mae, places=2)
        
        # Perfect prediction
        perfect_mae = calculate_mae(self.perfect_actual, self.perfect_predicted)
        self.assertEqual(perfect_mae, 0)
    
    def test_rmse(self):
        """Test Root Mean Squared Error calculation."""
        # Normal case
        rmse = calculate_rmse(self.actual, self.predicted)
        expected_rmse = 0.2
        self.assertAlmostEqual(rmse, expected_rmse, places=2)
        
        # Perfect prediction
        perfect_rmse = calculate_rmse(self.perfect_actual, self.perfect_predicted)
        self.assertEqual(perfect_rmse, 0)
    
    def test_mape(self):
        """Test Mean Absolute Percentage Error calculation."""
        # Normal case
        mape = calculate_mape(self.actual, self.predicted)
        expected_mape = 5.33
        self.assertAlmostEqual(mape, expected_mape, places=2)
        
        # Perfect prediction
        perfect_mape = calculate_mape(self.perfect_actual, self.perfect_predicted)
        self.assertEqual(perfect_mape, 0)
        
        # Case with zeros
        zero_mape = calculate_mape(self.actual_with_zeros, self.predicted_with_zeros)
        # We expect this to handle zeros properly without returning NaN
        self.assertFalse(np.isnan(zero_mape))
    
    def test_smape(self):
        """Test Symmetric Mean Absolute Percentage Error calculation."""
        # Normal case
        smape = calculate_smape(self.actual, self.predicted)
        expected_smape = 5.12  # Approximate value
        self.assertAlmostEqual(smape, expected_smape, places=2)
        
        # Perfect prediction
        perfect_smape = calculate_smape(self.perfect_actual, self.perfect_predicted)
        self.assertEqual(perfect_smape, 0)
        
        # Case with zeros
        zero_smape = calculate_smape(self.actual_with_zeros, self.predicted_with_zeros)
        # We expect this to handle zeros properly without returning NaN
        self.assertFalse(np.isnan(zero_smape))
    
    def test_r2(self):
        """Test R² score calculation."""
        # Normal case
        r2 = calculate_r2(self.actual, self.predicted)
        expected_r2 = 0.98  # Approximate value
        self.assertAlmostEqual(r2, expected_r2, places=2)
        
        # Perfect prediction
        perfect_r2 = calculate_r2(self.perfect_actual, self.perfect_predicted)
        self.assertEqual(perfect_r2, 1)
    
    def test_all_metrics(self):
        """Test calculation of all metrics at once."""
        # Normal case
        metrics = calculate_all_metrics(self.actual, self.predicted)
        
        # Check if all metrics are present
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertIn('SMAPE', metrics)
        self.assertIn('R2', metrics)
        
        # Check if values are as expected
        self.assertAlmostEqual(metrics['MAE'], 0.18, places=2)
        self.assertAlmostEqual(metrics['RMSE'], 0.2, places=2)
        self.assertAlmostEqual(metrics['MAPE'], 5.33, places=2)
        self.assertAlmostEqual(metrics['SMAPE'], 5.12, places=2)
        self.assertAlmostEqual(metrics['R2'], 0.98, places=2)
        
        # Test with arrays of different lengths
        longer_predicted = np.array([1.1, 2.2, 2.9, 4.1, 5.2, 6.0])
        metrics_diff_len = calculate_all_metrics(self.actual, longer_predicted)
        
        # Should use the minimum length and calculate properly
        self.assertAlmostEqual(metrics_diff_len['MAE'], 0.18, places=2)
        
        # Test with empty arrays
        empty_metrics = calculate_all_metrics(np.array([]), np.array([]))
        
        # Should return NaN for all metrics
        self.assertTrue(np.isnan(empty_metrics['MAE']))
        self.assertTrue(np.isnan(empty_metrics['RMSE']))
        self.assertTrue(np.isnan(empty_metrics['MAPE']))
        self.assertTrue(np.isnan(empty_metrics['SMAPE']))
        self.assertTrue(np.isnan(empty_metrics['R2']))


if __name__ == '__main__':
    unittest.main()
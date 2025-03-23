"""Tests for the UncertaintyQuantifier class."""

import unittest
import numpy as np
from core.theory.temporal_modeling import UncertaintyQuantifier

class TestUncertaintyQuantifier(unittest.TestCase):
    """Test cases for UncertaintyQuantifier class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.time_points = 100
        self.data = np.random.normal(10, 2, self.time_points)  # Normal distribution with mean 10 and std 2
        self.quantifier = UncertaintyQuantifier()

    def test_initialization(self):
        """Test initialization with different methods."""
        # Test default initialization
        quantifier = UncertaintyQuantifier()
        self.assertIsInstance(quantifier, UncertaintyQuantifier)
        self.assertEqual(quantifier.method, "bayesian")

        # Test with different methods
        methods = ["bayesian", "frequentist", "bootstrap", "monte_carlo"]
        for method in methods:
            quantifier = UncertaintyQuantifier(method=method)
            self.assertEqual(quantifier.method, method)

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        # Test frequentist method
        result = self.quantifier.compute_confidence_interval(
            self.data, confidence_level=0.95, method="frequentist"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("mean", result)
        self.assertIn("std", result)
        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertTrue(result["lower"] < np.mean(self.data) < result["upper"])

        # Test bootstrap method
        result = self.quantifier.compute_confidence_interval(
            self.data, confidence_level=0.95, method="bootstrap"
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result["lower"] < np.mean(self.data) < result["upper"])

        # Test Bayesian method
        result = self.quantifier.compute_confidence_interval(
            self.data, confidence_level=0.95, method="bayesian"
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result["lower"] < np.mean(self.data) < result["upper"])

    def test_prediction_interval(self):
        """Test prediction interval computation."""
        # Test frequentist method
        result = self.quantifier.compute_prediction_interval(
            self.data, confidence_level=0.95, method="frequentist"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("mean", result)
        self.assertIn("std", result)
        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertTrue(result["lower"] < np.mean(self.data) < result["upper"])

        # Test bootstrap method
        result = self.quantifier.compute_prediction_interval(
            self.data, confidence_level=0.95, method="bootstrap"
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result["lower"] < np.mean(self.data) < result["upper"])

    def test_uncertainty_propagation(self):
        """Test uncertainty propagation."""
        # Define a simple function to propagate uncertainty through
        def test_function(x):
            return np.mean(x) * 2

        uncertainties = np.ones_like(self.data) * 0.1
        result = self.quantifier.propagate_uncertainty(
            self.data, test_function, uncertainties
        )
        self.assertIsInstance(result, dict)
        self.assertIn("mean", result)
        self.assertIn("std", result)
        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertTrue(result["lower"] < result["mean"] < result["upper"])

    def test_error_decomposition(self):
        """Test error decomposition."""
        # Generate synthetic model predictions and uncertainties
        predictions = self.data + np.random.normal(0, 0.5, self.time_points)
        uncertainties = np.ones_like(self.data) * 0.1

        result = self.quantifier.compute_error_decomposition(
            self.data, predictions, uncertainties
        )
        self.assertIsInstance(result, dict)
        self.assertIn("total_uncertainty", result)
        self.assertIn("aleatory_uncertainty", result)
        self.assertIn("epistemic_uncertainty", result)
        self.assertIn("aleatory_fraction", result)
        self.assertIn("epistemic_fraction", result)
        
        # Check that fractions sum to approximately 1
        self.assertAlmostEqual(
            result["aleatory_fraction"] + result["epistemic_fraction"], 
            1.0, 
            places=6
        )

    def test_input_validation(self):
        """Test input validation."""
        # Test with insufficient data points
        with self.assertRaises(ValueError):
            self.quantifier.compute_confidence_interval(np.array([1.0]))

        with self.assertRaises(ValueError):
            self.quantifier.compute_prediction_interval(np.array([1.0]))

if __name__ == '__main__':
    unittest.main() 
"""Tests for the CausalInferenceAnalyzer class."""

import unittest
import numpy as np
import pandas as pd
from core.theory.temporal_modeling import CausalInferenceAnalyzer

class TestCausalInferenceAnalyzer(unittest.TestCase):
    """Test cases for CausalInferenceAnalyzer class."""

    def setUp(self):
        """Set up test data."""
        # Generate synthetic time series data
        np.random.seed(42)
        self.time_points = 100
        self.x = np.sin(np.linspace(0, 10, self.time_points)) + np.random.normal(0, 0.1, self.time_points)
        self.y = np.roll(self.x, 5) + np.random.normal(0, 0.1, self.time_points)  # y is causally influenced by x
        self.data = pd.DataFrame({
            'x': self.x,
            'y': self.y,
            'timestamp': pd.date_range(start='2024-01-01', periods=self.time_points, freq='h')
        })
        self.analyzer = CausalInferenceAnalyzer(self.data)

    def test_initialization(self):
        """Test initialization with different data types."""
        # Test with DataFrame
        analyzer = CausalInferenceAnalyzer(self.data)
        self.assertIsInstance(analyzer, CausalInferenceAnalyzer)

        # Test with numpy array
        data_array = np.column_stack((self.x, self.y))
        analyzer = CausalInferenceAnalyzer(data_array)
        self.assertIsInstance(analyzer, CausalInferenceAnalyzer)

    def test_granger_causality(self):
        """Test Granger causality analysis."""
        result = self.analyzer.analyze_granger_causality('x', 'y', max_lag=10)
        self.assertIsInstance(result, dict)
        self.assertIn('p_value', result)
        self.assertIn('f_statistic', result)
        self.assertIn('optimal_lag', result)

    def test_transfer_entropy(self):
        """Test transfer entropy calculation."""
        te = self.analyzer.compute_transfer_entropy('x', 'y')
        self.assertIsInstance(te, float)
        self.assertGreaterEqual(te, 0)  # Transfer entropy should be non-negative

    def test_convergent_cross_mapping(self):
        """Test convergent cross mapping."""
        ccm_result = self.analyzer.analyze_convergent_cross_mapping('x', 'y')
        self.assertIsInstance(ccm_result, dict)
        self.assertIn('correlation', ccm_result)
        self.assertIn('significance', ccm_result)

    def test_causal_impact(self):
        """Test causal impact analysis."""
        impact = self.analyzer.analyze_causal_impact('x', 'y', pre_period=[0, 50], post_period=[51, 99])
        self.assertIsInstance(impact, dict)
        self.assertIn('average_effect', impact)
        self.assertIn('confidence_interval', impact)

    def test_trigger_identification(self):
        """Test migraine trigger identification."""
        triggers = self.analyzer.identify_triggers(['x'], 'y')
        self.assertIsInstance(triggers, dict)
        self.assertIn('potential_triggers', triggers)
        self.assertIn('confidence_scores', triggers)

if __name__ == '__main__':
    unittest.main() 
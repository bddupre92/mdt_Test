"""Unit tests for multimodal integration components.

This module contains tests for:
1. Feature interaction analysis
2. Missing data handling
3. Reliability modeling
4. Integration utilities
"""

import unittest
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.datasets import make_classification
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.theory.multimodal_integration import (
    CrossModalInteractionAnalyzer,
    MultimodalMissingDataHandler,
    MultimodalReliabilityModel,
    ModalityData
)

# Test data generation utilities
def generate_test_data(n_samples=100, n_features=5, n_modalities=3):
    """Generate synthetic test data for multiple modalities."""
    data_sources = []
    modality_labels = []
    
    for i in range(n_modalities):
        X, _ = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_redundant=1,
            n_informative=3,
            random_state=42 + i
        )
        data_sources.append(X)
        modality_labels.append(f"modality_{i}")
    
    return data_sources, modality_labels

def generate_temporal_data(n_samples=100, n_features=3, n_modalities=2):
    """Generate synthetic temporal data."""
    time = np.linspace(0, 10, n_samples)
    data_sources = []
    
    for i in range(n_modalities):
        # Generate sinusoidal data with different frequencies
        X = np.zeros((n_samples, n_features))
        for j in range(n_features):
            freq = 0.5 * (i + 1) * (j + 1)
            X[:, j] = np.sin(2 * np.pi * freq * time) + np.random.normal(0, 0.1, n_samples)
        data_sources.append(X)
    
    return data_sources, time

class TestFeatureInteraction(unittest.TestCase):
    """Tests for feature interaction analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interaction_analyzer = CrossModalInteractionAnalyzer(
            interaction_method='correlation',
            significance_level=0.05,
            min_correlation=0.1
        )
        self.test_data = generate_test_data()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.interaction_analyzer.interaction_method, 'correlation')
        self.assertEqual(self.interaction_analyzer.significance_level, 0.05)
        self.assertEqual(self.interaction_analyzer.min_correlation, 0.1)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            CrossModalInteractionAnalyzer(interaction_method='invalid_method')
            
    def test_analyze_interactions(self):
        """Test interaction analysis."""
        data_sources, modality_labels = self.test_data
        
        # Create ModalityData objects
        modality_data = [
            ModalityData(data=X, modality_type=label)
            for X, label in zip(data_sources, modality_labels)
        ]
        
        # Analyze interactions
        results = self.interaction_analyzer.analyze_interactions(*modality_data)
        
        # Check results structure
        self.assertIn('interaction_matrix', results)
        self.assertIn('p_values', results)
        self.assertIn('significant_pairs', results)
        self.assertIn('graph', results)
        self.assertIn('clusters', results)
        
        # Check dimensions
        total_features = sum(X.shape[1] for X in data_sources)
        self.assertEqual(results['interaction_matrix'].shape, (total_features, total_features))
        
        # Test empty data
        with self.assertRaises(ValueError):
            empty_array = np.array([]).reshape(0, 0)
            self.interaction_analyzer.analyze_interactions(empty_array)
            
        # Test incompatible data
        data1 = np.random.randn(100, 5)
        data2 = np.random.randn(50, 5)  # Different number of samples
        with self.assertRaises(ValueError):
            self.interaction_analyzer.analyze_interactions(data1, data2)

    def test_visualization(self):
        """Test visualization methods."""
        data_sources, _ = self.test_data
        results = self.interaction_analyzer.analyze_interactions(*data_sources)
        
        # Test different visualization types
        vis_results = self.interaction_analyzer.visualize_interactions(
            results, plot_type='all'
        )
        
        self.assertIn('heatmap', vis_results)
        self.assertIn('network', vis_results)
        self.assertIn('dendrogram', vis_results)

class TestMissingData(unittest.TestCase):
    """Tests for missing data handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.missing_handler = MultimodalMissingDataHandler(
            imputation_method='auto',
            max_missing_ratio=0.3
        )
        data_sources, modality_labels = generate_test_data()
        
        # Introduce missing values
        for X in data_sources:
            mask = np.random.random(X.shape) < 0.2
            X[mask] = np.nan
            
        self.test_data_with_missing = (data_sources, modality_labels)
    
    def test_initialization(self):
        """Test handler initialization."""
        self.assertEqual(self.missing_handler.imputation_method, 'auto')
        self.assertEqual(self.missing_handler.max_missing_ratio, 0.3)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            MultimodalMissingDataHandler(imputation_method='invalid_method')
        
    def test_detect_missing_patterns(self):
        """Test missing pattern detection."""
        data_sources, _ = self.test_data_with_missing
        
        patterns = self.missing_handler.detect_missing_patterns(*data_sources)
        
        self.assertIn('patterns', patterns)
        self.assertIn('frequencies', patterns)
        self.assertIn('modality_stats', patterns)
        self.assertIn('temporal_stats', patterns)
        
    def test_imputation(self):
        """Test data imputation."""
        data_sources, _ = self.test_data_with_missing
        
        # Perform imputation
        imputed_arrays, uncertainties = self.missing_handler.impute(*data_sources)
        
        # Check results
        self.assertEqual(len(imputed_arrays), len(data_sources))
        for X_imp, X_orig in zip(imputed_arrays, data_sources):
            self.assertEqual(X_imp.shape, X_orig.shape)
            self.assertFalse(np.any(np.isnan(X_imp)))
            
    def test_imputation_methods(self):
        """Test different imputation methods."""
        data_sources, _ = self.test_data_with_missing
        
        methods = ['interpolation', 'knn', 'mice', 'pattern']
        for method in methods:
            self.missing_handler.imputation_method = method
            imputed, uncertainties = self.missing_handler.impute(*data_sources)
            self.assertTrue(all(not np.any(np.isnan(X)) for X in imputed))

class TestReliabilityModeling(unittest.TestCase):
    """Tests for reliability modeling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reliability_model = MultimodalReliabilityModel(
            reliability_method='auto',
            temporal_window=100
        )
        self.temporal_test_data = generate_temporal_data()
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.reliability_model.reliability_method, 'auto')
        self.assertEqual(self.reliability_model.temporal_window, 100)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            MultimodalReliabilityModel(reliability_method='invalid_method')
        
    def test_assess_reliability(self):
        """Test reliability assessment."""
        data_sources, temporal_info = self.temporal_test_data
        
        # Assess reliability
        scores = self.reliability_model.assess_reliability(
            *data_sources,
            temporal_info=temporal_info
        )
        
        # Check results
        self.assertEqual(len(scores), len(data_sources))
        self.assertTrue(all(0 <= score <= 1 for score in scores.values()))
        
    def test_reliability_methods(self):
        """Test different reliability assessment methods."""
        data_sources, temporal_info = self.temporal_test_data
        
        methods = ['auto', 'signal_quality', 'statistical', 'historical', 'ensemble',
                  'adaptive', 'temporal', 'conflict']
        for method in methods:
            self.reliability_model.reliability_method = method
            scores = self.reliability_model.assess_reliability(
                *data_sources,
                temporal_info=temporal_info
            )
            self.assertTrue(all(0 <= score <= 1 for score in scores.values()))
            
    def test_reliability_updates(self):
        """Test reliability score updates."""
        data_sources, _ = self.temporal_test_data
        
        # Initial assessment
        scores = self.reliability_model.assess_reliability(*data_sources)
        
        # Update with new evidence
        new_evidence = {
            'prediction_errors': {f"modality_{i}": 0.1 for i in range(len(data_sources))},
            'conflict_indicators': {f"modality_0": ["modality_1"]},
            'quality_updates': {
                f"modality_{i}": {'completeness': 0.9, 'stability': 0.8}
                for i in range(len(data_sources))
            }
        }
        
        updated_scores = self.reliability_model.update_reliability(scores, new_evidence)
        
        # Check updates
        self.assertEqual(len(updated_scores), len(scores))
        self.assertTrue(all(score >= self.reliability_model.min_confidence 
                          for score in updated_scores.values()))

    def test_error_handling(self):
        """Test error handling in components."""
        # Test invalid method directly instead of in an instance
        with self.assertRaises(ValueError):
            model = MultimodalReliabilityModel(reliability_method='invalid_method')
            
        # Test empty data array
        model = MultimodalReliabilityModel()
        with self.assertRaises(ValueError):
            model.assess_reliability(np.array([]))
            
        # Test conflicting temporal data
        data = np.random.randn(10, 3)
        invalid_temporal = np.random.randn(5)  # Wrong length
        with self.assertRaises(ValueError):
            model.assess_reliability(data, temporal_info=invalid_temporal)

class TestIntegration(unittest.TestCase):
    """Integration tests for multimodal components."""
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration of components."""
        # Generate test data
        data_sources, modality_labels = generate_test_data()
        temporal_info = np.arange(len(data_sources[0]))
        
        # Create components
        interaction_analyzer = CrossModalInteractionAnalyzer()
        missing_handler = MultimodalMissingDataHandler()
        reliability_model = MultimodalReliabilityModel()
        
        # Analyze interactions
        interaction_results = interaction_analyzer.analyze_interactions(*data_sources)
        
        # Handle missing data
        imputed_data, uncertainties = missing_handler.impute(
            *data_sources,
            temporal_info=temporal_info
        )
        
        # Assess reliability
        reliability_scores = reliability_model.assess_reliability(
            *imputed_data,
            temporal_info=temporal_info
        )
        
        # Verify integration
        self.assertEqual(len(reliability_scores), len(data_sources))
        self.assertTrue(all(not np.any(np.isnan(X)) for X in imputed_data))
        self.assertIn('interaction_matrix', interaction_results)

if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 
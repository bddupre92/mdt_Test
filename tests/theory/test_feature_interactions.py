"""
Unit tests for migraine feature interaction analysis.

Tests the functionality of the feature interaction analyzer including:
- Prodrome indicator analysis
- Trigger interaction detection
- Feature importance ranking
- Cross-modal correlation analysis
- Temporal lead/lag relationships
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from scipy import signal
from typing import Dict, Tuple

from core.theory.multimodal_integration import ModalityData
from core.theory.migraine_adaptation.feature_interactions import FeatureInteractionAnalyzer

def generate_test_data(n_samples: int = 1000) -> Dict[str, ModalityData]:
    """Generate synthetic test data for multiple modalities."""
    # Time points
    timestamps = np.array([datetime.now() + timedelta(minutes=i) for i in range(n_samples)])
    
    # ECG data with simulated HRV changes
    t = np.linspace(0, n_samples/100, n_samples)
    ecg_base = np.sin(2 * np.pi * 1.2 * t)  # Base ECG-like signal
    hrv_mod = np.sin(2 * np.pi * 0.01 * t)  # Slow HRV modulation
    ecg_data = ecg_base * (1 + 0.2 * hrv_mod)
    
    # EEG data with simulated alpha and theta bands
    alpha = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    theta = 0.5 * np.sin(2 * np.pi * 6 * t)  # 6 Hz theta
    eeg_data = alpha + theta + 0.1 * np.random.randn(n_samples)
    
    # Skin conductance with simulated responses
    sc_base = 5 + 0.1 * np.random.randn(n_samples)
    sc_responses = np.zeros_like(t)
    for i in range(5):
        peak_loc = (i + 1) * n_samples/6
        sc_responses += 2.0 * np.exp(-(t*100 - peak_loc)**2 / 10000)
    sc_data = sc_base + sc_responses
    
    # Create ModalityData objects
    data_sources = {
        'ecg': ModalityData(
            data=ecg_data,
            modality_type='ecg',
            timestamps=timestamps,
            metadata={'sampling_rate': 100}
        ),
        'eeg': ModalityData(
            data=eeg_data,
            modality_type='eeg',
            timestamps=timestamps,
            metadata={'sampling_rate': 100}
        ),
        'skin_conductance': ModalityData(
            data=sc_data,
            modality_type='skin_conductance',
            timestamps=timestamps,
            metadata={'sampling_rate': 100}
        )
    }
    
    return data_sources

def generate_trigger_data(n_samples: int = 1000) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generate synthetic trigger and response data."""
    t = np.linspace(0, n_samples/100, n_samples)
    
    # Generate triggers
    stress = 0.7 * np.sin(2 * np.pi * 0.01 * t) + 0.3 * np.random.randn(n_samples)
    sleep = 0.8 * np.sin(2 * np.pi * 0.005 * t + np.pi/4) + 0.2 * np.random.randn(n_samples)
    diet = 0.6 * np.sin(2 * np.pi * 0.007 * t + np.pi/3) + 0.4 * np.random.randn(n_samples)
    
    # Generate physiological responses with some delay
    hr_response = np.roll(0.6 * stress + 0.4 * sleep, 10) + 0.2 * np.random.randn(n_samples)
    bp_response = np.roll(0.5 * stress + 0.3 * diet, 15) + 0.2 * np.random.randn(n_samples)
    
    triggers = {
        'stress': stress,
        'sleep': sleep,
        'diet': diet
    }
    
    responses = {
        'heart_rate': hr_response,
        'blood_pressure': bp_response
    }
    
    return triggers, responses

class TestFeatureInteractionAnalyzer(unittest.TestCase):
    """Test suite for feature interaction analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FeatureInteractionAnalyzer()
        self.data_sources = generate_test_data()
        self.triggers, self.responses = generate_trigger_data()
        
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.significance_threshold, 0.05)
        self.assertEqual(self.analyzer.min_correlation, 0.2)
        self.assertEqual(self.analyzer.max_time_lag, 48)
        
    def test_analyze_prodrome_indicators(self):
        """Test prodrome indicator analysis."""
        results = self.analyzer.analyze_prodrome_indicators(self.data_sources)
        
        # Check results structure
        self.assertIn('indicators', results)
        self.assertIn('temporal_patterns', results)
        self.assertIn('significance', results)
        self.assertIn('feature_importance', results)
        self.assertIn('cross_modal_interactions', results)
        
        # Check indicators
        self.assertIsInstance(results['indicators'], list)
        if results['indicators']:
            indicator = results['indicators'][0]
            self.assertIn('modality', indicator)
            self.assertIn('feature', indicator)
            self.assertIn('significance', indicator)
            self.assertIn('temporal_pattern', indicator)
        
        # Check temporal patterns
        for modality, patterns in results['temporal_patterns'].items():
            self.assertIsInstance(patterns, dict)
            
        # Check significance scores
        for modality, scores in results['significance'].items():
            self.assertIsInstance(scores, dict)
            for score in scores.values():
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)
    
    def test_detect_trigger_interactions(self):
        """Test trigger interaction detection."""
        interactions = self.analyzer.detect_trigger_interactions(
            self.triggers,
            self.responses
        )
        
        # Check pairwise interactions
        for key, value in interactions.items():
            if key != 'multi_trigger':
                self.assertIn('correlation', value)
                self.assertIn('mutual_information', value)
                self.assertIn('time_lag', value)
                self.assertIn('significance', value)
                
                self.assertGreaterEqual(value['significance'], 0)
                self.assertLessEqual(value['significance'], 1)
                
        # Check multi-trigger analysis
        if 'multi_trigger' in interactions:
            multi = interactions['multi_trigger']
            self.assertIn('interaction_matrix', multi)
            self.assertIn('trigger_names', multi)
            self.assertIn('synergy_scores', multi)
    
    def test_rank_feature_importance(self):
        """Test feature importance ranking."""
        # Generate synthetic migraine occurrences
        n_samples = len(next(iter(self.triggers.values())))
        migraine_occurrences = np.zeros(n_samples)
        migraine_occurrences[::100] = 1  # Simulate migraines every 100 samples
        
        # Rank features
        ranked_features = self.analyzer.rank_feature_importance(
            self.triggers,
            migraine_occurrences
        )
        
        # Check results
        self.assertIsInstance(ranked_features, list)
        self.assertEqual(len(ranked_features), len(self.triggers))
        
        # Check ranking structure
        for feature_name, importance in ranked_features:
            self.assertIn(feature_name, self.triggers)
            self.assertGreaterEqual(importance, 0)
            self.assertLessEqual(importance, 1)
        
        # Check ordering
        importances = [score for _, score in ranked_features]
        self.assertEqual(importances, sorted(importances, reverse=True))
    
    def test_time_alignment(self):
        """Test time series alignment."""
        # Create data with different time windows
        n_samples = 1000
        base_time = datetime.now()
        times1 = np.array([base_time + timedelta(minutes=i) for i in range(n_samples)])
        times2 = np.array([base_time + timedelta(minutes=i+100) for i in range(n_samples)])
        
        data1 = ModalityData(
            data=np.random.randn(n_samples),
            modality_type='test1',
            timestamps=times1
        )
        
        data2 = ModalityData(
            data=np.random.randn(n_samples),
            modality_type='test2',
            timestamps=times2
        )
        
        data_sources = {'test1': data1, 'test2': data2}
        
        # Test alignment
        aligned = self.analyzer._align_time_series(data_sources)
        
        self.assertIn('test1', aligned)
        self.assertIn('test2', aligned)
        self.assertEqual(len(aligned['test1']), len(aligned['test2']))
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with empty data
        empty_data = {'empty': ModalityData(
            data=np.array([]),
            modality_type='empty'
        )}
        results = self.analyzer.analyze_prodrome_indicators(empty_data)
        self.assertEqual(len(results['indicators']), 0)
        
        # Test with invalid time window
        future_time = datetime.now() + timedelta(days=1)
        past_time = datetime.now() - timedelta(days=1)
        results = self.analyzer.analyze_prodrome_indicators(
            self.data_sources,
            time_window=(future_time.timestamp(), past_time.timestamp())
        )
        self.assertEqual(len(results['indicators']), 0)
        
        # Test with mismatched data lengths
        triggers_mismatched = {
            'trigger1': np.random.randn(100),
            'trigger2': np.random.randn(200)
        }
        responses_mismatched = {
            'response1': np.random.randn(150)
        }
        interactions = self.analyzer.detect_trigger_interactions(
            triggers_mismatched,
            responses_mismatched
        )
        self.assertIsInstance(interactions, dict)

if __name__ == '__main__':
    unittest.main() 
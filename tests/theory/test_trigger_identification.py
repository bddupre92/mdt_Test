"""
Unit tests for migraine trigger identification.

Tests the functionality of the trigger identification analyzer including:
- Trigger identification from time series data
- Trigger sensitivity analysis
- Temporal pattern recognition
- Trigger interaction analysis
- Personalized trigger profiling
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from core.theory.migraine_adaptation.trigger_identification import (
    TriggerIdentificationAnalyzer,
    TriggerEvent,
    TriggerProfile
)

def generate_synthetic_data(n_samples: int = 1000) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Generate synthetic data for testing."""
    # Generate timestamps
    timestamps = np.array([
        datetime.now() + timedelta(hours=i)
        for i in range(n_samples)
    ])
    
    # Generate trigger data
    t = np.linspace(0, n_samples/24, n_samples)  # Time in days
    
    # Stress trigger with daily pattern
    stress = 0.7 * np.sin(2 * np.pi * t) + 0.3 * np.random.randn(n_samples)
    
    # Sleep trigger with weekly pattern
    sleep = 0.8 * np.sin(2 * np.pi * t/7 + np.pi/4) + 0.2 * np.random.randn(n_samples)
    
    # Diet trigger with irregular pattern
    diet = 0.6 * np.sin(2 * np.pi * t/3 + np.pi/3) + 0.4 * np.random.randn(n_samples)
    
    triggers = {
        'stress': stress,
        'sleep': sleep,
        'diet': diet
    }
    
    # Generate symptom data with lagged responses to triggers
    pain = (
        0.4 * np.roll(stress, 12) +  # 12-hour lag for stress
        0.3 * np.roll(sleep, 24) +   # 24-hour lag for sleep
        0.2 * np.roll(diet, 6)       # 6-hour lag for diet
    )
    
    nausea = (
        0.3 * np.roll(stress, 8) +   # 8-hour lag for stress
        0.2 * np.roll(sleep, 16) +   # 16-hour lag for sleep
        0.4 * np.roll(diet, 4)       # 4-hour lag for diet
    )
    
    symptoms = {
        'pain': pain,
        'nausea': nausea
    }
    
    return triggers, symptoms, timestamps

def generate_trigger_history(n_events: int = 100) -> Tuple[List[TriggerEvent], List[datetime]]:
    """Generate synthetic trigger history and migraine events."""
    # Generate base timestamp
    base_time = datetime.now()
    
    # Generate trigger events
    trigger_types = ['stress', 'sleep', 'diet']
    trigger_events = []
    
    for i in range(n_events):
        event_time = base_time + timedelta(hours=i*8)  # Events every 8 hours
        trigger_type = trigger_types[i % len(trigger_types)]
        
        # Add some randomness to intensity
        intensity = 0.5 + 0.5 * np.random.random()
        
        event = TriggerEvent(
            trigger_type=trigger_type,
            timestamp=event_time,
            intensity=intensity,
            duration=timedelta(hours=2),
            confidence=0.8,
            associated_symptoms=['pain', 'nausea'],
            context={'location': 'home', 'activity': 'work'}
        )
        trigger_events.append(event)
    
    # Generate migraine events (some related to triggers, some random)
    migraine_events = []
    for i in range(n_events // 3):  # Fewer migraines than triggers
        if i % 2 == 0:
            # Related to a trigger
            trigger_time = trigger_events[i*3].timestamp
            migraine_time = trigger_time + timedelta(hours=12)  # 12-hour lag
        else:
            # Random migraine
            hours = np.random.randint(0, n_events*8)
            migraine_time = base_time + timedelta(hours=hours)
        
        migraine_events.append(migraine_time)
    
    return trigger_events, sorted(migraine_events)

class TestTriggerIdentification(unittest.TestCase):
    """Test suite for trigger identification analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TriggerIdentificationAnalyzer()
        self.triggers, self.symptoms, self.timestamps = generate_synthetic_data()
        self.trigger_history, self.migraine_events = generate_trigger_history()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.causal_threshold, 0.05)
        self.assertEqual(self.analyzer.sensitivity_window, 48)
        self.assertEqual(self.analyzer.min_confidence, 0.7)
        self.assertEqual(self.analyzer.max_interaction_order, 3)
    
    def test_identify_triggers(self):
        """Test trigger identification from time series data."""
        results = self.analyzer.identify_triggers(
            self.symptoms,
            self.triggers,
            self.timestamps
        )
        
        # Check results structure
        self.assertIn('triggers', results)
        self.assertIn('causal_scores', results)
        self.assertIn('sensitivity_thresholds', results)
        self.assertIn('temporal_patterns', results)
        self.assertIn('interaction_effects', results)
        self.assertIn('confidence_scores', results)
        
        # Check identified triggers
        self.assertGreater(len(results['triggers']), 0)
        for trigger in results['triggers']:
            self.assertIn(trigger, self.triggers)
            
        # Check confidence scores
        for trigger in results['triggers']:
            self.assertGreaterEqual(results['confidence_scores'][trigger], self.analyzer.min_confidence)
    
    def test_analyze_trigger_sensitivity(self):
        """Test trigger sensitivity analysis."""
        # Test for a single trigger
        trigger_data = self.triggers['stress']
        results = self.analyzer.analyze_trigger_sensitivity(
            trigger_data,
            self.symptoms
        )
        
        # Check results structure
        self.assertIn('thresholds', results)
        self.assertIn('temporal_sensitivity', results)
        self.assertIn('confidence_intervals', results)
        
        # Test with baseline period
        baseline_period = (
            self.timestamps[0],
            self.timestamps[len(self.timestamps)//4]
        )
        results_with_baseline = self.analyzer.analyze_trigger_sensitivity(
            trigger_data,
            self.symptoms,
            baseline_period
        )
        
        self.assertIn('baseline', results_with_baseline)
    
    def test_generate_trigger_profile(self):
        """Test personalized trigger profile generation."""
        profile = self.analyzer.generate_trigger_profile(
            self.trigger_history,
            self.migraine_events
        )
        
        # Check profile structure
        self.assertIsInstance(profile, TriggerProfile)
        self.assertTrue(profile.trigger_sensitivities)
        self.assertTrue(profile.interaction_effects)
        self.assertTrue(profile.temporal_patterns)
        self.assertTrue(profile.threshold_ranges)
        self.assertTrue(profile.confidence_scores)
        
        # Check trigger types
        trigger_types = set(event.trigger_type for event in self.trigger_history)
        for trigger_type in trigger_types:
            self.assertIn(trigger_type, profile.trigger_sensitivities)
            self.assertIn(trigger_type, profile.threshold_ranges)
            self.assertIn(trigger_type, profile.confidence_scores)
    
    def test_temporal_pattern_detection(self):
        """Test temporal pattern detection in trigger data."""
        # Test for a single trigger
        patterns = self.analyzer._detect_temporal_patterns(
            self.triggers['stress'],
            self.symptoms,
            self.timestamps
        )
        
        # Check pattern structure
        self.assertIn('daily', patterns)
        self.assertIn('weekly', patterns)
        self.assertIn('temporal_relationships', patterns)
        
        # Check daily pattern
        daily = patterns['daily']
        self.assertIn('hourly_means', daily)
        self.assertIn('peak_hours', daily)
        self.assertIn('strength', daily)
        
        # Check temporal relationships
        relationships = patterns['temporal_relationships']
        for symptom in self.symptoms:
            self.assertIn(symptom, relationships)
            self.assertIn('lag', relationships[symptom])
            self.assertIn('correlation', relationships[symptom])
    
    def test_trigger_interactions(self):
        """Test trigger interaction analysis."""
        interactions = self.analyzer._analyze_trigger_interactions(
            self.triggers,
            self.symptoms,
            self.timestamps
        )
        
        # Check pairwise interactions
        trigger_names = list(self.triggers.keys())
        for i, name1 in enumerate(trigger_names):
            for j, name2 in enumerate(trigger_names):
                if i < j:
                    key = f"{name1}-{name2}"
                    self.assertIn(key, interactions)
                    self.assertGreaterEqual(interactions[key], 0)
                    self.assertLessEqual(interactions[key], 1)
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with empty data
        empty_results = self.analyzer.identify_triggers(
            {},
            {},
            np.array([])
        )
        self.assertEqual(len(empty_results['triggers']), 0)
        
        # Test with single sample
        single_sample = {
            'trigger': np.array([1.0]),
            'symptom': np.array([1.0])
        }
        single_time = np.array([datetime.now()])
        single_results = self.analyzer.identify_triggers(
            {'symptom': single_sample['symptom']},
            {'trigger': single_sample['trigger']},
            single_time
        )
        self.assertEqual(len(single_results['triggers']), 0)
        
        # Test with invalid timestamps (should handle gracefully)
        invalid_results = self.analyzer.identify_triggers(
            self.symptoms,
            self.triggers,
            np.array([0, -1, 2])  # Invalid timestamps
        )
        self.assertEqual(len(invalid_results['triggers']), 0)
        self.assertIn('temporal_patterns', invalid_results)
        self.assertEqual(invalid_results['temporal_patterns'], {})
    
    def test_historical_pattern_analysis(self):
        """Test historical pattern analysis."""
        patterns = self.analyzer._analyze_temporal_trigger_patterns(
            self.trigger_history,
            self.migraine_events
        )
        
        # Check pattern structure for each trigger type
        trigger_types = set(event.trigger_type for event in self.trigger_history)
        for trigger_type in trigger_types:
            self.assertIn(trigger_type, patterns)
            type_patterns = patterns[trigger_type]
            
            # Check daily patterns
            self.assertIn('daily', type_patterns)
            self.assertIn('strength', type_patterns['daily'])
            self.assertIn('peak_hours', type_patterns['daily'])
            
            # Check weekly patterns
            self.assertIn('weekly', type_patterns)
            self.assertIn('strength', type_patterns['weekly'])
            self.assertIn('peak_days', type_patterns['weekly'])
            
            # Check seasonal patterns
            self.assertIn('seasonal', type_patterns)
            self.assertIn('strength', type_patterns['seasonal'])
            self.assertIn('peak_months', type_patterns['seasonal'])
    
    def test_threshold_calculation(self):
        """Test threshold calculation for triggers."""
        thresholds = self.analyzer._calculate_threshold_ranges(
            self.trigger_history,
            self.migraine_events
        )
        
        # Check thresholds for each trigger type
        trigger_types = set(event.trigger_type for event in self.trigger_history)
        for trigger_type in trigger_types:
            self.assertIn(trigger_type, thresholds)
            lower, upper = thresholds[trigger_type]
            
            # Check threshold values
            self.assertLessEqual(lower, upper)
            self.assertGreaterEqual(lower, 0)
            self.assertLessEqual(upper, 1)
    
    def test_profile_confidence(self):
        """Test confidence score calculation for trigger profiles."""
        # First get temporal patterns
        patterns = self.analyzer._analyze_temporal_trigger_patterns(
            self.trigger_history,
            self.migraine_events
        )
        
        # Calculate confidence scores
        confidence = self.analyzer._calculate_profile_confidence(
            self.trigger_history,
            self.migraine_events,
            patterns
        )
        
        # Check confidence scores
        trigger_types = set(event.trigger_type for event in self.trigger_history)
        for trigger_type in trigger_types:
            self.assertIn(trigger_type, confidence)
            score = confidence[trigger_type]
            
            # Check score range
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main() 
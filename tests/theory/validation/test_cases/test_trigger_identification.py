"""
Test Cases for Trigger Identification Components.

This module provides test cases for validating the theoretical correctness
of trigger identification mechanisms using synthetic data generators.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from tests.theory.validation.test_harness import TestCase, CoreLayerHarness
from tests.theory.validation.synthetic_generators.patient_generators import (
    PatientGenerator, LongitudinalDataGenerator
)
from tests.theory.validation.synthetic_generators.trigger_generators import (
    TriggerGenerator, TriggerProfile, TriggerInteractionGenerator
)
from core.theory.migraine_adaptation.trigger_identification import (
    TriggerIdentifier,
    CausalAnalyzer,
    SensitivityAnalyzer,
    InteractionAnalyzer
)

class TestTriggerCausality:
    """Test cases for trigger causality analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.identifier = TriggerIdentifier()
        self.causal = CausalAnalyzer()
        self.trigger_gen = TriggerGenerator()
        self.patient_gen = PatientGenerator(num_patients=1)
    
    def test_causal_discovery(self):
        """Test discovery of causal relationships between triggers and migraines."""
        # Generate synthetic trigger-migraine data with known causal relationships
        duration_days = 90  # 3 months
        causal_structure = {
            "primary_triggers": [
                {
                    "name": "stress",
                    "strength": 0.8,
                    "lag": 6,     # hours
                    "duration": 12  # hours
                },
                {
                    "name": "sleep_disruption",
                    "strength": 0.7,
                    "lag": 12,    # hours
                    "duration": 24  # hours
                }
            ],
            "secondary_triggers": [
                {
                    "name": "caffeine",
                    "strength": 0.4,
                    "lag": 4,     # hours
                    "duration": 8  # hours
                },
                {
                    "name": "weather",
                    "strength": 0.3,
                    "lag": 24,    # hours
                    "duration": 48  # hours
                }
            ],
            "non_causal_factors": [
                "exercise",
                "diet",
                "screen_time"
            ]
        }
        
        # Generate data
        trigger_data = self.trigger_gen.generate_causal_data(
            duration_days=duration_days,
            causal_structure=causal_structure,
            sampling_rate="hourly"
        )
        
        # Create test case
        test_case = TestCase(
            name="causal_discovery",
            inputs={
                "trigger_data": trigger_data['triggers'],
                "migraine_data": trigger_data['migraines'],
                "temporal_resolution": "hourly",
                "analysis_parameters": {
                    "max_lag": 48,        # hours
                    "min_strength": 0.2,
                    "confidence_level": 0.95
                }
            },
            expected_outputs={
                "causal_identification": {
                    "primary_triggers": ["stress", "sleep_disruption"],
                    "secondary_triggers": ["caffeine", "weather"],
                    "non_causal": ["exercise", "diet", "screen_time"]
                },
                "causal_metrics": {
                    "primary": {
                        "strength": (0.7, 0.9),
                        "lag": (4, 8),      # hours
                        "confidence": 0.95
                    },
                    "secondary": {
                        "strength": (0.3, 0.5),
                        "lag": (2, 24),     # hours
                        "confidence": 0.8
                    }
                },
                "temporal_characteristics": {
                    "detection_latency": 2.0,  # hours
                    "persistence": 0.8,
                    "stability": 0.85
                }
            },
            tolerance={
                "strength": 0.1,    # ±10% tolerance
                "lag": 2.0,         # ±2 hours tolerance
                "metrics": 0.1      # ±10% tolerance
            },
            metadata={
                "description": "Validate causal trigger discovery",
                "true_structure": causal_structure,
                "data_quality": "high"
            }
        )
        
        # Create validation function
        def validate_causality(
            trigger_data: Dict[str, np.ndarray],
            migraine_data: Dict[str, np.ndarray],
            temporal_resolution: str,
            analysis_parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Perform causal discovery
            causal_results = self.causal.discover_causal_triggers(
                triggers=trigger_data,
                migraines=migraine_data,
                resolution=temporal_resolution,
                parameters=analysis_parameters
            )
            
            # Calculate causal metrics
            metrics = self.causal.calculate_causal_metrics(
                results=causal_results,
                true_structure=test_case.metadata['true_structure']
            )
            
            # Analyze temporal characteristics
            temporal = self.causal.analyze_temporal_characteristics(
                results=causal_results,
                data_resolution=temporal_resolution
            )
            
            return {
                "causal_identification": {
                    "primary_triggers": causal_results['primary'],
                    "secondary_triggers": causal_results['secondary'],
                    "non_causal": causal_results['non_causal']
                },
                "causal_metrics": metrics,
                "temporal_characteristics": temporal
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("causal_discovery", validate_causality)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestTriggerSensitivity:
    """Test cases for trigger sensitivity analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.identifier = TriggerIdentifier()
        self.sensitivity = SensitivityAnalyzer()
        self.profile_gen = TriggerProfile()
    
    def test_sensitivity_analysis(self):
        """Test analysis of patient-specific trigger sensitivities."""
        # Generate synthetic patient profile with known sensitivities
        sensitivity_profile = {
            "high_sensitivity": {
                "stress": {
                    "threshold": 0.4,    # Activation threshold
                    "response": 0.8,     # Response strength
                    "variability": 0.1   # Day-to-day variability
                },
                "bright_light": {
                    "threshold": 0.5,
                    "response": 0.7,
                    "variability": 0.15
                }
            },
            "moderate_sensitivity": {
                "dehydration": {
                    "threshold": 0.6,
                    "response": 0.5,
                    "variability": 0.2
                },
                "noise": {
                    "threshold": 0.7,
                    "response": 0.4,
                    "variability": 0.25
                }
            },
            "low_sensitivity": {
                "caffeine": {
                    "threshold": 0.8,
                    "response": 0.2,
                    "variability": 0.3
                }
            }
        }
        
        # Generate data
        profile_data = self.profile_gen.generate_sensitivity_data(
            duration_days=60,
            sensitivity_profile=sensitivity_profile
        )
        
        # Create test case
        test_case = TestCase(
            name="sensitivity_analysis",
            inputs={
                "trigger_exposures": profile_data['exposures'],
                "migraine_events": profile_data['migraines'],
                "analysis_window": 30,  # days
                "sensitivity_parameters": {
                    "min_exposures": 10,
                    "threshold_step": 0.1,
                    "confidence_level": 0.95
                }
            },
            expected_outputs={
                "sensitivity_classification": {
                    "high": ["stress", "bright_light"],
                    "moderate": ["dehydration", "noise"],
                    "low": ["caffeine"]
                },
                "threshold_estimation": {
                    "stress": (0.35, 0.45),
                    "bright_light": (0.45, 0.55),
                    "dehydration": (0.55, 0.65),
                    "noise": (0.65, 0.75),
                    "caffeine": (0.75, 0.85)
                },
                "response_characteristics": {
                    "strength_estimation": {
                        "accuracy": 0.9,
                        "precision": 0.85,
                        "recall": 0.85
                    },
                    "variability_estimation": {
                        "accuracy": 0.85,
                        "temporal_stability": 0.8
                    }
                }
            },
            tolerance={
                "thresholds": 0.05,   # ±5% tolerance
                "strength": 0.1,      # ±10% tolerance
                "variability": 0.1    # ±10% tolerance
            },
            metadata={
                "description": "Validate trigger sensitivity analysis",
                "true_profile": sensitivity_profile,
                "analysis_method": "probabilistic"
            }
        )
        
        # Create validation function
        def validate_sensitivity(
            trigger_exposures: Dict[str, np.ndarray],
            migraine_events: Dict[str, np.ndarray],
            analysis_window: int,
            sensitivity_parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Perform sensitivity analysis
            sensitivity_results = self.sensitivity.analyze_sensitivities(
                exposures=trigger_exposures,
                migraines=migraine_events,
                window_days=analysis_window,
                parameters=sensitivity_parameters
            )
            
            # Classify sensitivities
            classification = self.sensitivity.classify_triggers(
                results=sensitivity_results,
                true_profile=test_case.metadata['true_profile']
            )
            
            # Estimate thresholds
            thresholds = self.sensitivity.estimate_thresholds(
                results=sensitivity_results,
                method=test_case.metadata['analysis_method']
            )
            
            # Analyze response characteristics
            response = self.sensitivity.analyze_responses(
                results=sensitivity_results,
                true_profile=test_case.metadata['true_profile']
            )
            
            return {
                "sensitivity_classification": classification,
                "threshold_estimation": thresholds,
                "response_characteristics": response
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("sensitivity_analysis", validate_sensitivity)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestTriggerInteractions:
    """Test cases for multi-trigger interaction analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.identifier = TriggerIdentifier()
        self.interaction = InteractionAnalyzer()
        self.interaction_gen = TriggerInteractionGenerator()
    
    def test_interaction_detection(self):
        """Test detection of interactions between multiple triggers."""
        # Generate synthetic data with known trigger interactions
        interaction_patterns = {
            "synergistic": [
                {
                    "triggers": ["stress", "sleep_disruption"],
                    "strength": 0.9,  # Combined effect stronger than individual
                    "temporal_pattern": "concurrent"
                },
                {
                    "triggers": ["bright_light", "noise"],
                    "strength": 0.8,
                    "temporal_pattern": "sequential"
                }
            ],
            "antagonistic": [
                {
                    "triggers": ["caffeine", "exercise"],
                    "strength": -0.3,  # Exercise reduces caffeine effect
                    "temporal_pattern": "overlapping"
                }
            ],
            "independent": [
                {
                    "triggers": ["weather", "diet"],
                    "strength": 0.0,
                    "temporal_pattern": "any"
                }
            ]
        }
        
        # Generate data
        interaction_data = self.interaction_gen.generate_interaction_data(
            duration_days=90,
            interaction_patterns=interaction_patterns
        )
        
        # Create test case
        test_case = TestCase(
            name="interaction_detection",
            inputs={
                "trigger_data": interaction_data['triggers'],
                "migraine_data": interaction_data['migraines'],
                "analysis_parameters": {
                    "max_interaction_order": 3,
                    "min_support": 0.1,
                    "significance_level": 0.05
                }
            },
            expected_outputs={
                "interaction_discovery": {
                    "synergistic": [
                        ["stress", "sleep_disruption"],
                        ["bright_light", "noise"]
                    ],
                    "antagonistic": [
                        ["caffeine", "exercise"]
                    ],
                    "independent": [
                        ["weather", "diet"]
                    ]
                },
                "interaction_metrics": {
                    "synergistic": {
                        "strength": (0.8, 1.0),
                        "reliability": 0.9,
                        "temporal_consistency": 0.85
                    },
                    "antagonistic": {
                        "strength": (-0.4, -0.2),
                        "reliability": 0.8,
                        "temporal_consistency": 0.8
                    }
                },
                "pattern_characteristics": {
                    "temporal_patterns": {
                        "accuracy": 0.9,
                        "pattern_stability": 0.85
                    },
                    "interaction_dynamics": {
                        "onset_detection": 0.85,
                        "duration_estimation": 0.8
                    }
                }
            },
            tolerance={
                "strength": 0.1,     # ±10% tolerance
                "reliability": 0.1,   # ±10% tolerance
                "patterns": 0.15     # ±15% tolerance
            },
            metadata={
                "description": "Validate trigger interaction detection",
                "true_patterns": interaction_patterns,
                "detection_method": "information_theoretic"
            }
        )
        
        # Create validation function
        def validate_interactions(
            trigger_data: Dict[str, np.ndarray],
            migraine_data: Dict[str, np.ndarray],
            analysis_parameters: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Detect interactions
            interaction_results = self.interaction.detect_interactions(
                triggers=trigger_data,
                migraines=migraine_data,
                parameters=analysis_parameters
            )
            
            # Calculate interaction metrics
            metrics = self.interaction.calculate_metrics(
                results=interaction_results,
                true_patterns=test_case.metadata['true_patterns']
            )
            
            # Analyze pattern characteristics
            patterns = self.interaction.analyze_patterns(
                results=interaction_results,
                method=test_case.metadata['detection_method']
            )
            
            return {
                "interaction_discovery": {
                    "synergistic": interaction_results['synergistic'],
                    "antagonistic": interaction_results['antagonistic'],
                    "independent": interaction_results['independent']
                },
                "interaction_metrics": metrics,
                "pattern_characteristics": patterns
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("interaction_detection", validate_interactions)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 
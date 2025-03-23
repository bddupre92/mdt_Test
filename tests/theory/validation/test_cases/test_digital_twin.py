"""
Test Cases for Digital Twin Components.

This module provides test cases for validating the theoretical correctness
of the digital twin model using synthetic data generators.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from tests.theory.validation.test_harness import TestCase, CoreLayerHarness
from tests.theory.validation.synthetic_generators.patient_generators import (
    PatientGenerator, LongitudinalDataGenerator
)
from core.theory.migraine_adaptation.digital_twin import (
    DigitalTwin,
    StateEstimator,
    InterventionSimulator,
    AdaptationEngine
)

class TestDigitalTwinState:
    """Test cases for digital twin state representation and estimation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.twin = DigitalTwin()
        self.estimator = StateEstimator()
        self.patient_gen = PatientGenerator(num_patients=1)
    
    def test_state_initialization(self):
        """Test initialization of digital twin state from patient data."""
        # Generate synthetic patient data with known characteristics
        profile = self.patient_gen.generate_profile(
            risk_factors={
                "age": 35,
                "gender": "F",
                "family_history": True,
                "comorbidities": ["anxiety", "insomnia"]
            },
            trigger_sensitivities={
                "stress": 0.8,
                "sleep_disruption": 0.7,
                "weather_changes": 0.5,
                "bright_lights": 0.4
            },
            baseline_characteristics={
                "migraine_frequency": 4.0,  # per month
                "typical_severity": 7.0,    # 0-10 scale
                "typical_duration": 24.0    # hours
            }
        )
        
        # Create test case
        test_case = TestCase(
            name="state_initialization",
            inputs={
                "patient_profile": profile,
                "initialization_period": 30,  # days
                "required_data_types": [
                    "physiological",
                    "environmental",
                    "behavioral"
                ]
            },
            expected_outputs={
                "state_completeness": {
                    "risk_factors": 0.95,      # 95% complete
                    "trigger_profile": 0.9,     # 90% complete
                    "baseline_state": 0.85      # 85% complete
                },
                "state_accuracy": {
                    "risk_assessment": 0.9,     # 90% accurate
                    "trigger_sensitivity": 0.85, # 85% accurate
                    "pattern_detection": 0.8     # 80% accurate
                },
                "uncertainty_quantification": {
                    "risk_uncertainty": 0.15,    # ±15% uncertainty
                    "sensitivity_uncertainty": 0.2,
                    "pattern_uncertainty": 0.25
                }
            },
            tolerance={
                "completeness": 0.1,   # ±10% tolerance
                "accuracy": 0.1,       # ±10% tolerance
                "uncertainty": 0.05    # ±5% tolerance
            },
            metadata={
                "description": "Validate digital twin state initialization",
                "true_profile": profile,
                "data_requirements": {
                    "physiological": ["ecg", "eeg", "gsr"],
                    "environmental": ["weather", "light", "noise"],
                    "behavioral": ["sleep", "activity", "stress"]
                }
            }
        )
        
        # Create validation function
        def validate_initialization(
            patient_profile: Dict[str, Any],
            initialization_period: int,
            required_data_types: List[str]
        ) -> Dict[str, Any]:
            # Initialize digital twin
            twin_state = self.twin.initialize_state(
                profile=patient_profile,
                period_days=initialization_period,
                required_data=required_data_types
            )
            
            # Evaluate state completeness
            completeness = self.estimator.evaluate_completeness(
                state=twin_state,
                requirements=test_case.metadata['data_requirements']
            )
            
            # Assess state accuracy
            accuracy = self.estimator.assess_accuracy(
                state=twin_state,
                true_profile=test_case.metadata['true_profile']
            )
            
            # Quantify uncertainty
            uncertainty = self.estimator.quantify_uncertainty(
                state=twin_state,
                confidence_level=0.95
            )
            
            return {
                "state_completeness": completeness,
                "state_accuracy": accuracy,
                "uncertainty_quantification": uncertainty
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("state_initialization", validate_initialization)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_state_update(self):
        """Test updating of digital twin state with new data."""
        # Generate initial state and update data
        duration_days = 7
        update_events = [
            {
                "type": "migraine_episode",
                "timestamp": "2024-03-15T14:30:00",
                "severity": 8,
                "duration": 18,  # hours
                "triggers": ["stress", "poor_sleep"]
            },
            {
                "type": "intervention",
                "timestamp": "2024-03-15T15:00:00",
                "intervention": "medication",
                "effectiveness": 0.7
            },
            {
                "type": "lifestyle_change",
                "timestamp": "2024-03-16T00:00:00",
                "change": "sleep_schedule",
                "impact": 0.4
            }
        ]
        
        # Create test case
        test_case = TestCase(
            name="state_update",
            inputs={
                "current_state": self.twin.get_initial_state(),
                "update_events": update_events,
                "update_period": duration_days,
                "update_types": ["episodes", "interventions", "lifestyle"]
            },
            expected_outputs={
                "update_quality": {
                    "state_consistency": 0.9,    # State remains consistent
                    "temporal_coherence": 0.85,  # Updates are temporally coherent
                    "causal_validity": 0.9      # Updates respect causal relationships
                },
                "adaptation_metrics": {
                    "trigger_sensitivity": {
                        "stress": (0.7, 0.9),    # Updated sensitivity range
                        "poor_sleep": (0.6, 0.8)
                    },
                    "intervention_efficacy": {
                        "medication": (0.6, 0.8)  # Updated efficacy range
                    },
                    "pattern_evolution": {
                        "detection_rate": 0.85,
                        "false_alarm_rate": 0.15
                    }
                },
                "uncertainty_updates": {
                    "sensitivity_reduction": 0.2,  # 20% reduction in uncertainty
                    "efficacy_reduction": 0.15,
                    "pattern_reduction": 0.1
                }
            },
            tolerance={
                "quality": 0.1,      # ±10% tolerance
                "metrics": 0.15,     # ±15% tolerance
                "uncertainty": 0.05  # ±5% tolerance
            },
            metadata={
                "description": "Validate digital twin state updates",
                "true_events": update_events,
                "update_frequency": "daily"
            }
        )
        
        # Create validation function
        def validate_update(
            current_state: Dict[str, Any],
            update_events: List[Dict[str, Any]],
            update_period: int,
            update_types: List[str]
        ) -> Dict[str, Any]:
            # Perform state update
            updated_state = self.twin.update_state(
                current_state=current_state,
                events=update_events,
                period_days=update_period,
                update_types=update_types
            )
            
            # Evaluate update quality
            quality_metrics = self.estimator.evaluate_update_quality(
                previous_state=current_state,
                updated_state=updated_state,
                events=update_events
            )
            
            # Assess adaptation
            adaptation_metrics = self.estimator.assess_adaptation(
                previous_state=current_state,
                updated_state=updated_state,
                period_days=update_period
            )
            
            # Calculate uncertainty reduction
            uncertainty_reduction = self.estimator.calculate_uncertainty_reduction(
                previous_state=current_state,
                updated_state=updated_state
            )
            
            return {
                "update_quality": quality_metrics,
                "adaptation_metrics": adaptation_metrics,
                "uncertainty_updates": uncertainty_reduction
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("state_update", validate_update)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestInterventionSimulation:
    """Test cases for intervention simulation in digital twin."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.twin = DigitalTwin()
        self.simulator = InterventionSimulator()
        self.adaptation = AdaptationEngine()
    
    def test_intervention_efficacy(self):
        """Test simulation of intervention efficacy."""
        # Generate synthetic intervention scenarios
        interventions = [
            {
                "type": "medication",
                "name": "sumatriptan",
                "timing": "onset",
                "dosage": "100mg",
                "expected_efficacy": 0.75
            },
            {
                "type": "behavioral",
                "name": "stress_reduction",
                "timing": "preventive",
                "duration": "30min",
                "expected_efficacy": 0.6
            },
            {
                "type": "environmental",
                "name": "light_reduction",
                "timing": "trigger_response",
                "intensity": "high",
                "expected_efficacy": 0.5
            }
        ]
        
        # Create test case
        test_case = TestCase(
            name="intervention_simulation",
            inputs={
                "patient_state": self.twin.get_current_state(),
                "interventions": interventions,
                "simulation_period": 30,  # days
                "simulation_iterations": 100
            },
            expected_outputs={
                "efficacy_assessment": {
                    "medication": {
                        "mean_efficacy": 0.75,
                        "response_time": 45,  # minutes
                        "duration": 24      # hours
                    },
                    "behavioral": {
                        "mean_efficacy": 0.6,
                        "onset_delay": 2,   # days
                        "sustainability": 0.8
                    },
                    "environmental": {
                        "mean_efficacy": 0.5,
                        "trigger_reduction": 0.7,
                        "rebound_risk": 0.2
                    }
                },
                "simulation_quality": {
                    "convergence": 0.9,     # Simulation convergence
                    "stability": 0.85,      # Result stability
                    "reproducibility": 0.95  # Result reproducibility
                },
                "uncertainty_bounds": {
                    "efficacy": 0.1,        # ±10% uncertainty
                    "timing": 0.15,         # ±15% uncertainty
                    "duration": 0.2         # ±20% uncertainty
                }
            },
            tolerance={
                "efficacy": 0.1,     # ±10% tolerance
                "quality": 0.1,      # ±10% tolerance
                "uncertainty": 0.05  # ±5% tolerance
            },
            metadata={
                "description": "Validate intervention simulation",
                "true_interventions": interventions,
                "simulation_config": {
                    "random_seed": 42,
                    "confidence_level": 0.95
                }
            }
        )
        
        # Create validation function
        def validate_simulation(
            patient_state: Dict[str, Any],
            interventions: List[Dict[str, Any]],
            simulation_period: int,
            simulation_iterations: int
        ) -> Dict[str, Any]:
            # Run intervention simulations
            simulation_results = self.simulator.simulate_interventions(
                state=patient_state,
                interventions=interventions,
                period_days=simulation_period,
                n_iterations=simulation_iterations
            )
            
            # Assess efficacy
            efficacy_assessment = self.simulator.assess_efficacy(
                results=simulation_results,
                true_interventions=test_case.metadata['true_interventions']
            )
            
            # Evaluate simulation quality
            quality_metrics = self.simulator.evaluate_quality(
                results=simulation_results,
                config=test_case.metadata['simulation_config']
            )
            
            # Calculate uncertainty bounds
            uncertainty_bounds = self.simulator.calculate_uncertainty_bounds(
                results=simulation_results,
                confidence_level=0.95
            )
            
            return {
                "efficacy_assessment": efficacy_assessment,
                "simulation_quality": quality_metrics,
                "uncertainty_bounds": uncertainty_bounds
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("intervention_simulation", validate_simulation)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_adaptation_mechanisms(self):
        """Test adaptation mechanisms in the digital twin."""
        # Generate adaptation scenarios
        adaptation_scenarios = [
            {
                "type": "trigger_sensitivity",
                "trigger": "stress",
                "change": 0.2,  # 20% increase in sensitivity
                "timeframe": 14  # days
            },
            {
                "type": "treatment_response",
                "intervention": "sumatriptan",
                "change": -0.1,  # 10% decrease in efficacy
                "timeframe": 30  # days
            },
            {
                "type": "pattern_evolution",
                "pattern": "frequency",
                "change": 0.3,  # 30% increase in frequency
                "timeframe": 21  # days
            }
        ]
        
        # Create test case
        test_case = TestCase(
            name="adaptation_mechanisms",
            inputs={
                "initial_state": self.twin.get_current_state(),
                "adaptation_scenarios": adaptation_scenarios,
                "monitoring_period": 30,  # days
                "adaptation_parameters": {
                    "learning_rate": 0.1,
                    "memory_length": 90,  # days
                    "threshold": 0.2
                }
            },
            expected_outputs={
                "adaptation_performance": {
                    "detection_rate": 0.9,    # Change detection rate
                    "false_alarm_rate": 0.1,  # False adaptation rate
                    "adaptation_speed": 0.8   # Speed of adaptation
                },
                "mechanism_evaluation": {
                    "trigger_adaptation": {
                        "accuracy": 0.85,
                        "stability": 0.8,
                        "responsiveness": 0.9
                    },
                    "treatment_adaptation": {
                        "accuracy": 0.8,
                        "stability": 0.85,
                        "responsiveness": 0.85
                    },
                    "pattern_adaptation": {
                        "accuracy": 0.75,
                        "stability": 0.8,
                        "responsiveness": 0.8
                    }
                },
                "adaptation_metrics": {
                    "learning_curve": (0.6, 0.9),  # Learning progression
                    "stability_index": 0.85,       # Adaptation stability
                    "generalization": 0.8         # Generalization capability
                }
            },
            tolerance={
                "performance": 0.1,   # ±10% tolerance
                "evaluation": 0.15,   # ±15% tolerance
                "metrics": 0.1       # ±10% tolerance
            },
            metadata={
                "description": "Validate adaptation mechanisms",
                "true_scenarios": adaptation_scenarios,
                "adaptation_config": {
                    "min_data_points": 10,
                    "confidence_threshold": 0.9
                }
            }
        )
        
        # Create validation function
        def validate_adaptation(
            initial_state: Dict[str, Any],
            adaptation_scenarios: List[Dict[str, Any]],
            monitoring_period: int,
            adaptation_parameters: Dict[str, float]
        ) -> Dict[str, Any]:
            # Run adaptation simulation
            adapted_state = self.adaptation.simulate_adaptation(
                initial_state=initial_state,
                scenarios=adaptation_scenarios,
                period_days=monitoring_period,
                parameters=adaptation_parameters
            )
            
            # Evaluate adaptation performance
            performance_metrics = self.adaptation.evaluate_performance(
                initial_state=initial_state,
                adapted_state=adapted_state,
                true_scenarios=test_case.metadata['true_scenarios']
            )
            
            # Assess adaptation mechanisms
            mechanism_evaluation = self.adaptation.evaluate_mechanisms(
                adapted_state=adapted_state,
                scenarios=adaptation_scenarios,
                config=test_case.metadata['adaptation_config']
            )
            
            # Calculate adaptation metrics
            adaptation_metrics = self.adaptation.calculate_metrics(
                initial_state=initial_state,
                adapted_state=adapted_state,
                period_days=monitoring_period
            )
            
            return {
                "adaptation_performance": performance_metrics,
                "mechanism_evaluation": mechanism_evaluation,
                "adaptation_metrics": adaptation_metrics
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("adaptation_mechanisms", validate_adaptation)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 
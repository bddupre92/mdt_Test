"""
Integration Tests for End-to-End Workflows.

This module provides test cases for validating complete workflows from data ingestion
through prediction, trigger identification, and visualization in the system.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from tests.theory.validation.test_harness import IntegrationTestCase, IntegrationTestHarness
from tests.theory.validation.synthetic_generators import (
    PatientDataGenerator,
    SensorDataGenerator,
    EnvironmentalDataGenerator
)
from core.data_ingestion import DataIngestionPipeline
from core.theory.temporal_modeling import PredictionEngine
from core.theory.migraine_adaptation import TriggerIdentifier
from core.application.alert_system import AlertManager
from core.application.visualization import VisualizationEngine
from core.digital_twin import DigitalTwinManager

class TestEndToEndWorkflows:
    """Test cases for complete end-to-end system workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ingestion = DataIngestionPipeline()
        self.prediction = PredictionEngine()
        self.trigger_id = TriggerIdentifier()
        self.alerts = AlertManager()
        self.visualization = VisualizationEngine()
        self.twin_manager = DigitalTwinManager()
        
        # Data generators
        self.patient_gen = PatientDataGenerator()
        self.sensor_gen = SensorDataGenerator()
        self.env_gen = EnvironmentalDataGenerator()
    
    def test_data_ingestion_to_prediction(self):
        """Test complete workflow from data ingestion to prediction generation."""
        # Generate test data
        test_duration = 30  # days
        test_data = {
            "physiological": self.sensor_gen.generate_sensor_data(test_duration),
            "environmental": self.env_gen.generate_environmental_data(test_duration),
            "patient_records": self.patient_gen.generate_patient_history(test_duration)
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="ingestion_to_prediction",
            inputs={
                "raw_data": test_data,
                "pipeline_config": {
                    "batch_size": 24,  # hours
                    "validation_rules": ["completeness", "consistency", "range"],
                    "processing_steps": ["clean", "normalize", "aggregate"]
                },
                "prediction_params": {
                    "horizon": 48,     # hours
                    "features": ["all"],
                    "confidence_threshold": 0.8
                }
            },
            expected_outputs={
                "data_quality": {
                    "completeness": 0.95,
                    "consistency": 0.9,
                    "validity": 0.95
                },
                "prediction_quality": {
                    "accuracy": 0.85,
                    "reliability": 0.9,
                    "lead_time": 24  # hours
                },
                "system_metrics": {
                    "processing_time": 300,  # seconds
                    "memory_usage": 2.0,     # GB
                    "pipeline_reliability": 0.99
                }
            },
            tolerance={
                "quality": 0.1,     # ±10% tolerance
                "timing": 60,       # ±60 seconds tolerance
                "resources": 0.5    # ±0.5 GB tolerance
            },
            metadata={
                "description": "Validate data ingestion to prediction workflow",
                "environment": "test",
                "data_volume": "medium"
            }
        )
        
        # Create validation function
        def validate_workflow(
            raw_data: Dict[str, Any],
            pipeline_config: Dict[str, Any],
            prediction_params: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Run data ingestion pipeline
            processed_data = self.ingestion.process_data(
                data=raw_data,
                config=pipeline_config
            )
            
            # Validate data quality
            quality_metrics = self.ingestion.validate_data(
                data=processed_data,
                rules=pipeline_config["validation_rules"]
            )
            
            # Generate predictions
            predictions = self.prediction.generate_predictions(
                data=processed_data,
                parameters=prediction_params
            )
            
            # Evaluate prediction quality
            prediction_metrics = self.prediction.evaluate_predictions(
                predictions=predictions,
                actual_data=raw_data["patient_records"]
            )
            
            # Monitor system performance
            system_metrics = self.ingestion.get_system_metrics()
            
            return {
                "data_quality": quality_metrics,
                "prediction_quality": prediction_metrics,
                "system_metrics": system_metrics
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("ingestion_to_prediction", validate_workflow)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_trigger_identification_to_alerts(self):
        """Test complete workflow from trigger identification to alert generation."""
        # Generate test scenarios
        test_duration = 14  # days
        trigger_scenarios = {
            "known_triggers": [
                {"type": "stress", "pattern": "acute", "strength": 0.8},
                {"type": "sleep", "pattern": "disrupted", "strength": 0.7},
                {"type": "weather", "pattern": "pressure_change", "strength": 0.6}
            ],
            "patient_sensitivity": {
                "stress": "high",
                "sleep": "medium",
                "weather": "low"
            },
            "temporal_patterns": {
                "daily": ["stress", "sleep"],
                "weekly": ["weather"],
                "random": ["other"]
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="triggers_to_alerts",
            inputs={
                "patient_data": self.patient_gen.generate_patient_data(test_duration),
                "trigger_scenarios": trigger_scenarios,
                "alert_config": {
                    "threshold": 0.7,
                    "channels": ["push", "email"],
                    "frequency": "optimal"
                }
            },
            expected_outputs={
                "trigger_identification": {
                    "accuracy": 0.85,
                    "detection_rate": 0.9,
                    "false_positive_rate": 0.1
                },
                "alert_effectiveness": {
                    "timeliness": 0.9,
                    "relevance": 0.85,
                    "actionability": 0.8
                },
                "workflow_metrics": {
                    "end_to_end_latency": 5.0,  # minutes
                    "processing_efficiency": 0.9,
                    "system_reliability": 0.95
                }
            },
            tolerance={
                "accuracy": 0.1,    # ±10% tolerance
                "timing": 2.0,      # ±2 minutes tolerance
                "efficiency": 0.1   # ±10% tolerance
            },
            metadata={
                "description": "Validate trigger identification to alerts workflow",
                "patient_profile": "standard",
                "alert_priority": "balanced"
            }
        )
        
        # Create validation function
        def validate_workflow(
            patient_data: Dict[str, Any],
            trigger_scenarios: Dict[str, Dict[str, Any]],
            alert_config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Identify triggers
            identified_triggers = self.trigger_id.identify_triggers(
                data=patient_data,
                scenarios=trigger_scenarios
            )
            
            # Evaluate trigger identification
            trigger_metrics = self.trigger_id.evaluate_identification(
                identified=identified_triggers,
                expected=trigger_scenarios["known_triggers"]
            )
            
            # Generate alerts
            alerts = self.alerts.generate_alerts(
                triggers=identified_triggers,
                config=alert_config
            )
            
            # Evaluate alert effectiveness
            alert_metrics = self.alerts.evaluate_effectiveness(
                alerts=alerts,
                patient_data=patient_data
            )
            
            # Measure workflow metrics
            workflow_metrics = {
                "end_to_end_latency": self.measure_latency(),
                "processing_efficiency": self.measure_efficiency(),
                "system_reliability": self.measure_reliability()
            }
            
            return {
                "trigger_identification": trigger_metrics,
                "alert_effectiveness": alert_metrics,
                "workflow_metrics": workflow_metrics
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("triggers_to_alerts", validate_workflow)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_digital_twin_to_visualization(self):
        """Test complete workflow from digital twin updates to visualization generation."""
        # Generate test scenarios
        test_duration = 7  # days
        visualization_scenarios = {
            "temporal_views": {
                "daily_patterns": ["migraine_intensity", "trigger_exposure"],
                "weekly_trends": ["recovery_time", "medication_efficacy"],
                "monthly_summary": ["trigger_frequency", "intervention_success"]
            },
            "interaction_modes": {
                "exploration": ["zoom", "filter", "details"],
                "comparison": ["overlay", "side-by-side"],
                "analysis": ["correlation", "pattern_detection"]
            },
            "update_frequency": {
                "real_time": ["current_state", "alerts"],
                "periodic": ["trends", "patterns"],
                "on_demand": ["historical", "analysis"]
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="twin_to_visualization",
            inputs={
                "twin_state": self.twin_manager.get_current_state(),
                "visualization_scenarios": visualization_scenarios,
                "update_config": {
                    "frequency": "real_time",
                    "detail_level": "high",
                    "cache_policy": "smart"
                }
            },
            expected_outputs={
                "visualization_quality": {
                    "accuracy": 0.95,
                    "completeness": 0.9,
                    "responsiveness": 0.85
                },
                "interaction_metrics": {
                    "update_latency": 0.1,    # seconds
                    "smoothness": 0.9,
                    "consistency": 0.95
                },
                "system_performance": {
                    "rendering_time": 0.2,     # seconds
                    "memory_efficiency": 0.8,
                    "cache_hit_rate": 0.9
                }
            },
            tolerance={
                "quality": 0.05,    # ±5% tolerance
                "latency": 0.05,    # ±50ms tolerance
                "performance": 0.1  # ±10% tolerance
            },
            metadata={
                "description": "Validate digital twin to visualization workflow",
                "visualization_mode": "interactive",
                "device_target": "all"
            }
        )
        
        # Create validation function
        def validate_workflow(
            twin_state: Dict[str, Any],
            visualization_scenarios: Dict[str, Dict[str, Any]],
            update_config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Update digital twin
            updated_state = self.twin_manager.update_state(
                current_state=twin_state,
                config=update_config
            )
            
            # Generate visualizations
            visualizations = self.visualization.generate_visualizations(
                state=updated_state,
                scenarios=visualization_scenarios
            )
            
            # Evaluate visualization quality
            quality_metrics = self.visualization.evaluate_quality(
                visualizations=visualizations,
                source_state=updated_state
            )
            
            # Measure interaction performance
            interaction_metrics = self.visualization.measure_interaction(
                visualizations=visualizations,
                scenarios=visualization_scenarios
            )
            
            # Monitor system performance
            performance_metrics = self.visualization.monitor_performance(
                visualizations=visualizations,
                config=update_config
            )
            
            return {
                "visualization_quality": quality_metrics,
                "interaction_metrics": interaction_metrics,
                "system_performance": performance_metrics
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("twin_to_visualization", validate_workflow)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 
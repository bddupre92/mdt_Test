"""
Integration Tests for Cross-Component Interactions.

This module provides test cases for validating interactions between different
components and layers of the system, including data transformation and state management.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from tests.theory.validation.test_harness import IntegrationTestCase, IntegrationTestHarness
from tests.theory.validation.synthetic_generators import (
    DataTransformGenerator,
    StateGenerator,
    CommunicationGenerator
)
from core.theory.temporal_modeling import ModelingLayer
from core.theory.migraine_adaptation import AdaptationLayer
from core.application.service_layer import ServiceLayer
from core.data_ingestion import DataLayer
from core.digital_twin import StateManager

class TestCrossComponentInteractions:
    """Test cases for interactions between system components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.modeling = ModelingLayer()
        self.adaptation = AdaptationLayer()
        self.service = ServiceLayer()
        self.data = DataLayer()
        self.state = StateManager()
        
        # Generators
        self.transform_gen = DataTransformGenerator()
        self.state_gen = StateGenerator()
        self.comm_gen = CommunicationGenerator()
    
    def test_inter_layer_communication(self):
        """Test communication and data flow between system layers."""
        # Generate test scenarios
        communication_scenarios = {
            "data_to_modeling": {
                "data_types": ["physiological", "environmental", "behavioral"],
                "frequency": "real-time",
                "validation": "strict"
            },
            "modeling_to_adaptation": {
                "models": ["prediction", "classification", "pattern"],
                "update_mode": "incremental",
                "synchronization": "automatic"
            },
            "adaptation_to_service": {
                "outputs": ["alerts", "recommendations", "visualizations"],
                "delivery": "immediate",
                "feedback": "enabled"
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="layer_communication",
            inputs={
                "test_data": self.comm_gen.generate_layer_data(),
                "scenarios": communication_scenarios,
                "config": {
                    "timeout": 30,  # seconds
                    "retry_policy": "exponential",
                    "monitoring": "detailed"
                }
            },
            expected_outputs={
                "communication_metrics": {
                    "latency": {
                        "data_to_modeling": 0.1,    # seconds
                        "modeling_to_adaptation": 0.2,
                        "adaptation_to_service": 0.15
                    },
                    "reliability": {
                        "data_flow": 0.99,
                        "model_updates": 0.95,
                        "service_delivery": 0.98
                    },
                    "consistency": {
                        "data_validation": 0.95,
                        "state_synchronization": 0.9,
                        "output_verification": 0.95
                    }
                },
                "error_handling": {
                    "recovery_rate": 0.95,
                    "error_isolation": 0.9,
                    "cascade_prevention": 0.85
                },
                "system_stability": {
                    "throughput": 1000,  # operations/second
                    "resource_balance": 0.8,
                    "deadlock_freedom": 1.0
                }
            },
            tolerance={
                "latency": 0.05,    # ±50ms tolerance
                "reliability": 0.02, # ±2% tolerance
                "stability": 0.1    # ±10% tolerance
            },
            metadata={
                "description": "Validate inter-layer communication",
                "test_duration": "1h",
                "load_profile": "variable"
            }
        )
        
        # Create validation function
        def validate_communication(
            test_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Test data layer to modeling layer
            data_modeling = self.modeling.receive_data(
                data=test_data["physiological"],
                source="data_layer",
                config=config
            )
            
            # Test modeling layer to adaptation layer
            modeling_adaptation = self.adaptation.receive_models(
                models=data_modeling["models"],
                source="modeling_layer",
                config=config
            )
            
            # Test adaptation layer to service layer
            adaptation_service = self.service.receive_outputs(
                outputs=modeling_adaptation["outputs"],
                source="adaptation_layer",
                config=config
            )
            
            # Calculate metrics
            metrics = {
                "communication_metrics": {
                    "latency": self.measure_latencies(),
                    "reliability": self.measure_reliability(),
                    "consistency": self.measure_consistency()
                },
                "error_handling": self.evaluate_error_handling(),
                "system_stability": self.measure_stability()
            }
            
            return metrics
        
        # Create harness and validate
        harness = IntegrationTestHarness("layer_communication", validate_communication)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_data_transformation(self):
        """Test data transformation and validation between components."""
        # Generate test scenarios
        transformation_scenarios = {
            "format_conversions": {
                "types": ["numerical", "categorical", "temporal"],
                "operations": ["normalize", "encode", "aggregate"],
                "validation": ["range", "type", "consistency"]
            },
            "feature_engineering": {
                "extractors": ["statistical", "temporal", "domain"],
                "selectors": ["importance", "correlation", "redundancy"],
                "validators": ["completeness", "relevance", "quality"]
            },
            "state_transformations": {
                "mappings": ["raw_to_processed", "processed_to_model", "model_to_view"],
                "validations": ["schema", "constraints", "relationships"],
                "optimizations": ["caching", "indexing", "compression"]
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="data_transformation",
            inputs={
                "raw_data": self.transform_gen.generate_raw_data(),
                "scenarios": transformation_scenarios,
                "config": {
                    "validation_level": "strict",
                    "optimization_mode": "balanced",
                    "error_handling": "robust"
                }
            },
            expected_outputs={
                "transformation_quality": {
                    "accuracy": 0.95,
                    "consistency": 0.9,
                    "completeness": 0.95
                },
                "validation_metrics": {
                    "schema_compliance": 1.0,
                    "constraint_satisfaction": 0.95,
                    "relationship_integrity": 0.9
                },
                "performance_metrics": {
                    "processing_time": 0.5,  # seconds
                    "memory_usage": 500,     # MB
                    "throughput": 5000       # records/second
                }
            },
            tolerance={
                "quality": 0.05,    # ±5% tolerance
                "validation": 0.02, # ±2% tolerance
                "performance": 0.1  # ±10% tolerance
            },
            metadata={
                "description": "Validate data transformation processes",
                "data_volume": "medium",
                "complexity": "high"
            }
        )
        
        # Create validation function
        def validate_transformation(
            raw_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Perform format conversions
            converted_data = self.data.convert_formats(
                data=raw_data,
                scenarios=scenarios["format_conversions"],
                config=config
            )
            
            # Engineer features
            engineered_data = self.modeling.engineer_features(
                data=converted_data,
                scenarios=scenarios["feature_engineering"],
                config=config
            )
            
            # Transform state representations
            transformed_state = self.state.transform_state(
                data=engineered_data,
                scenarios=scenarios["state_transformations"],
                config=config
            )
            
            # Calculate metrics
            quality = self.evaluate_transformation_quality(transformed_state)
            validation = self.validate_transformations(transformed_state)
            performance = self.measure_transformation_performance()
            
            return {
                "transformation_quality": quality,
                "validation_metrics": validation,
                "performance_metrics": performance
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("data_transformation", validate_transformation)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_state_management(self):
        """Test state management and synchronization across components."""
        # Generate test scenarios
        state_scenarios = {
            "state_transitions": {
                "types": ["patient", "model", "system"],
                "triggers": ["update", "sync", "reset"],
                "validation": ["consistency", "completeness"]
            },
            "synchronization": {
                "strategies": ["immediate", "periodic", "conditional"],
                "scope": ["local", "global", "hierarchical"],
                "conflict_resolution": ["timestamp", "priority", "merge"]
            },
            "persistence": {
                "storage": ["memory", "cache", "disk"],
                "durability": ["transient", "persistent"],
                "recovery": ["checkpoint", "journal", "snapshot"]
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="state_management",
            inputs={
                "initial_state": self.state_gen.generate_initial_state(),
                "scenarios": state_scenarios,
                "config": {
                    "sync_interval": 5,    # seconds
                    "consistency_model": "eventual",
                    "recovery_mode": "automatic"
                }
            },
            expected_outputs={
                "state_integrity": {
                    "consistency": 0.95,
                    "completeness": 0.9,
                    "correctness": 0.95
                },
                "sync_metrics": {
                    "latency": 0.2,        # seconds
                    "success_rate": 0.98,
                    "conflict_rate": 0.02
                },
                "recovery_metrics": {
                    "time_to_recover": 2.0, # seconds
                    "data_loss": 0.0,
                    "state_validity": 1.0
                }
            },
            tolerance={
                "integrity": 0.05,   # ±5% tolerance
                "sync": 0.1,         # ±100ms tolerance
                "recovery": 0.5      # ±500ms tolerance
            },
            metadata={
                "description": "Validate state management system",
                "state_complexity": "high",
                "update_frequency": "high"
            }
        )
        
        # Create validation function
        def validate_state_management(
            initial_state: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Test state transitions
            transitioned_state = self.state.handle_transitions(
                state=initial_state,
                scenarios=scenarios["state_transitions"],
                config=config
            )
            
            # Test state synchronization
            synced_state = self.state.synchronize_state(
                state=transitioned_state,
                scenarios=scenarios["synchronization"],
                config=config
            )
            
            # Test state persistence
            persisted_state = self.state.persist_state(
                state=synced_state,
                scenarios=scenarios["persistence"],
                config=config
            )
            
            # Calculate metrics
            integrity = self.evaluate_state_integrity(persisted_state)
            sync = self.measure_sync_metrics()
            recovery = self.test_recovery_metrics()
            
            return {
                "state_integrity": integrity,
                "sync_metrics": sync,
                "recovery_metrics": recovery
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("state_management", validate_state_management)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 
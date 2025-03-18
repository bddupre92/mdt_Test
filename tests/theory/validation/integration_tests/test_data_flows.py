"""
Integration Tests for Data Flow Validation.

This module provides test cases for validating data flows between different modules,
including physiological, environmental, and patient feedback data.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from tests.theory.validation.test_harness import IntegrationTestCase, IntegrationTestHarness
from tests.theory.validation.synthetic_generators import (
    PhysiologicalDataGenerator,
    EnvironmentalDataGenerator,
    FeedbackDataGenerator
)
from core.data_ingestion import DataPipeline
from core.theory.temporal_modeling import DataProcessor
from core.theory.migraine_adaptation import FeedbackAnalyzer
from core.digital_twin import DataIntegrator

class TestDataFlows:
    """Test cases for data flows between system modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pipeline = DataPipeline()
        self.processor = DataProcessor()
        self.analyzer = FeedbackAnalyzer()
        self.integrator = DataIntegrator()
        
        # Data generators
        self.physio_gen = PhysiologicalDataGenerator()
        self.env_gen = EnvironmentalDataGenerator()
        self.feedback_gen = FeedbackDataGenerator()
    
    def test_physiological_data_flow(self):
        """Test flow of physiological data through the system."""
        # Generate test scenarios
        physio_scenarios = {
            "data_sources": {
                "ecg": {
                    "sampling_rate": 250,  # Hz
                    "duration": 3600,      # seconds
                    "channels": ["lead_i", "lead_ii", "lead_iii"]
                },
                "eeg": {
                    "sampling_rate": 500,  # Hz
                    "duration": 3600,      # seconds
                    "channels": ["frontal", "temporal", "occipital"]
                },
                "gsr": {
                    "sampling_rate": 100,  # Hz
                    "duration": 3600,      # seconds
                    "channels": ["conductance", "resistance"]
                }
            },
            "processing_stages": {
                "preprocessing": ["filtering", "artifact_removal", "normalization"],
                "feature_extraction": ["temporal", "spectral", "statistical"],
                "integration": ["fusion", "synchronization", "validation"]
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="physiological_flow",
            inputs={
                "raw_data": self.physio_gen.generate_data(3600),  # 1 hour
                "scenarios": physio_scenarios,
                "config": {
                    "buffer_size": 1024,    # samples
                    "processing_mode": "streaming",
                    "quality_threshold": 0.9
                }
            },
            expected_outputs={
                "data_quality": {
                    "signal_quality": 0.95,
                    "noise_level": 0.05,
                    "artifact_rate": 0.02
                },
                "processing_metrics": {
                    "latency": 0.1,         # seconds
                    "throughput": 10000,    # samples/second
                    "accuracy": 0.98
                },
                "integration_metrics": {
                    "sync_accuracy": 0.99,
                    "data_consistency": 0.95,
                    "fusion_quality": 0.9
                }
            },
            tolerance={
                "quality": 0.05,    # ±5% tolerance
                "timing": 0.02,     # ±20ms tolerance
                "accuracy": 0.02    # ±2% tolerance
            },
            metadata={
                "description": "Validate physiological data flow",
                "data_type": "continuous",
                "criticality": "high"
            }
        )
        
        # Create validation function
        def validate_flow(
            raw_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Process raw physiological data
            processed_data = self.processor.process_physiological(
                data=raw_data,
                stages=scenarios["processing_stages"],
                config=config
            )
            
            # Evaluate data quality
            quality = self.pipeline.evaluate_quality(
                data=processed_data,
                sources=scenarios["data_sources"]
            )
            
            # Measure processing performance
            processing = self.processor.measure_performance(
                data=processed_data,
                config=config
            )
            
            # Assess data integration
            integration = self.integrator.assess_integration(
                data=processed_data,
                sources=scenarios["data_sources"]
            )
            
            return {
                "data_quality": quality,
                "processing_metrics": processing,
                "integration_metrics": integration
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("physiological_flow", validate_flow)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_environmental_data_flow(self):
        """Test flow of environmental data through the system."""
        # Generate test scenarios
        env_scenarios = {
            "data_sources": {
                "weather": {
                    "parameters": ["temperature", "pressure", "humidity"],
                    "frequency": "hourly",
                    "location": "local"
                },
                "indoor": {
                    "parameters": ["light", "noise", "air_quality"],
                    "frequency": "minute",
                    "location": "room"
                },
                "activity": {
                    "parameters": ["movement", "location", "context"],
                    "frequency": "continuous",
                    "location": "personal"
                }
            },
            "processing_stages": {
                "collection": ["sampling", "validation", "aggregation"],
                "analysis": ["trending", "correlation", "anomaly_detection"],
                "contextualization": ["location_binding", "time_alignment", "event_correlation"]
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="environmental_flow",
            inputs={
                "raw_data": self.env_gen.generate_data(24),  # 24 hours
                "scenarios": env_scenarios,
                "config": {
                    "sampling_interval": 60,  # seconds
                    "aggregation_window": 3600,  # seconds
                    "correlation_threshold": 0.7
                }
            },
            expected_outputs={
                "data_completeness": {
                    "temporal": 0.98,
                    "spatial": 0.95,
                    "contextual": 0.9
                },
                "analysis_metrics": {
                    "trend_accuracy": 0.9,
                    "correlation_strength": 0.8,
                    "anomaly_detection": 0.85
                },
                "context_quality": {
                    "location_accuracy": 0.95,
                    "temporal_alignment": 0.98,
                    "event_correlation": 0.9
                }
            },
            tolerance={
                "completeness": 0.05,  # ±5% tolerance
                "accuracy": 0.1,       # ±10% tolerance
                "correlation": 0.1     # ±10% tolerance
            },
            metadata={
                "description": "Validate environmental data flow",
                "data_type": "mixed",
                "scope": "comprehensive"
            }
        )
        
        # Create validation function
        def validate_flow(
            raw_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Process environmental data
            processed_data = self.processor.process_environmental(
                data=raw_data,
                stages=scenarios["processing_stages"],
                config=config
            )
            
            # Evaluate completeness
            completeness = self.pipeline.evaluate_completeness(
                data=processed_data,
                sources=scenarios["data_sources"]
            )
            
            # Analyze environmental patterns
            analysis = self.processor.analyze_patterns(
                data=processed_data,
                config=config
            )
            
            # Assess contextualization
            context = self.integrator.assess_context(
                data=processed_data,
                sources=scenarios["data_sources"]
            )
            
            return {
                "data_completeness": completeness,
                "analysis_metrics": analysis,
                "context_quality": context
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("environmental_flow", validate_flow)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_patient_feedback_flow(self):
        """Test flow of patient feedback data through the system."""
        # Generate test scenarios
        feedback_scenarios = {
            "data_types": {
                "symptoms": {
                    "categories": ["pain", "aura", "nausea"],
                    "scale": "0-10",
                    "frequency": "event_based"
                },
                "triggers": {
                    "categories": ["stress", "sleep", "diet"],
                    "confidence": "1-5",
                    "frequency": "daily"
                },
                "interventions": {
                    "categories": ["medication", "lifestyle", "preventive"],
                    "effectiveness": "1-5",
                    "frequency": "as_needed"
                }
            },
            "analysis_stages": {
                "validation": ["completeness", "consistency", "credibility"],
                "integration": ["temporal_alignment", "cross_validation", "enrichment"],
                "learning": ["pattern_recognition", "effectiveness_analysis", "adaptation"]
            }
        }
        
        # Create test case
        test_case = IntegrationTestCase(
            name="feedback_flow",
            inputs={
                "feedback_data": self.feedback_gen.generate_feedback(30),  # 30 days
                "scenarios": feedback_scenarios,
                "config": {
                    "validation_rules": "strict",
                    "learning_rate": 0.1,
                    "adaptation_threshold": 0.8
                }
            },
            expected_outputs={
                "feedback_quality": {
                    "completeness": 0.9,
                    "consistency": 0.85,
                    "credibility": 0.95
                },
                "integration_metrics": {
                    "temporal_accuracy": 0.95,
                    "cross_validation": 0.9,
                    "enrichment_quality": 0.85
                },
                "learning_metrics": {
                    "pattern_recognition": 0.8,
                    "effectiveness_score": 0.85,
                    "adaptation_rate": 0.75
                }
            },
            tolerance={
                "quality": 0.1,      # ±10% tolerance
                "integration": 0.05,  # ±5% tolerance
                "learning": 0.1      # ±10% tolerance
            },
            metadata={
                "description": "Validate patient feedback flow",
                "feedback_mode": "mixed",
                "analysis_depth": "detailed"
            }
        )
        
        # Create validation function
        def validate_flow(
            feedback_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Process feedback data
            processed_feedback = self.analyzer.process_feedback(
                data=feedback_data,
                types=scenarios["data_types"],
                config=config
            )
            
            # Evaluate feedback quality
            quality = self.analyzer.evaluate_quality(
                feedback=processed_feedback,
                rules=config["validation_rules"]
            )
            
            # Assess integration
            integration = self.integrator.assess_feedback_integration(
                feedback=processed_feedback,
                stages=scenarios["analysis_stages"]
            )
            
            # Measure learning effectiveness
            learning = self.analyzer.measure_learning(
                feedback=processed_feedback,
                config=config
            )
            
            return {
                "feedback_quality": quality,
                "integration_metrics": integration,
                "learning_metrics": learning
            }
        
        # Create harness and validate
        harness = IntegrationTestHarness("feedback_flow", validate_flow)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 
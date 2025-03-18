"""
Performance Benchmarks for System Adaptability.

This module provides benchmarks for measuring system adaptability,
including pattern drift handling, load variation response, and recovery time.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor

from tests.performance.benchmark_harness import BenchmarkCase, BenchmarkHarness
from tests.performance.data_generators import (
    DriftGenerator,
    LoadGenerator,
    RecoveryScenarioGenerator
)
from core.adaptation import (
    DriftDetector,
    LoadBalancer,
    RecoveryManager
)
from core.monitoring import (
    AdaptationMonitor,
    SystemMetrics,
    PerformanceTracker
)

class TestSystemAdaptability:
    """Benchmark cases for system adaptability."""
    
    def setup_method(self):
        """Set up benchmark fixtures."""
        self.detector = DriftDetector()
        self.balancer = LoadBalancer()
        self.recovery = RecoveryManager()
        self.monitor = AdaptationMonitor()
        self.metrics = SystemMetrics()
        self.tracker = PerformanceTracker()
        
        # Data generators
        self.drift_gen = DriftGenerator()
        self.load_gen = LoadGenerator()
        self.scenario_gen = RecoveryScenarioGenerator()
    
    def test_pattern_drift_handling(self):
        """Benchmark pattern drift detection and adaptation."""
        # Define benchmark case
        benchmark_case = BenchmarkCase(
            name="pattern_drift",
            inputs={
                "drift_scenarios": [
                    {
                        "type": "gradual",
                        "duration": "24h",
                        "magnitude": 0.3
                    },
                    {
                        "type": "sudden",
                        "duration": "1h",
                        "magnitude": 0.8
                    },
                    {
                        "type": "seasonal",
                        "duration": "7d",
                        "magnitude": 0.5
                    }
                ],
                "data_streams": self.drift_gen.generate_data_streams(),
                "config": {
                    "detection_window": "6h",
                    "confidence_threshold": 0.95,
                    "adaptation_rate": 0.1
                }
            },
            expected_performance={
                "detection_metrics": {
                    "accuracy": 0.95,
                    "latency": 3600,  # seconds
                    "false_positive_rate": 0.05
                },
                "adaptation_metrics": {
                    "response_time": 300,  # seconds
                    "convergence_rate": 0.9,
                    "stability": 0.95
                },
                "impact_metrics": {
                    "performance_impact": 0.1,
                    "resource_overhead": 0.15,
                    "recovery_time": 1800  # seconds
                }
            },
            tolerance={
                "detection": {
                    "accuracy": 0.05,
                    "latency": 600,  # seconds
                    "false_positives": 0.02
                },
                "adaptation": 0.1,
                "impact": 0.05
            },
            metadata={
                "description": "Evaluate pattern drift handling capabilities",
                "scenario_count": 3,
                "total_duration": "7d"
            }
        )
        
        # Create benchmark function
        def run_drift_benchmarks(
            drift_scenarios: List[Dict[str, Any]],
            data_streams: Dict[str, Any],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            results = {}
            
            # Test each drift scenario
            for scenario in drift_scenarios:
                # Configure detector
                self.detector.configure(
                    window=config["detection_window"],
                    threshold=config["confidence_threshold"],
                    adaptation_rate=config["adaptation_rate"]
                )
                
                # Start monitoring
                self.monitor.start_monitoring()
                
                # Process data stream
                stream = data_streams[scenario["type"]]
                detection_results = self.detector.process_stream(
                    stream=stream,
                    duration=scenario["duration"]
                )
                
                # Collect metrics
                metrics = self.monitor.stop_monitoring()
                
                # Calculate performance metrics
                scenario_results = {
                    "detection": self.metrics.calculate_detection_metrics(
                        results=detection_results,
                        ground_truth=stream["drift_points"]
                    ),
                    "adaptation": self.metrics.calculate_adaptation_metrics(
                        results=detection_results,
                        metrics=metrics
                    ),
                    "impact": self.metrics.calculate_impact_metrics(
                        before=metrics["pre_drift"],
                        after=metrics["post_drift"]
                    )
                }
                
                results[scenario["type"]] = scenario_results
            
            return results
        
        # Create harness and run benchmarks
        harness = BenchmarkHarness("pattern_drift", run_drift_benchmarks)
        harness.add_benchmark_case(benchmark_case)
        results = harness.run_all()
        
        return results
    
    def test_load_variation_response(self):
        """Benchmark system response to load variations."""
        # Define benchmark case
        benchmark_case = BenchmarkCase(
            name="load_variation",
            inputs={
                "load_patterns": [
                    {
                        "type": "spike",
                        "base_load": 100,
                        "peak_load": 1000,
                        "duration": "30m"
                    },
                    {
                        "type": "gradual_increase",
                        "start_load": 100,
                        "end_load": 500,
                        "duration": "2h"
                    },
                    {
                        "type": "oscillating",
                        "min_load": 100,
                        "max_load": 300,
                        "period": "15m",
                        "duration": "1h"
                    }
                ],
                "system_config": {
                    "min_instances": 1,
                    "max_instances": 10,
                    "scaling_threshold": 0.7,
                    "cooldown_period": 300  # seconds
                }
            },
            expected_performance={
                "scaling_metrics": {
                    "response_time": 120,  # seconds
                    "accuracy": 0.9,
                    "stability": 0.95
                },
                "performance_metrics": {
                    "latency_increase": 0.2,
                    "throughput_maintenance": 0.95,
                    "resource_efficiency": 0.85
                },
                "cost_metrics": {
                    "resource_utilization": 0.8,
                    "scaling_efficiency": 0.9,
                    "cost_effectiveness": 0.85
                }
            },
            tolerance={
                "scaling": {
                    "time": 30,  # seconds
                    "accuracy": 0.1
                },
                "performance": 0.1,
                "cost": 0.1
            },
            metadata={
                "description": "Evaluate load variation handling",
                "pattern_count": 3,
                "total_duration": "4h"
            }
        )
        
        # Create benchmark function
        def run_load_benchmarks(
            load_patterns: List[Dict[str, Any]],
            system_config: Dict[str, Any]
        ) -> Dict[str, Any]:
            results = {}
            
            # Configure load balancer
            self.balancer.configure(**system_config)
            
            # Test each load pattern
            for pattern in load_patterns:
                # Generate load
                load_data = self.load_gen.generate_load(
                    pattern_type=pattern["type"],
                    **{k: v for k, v in pattern.items() if k != "type"}
                )
                
                # Start monitoring
                self.monitor.start_monitoring()
                
                # Process load
                scaling_results = self.balancer.handle_load(
                    load_data=load_data,
                    duration=pattern["duration"]
                )
                
                # Collect metrics
                metrics = self.monitor.stop_monitoring()
                
                # Calculate performance metrics
                pattern_results = {
                    "scaling": self.metrics.calculate_scaling_metrics(
                        results=scaling_results,
                        config=system_config
                    ),
                    "performance": self.metrics.calculate_performance_metrics(
                        before=metrics["pre_scaling"],
                        after=metrics["post_scaling"]
                    ),
                    "cost": self.metrics.calculate_cost_metrics(
                        scaling_results=scaling_results,
                        performance_metrics=metrics
                    )
                }
                
                results[pattern["type"]] = pattern_results
            
            return results
        
        # Create harness and run benchmarks
        harness = BenchmarkHarness("load_variation", run_load_benchmarks)
        harness.add_benchmark_case(benchmark_case)
        results = harness.run_all()
        
        return results
    
    def test_recovery_time_measurement(self):
        """Benchmark system recovery time from various scenarios."""
        # Define benchmark case
        benchmark_case = BenchmarkCase(
            name="recovery_time",
            inputs={
                "failure_scenarios": [
                    {
                        "type": "component_failure",
                        "component": "data_processor",
                        "duration": "15m"
                    },
                    {
                        "type": "resource_exhaustion",
                        "resource": "memory",
                        "threshold": 0.9,
                        "duration": "10m"
                    },
                    {
                        "type": "cascade_failure",
                        "components": ["cache", "database", "api"],
                        "duration": "30m"
                    }
                ],
                "recovery_config": {
                    "max_attempts": 3,
                    "timeout": 600,  # seconds
                    "backoff_factor": 2
                }
            },
            expected_performance={
                "recovery_metrics": {
                    "success_rate": 0.95,
                    "time_to_recover": 300,  # seconds
                    "stability_post_recovery": 0.9
                },
                "service_metrics": {
                    "availability": 0.99,
                    "degradation_level": 0.2,
                    "data_consistency": 0.99
                },
                "resource_metrics": {
                    "utilization_during_recovery": 0.7,
                    "overhead": 0.15,
                    "efficiency": 0.85
                }
            },
            tolerance={
                "recovery": {
                    "success": 0.05,
                    "time": 60  # seconds
                },
                "service": 0.01,
                "resource": 0.1
            },
            metadata={
                "description": "Evaluate system recovery capabilities",
                "scenario_count": 3,
                "total_duration": "1h"
            }
        )
        
        # Create benchmark function
        def run_recovery_benchmarks(
            failure_scenarios: List[Dict[str, Any]],
            recovery_config: Dict[str, Any]
        ) -> Dict[str, Any]:
            results = {}
            
            # Configure recovery manager
            self.recovery.configure(**recovery_config)
            
            # Test each failure scenario
            for scenario in failure_scenarios:
                # Generate scenario
                failure_data = self.scenario_gen.generate_scenario(
                    scenario_type=scenario["type"],
                    **{k: v for k, v in scenario.items() if k != "type"}
                )
                
                # Start monitoring
                self.monitor.start_monitoring()
                
                # Execute recovery
                recovery_results = self.recovery.handle_failure(
                    scenario=failure_data,
                    duration=scenario["duration"]
                )
                
                # Collect metrics
                metrics = self.monitor.stop_monitoring()
                
                # Calculate performance metrics
                scenario_results = {
                    "recovery": self.metrics.calculate_recovery_metrics(
                        results=recovery_results,
                        config=recovery_config
                    ),
                    "service": self.metrics.calculate_service_metrics(
                        before=metrics["pre_failure"],
                        during=metrics["during_failure"],
                        after=metrics["post_recovery"]
                    ),
                    "resource": self.metrics.calculate_resource_metrics(
                        recovery_phase=metrics["recovery_phase"]
                    )
                }
                
                results[scenario["type"]] = scenario_results
            
            return results
        
        # Create harness and run benchmarks
        harness = BenchmarkHarness("recovery_time", run_recovery_benchmarks)
        harness.add_benchmark_case(benchmark_case)
        results = harness.run_all()
        
        return results
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get system hardware information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None
        } 
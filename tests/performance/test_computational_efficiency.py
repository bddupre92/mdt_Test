"""
Performance Benchmarks for Computational Efficiency.

This module provides benchmarks for measuring computational efficiency,
including processing time analysis, memory profiling, and scalability testing.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import psutil
import os
import cProfile
import memory_profiler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from tests.performance.benchmark_harness import BenchmarkCase, BenchmarkHarness
from tests.performance.data_generators import (
    LoadGenerator,
    ScalabilityDataGenerator
)
from core.processing import (
    DataProcessor,
    ParallelProcessor,
    CacheManager
)
from core.optimization import (
    OptimizationEngine,
    ResourceManager
)

class TestComputationalEfficiency:
    """Benchmark cases for computational efficiency and resource utilization."""
    
    def setup_method(self):
        """Set up benchmark fixtures."""
        self.processor = DataProcessor()
        self.parallel = ParallelProcessor()
        self.cache = CacheManager()
        self.optimizer = OptimizationEngine()
        self.resources = ResourceManager()
        
        # Load generators
        self.load_gen = LoadGenerator()
        self.scale_gen = ScalabilityDataGenerator()
    
    def test_processing_time(self):
        """Benchmark processing time for various operations."""
        # Define processing scenarios
        processing_scenarios = {
            "data_volumes": [
                {"size": "small", "records": 1000},
                {"size": "medium", "records": 10000},
                {"size": "large", "records": 100000}
            ],
            "operation_types": [
                "data_transformation",
                "feature_extraction",
                "model_inference",
                "optimization"
            ],
            "complexity_levels": [
                {"level": "simple", "operations": 10},
                {"level": "moderate", "operations": 100},
                {"level": "complex", "operations": 1000}
            ]
        }
        
        # Create benchmark case
        benchmark_case = BenchmarkCase(
            name="processing_time",
            inputs={
                "test_data": self.load_gen.generate_processing_data(processing_scenarios),
                "scenarios": processing_scenarios,
                "config": {
                    "profiling": True,
                    "timing_resolution": "microsecond",
                    "repeat_count": 10
                }
            },
            expected_performance={
                "execution_time": {
                    "small": {
                        "simple": 0.1,     # seconds
                        "moderate": 1.0,
                        "complex": 10.0
                    },
                    "medium": {
                        "simple": 1.0,
                        "moderate": 10.0,
                        "complex": 100.0
                    },
                    "large": {
                        "simple": 10.0,
                        "moderate": 100.0,
                        "complex": 1000.0
                    }
                },
                "operation_metrics": {
                    "throughput": 1000,     # operations/second
                    "latency": 0.001,       # seconds
                    "utilization": 0.8
                },
                "bottleneck_analysis": {
                    "cpu_bound": 0.7,
                    "io_bound": 0.2,
                    "memory_bound": 0.1
                }
            },
            tolerance={
                "timing": 0.1,      # ±10% tolerance
                "throughput": 0.1,  # ±10% tolerance
                "latency": 0.001    # ±1ms tolerance
            },
            metadata={
                "description": "Measure processing time efficiency",
                "hardware_specs": self._get_hardware_info(),
                "profiling_method": "detailed"
            }
        )
        
        # Create benchmark function
        def run_processing_benchmarks(
            test_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            results = {}
            profiler = cProfile.Profile()
            
            # Test each data volume
            for volume in scenarios["data_volumes"]:
                volume_results = {}
                
                # Test each operation type
                for op_type in scenarios["operation_types"]:
                    op_results = {}
                    
                    # Test each complexity level
                    for complexity in scenarios["complexity_levels"]:
                        # Profile execution
                        profiler.enable()
                        start_time = time.time()
                        
                        # Run operations
                        operation_results = self.processor.process_data(
                            data=test_data[volume["size"]][op_type],
                            operations=complexity["operations"],
                            config=config
                        )
                        
                        runtime = time.time() - start_time
                        profiler.disable()
                        
                        # Analyze performance
                        metrics = self.resources.analyze_performance(
                            profiler.getstats(),
                            runtime=runtime,
                            operation_count=complexity["operations"]
                        )
                        
                        # Identify bottlenecks
                        bottlenecks = self.resources.identify_bottlenecks(
                            metrics=metrics,
                            threshold=0.8
                        )
                        
                        op_results[complexity["level"]] = {
                            "execution_time": runtime,
                            "operation_metrics": metrics,
                            "bottlenecks": bottlenecks,
                            "profile": profiler.getstats()
                        }
                    
                    volume_results[op_type] = op_results
                
                results[volume["size"]] = volume_results
            
            return results
        
        # Create harness and run benchmarks
        harness = BenchmarkHarness("processing_time", run_processing_benchmarks)
        harness.add_benchmark_case(benchmark_case)
        results = harness.run_all()
        
        return results
    
    def test_memory_usage(self):
        """Benchmark memory usage and profiling."""
        # Define memory scenarios
        memory_scenarios = {
            "data_sizes": [
                {"size": "small", "mb": 100},
                {"size": "medium", "mb": 1000},
                {"size": "large", "mb": 5000}
            ],
            "operation_patterns": [
                "sequential_processing",
                "parallel_processing",
                "cached_operations"
            ],
            "memory_pressure": [
                {"level": "low", "utilization": 0.3},
                {"level": "medium", "utilization": 0.6},
                {"level": "high", "utilization": 0.9}
            ]
        }
        
        # Create benchmark case
        benchmark_case = BenchmarkCase(
            name="memory_usage",
            inputs={
                "test_data": self.load_gen.generate_memory_data(memory_scenarios),
                "scenarios": memory_scenarios,
                "config": {
                    "profiling": True,
                    "sampling_rate": 0.1,  # seconds
                    "gc_control": True
                }
            },
            expected_performance={
                "memory_footprint": {
                    "small": {
                        "baseline": 100,    # MB
                        "peak": 200,
                        "average": 150
                    },
                    "medium": {
                        "baseline": 500,
                        "peak": 1000,
                        "average": 750
                    },
                    "large": {
                        "baseline": 2000,
                        "peak": 4000,
                        "average": 3000
                    }
                },
                "allocation_patterns": {
                    "heap_growth": 0.5,     # Growth rate
                    "fragmentation": 0.2,
                    "collection_efficiency": 0.8
                },
                "cache_effectiveness": {
                    "hit_rate": 0.8,
                    "eviction_rate": 0.2,
                    "memory_savings": 0.4
                }
            },
            tolerance={
                "memory": 0.1,     # ±10% tolerance
                "timing": 0.1,     # ±10% tolerance
                "efficiency": 0.05  # ±5% tolerance
            },
            metadata={
                "description": "Profile memory usage patterns",
                "memory_total": psutil.virtual_memory().total,
                "profiling_method": "sampling"
            }
        )
        
        # Create benchmark function
        def run_memory_benchmarks(
            test_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            results = {}
            
            # Test each data size
            for size in scenarios["data_sizes"]:
                size_results = {}
                
                # Test each operation pattern
                for pattern in scenarios["operation_patterns"]:
                    pattern_results = {}
                    
                    # Test each memory pressure level
                    for pressure in scenarios["memory_pressure"]:
                        # Profile memory usage
                        @memory_profiler.profile
                        def run_memory_test():
                            # Configure memory pressure
                            self.resources.set_memory_pressure(
                                level=pressure["utilization"]
                            )
                            
                            # Run operations
                            if pattern == "parallel_processing":
                                return self._run_parallel_test(
                                    data=test_data[size["size"]],
                                    config=config
                                )
                            elif pattern == "cached_operations":
                                return self._run_cached_test(
                                    data=test_data[size["size"]],
                                    config=config
                                )
                            else:
                                return self._run_sequential_test(
                                    data=test_data[size["size"]],
                                    config=config
                                )
                        
                        # Run test and collect metrics
                        memory_profile = run_memory_test()
                        
                        # Analyze memory patterns
                        memory_metrics = self.resources.analyze_memory(
                            profile=memory_profile,
                            data_size=size["mb"]
                        )
                        
                        # Evaluate cache effectiveness
                        cache_metrics = self.cache.evaluate_effectiveness(
                            pattern=pattern,
                            pressure=pressure["level"]
                        )
                        
                        pattern_results[pressure["level"]] = {
                            "memory_metrics": memory_metrics,
                            "cache_metrics": cache_metrics,
                            "profile": memory_profile
                        }
                    
                    size_results[pattern] = pattern_results
                
                results[size["size"]] = size_results
            
            return results
        
        # Create harness and run benchmarks
        harness = BenchmarkHarness("memory_usage", run_memory_benchmarks)
        harness.add_benchmark_case(benchmark_case)
        results = harness.run_all()
        
        return results
    
    def test_scalability(self):
        """Benchmark system scalability under various loads."""
        # Define scalability scenarios
        scalability_scenarios = {
            "load_patterns": [
                {"type": "linear", "step": 1000},
                {"type": "exponential", "base": 2},
                {"type": "random", "range": [100, 10000]}
            ],
            "concurrency_levels": [1, 2, 4, 8, 16, 32],
            "resource_configs": [
                {"cpu_limit": 0.5, "memory_limit": 0.5},
                {"cpu_limit": 0.75, "memory_limit": 0.75},
                {"cpu_limit": 1.0, "memory_limit": 1.0}
            ]
        }
        
        # Create benchmark case
        benchmark_case = BenchmarkCase(
            name="scalability",
            inputs={
                "test_data": self.scale_gen.generate_scalability_data(scalability_scenarios),
                "scenarios": scalability_scenarios,
                "config": {
                    "duration": 3600,  # seconds
                    "sampling_rate": 1.0,
                    "monitoring": True
                }
            },
            expected_performance={
                "throughput_scaling": {
                    "linear": {
                        "efficiency": 0.9,
                        "speedup": 0.8,
                        "saturation_point": 16
                    },
                    "exponential": {
                        "efficiency": 0.8,
                        "speedup": 0.7,
                        "saturation_point": 8
                    },
                    "random": {
                        "efficiency": 0.85,
                        "speedup": 0.75,
                        "saturation_point": 12
                    }
                },
                "resource_utilization": {
                    "cpu_efficiency": 0.8,
                    "memory_efficiency": 0.7,
                    "io_efficiency": 0.9
                },
                "system_stability": {
                    "error_rate": 0.01,
                    "response_time": 0.1,  # seconds
                    "recovery_time": 1.0   # seconds
                }
            },
            tolerance={
                "scaling": 0.1,     # ±10% tolerance
                "efficiency": 0.1,  # ±10% tolerance
                "stability": 0.05   # ±5% tolerance
            },
            metadata={
                "description": "Evaluate system scalability",
                "test_duration": "1h",
                "monitoring_interval": "1s"
            }
        )
        
        # Create benchmark function
        def run_scalability_benchmarks(
            test_data: Dict[str, Any],
            scenarios: Dict[str, Dict[str, Any]],
            config: Dict[str, Any]
        ) -> Dict[str, Any]:
            results = {}
            
            # Test each load pattern
            for pattern in scenarios["load_patterns"]:
                pattern_results = {}
                
                # Test each concurrency level
                for concurrency in scenarios["concurrency_levels"]:
                    concurrency_results = {}
                    
                    # Test each resource configuration
                    for resource_config in scenarios["resource_configs"]:
                        # Configure resources
                        self.resources.configure(
                            cpu_limit=resource_config["cpu_limit"],
                            memory_limit=resource_config["memory_limit"]
                        )
                        
                        # Run parallel processing
                        with ProcessPoolExecutor(max_workers=concurrency) as executor:
                            start_time = time.time()
                            futures = []
                            
                            # Submit tasks
                            for batch in test_data[pattern["type"]]:
                                futures.append(
                                    executor.submit(
                                        self.parallel.process_batch,
                                        batch=batch,
                                        config=config
                                    )
                                )
                            
                            # Collect results
                            results = [f.result() for f in futures]
                            runtime = time.time() - start_time
                        
                        # Measure scaling metrics
                        scaling_metrics = self.resources.analyze_scaling(
                            results=results,
                            concurrency=concurrency,
                            runtime=runtime
                        )
                        
                        # Monitor resource utilization
                        utilization = self.resources.monitor_utilization(
                            duration=config["duration"],
                            interval=config["sampling_rate"]
                        )
                        
                        # Evaluate system stability
                        stability = self.resources.evaluate_stability(
                            metrics=scaling_metrics,
                            utilization=utilization
                        )
                        
                        concurrency_results[f"resources_{resource_config['cpu_limit']}"] = {
                            "scaling_metrics": scaling_metrics,
                            "utilization": utilization,
                            "stability": stability
                        }
                    
                    pattern_results[f"concurrency_{concurrency}"] = concurrency_results
                
                results[pattern["type"]] = pattern_results
            
            return results
        
        # Create harness and run benchmarks
        harness = BenchmarkHarness("scalability", run_scalability_benchmarks)
        harness.add_benchmark_case(benchmark_case)
        results = harness.run_all()
        
        return results
    
    def _run_parallel_test(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run parallel processing test."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                self.processor.process_chunk,
                data["chunks"],
                [config] * len(data["chunks"])
            ))
        return {"results": results}
    
    def _run_cached_test(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run cached operations test."""
        results = []
        for chunk in data["chunks"]:
            cache_key = self.cache.compute_key(chunk)
            if self.cache.has(cache_key):
                results.append(self.cache.get(cache_key))
            else:
                result = self.processor.process_chunk(chunk, config)
                self.cache.set(cache_key, result)
                results.append(result)
        return {"results": results}
    
    def _run_sequential_test(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run sequential processing test."""
        results = []
        for chunk in data["chunks"]:
            results.append(self.processor.process_chunk(chunk, config))
        return {"results": results}
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get system hardware information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None
        } 
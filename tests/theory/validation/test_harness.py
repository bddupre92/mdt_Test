"""
Test harness for theoretical component validation.

This module provides the foundational testing framework for validating
theoretical components of the migraine prediction system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
import json
import numpy as np
import datetime

@dataclass
class TestCase:
    """Base test case definition for validation tests."""
    
    name: str
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    tolerance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary for serialization."""
        return {
            "name": self.name,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "tolerance": self.tolerance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create test case from dictionary."""
        return cls(
            name=data["name"],
            inputs=data["inputs"],
            expected_outputs=data["expected_outputs"],
            tolerance=data.get("tolerance", {}),
            metadata=data.get("metadata", {})
        )

@dataclass
class BenchmarkCase:
    """Benchmark case definition for performance testing."""
    
    name: str
    inputs: Dict[str, Any]
    expected_performance: Dict[str, Any]
    tolerance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark case to dictionary for serialization."""
        return {
            "name": self.name,
            "inputs": self.inputs,
            "expected_performance": self.expected_performance,
            "tolerance": self.tolerance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkCase':
        """Create benchmark case from dictionary."""
        return cls(
            name=data["name"],
            inputs=data["inputs"],
            expected_performance=data["expected_performance"],
            tolerance=data.get("tolerance", {}),
            metadata=data.get("metadata", {})
        )

@dataclass
class IntegrationTestCase:
    """Integration test case definition for cross-component testing."""
    
    name: str
    components: List[str]
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    expected_interactions: Dict[str, Any] = field(default_factory=dict)
    tolerance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert integration test case to dictionary for serialization."""
        return {
            "name": self.name,
            "components": self.components,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "expected_interactions": self.expected_interactions,
            "tolerance": self.tolerance,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegrationTestCase':
        """Create integration test case from dictionary."""
        return cls(
            name=data["name"],
            components=data["components"],
            inputs=data["inputs"],
            expected_outputs=data["expected_outputs"],
            expected_interactions=data.get("expected_interactions", {}),
            tolerance=data.get("tolerance", {}),
            metadata=data.get("metadata", {})
        )

class ValidationMetrics:
    """Utility for validation metrics calculation."""
    
    @staticmethod
    def calculate_accuracy(expected: np.ndarray, actual: np.ndarray) -> float:
        """Calculate accuracy between expected and actual values."""
        if len(expected) != len(actual):
            raise ValueError("Expected and actual arrays must have same length")
        
        return np.mean(expected == actual)
    
    @staticmethod
    def calculate_mae(expected: np.ndarray, actual: np.ndarray) -> float:
        """Calculate mean absolute error."""
        if len(expected) != len(actual):
            raise ValueError("Expected and actual arrays must have same length")
        
        return np.mean(np.abs(expected - actual))
    
    @staticmethod
    def calculate_rmse(expected: np.ndarray, actual: np.ndarray) -> float:
        """Calculate root mean squared error."""
        if len(expected) != len(actual):
            raise ValueError("Expected and actual arrays must have same length")
        
        return np.sqrt(np.mean((expected - actual) ** 2))
    
    @staticmethod
    def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Calculate correlation coefficient."""
        if len(x) != len(y):
            raise ValueError("Input arrays must have same length")
        
        return np.corrcoef(x, y)[0, 1]

class TestHarness:
    """Base test harness functionality."""
    
    def __init__(self, name: str = "Test Harness"):
        """Initialize test harness."""
        self.name = name
        self.test_cases = []
        self.results = {}
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the harness."""
        self.test_cases.append(test_case)
    
    def run_test(self, test_case: TestCase, test_func: Callable) -> Dict[str, Any]:
        """Run a test function on a test case."""
        try:
            result = test_func(test_case)
            self.results[test_case.name] = {
                "status": "pass" if result.get("success", False) else "fail",
                "metrics": result.get("metrics", {}),
                "details": result.get("details", {})
            }
            return self.results[test_case.name]
        except Exception as e:
            self.results[test_case.name] = {
                "status": "error",
                "error": str(e),
                "details": {"exception_type": type(e).__name__}
            }
            return self.results[test_case.name]
    
    def run_all_tests(self, test_func: Callable) -> Dict[str, Dict[str, Any]]:
        """Run all test cases."""
        for test_case in self.test_cases:
            self.run_test(test_case, test_func)
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "pass")
        failed = sum(1 for r in self.results.values() if r["status"] == "fail")
        errors = sum(1 for r in self.results.values() if r["status"] == "error")
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0
        }
    
    def save_results(self, file_path: str) -> None:
        """Save test results to a file."""
        results_with_summary = {
            "summary": self.get_summary(),
            "results": self.results
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_with_summary, f, indent=2)

class BenchmarkHarness(TestHarness):
    """Harness for performance benchmarking."""
    
    def __init__(self, name: str = "Benchmark Harness"):
        """Initialize benchmark harness."""
        super().__init__(name)
        self.benchmark_cases = []
    
    def add_benchmark_case(self, benchmark_case: BenchmarkCase) -> None:
        """Add a benchmark case to the harness."""
        self.benchmark_cases.append(benchmark_case)
    
    def run_benchmark(self, benchmark_case: BenchmarkCase, benchmark_func: Callable) -> Dict[str, Any]:
        """Run a benchmark function on a benchmark case."""
        try:
            start_time = datetime.datetime.now()
            result = benchmark_func(benchmark_case)
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.results[benchmark_case.name] = {
                "status": "pass" if result.get("success", False) else "fail",
                "metrics": result.get("metrics", {}),
                "performance": {
                    **result.get("performance", {}),
                    "execution_time": execution_time
                },
                "details": result.get("details", {})
            }
            return self.results[benchmark_case.name]
        except Exception as e:
            self.results[benchmark_case.name] = {
                "status": "error",
                "error": str(e),
                "details": {"exception_type": type(e).__name__}
            }
            return self.results[benchmark_case.name]
    
    def run_all_benchmarks(self, benchmark_func: Callable) -> Dict[str, Dict[str, Any]]:
        """Run all benchmark cases."""
        for benchmark_case in self.benchmark_cases:
            self.run_benchmark(benchmark_case, benchmark_func)
        return self.results

class IntegrationTestHarness(TestHarness):
    """Harness for integration testing."""
    
    def __init__(self, name: str = "Integration Test Harness"):
        """Initialize integration test harness."""
        super().__init__(name)
        self.integration_test_cases = []
    
    def add_integration_test_case(self, integration_test_case: IntegrationTestCase) -> None:
        """Add an integration test case to the harness."""
        self.integration_test_cases.append(integration_test_case)
    
    def run_integration_test(self, integration_test_case: IntegrationTestCase, test_func: Callable) -> Dict[str, Any]:
        """Run an integration test function on an integration test case."""
        try:
            result = test_func(integration_test_case)
            self.results[integration_test_case.name] = {
                "status": "pass" if result.get("success", False) else "fail",
                "metrics": result.get("metrics", {}),
                "interactions": result.get("interactions", {}),
                "details": result.get("details", {})
            }
            return self.results[integration_test_case.name]
        except Exception as e:
            self.results[integration_test_case.name] = {
                "status": "error",
                "error": str(e),
                "details": {"exception_type": type(e).__name__}
            }
            return self.results[integration_test_case.name]
    
    def run_all_integration_tests(self, test_func: Callable) -> Dict[str, Dict[str, Any]]:
        """Run all integration test cases."""
        for integration_test_case in self.integration_test_cases:
            self.run_integration_test(integration_test_case, test_func)
        return self.results 
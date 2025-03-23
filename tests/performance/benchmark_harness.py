"""
Benchmark harness for performance testing.

This module provides tools for benchmarking performance of various components
including computational efficiency, prediction accuracy, and system adaptability.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
import time
import datetime
import numpy as np
import json

@dataclass
class BenchmarkCase:
    """Data class representing a performance benchmark case."""
    
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

class BenchmarkHarness:
    """Harness for performance benchmarking."""
    
    def __init__(self, name: str = "Benchmark Harness"):
        """Initialize benchmark harness."""
        self.name = name
        self.benchmark_cases = []
        self.results = {}
    
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
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
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
        """Save benchmark results to a file."""
        results_with_summary = {
            "summary": self.get_summary(),
            "results": self.results
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_with_summary, f, indent=2) 
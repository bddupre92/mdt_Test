"""
Benchmark module for the MoE Framework.

This package provides tools for measuring and comparing the performance of
different MoE configurations, integration strategies, and gating networks.
"""

from moe_framework.benchmark.performance_benchmarks import (
    BenchmarkRunner,
    run_standard_benchmark_suite
)

__all__ = [
    'BenchmarkRunner',
    'run_standard_benchmark_suite'
]

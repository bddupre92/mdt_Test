"""
Baseline comparison module for the Meta Optimizer framework

This module provides tools for comparing the Meta Optimizer against
baseline algorithm selection methods, such as the SATzilla-inspired selector.
"""

__version__ = "0.1.0"

from .comparison_runner import BaselineComparison
from .benchmark_utils import (
    get_benchmark_function,
    get_all_benchmark_functions,
    get_dynamic_benchmark_functions,
    BenchmarkFunction
)
from .visualization import ComparisonVisualizer

# Import training modules
from .training import train_selector, feature_analysis

# Import visualization tools if available
try:
    from .visualization import (
        plot_performance_comparison,
        plot_algorithm_selection_frequency,
        plot_convergence_curves,
        plot_radar_chart,
        plot_boxplots,
        plot_heatmap
    )
    
    # Check if ComparisonVisualizer class exists
    try:
        from .visualization import ComparisonVisualizer
        __all__ = [
            'BaselineComparison', 
            'ComparisonVisualizer',
            'get_benchmark_function',
            'get_all_benchmark_functions',
            'get_dynamic_benchmark_functions',
            'BenchmarkFunction',
            'train_selector',
            'feature_analysis'
        ]
    except ImportError:
        # ComparisonVisualizer not found
        __all__ = [
            'BaselineComparison',
            'get_benchmark_function',
            'get_all_benchmark_functions',
            'get_dynamic_benchmark_functions',
            'BenchmarkFunction',
            'train_selector',
            'feature_analysis'
        ]
except ImportError:
    # Visualization module not found
    __all__ = [
        'BaselineComparison',
        'get_benchmark_function',
        'get_all_benchmark_functions',
        'get_dynamic_benchmark_functions',
        'BenchmarkFunction',
        'train_selector',
        'feature_analysis'
    ] 
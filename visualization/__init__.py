"""
Visualization package for the meta-optimizer framework.
"""

# Import core visualization components
from .live_visualization import LiveOptimizationMonitor
from .optimizer_analysis import OptimizerAnalyzer
from .drift_analysis import DriftAnalyzer

# Import algorithm selection visualization if available
try:
    from .algorithm_selection_viz import AlgorithmSelectionVisualizer
except ImportError:
    pass

__all__ = [
    'LiveOptimizationMonitor',
    'OptimizerAnalyzer',
    'DriftAnalyzer',
    'AlgorithmSelectionVisualizer'
]

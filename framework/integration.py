"""
Framework integration module for coordinating optimization, drift detection, and evaluation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt

from drift_detection.detector import DriftDetector
from evaluation.framework_evaluator import FrameworkEvaluator
from optimization.state_tracking import OptimizerStateTracker

class FrameworkIntegrator:
    """Coordinates framework components and manages their interaction."""
    
    def __init__(self, 
                 window_size: int = 50,
                 drift_threshold: float = 1.8,
                 significance_level: float = 0.01):
        """Initialize framework components."""
        self.drift_detector = DriftDetector(
            window_size=window_size,
            drift_threshold=drift_threshold,
            significance_level=significance_level
        )
        self.evaluator = FrameworkEvaluator()
        self.state_tracker = OptimizerStateTracker(window_size=window_size)
        self.feature_names = None
        
    def integrate_optimization_step(self,
                                 optimizer_state: Dict,
                                 features: np.ndarray,
                                 predictions: np.ndarray,
                                 true_values: Optional[np.ndarray] = None) -> Dict:
        """Integrate a single optimization step with drift detection and evaluation."""
        # Update optimizer state tracking
        self.state_tracker.update_state(
            parameters=optimizer_state.get('parameters', {}),
            fitness=optimizer_state.get('fitness', float('inf')),
            generation=optimizer_state.get('generation', 0),
            gradient=optimizer_state.get('gradient', None)
        )
        
        # Perform drift detection
        drift_detected, severity, info = self.drift_detector.detect_drift(
            curr_data=features,
            ref_data=self.drift_detector.reference_window
        )
        
        # Track drift event if detected
        if drift_detected:
            drift_info = {
                'severity': severity,
                'mean_shift': info.get('mean_shift', 0),
                'p_value': info.get('p_value', 1),
                'drifted_features': self._identify_drifted_features(features)
            }
            self.evaluator.track_drift_event(drift_info)
        
        # Evaluate predictions if true values are available
        metrics = {}
        if true_values is not None:
            metrics = self.evaluator.evaluate_prediction_performance(
                true_values, predictions
            )
        
        # Track feature importance if available
        if 'feature_importance' in optimizer_state and self.feature_names:
            self.evaluator.track_feature_importance(
                self.feature_names,
                optimizer_state['feature_importance']
            )
        
        return {
            'drift_detected': drift_detected,
            'drift_severity': severity,
            'optimization_metrics': self.state_tracker.get_state_summary(),
            'prediction_metrics': metrics
        }
    
    def _identify_drifted_features(self, features: np.ndarray) -> List[str]:
        """Identify which features have drifted significantly."""
        if self.feature_names is None:
            return []
            
        ref_mean = np.mean(self.drift_detector.reference_window, axis=0)
        curr_mean = np.mean(features, axis=0)
        
        # Calculate normalized differences
        diffs = np.abs(curr_mean - ref_mean) / (np.std(features, axis=0) + 1e-6)
        
        # Identify features with significant drift
        drifted_indices = np.where(diffs > self.drift_detector.drift_threshold)[0]
        return [self.feature_names[i] for i in drifted_indices]
    
    def generate_visualizations(self, output_dir: str) -> None:
        """Generate comprehensive framework visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate framework performance plot
        self.evaluator.plot_framework_performance(
            os.path.join(output_dir, 'framework_performance.png')
        )
        
        # Generate drift analysis plot
        self.evaluator.plot_drift_analysis(
            os.path.join(output_dir, 'drift_analysis.png')
        )
        
        # Generate optimization landscape plot
        self._plot_optimization_landscape(
            os.path.join(output_dir, 'optimization_landscape.png')
        )
    
    def _plot_optimization_landscape(self, save_path: str) -> None:
        """Plot optimization landscape characteristics."""
        state_summary = self.state_tracker.get_state_summary()
        if not state_summary:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Parameter Adaptation
        ax = axes[0, 0]
        params = state_summary.get('parameter_adaptation', {})
        if params:
            param_names = list(params.keys())
            current_values = [p['current'] for p in params.values()]
            ax.bar(param_names, current_values)
            ax.set_title('Current Parameter Values')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Convergence Metrics
        ax = axes[0, 1]
        conv_metrics = state_summary.get('convergence_metrics', {})
        if conv_metrics:
            metric_names = list(conv_metrics.keys())
            metric_values = list(conv_metrics.values())
            ax.bar(metric_names, metric_values)
            ax.set_title('Convergence Metrics')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Landscape Metrics
        ax = axes[1, 0]
        landscape = state_summary.get('landscape_metrics', {})
        if landscape:
            metric_names = list(landscape.keys())
            metric_values = list(landscape.values())
            ax.bar(metric_names, metric_values)
            ax.set_title('Landscape Metrics')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Fitness History
        ax = axes[1, 1]
        if hasattr(self.state_tracker, 'fitness_history'):
            ax.plot(self.state_tracker.fitness_history)
            ax.set_title('Fitness History')
            ax.set_xlabel('Step')
            ax.set_ylabel('Fitness')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def save_state(self, save_path: str) -> None:
        """Save framework state to file."""
        state = {
            'optimizer_state': self.state_tracker.get_state_summary(),
            'performance_report': self.evaluator.generate_performance_report(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, load_path: str) -> None:
        """Load framework state from file."""
        with open(load_path, 'r') as f:
            state = json.load(f)
            
        # Restore relevant state components
        if 'optimizer_state' in state:
            self.state_tracker.update_state(**state['optimizer_state'])

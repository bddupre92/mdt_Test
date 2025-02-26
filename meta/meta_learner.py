"""
meta_learner.py
-------------
Meta-learner implementation that combines multiple optimization algorithms
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from models.model_factory import ModelFactory

class MetaLearner:
    def __init__(self):
        self.algorithms = []
        self.best_config = None
        self.best_score = float('-inf')
        self.feature_names = None
        
        # Enhanced parameter ranges
        self.param_ranges = {
            'n_estimators': (100, 500),  # Increased from (50, 200)
            'max_depth': (5, 25),        # Increased from (3, 15)
            'min_samples_split': (2, 30), # Increased from (2, 20)
            'min_samples_leaf': (1, 15),  # Added parameter
            'max_features': ['sqrt', 'log2', None]  # Added parameter
        }
        
        # Performance tracking
        self.eval_history = []
        self.param_importance = {}
    
    def set_algorithms(self, algorithms: List[Any]) -> None:
        """Set optimization algorithms to use"""
        self.algorithms = algorithms
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                context: Optional[Dict] = None,
                progress_callback: Optional[Callable[[int, float], None]] = None,
                max_iterations: int = 100,
                feature_names: Optional[List[str]] = None) -> None:
        """
        Run optimization process
        
        Args:
            X: Features
            y: Labels
            context: Optional context dictionary
            progress_callback: Optional callback for progress updates
            max_iterations: Maximum iterations for optimization
            feature_names: Optional list of feature names
        """
        if context is None:
            context = {}
            
        self.feature_names = feature_names
        current_iteration = 0
        
        for algorithm in self.algorithms:
            def objective_function(params):
                nonlocal current_iteration
                
                # Scale parameters to actual ranges
                scaled_params = {
                    'n_estimators': int(params[0] * (self.param_ranges['n_estimators'][1] - self.param_ranges['n_estimators'][0]) + self.param_ranges['n_estimators'][0]),
                    'max_depth': int(params[1] * (self.param_ranges['max_depth'][1] - self.param_ranges['max_depth'][0]) + self.param_ranges['max_depth'][0]),
                    'min_samples_split': int(params[2] * (self.param_ranges['min_samples_split'][1] - self.param_ranges['min_samples_split'][0]) + self.param_ranges['min_samples_split'][0]),
                    'min_samples_leaf': int(params[3] * (self.param_ranges['min_samples_leaf'][1] - self.param_ranges['min_samples_leaf'][0]) + self.param_ranges['min_samples_leaf'][0]),
                    'max_features': self.param_ranges['max_features'][int(params[4] * len(self.param_ranges['max_features']))]
                }
                
                score = self._evaluate_config(scaled_params, X, y)
                
                # Track evaluation
                self.eval_history.append({
                    'iteration': current_iteration,
                    'params': scaled_params,
                    'score': score
                })
                
                if progress_callback:
                    progress_callback(current_iteration, score)
                current_iteration += 1
                
                return -score  # Negative because optimizers minimize
            
            # Run optimization
            best_params, _ = algorithm.optimize(
                objective_function,
                max_evals=max_iterations
            )
            
            # Update best configuration if better
            performance = -objective_function(best_params)
            if performance > self.best_score:
                self.best_score = performance
                self.best_config = self._scale_parameters(best_params)
                
        # Calculate parameter importance
        self._update_param_importance()
    
    def _scale_parameters(self, params):
        """Scale parameters to actual ranges"""
        return {
            'n_estimators': int(params[0] * (self.param_ranges['n_estimators'][1] - self.param_ranges['n_estimators'][0]) + self.param_ranges['n_estimators'][0]),
            'max_depth': int(params[1] * (self.param_ranges['max_depth'][1] - self.param_ranges['max_depth'][0]) + self.param_ranges['max_depth'][0]),
            'min_samples_split': int(params[2] * (self.param_ranges['min_samples_split'][1] - self.param_ranges['min_samples_split'][0]) + self.param_ranges['min_samples_split'][0]),
            'min_samples_leaf': int(params[3] * (self.param_ranges['min_samples_leaf'][1] - self.param_ranges['min_samples_leaf'][0]) + self.param_ranges['min_samples_leaf'][0]),
            'max_features': self.param_ranges['max_features'][int(params[4] * len(self.param_ranges['max_features']))]
        }
    
    def _evaluate_config(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate a configuration using cross-validation
        
        Args:
            params: Model configuration to evaluate
            X: Training features
            y: Training labels
            
        Returns:
            Mean validation score
        """
        # Simple holdout validation for speed
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        factory = ModelFactory()
        model = factory.create_model(params)
        model.fit(X_train, y_train, feature_names=self.feature_names)
        return model.score(X_val, y_val)
    
    def _update_param_importance(self):
        """Calculate parameter importance based on evaluation history"""
        if not self.eval_history:
            return
            
        # Convert history to numpy arrays
        scores = np.array([h['score'] for h in self.eval_history])
        
        # Calculate importance for each parameter
        for param in self.param_ranges.keys():
            values = np.array([h['params'][param] for h in self.eval_history])
            correlation = np.corrcoef(values, scores)[0, 1]
            self.param_importance[param] = abs(correlation)
    
    def get_best_configuration(self) -> Dict[str, Any]:
        """Get best configuration found during optimization"""
        return self.best_config
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get detailed optimization statistics"""
        if not self.eval_history:
            return {}
            
        scores = [h['score'] for h in self.eval_history]
        iterations = [h['iteration'] for h in self.eval_history]
        
        # Calculate convergence metrics
        convergence_rate = (max(scores) - scores[0]) / len(scores)
        plateau_threshold = 0.001
        plateau_count = sum(1 for i in range(1, len(scores))
                          if abs(scores[i] - scores[i-1]) < plateau_threshold)
        
        # Calculate exploration metrics
        param_ranges = {}
        for param in self.param_ranges.keys():
            values = [h['params'][param] for h in self.eval_history]
            if isinstance(values[0], (int, float)):
                param_ranges[param] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            else:
                # For categorical parameters
                unique_values = set(values)
                param_ranges[param] = {
                    'unique_values': list(unique_values),
                    'most_common': max(set(values), key=values.count)
                }
        
        # Calculate performance improvement
        initial_window = scores[:5]
        final_window = scores[-5:]
        improvement = (np.mean(final_window) - np.mean(initial_window)) / np.mean(initial_window) * 100
        
        return {
            'best_score': self.best_score,
            'evaluations': len(self.eval_history),
            'param_importance': self.param_importance,
            'convergence': [h['score'] for h in self.eval_history],
            'convergence_rate': convergence_rate,
            'plateau_percentage': (plateau_count / len(scores)) * 100,
            'param_ranges': param_ranges,
            'performance_improvement': improvement,
            'final_performance': {
                'mean': np.mean(final_window),
                'std': np.std(final_window),
                'stability': 1 - (np.std(final_window) / np.mean(final_window))
            },
            'exploration_coverage': {
                param: (ranges['max'] - ranges['min']) / 
                      (self.param_ranges[param][1] - self.param_ranges[param][0])
                for param, ranges in param_ranges.items()
                if isinstance(ranges, dict) and 'min' in ranges
            }
        }

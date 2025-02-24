"""
Meta-optimizer that learns to select the best optimization algorithm.
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from pathlib import Path
import os

from .optimization_history import OptimizationHistory
from .problem_analysis import ProblemAnalyzer
from .selection_tracker import SelectionTracker


class MetaOptimizer:
    """Meta-optimizer that learns to select the best optimization algorithm."""
    
    def __init__(self, 
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 optimizers: Dict,
                 history_file: Optional[str] = None,
                 selection_file: Optional[str] = None):
        """
        Initialize meta-optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            optimizers: Dictionary of optimizers to choose from
            history_file: Optional path to save/load optimization history
            selection_file: Optional path to save/load selection history
        """
        self.dim = dim
        self.bounds = bounds
        self.optimizers = optimizers
        
        # Initialize components
        self.history = OptimizationHistory(history_file)
        self.analyzer = ProblemAnalyzer(bounds, dim)
        self.selection_tracker = SelectionTracker(selection_file)
        
        # Tracking variables
        self.total_evaluations = 0
        self._current_iteration = 0
        self.current_features = None
        self.current_problem_type = None
        
        # Learning parameters
        self.min_exploration_rate = 0.1
        self.exploration_decay = 0.995  # Slower decay for more stable learning
        self.confidence_threshold = 0.7
        
    def _calculate_exploration_rate(self) -> float:
        """Calculate adaptive exploration rate based on progress and performance."""
        # Base rate decays with iterations
        base_rate = max(
            self.min_exploration_rate,
            1.0 - (self._current_iteration / 1000)  # Assuming max 1000 iterations
        )
        
        if not self.current_problem_type:
            return base_rate
            
        # Get selection statistics for current problem
        stats = self.selection_tracker.get_selection_stats(self.current_problem_type)
        if stats.empty:
            return base_rate
            
        # If we have a highly successful optimizer, reduce exploration
        max_success_rate = stats['success_rate'].max()
        if max_success_rate > 0.8:
            return base_rate * 0.5
            
        # If all optimizers performing poorly, increase exploration
        if max_success_rate < 0.2:
            return min(0.8, base_rate * 2.0)
            
        return base_rate
        
    def _select_optimizer(self, context: Dict[str, Any]) -> str:
        """
        Select an optimizer based on problem features and history.
        
        Args:
            context: Problem context
            
        Returns:
            Name of selected optimizer
        """
        if self.current_features is None:
            return np.random.choice(list(self.optimizers.keys()))
            
        # Calculate exploration rate
        exploration_rate = self._calculate_exploration_rate()
            
        # Decide between exploration and exploitation
        if np.random.random() < exploration_rate:
            return np.random.choice(list(self.optimizers.keys()))
            
        # First try to use selection history if available
        if self.current_problem_type:
            correlations = self.selection_tracker.get_feature_correlations(self.current_problem_type)
            if correlations:
                # Calculate weighted scores for each optimizer
                scores = {}
                for opt, feat_corrs in correlations.items():
                    score = 0.0
                    for feat, corr in feat_corrs.items():
                        # Weight the feature by its correlation with success
                        score += self.current_features[feat] * corr
                    scores[opt] = score
                    
                if scores:
                    # Select optimizer with highest score
                    best_opt = max(scores.items(), key=lambda x: x[1])[0]
                    return best_opt
            
        # Fallback to optimization history
        similar_problems = self.history.find_similar_problems(self.current_features)
        
        if similar_problems:
            # Use historical performance to weight selection
            optimizer_scores = {}
            for sim_score, record in similar_problems:
                if record['optimizer'] not in optimizer_scores:
                    optimizer_scores[record['optimizer']] = 0.0
                # Higher score is better (we minimize objective)
                score = 1.0 / (1.0 + record['performance'])
                optimizer_scores[record['optimizer']] += sim_score * score
                
            # Select best performing optimizer
            best_optimizer = max(optimizer_scores.items(), key=lambda x: x[1])[0]
            confidence = max(optimizer_scores.values()) / sum(optimizer_scores.values())
            
            if confidence > self.confidence_threshold:
                return best_optimizer
                
        # Fallback to feature-based selection
        if self.current_features['modality'] > 1.5:
            if self.current_features['ruggedness'] > 0.5:
                return 'de'  # Good for rugged, multimodal problems
            else:
                return 'gwo'  # Good for smooth, multimodal problems
        else:
            if self.current_features['separability'] > 0.7:
                return 'surrogate'  # Good for separable problems
            else:
                return 'aco'  # Good for non-separable problems
    
    def _optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Internal optimization method"""
        if max_evals is None:
            max_evals = 1000
            
        # Analyze problem features
        self.current_features = self.analyzer.analyze_features(objective_func)
        
        # Try to determine problem type from context
        self.current_problem_type = context.get('problem_type') if context else None
        
        best_solution = None
        best_score = float('inf')
        n_evals = 0
        self.total_evaluations = 0  # Reset counter
        self.optimization_history = []  # Reset optimization history
        
        # Create a wrapped objective function to count evaluations
        def wrapped_objective(x):
            self.total_evaluations += 1
            value = objective_func(x)
            if record_history:
                self.optimization_history.append(float(value))  # Store only the score
            return value
        
        while n_evals < max_evals:
            # Select optimizer
            selected_optimizer = self._select_optimizer(context or {})
            optimizer = self.optimizers[selected_optimizer]
            
            try:
                # Reset optimizer state
                optimizer.reset()
                
                # Run optimization with remaining budget
                remaining_evals = max_evals - n_evals
                solution = optimizer.optimize(
                    wrapped_objective,
                    max_evals=remaining_evals,
                    record_history=True
                )
                
                if solution is None:
                    continue
                    
                score = objective_func(solution)  # One final evaluation
                self.total_evaluations += 1
                if record_history:
                    self.optimization_history.append(float(score))  # Store final score
                
                # Update optimizer history
                success = score < 1e-4  # Consider solutions below threshold as successful
                self.history.add_record(
                    self.current_features,
                    selected_optimizer,
                    score,
                    success
                )
                
                # Track selection if we know the problem type
                if self.current_problem_type:
                    self.selection_tracker.record_selection(
                        self.current_problem_type,
                        selected_optimizer,
                        self.current_features,
                        success,
                        score
                    )
                
                # Update best solution
                if score < best_score:
                    best_score = score
                    best_solution = solution.copy()
                    
                # Early stopping if we found a good solution
                if best_score < 1e-4:
                    break
                    
                # Update evaluations based on actual count
                n_evals = self.total_evaluations
                    
            except Exception as e:
                logging.error(f"Optimizer {selected_optimizer} failed: {str(e)}")
                continue
                
            self._current_iteration += 1
            
        # If no solution found, return best from random search
        if best_solution is None:
            logging.warning("No solution found from optimizers, using random search")
            X = np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds],
                size=(100, self.dim)
            )
            scores = np.array([wrapped_objective(x) for x in X])
            best_idx = np.argmin(scores)
            best_solution = X[best_idx]
            best_score = scores[best_idx]
            
        return best_solution, best_score
        
    def optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Optimize the objective function.
        
        Args:
            objective_func: Function to minimize
            max_evals: Maximum number of function evaluations
            record_history: Whether to record optimization history
            context: Additional context for optimization
            
        Returns:
            Best solution found
        """
        best_solution, _ = self._optimize(objective_func, max_evals, record_history, context)
        return best_solution
        
    def reset(self) -> None:
        """Reset optimizer state."""
        self._current_iteration = 0
        self.total_evaluations = 0
        self.current_features = None
        self.current_problem_type = None

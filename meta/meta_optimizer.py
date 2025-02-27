"""
Meta-optimizer that learns to select the best optimization algorithm.
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from pathlib import Path
import os
import concurrent.futures
from dataclasses import dataclass
from threading import Lock
import time

from .optimization_history import OptimizationHistory
from .problem_analysis import ProblemAnalyzer
from .selection_tracker import SelectionTracker
from optimizers.base_optimizer import BaseOptimizer


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    optimizer_name: str
    solution: np.ndarray
    score: float
    n_evals: int
    success: bool = False


class MetaOptimizer:
    """Meta-optimizer that learns to select the best optimization algorithm."""
    
    def __init__(self, 
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 optimizers: Dict[str, BaseOptimizer],
                 history_file: Optional[str] = None,
                 selection_file: Optional[str] = None,
                 n_parallel: int = 2):
        """
        Initialize meta-optimizer.
        
        Args:
            dim: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            optimizers: Dictionary of optimizers to choose from
            history_file: Optional path to save/load optimization history
            selection_file: Optional path to save/load selection history
            n_parallel: Number of optimizers to run in parallel
        """
        self.dim = dim
        self.bounds = bounds
        self.optimizers = optimizers
        self.n_parallel = min(n_parallel, len(optimizers))
        
        # Initialize components
        self.history = OptimizationHistory(history_file)
        self.analyzer = ProblemAnalyzer(bounds, dim)
        self.selection_tracker = SelectionTracker(selection_file)
        
        # Tracking variables
        self.total_evaluations = 0
        self._current_iteration = 0
        self.current_features = None
        self.current_problem_type = None
        self._eval_lock = Lock()
        
        # Learning parameters
        self.min_exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.confidence_threshold = 0.7

    def _calculate_exploration_rate(self) -> float:
        """Calculate adaptive exploration rate based on progress and performance."""
        # Get current performance metrics
        if not self.current_problem_type:
            return self.min_exploration_rate
            
        stats = self.selection_tracker.get_selection_stats(self.current_problem_type)
        if stats.empty:
            return 0.5  # Start with balanced exploration
            
        # Calculate success-based rate
        max_success_rate = stats['success_rate'].max()
        min_success_rate = stats['success_rate'].min()
        success_gap = max_success_rate - min_success_rate
        
        # Adjust exploration based on success distribution
        if max_success_rate > 0.8:
            # We have a very good optimizer, reduce exploration
            base_rate = 0.1
        elif success_gap > 0.4:
            # Clear performance differences, focus on exploitation
            base_rate = 0.2
        elif max_success_rate < 0.3:
            # All optimizers struggling, increase exploration
            base_rate = 0.8
        else:
            # Balanced exploration/exploitation
            base_rate = 0.4
            
        # Adjust for iteration progress
        progress = min(1.0, self._current_iteration / 1000)
        decay = np.exp(-3 * progress)  # Exponential decay
        
        # Combine factors
        return max(self.min_exploration_rate, base_rate * decay)
        
    def _select_optimizer(self, context: Dict[str, Any]) -> List[str]:
        """
        Select optimizers based on problem features and history.
        
        Args:
            context: Problem context
            
        Returns:
            List of selected optimizer names
        """
        if self.current_features is None:
            return list(np.random.choice(
                list(self.optimizers.keys()),
                size=self.n_parallel,
                replace=False
            ))
            
        # Calculate exploration rate
        exploration_rate = self._calculate_exploration_rate()
            
        selected_optimizers = []
        remaining_slots = self.n_parallel
        
        # First, try to use selection history
        if self.current_problem_type:
            correlations = self.selection_tracker.get_feature_correlations(self.current_problem_type)
            if correlations:
                # Calculate weighted scores for each optimizer
                scores = {}
                for opt, feat_corrs in correlations.items():
                    score = 0.0
                    for feat, corr in feat_corrs.items():
                        if feat in self.current_features:
                            # Weight the feature by its correlation with success
                            score += self.current_features[feat] * corr
                    scores[opt] = score
                    
                if scores:
                    # Select top performers based on scores
                    sorted_opts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    n_exploit = int(remaining_slots * (1 - exploration_rate))
                    
                    for opt, _ in sorted_opts[:n_exploit]:
                        selected_optimizers.append(opt)
                        remaining_slots -= 1
                        
        # Next, try to use optimization history
        if remaining_slots > 0 and len(self.history.records) > 0:
            # Find similar problems in history
            similar_records = self.history.find_similar_problems(
                self.current_features,
                k=min(10, len(self.history.records))
            )
            
            if similar_records:
                # Count optimizer successes
                opt_counts = {}
                for record in similar_records:
                    opt = record['optimizer']
                    if opt not in opt_counts:
                        opt_counts[opt] = {'success': 0, 'total': 0}
                    
                    opt_counts[opt]['total'] += 1
                    if record['success']:
                        opt_counts[opt]['success'] += 1
                
                # Calculate success rates
                success_rates = {
                    opt: counts['success'] / counts['total']
                    for opt, counts in opt_counts.items()
                    if counts['total'] > 0 and opt not in selected_optimizers
                }
                
                if success_rates:
                    # Select optimizers based on history
                    n_history = int(remaining_slots * 0.7)  # Use 70% of remaining slots
                    
                    # Convert to probabilities
                    total = sum(success_rates.values())
                    if total > 0:
                        probs = [success_rates[opt] / total for opt in success_rates.keys()]
                        
                        # Sample based on success rates
                        history_selections = np.random.choice(
                            list(success_rates.keys()),
                            size=min(n_history, len(success_rates)),
                            p=probs,
                            replace=False
                        )
                        selected_optimizers.extend(history_selections)
                        remaining_slots -= n_history
        
        # Fill remaining slots with random exploration
        if remaining_slots > 0:
            available_opts = [
                opt for opt in self.optimizers.keys()
                if opt not in selected_optimizers
            ]
            if available_opts:
                random_selections = np.random.choice(
                    available_opts,
                    size=remaining_slots,
                    replace=False
                )
                selected_optimizers.extend(random_selections)
        
        return selected_optimizers

    def _run_single_optimizer(self,
                            optimizer_name: str,
                            optimizer: BaseOptimizer,
                            objective_func: Callable,
                            max_evals: int,
                            record_history: bool = True) -> Optional[OptimizationResult]:
        """Run a single optimizer and return its results"""
        try:
            # Reset optimizer state
            optimizer.reset()
            
            # Set max evaluations
            optimizer.max_evals = max_evals
            
            # Create wrapped objective that ensures numpy array input
            def wrapped_objective(x):
                x = np.asarray(x)
                return float(objective_func(x))
            
            # Run optimization
            start_time = time.time()
            solution, score = optimizer._optimize(wrapped_objective)
            end_time = time.time()
            
            if solution is None:
                return None
                
            # Convert to numpy array and ensure float score
            solution = np.asarray(solution)
            score = float(score)
            
            with self._eval_lock:
                self.total_evaluations += optimizer.evaluations
                if record_history and hasattr(self, 'optimization_history'):
                    # Record optimization history
                    self.optimization_history.append(score)
                    if self.current_features:
                        self.history.add_record(
                            features=self.current_features,
                            optimizer=optimizer_name,
                            performance=score,
                            success=score < 1e-4
                        )
            
            success = score < 1e-4
            
            return OptimizationResult(
                optimizer_name=optimizer_name,
                solution=solution,
                score=score,
                n_evals=optimizer.evaluations,
                success=success
            )
            
        except Exception as e:
            logging.error(f"Optimizer {optimizer_name} failed: {str(e)}")
            return None

    def _optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Internal optimization method with parallel execution"""
        if max_evals is None:
            max_evals = 1000
            
        # Analyze problem features
        self.current_features = self.analyzer.analyze_features(objective_func)
        self.current_problem_type = context.get('problem_type') if context else None
        
        best_solution = None
        best_score = float('inf')
        self.total_evaluations = 0
        self.optimization_history = []
        
        while self.total_evaluations < max_evals:
            # Select optimizers to run in parallel
            selected_optimizers = self._select_optimizer(context or {})
            
            # Calculate remaining evaluations
            remaining_evals = max_evals - self.total_evaluations
            evals_per_optimizer = remaining_evals // len(selected_optimizers)
            
            # Run optimizers in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                future_to_opt = {
                    executor.submit(
                        self._run_single_optimizer,
                        opt_name,
                        self.optimizers[opt_name],
                        objective_func,
                        evals_per_optimizer,
                        record_history
                    ): opt_name
                    for opt_name in selected_optimizers
                }
                
                for future in concurrent.futures.as_completed(future_to_opt):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            
            # Process results
            for result in results:
                # Update optimizer history
                self.history.add_record(
                    self.current_features,
                    result.optimizer_name,
                    result.score,
                    result.success
                )
                
                # Track selection if we know the problem type
                if self.current_problem_type:
                    self.selection_tracker.record_selection(
                        self.current_problem_type,
                        result.optimizer_name,
                        self.current_features,
                        result.success,
                        result.score
                    )
                
                # Update best solution
                if result.score < best_score:
                    best_score = result.score
                    best_solution = result.solution.copy()
                    
                    # Early stopping if we found a good solution
                    if best_score < 1e-4:
                        return best_solution, best_score
            
            self._current_iteration += 1
            
            # Check if we've used up all evaluations
            if self.total_evaluations >= max_evals:
                break
            
        # If no solution found, return best from random search
        if best_solution is None:
            logging.warning("No solution found from optimizers, using random search")
            X = np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds],
                size=(100, self.dim)
            )
            scores = np.array([objective_func(x) for x in X])
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

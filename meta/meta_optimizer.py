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
from tqdm import tqdm  # Add this import at the top

from .optimization_history import OptimizationHistory
from .problem_analysis import ProblemAnalyzer
from .selection_tracker import SelectionTracker
from visualization.live_visualization import LiveOptimizationMonitor

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
                 optimizers: Dict[str, 'BaseOptimizer'],
                 history_file: Optional[str] = None,
                 selection_file: Optional[str] = None,
                 n_parallel: int = 2):
        self.dim = dim
        self.bounds = bounds
        self.optimizers = optimizers
        self.history_file = history_file
        self.selection_file = selection_file
        self.n_parallel = n_parallel
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        self.logger.setLevel(logging.DEBUG)
        
        # Log initialization parameters
        self.logger.info(f"Initializing MetaOptimizer with dim={dim}, n_parallel={n_parallel}")
        self.logger.debug(f"Bounds configuration: {bounds}")
        
        # Initialize optimization history
        self.history = OptimizationHistory(history_file) if history_file else None
        
        # Initialize selection tracker
        self.selection_tracker = SelectionTracker(selection_file) if selection_file else None
        
        # Initialize problem analyzer
        self.analyzer = ProblemAnalyzer(bounds, dim)
        
        # Track best solution
        self.best_solution = None
        self.best_score = float('inf')
        
        # Live visualization
        self.live_viz_monitor = None
        self.live_viz_enabled = False
        self.save_viz_path = None
        
        # Log available optimizers
        self.logger.debug(f"Available optimizers: {list(optimizers.keys())}")
        
        # Tracking variables
        self.total_evaluations = 0
        self._current_iteration = 0
        self.current_features = None
        self.current_problem_type = None
        self._eval_lock = Lock()
        
        # Results tracking
        self.convergence_curve = []
        
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
                            optimizer: 'BaseOptimizer',
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
            solution, score = optimizer.optimize(wrapped_objective)
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
            self.logger.error(f"Optimizer {optimizer_name} failed: {str(e)}")
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
            self.logger.warning("No solution found from optimizers, using random search")
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

    def _update_selection_tracker(self, results):
        """Update selection tracker with optimization results."""
        if self.selection_tracker is None:
            return
            
        for result in results:
            if 'optimizer_name' in result and 'score' in result:
                self.selection_tracker.update(
                    result['optimizer_name'],
                    result['score'],
                    result.get('success', False)
                )
                
    def enable_live_visualization(self, max_data_points: int = 1000, auto_show: bool = True, headless: bool = False):
        """
        Enable live visualization of the optimization process.
        
        Args:
            max_data_points: Maximum number of data points to store per optimizer
            auto_show: Whether to automatically show the plot when monitoring starts
            headless: Whether to run in headless mode (no display, save plots only)
        """
        from visualization.live_visualization import LiveOptimizationMonitor
        self.live_viz_monitor = LiveOptimizationMonitor(
            max_data_points=max_data_points, 
            auto_show=auto_show,
            headless=headless
        )
        self.live_viz_monitor.start_monitoring()
        self.live_viz_enabled = True
        self.logger.info("Live optimization visualization enabled")
        
    def disable_live_visualization(self, save_results: bool = False, results_path: str = None, data_path: str = None):
        """
        Disable live visualization and optionally save results.
        
        Args:
            save_results: Whether to save visualization results
            results_path: Path to save visualization image
            data_path: Path to save visualization data
        """
        if self.live_viz_enabled and self.live_viz_monitor:
            if save_results and results_path:
                self.live_viz_monitor.save_results(results_path)
                
            if save_results and data_path:
                self.live_viz_monitor.save_data(data_path)
                
            self.live_viz_monitor.stop_monitoring()
            self.live_viz_enabled = False
            self.logger.info("Live optimization visualization disabled")
        
    def _report_progress(self, optimizer_name, iteration, score, evaluations):
        """Report optimization progress to any active monitors."""
        # Report to live visualization if enabled
        if self.live_viz_enabled and self.live_viz_monitor:
            self.live_viz_monitor.update_data(
                optimizer=optimizer_name,
                iteration=iteration,
                score=score,
                evaluations=evaluations
            )

    def optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None,
                live_viz: bool = False,
                save_viz_results: bool = False,
                viz_results_path: Optional[str] = None,
                viz_data_path: Optional[str] = None,
                max_data_points: int = 1000,
                auto_show: bool = True,
                headless: bool = False):
        """
        Run optimization with all configured optimizers.
        
        Args:
            objective_func: Function to optimize
            max_evals: Maximum number of evaluations per optimizer
            context: Optional context information
            live_viz: Whether to enable live visualization
            save_viz_results: Whether to save visualization results
            viz_results_path: Path to save visualization results
            viz_data_path: Path to save visualization data
            max_data_points: Maximum number of data points to store per optimizer
            auto_show: Whether to automatically show the plot when monitoring starts
            headless: Whether to run in headless mode (no display, save plots only)
            
        Returns:
            Dictionary of optimization results
        """
        # Set up live visualization if requested
        if live_viz:
            self.enable_live_visualization(
                max_data_points=max_data_points,
                auto_show=auto_show,
                headless=headless
            )
            
        try:
            self.logger.info("Starting optimization process")
            self.logger.debug(f"Max evaluations: {max_evals}")
            self.logger.debug(f"Context: {context}")
            
            # Set progress callback for all optimizers if live visualization is enabled
            if live_viz:
                for name, optimizer in self.optimizers.items():
                    optimizer.progress_callback = self._report_progress
            
            # Initialize tracking variables
            best_score = float('inf')
            best_solution = None
            history = []
            
            if not self.optimizers:
                raise ValueError("No optimizers available")
                
            if not callable(objective_func):
                raise ValueError("Objective function must be callable")
            
            # Initialize optimization history
            history = []
            best_score = float('inf')
            best_solution = None
            
            # Track optimization progress
            for optimizer_name, optimizer in self.optimizers.items():
                self.logger.info(f"Running optimizer: {optimizer_name}")
                try:
                    # Validate optimizer
                    if not hasattr(optimizer, 'run') or not callable(getattr(optimizer, 'run')):
                        self.logger.error(f"Invalid optimizer {optimizer_name}: missing run method")
                        continue
                        
                    # Initialize optimizer with problem parameters
                    if hasattr(optimizer, 'reset') and callable(getattr(optimizer, 'reset')):
                        optimizer.reset()
                    if hasattr(optimizer, 'set_objective') and callable(getattr(optimizer, 'set_objective')):
                        optimizer.set_objective(objective_func)
                    
                    # Log optimizer state
                    self.logger.debug(f"Optimizer {optimizer_name} state: {getattr(optimizer, 'state', 'No state available')}")
                    
                    # Run optimization with validation
                    result = optimizer.run(objective_func=objective_func, max_evals=max_evals)
                    
                    # Validate optimization results
                    if not isinstance(result, dict):
                        self.logger.error(f"Invalid result type from {optimizer_name}: expected dict, got {type(result)}")
                        continue
                        
                    if 'score' not in result:
                        self.logger.error(f"Missing 'score' in results from {optimizer_name}")
                        continue
                    
                    # Log optimization results
                    self.logger.info(f"Optimizer {optimizer_name} completed with score: {result['score']:.3f}")
                    self.logger.debug(f"Optimizer {optimizer_name} full results: {result}")
                    
                    # Update best solution
                    if result['score'] < best_score:
                        best_score = result['score']
                        best_solution = result
                        self.best_score = best_score  # Set the instance attribute
                        self.best_solution = result.get('solution')  # Set the instance attribute
                        self.logger.info(f"New best solution found by {optimizer_name} with score: {best_score:.3f}")
                    
                    history.append(result)
                except Exception as e:
                    self.logger.error(f"Error in optimizer {optimizer_name}: {str(e)}")
                    continue
            
            # Check if we have any valid results
            if not history:
                self.logger.warning("No valid optimization results obtained")
                return {
                    'best_solution': None,
                    'history': [],
                    'best_score': float('inf'),
                    'error': 'No valid optimization results'
                }
            
            # Prepare final results
            final_results = {
                'best_solution': best_solution.get('solution') if best_solution else None,
                'best_score': best_score,
                'history': history,
                'total_evaluations': sum(result.get('evaluations', 0) for result in history)
            }
            
            # Set convergence_curve if any optimizer has it
            self.convergence_curve = []
            for result in history:
                if 'convergence_curve' in result:
                    self.convergence_curve = result.get('convergence_curve', [])
                    break
            
            self.logger.info(f"Optimization completed. Best score: {best_score:.3f}")
            self.logger.debug(f"Final results: {final_results}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Critical error in optimization process: {str(e)}")
            return {
                'best_solution': None,
                'history': [],
                'best_score': float('inf'),
                'error': str(e)
            }
        finally:
            # Clean up live visualization if it was enabled
            if live_viz:
                self.disable_live_visualization(save_results=save_viz_results, results_path=viz_results_path, data_path=viz_data_path)
                
            # Reset progress callbacks
            for name, optimizer in self.optimizers.items():
                optimizer.progress_callback = None
                
    def get_parameters(self) -> Dict[str, Any]:
        """Get optimizer parameters
        
        Returns:
            Dictionary of parameter settings
        """
        return {
            "dim": self.dim,
            "n_parallel": self.n_parallel,
            "optimizers": list(self.optimizers.keys())
        }

    def reset(self) -> None:
        """Reset optimizer state."""
        self._current_iteration = 0
        self.total_evaluations = 0
        self.current_features = None
        self.current_problem_type = None
        self.best_score = float('inf')
        self.best_solution = None
        self.convergence_curve = []

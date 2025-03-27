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
from tqdm import tqdm  
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil
import gc

from .optimization_history import OptimizationHistory
from .problem_analysis import ProblemAnalyzer
from .selection_tracker import SelectionTracker
from visualization.live_visualization import LiveOptimizationMonitor
from optimizers.base_optimizer import BaseOptimizer

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    optimizer_name: str
    solution: np.ndarray
    score: float
    n_evals: int
    success: bool = False


class MetaOptimizer(BaseOptimizer):
    """Meta-optimizer that learns to select the best optimization algorithm."""
    def __init__(self, 
                 dim: int,
                 bounds: List[Tuple[float, float]],
                 optimizers: Dict[str, 'BaseOptimizer'],
                 history_file: Optional[str] = None,
                 selection_file: Optional[str] = None,
                 n_parallel: int = 2,
                 budget_per_iteration: int = 100,
                 default_max_evals: int = 1000,
                 verbose: bool = False,
                 timeout: int = 60,
                 iteration_timeout: int = 10,
                 **kwargs):
        super().__init__(dim=dim, timeout=timeout, iteration_timeout=iteration_timeout, **kwargs)
        self.dim = dim
        self.bounds = bounds
        self.optimizers = optimizers
        self.history_file = history_file
        self.selection_file = selection_file
        self.n_parallel = n_parallel
        self.budget_per_iteration = budget_per_iteration
        self.default_max_evals = default_max_evals
        self.logger = logging.getLogger('MetaOptimizer')
        if not self.logger.handlers:  
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set log level based on verbose flag
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
        
        # Configure logging
        # self.logger.setLevel(logging.DEBUG)
        
        # Log initialization parameters
        self.logger.info(f"Initializing MetaOptimizer with dim={dim}, n_parallel={n_parallel}")
        
        # Initialize optimization history
        self.history = OptimizationHistory(history_file)
        
        # Initialize selection tracker
        self.selection_tracker = SelectionTracker(selection_file)
        
        # Initialize state variables
        self.objective_func = None
        self.max_evals = None
        self.best_solution = None
        self.best_score = float('inf')
        self.total_evaluations = 0
        self.start_time = 0
        self.end_time = 0
        self.convergence_curve = []
        self.optimization_history = []
        
        # Problem features
        self.current_features = None
        self.current_problem_type = None
        
        # Initialize problem analyzer
        self.analyzer = ProblemAnalyzer(bounds, dim)
        
        # Live visualization
        self.live_viz_monitor = None
        self.enable_viz = False
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
        
        self.max_memory_gb = kwargs.get('max_memory_gb', 4)  # Add memory limit
        self.stop_flag = threading.Event()
        self.current_optimizer = None
        
        # Configure logging with more detail
        logging.basicConfig(level=logging.DEBUG)
        self.logger.setLevel(logging.DEBUG)

        self.visualization_enabled = kwargs.get('visualization_enabled', True)

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
                for similarity, record in similar_records:
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
                
    def optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None):
        """Run optimization with enhanced timeout and resource monitoring"""
        self.logger.info("Starting optimization with enhanced monitoring")
        start_time = time.time()
        
        def check_resources():
            """Check system resources"""
            process = psutil.Process(os.getpid())
            memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
            if memory_gb > self.max_memory_gb:
                self.logger.warning(f"Memory usage ({memory_gb:.2f}GB) exceeded limit ({self.max_memory_gb}GB)")
                return False
            return True

        def run_single_optimizer(name, optimizer):
            """Run a single optimizer with timeout"""
            if self.stop_flag.is_set():
                return None
                
            self.logger.info(f"Starting optimizer: {name}")
            self.current_optimizer = optimizer
            
            try:
                # Set iteration budget
                iter_max_evals = max_evals // len(self.optimizers) if max_evals else 1000
                
                # Run with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(optimizer.optimize, objective_func, iter_max_evals)
                    try:
                        result = future.result(timeout=self.iteration_timeout)
                        if isinstance(result, tuple):
                            return result[0], result[1]  # solution, score
                        return result, objective_func(result)
                    except TimeoutError:
                        self.logger.warning(f"Optimizer {name} timed out")
                        return None
                    except Exception as e:
                        self.logger.error(f"Error in optimizer {name}: {str(e)}")
                        return None
            except Exception as e:
                self.logger.error(f"Failed to run optimizer {name}: {str(e)}")
                return None

        try:
            best_solution = None
            best_score = float('inf')
            
            # Run optimizers sequentially with timeouts
            for name, optimizer in self.optimizers.items():
                if time.time() - start_time > self.timeout:
                    self.logger.warning("Global timeout reached")
                    break
                    
                if not check_resources():
                    self.logger.warning("Resource limit exceeded")
                    break
                
                result = run_single_optimizer(name, optimizer)
                if result is not None:
                    solution, score = result
                    if score < best_score:
                        best_score = score
                        best_solution = solution
                        self.logger.info(f"New best score: {best_score} from {name}")
                
                # Force garbage collection between optimizers
                gc.collect()
                
            if best_solution is not None:
                self.best_solution = best_solution
                self.best_score = best_score
                return best_solution, best_score
            else:
                self.logger.error("No valid solution found")
                return None, float('inf')
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return None, float('inf')
        finally:
            self.stop_flag.clear()
            self.current_optimizer = None
            duration = time.time() - start_time
            self.logger.info(f"Optimization completed in {duration:.2f} seconds")

    def run(self, objective_func: Callable, max_evals: Optional[int] = None) -> Dict[str, Any]:
        """
        Run optimization method compatible with the Meta-Optimizer interface.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting Meta-Optimizer run")
        
        # Call the optimize method which contains our implementation
        self.optimize(objective_func, max_evals)
        
        # Return result in expected dictionary format
        return {
            'solution': self.best_solution,
            'score': self.best_score,
            'evaluations': self.total_evaluations,
            'runtime': (self.end_time - self.start_time) if hasattr(self, 'end_time') and self.end_time > 0 else 0,
            'convergence_curve': self.convergence_curve if hasattr(self, 'convergence_curve') else []
        }

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
        self.best_solution = None
        self.best_score = float('inf')
        self.total_evaluations = 0
        self.convergence_curve = []
        self.stop_flag.clear()
        self.current_optimizer = None

    def set_objective(self, objective_func: Callable):
        """Set the objective function for optimization.
        
        Args:
            objective_func: The objective function to optimize
        """
        self.logger.info("Setting objective function")
        self.objective_func = objective_func

    def _update_selection_strategy(self, optimizer_states: Dict[str, 'OptimizerState']):
        """
        Update optimizer selection strategy based on performance.
        
        Args:
            optimizer_states: Dictionary of optimizer states
        """
        # Extract metrics from optimizer states
        optimizer_metrics = {}
        for opt_name, state in optimizer_states.items():
            if hasattr(state, 'to_dict'):
                state_dict = state.to_dict()
                
                # Calculate convergence rate and success rate from state metrics
                convergence_rate = state_dict.get('convergence_rate', 0.0)
                stagnation_count = state_dict.get('stagnation_count', 0)
                iterations = state_dict.get('iterations', 1)
                success_rate = 1.0 - (stagnation_count / max(iterations, 1))
                
                optimizer_metrics[opt_name] = {
                    'convergence_rate': convergence_rate,
                    'success_rate': success_rate
                }
                
                # Classify problem type if not already done
                if not self.current_problem_type and self.current_features:
                    self.current_problem_type = self._classify_problem(self.current_features)
        
        # Update selection tracker with new information
        if self.current_problem_type:
            self.selection_tracker.update_correlations(
                self.current_problem_type,
                optimizer_states
            )

    def _extract_problem_features(self, objective_func: Callable) -> Dict[str, float]:
        """
        Extract features from the objective function to characterize the problem.
        
        Args:
            objective_func: Objective function to analyze
            
        Returns:
            Dictionary of problem features
        """
        # Use the ProblemAnalyzer to extract features
        analyzer = ProblemAnalyzer(self.bounds, self.dim)
        features = analyzer.analyze_features(objective_func)
        
        self.logger.debug(f"Extracted problem features: {features}")
        return features
        
    def _classify_problem(self, features: Dict[str, float]) -> str:
        """
        Classify the problem type based on features.
        
        Args:
            features: Problem features
            
        Returns:
            Problem type classification
        """
        # Simple classification based on key features
        if features['dimension'] > 10:
            problem_type = 'high_dimensional'
        elif features['modality'] > 5:
            problem_type = 'multimodal'
        elif features['ruggedness'] > 0.7:
            problem_type = 'rugged'
        elif features['convexity'] > 0.8:
            problem_type = 'convex'
        else:
            problem_type = 'general'
            
        self.logger.debug(f"Classified problem as: {problem_type}")
        return problem_type

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
        self.enable_viz = True
        self.logger.info("Live optimization visualization enabled")
        
    def disable_live_visualization(self, save_results: bool = False, results_path: str = None, data_path: str = None):
        """
        Disable live visualization and optionally save results.
        
        Args:
            save_results: Whether to save visualization results
            results_path: Path to save visualization image
            data_path: Path to save visualization data
        """
        if self.enable_viz and self.live_viz_monitor:
            if save_results and results_path:
                self.live_viz_monitor.save_results(results_path)
                
            if save_results and data_path:
                self.live_viz_monitor.save_data(data_path)
                
            self.live_viz_monitor.stop_monitoring()
            self.enable_viz = False
            self.logger.info("Live optimization visualization disabled")
        
    def _report_progress(self, optimizer_name, iteration, score, evaluations):
        """Report optimization progress to any active monitors."""
        # Report to live visualization if enabled
        if self.enable_viz and self.live_viz_monitor:
            self.live_viz_monitor.update_data(
                optimizer=optimizer_name,
                iteration=iteration,
                score=score,
                evaluations=evaluations
            )

    def stop(self):
        """Stop optimization"""
        self.stop_flag.set()
        if self.current_optimizer and hasattr(self.current_optimizer, 'stop'):
            self.current_optimizer.stop()

    def _optimize(self, objective_function, max_evals):
        self.logger.info("Algorithm selection visualization enabled")
        self.logger.debug(f"Available optimizers: {list(self.optimizers.keys())}")
        
        # Extract problem features
        features = self._extract_problem_features(objective_function)
        self.logger.debug(f"Extracted problem features: {features}")
        
        # Classify problem
        problem_type = self._classify_problem(features)
        self.logger.debug(f"Classified problem as: {problem_type}")
        self.logger.info(f"Problem classified as: {problem_type}")
        
        # Select initial optimizer
        selected_optimizers = self._select_optimizers(problem_type, features)
        self.logger.debug(f"Selected optimizers: {selected_optimizers}")
        
        best_solution = None
        best_score = float('inf')
        total_evals = 0
        
        start_time = time.time()
        
        try:
            for i, opt_name in enumerate(selected_optimizers, 1):
                if self._check_stop_criteria(total_evals, max_evals, best_score):
                    break
                    
                self.logger.info(f"Recorded selection of optimizer {opt_name} for iteration {i}")
                optimizer = self.optimizers[opt_name]
                optimizer.reset()
                
                remaining_evals = max_evals - total_evals
                if remaining_evals <= 0:
                    break
                    
                result = optimizer.optimize(objective_function, max_evals=remaining_evals)
                
                if result.success and result.best_score < best_score:
                    best_score = result.best_score
                    best_solution = result.best_solution
                    self.logger.info(f"New best solution from {opt_name}: {best_score}")
                
                total_evals += result.evaluations
                
                if time.time() - start_time > self.timeout:
                    self.logger.warning("Meta-optimization timeout reached")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error during meta-optimization: {str(e)}")
            if best_solution is None:
                raise
                
        return best_solution, best_score

    def _select_optimizers(self, problem_type, features):
        """Select appropriate optimizers based on problem classification."""
        # For demonstration, randomly select one optimizer
        selected = np.random.choice(list(self.optimizers.keys()), size=1)
        self.logger.info(f"Randomly selected 1 optimizers for demonstration: {selected}")
        return selected

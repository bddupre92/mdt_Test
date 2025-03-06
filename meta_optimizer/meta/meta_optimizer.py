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
import sys
from tqdm import tqdm  

from .optimization_history import OptimizationHistory
from .problem_analysis import ProblemAnalyzer
from .selection_tracker import SelectionTracker
from ..visualization.live_visualization import LiveOptimizationMonitor

# Import algorithm selection visualizer
try:
    from ...visualization.algorithm_selection_viz import AlgorithmSelectionVisualizer
    ALGORITHM_VIZ_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from visualization.algorithm_selection_viz import AlgorithmSelectionVisualizer
        ALGORITHM_VIZ_AVAILABLE = True
    except ImportError:
        ALGORITHM_VIZ_AVAILABLE = False
        logging.warning("Algorithm selection visualization not available. Using standard visualization.")

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
                 n_parallel: int = 2,
                 budget_per_iteration: int = 100,
                 default_max_evals: int = 1000,
                 verbose: bool = False):
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
        
        # Algorithm selection visualization
        self.algo_selection_viz = None
        self.enable_algo_viz = False
        self.visualize_algorithm_selection = False
        
        # Initialize algorithm selection visualization
        if ALGORITHM_VIZ_AVAILABLE:
            self.algo_selection_viz = AlgorithmSelectionVisualizer()
            self.enable_algo_viz = True
            self.visualize_algorithm_selection = True
            self.logger.info("Algorithm selection visualization enabled")
        
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
        Select which optimizer(s) to use based on problem features.
        
        Args:
            context: Additional context information
            
        Returns:
            List of selected optimizer names
        """
        # For this demo, let's ensure we always select optimizers
        # rather than waiting for the full learning algorithm
        selected_optimizers = []
        
        # Get available optimizers
        available_optimizers = list(self.optimizers.keys())
        
        # For demonstration purposes, randomly select 1-3 optimizers
        if len(available_optimizers) > 1:
            # Randomly select 1-3 optimizers for demonstration purposes
            n_select = np.random.randint(1, min(4, len(available_optimizers)))
            selected_optimizers = list(np.random.choice(available_optimizers, size=n_select, replace=False))
            self.logger.info(f"Randomly selected {n_select} optimizers for demonstration: {selected_optimizers}")
        else:
            # If there's only one optimizer, select it
            selected_optimizers = available_optimizers
            
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
                context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run optimization with all configured optimizers.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            context: Optional context information
            
        Returns:
            Best solution found (numpy array)
        """
        # Use default max evaluations if not specified
        max_evals = max_evals or self.default_max_evals
        
        # Set initial parameters
        self.reset()
        self.objective_func = objective_func
        self.max_evals = max_evals
        self.start_time = time.time()
        
        # Initialize optimization history if not present
        if not hasattr(self, 'optimization_history'):
            self.optimization_history = []
        
        # Extract problem features
        try:
            self.current_features = self._extract_problem_features(objective_func)
            
            # Classify problem type
            self.current_problem_type = self._classify_problem(self.current_features)
            self.logger.info(f"Problem classified as: {self.current_problem_type}")
        except Exception as e:
            self.logger.warning(f"Could not extract problem features: {e}")
            self.current_features = None
            self.current_problem_type = None
        
        # Store context in current features if provided
        if context:
            if not self.current_features:
                self.current_features = {}
            
            for key, value in context.items():
                self.current_features[key] = value
                
        # Main optimization loop
        while self.total_evaluations < max_evals:
            self._current_iteration += 1
            
            # Select optimizer to use for this iteration based on problem features
            selected_optimizers = self._select_optimizer(context or {})
            
            # Check if we need to select at least one
            if not selected_optimizers:
                # Fallback: select a random optimizer
                selected_optimizers = [np.random.choice(list(self.optimizers.keys()))]
                
            self.logger.debug(f"Selected optimizers: {selected_optimizers}")
            
            # Run each selected optimizer for a portion of the budget
            per_optimizer_budget = self.budget_per_iteration // len(selected_optimizers)
            optimizer_futures = {}
            
            # Record selections in algo visualization
            if hasattr(self, 'enable_algo_viz') and self.enable_algo_viz and hasattr(self, 'algo_selection_viz') and self.algo_selection_viz:
                for optimizer_name in selected_optimizers:
                    self.algo_selection_viz.record_selection(
                        iteration=self._current_iteration,
                        optimizer=optimizer_name,
                        problem_type=self.current_problem_type or "unknown",
                        score=self.best_score,
                        context=context
                    )
                    # Log that we've recorded a selection
                    self.logger.info(f"Recorded selection of optimizer {optimizer_name} for iteration {self._current_iteration}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_optimizers)) as executor:
                # Submit each optimizer task
                for optimizer_name in selected_optimizers:
                    if optimizer_name not in self.optimizers:
                        self.logger.warning(f"Selected optimizer {optimizer_name} not available")
                        continue
                        
                    optimizer = self.optimizers[optimizer_name]
                    optimizer_futures[executor.submit(
                        self._run_single_optimizer,
                        optimizer_name,
                        optimizer,
                        objective_func,
                        per_optimizer_budget
                    )] = optimizer_name
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(optimizer_futures):
                    optimizer_name = optimizer_futures[future]
                    try:
                        result = future.result()
                        if result:
                            # Update best solution if this optimizer found a better one
                            if result.score < self.best_score:
                                self.best_score = result.score
                                self.best_solution = result.solution
                                self.logger.info(
                                    f"New best solution from {optimizer_name}: {self.best_score:.10f}"
                                )
                                
                            # Track evaluations
                            self.total_evaluations += result.n_evals
                            
                            # Update convergence curve
                            if len(self.convergence_curve) == 0:
                                self.convergence_curve.append((0, result.score))
                            self.convergence_curve.append((self.total_evaluations, self.best_score))
                            
                            # Record history
                            self.optimization_history.append({
                                'iteration': self._current_iteration,
                                'selected_optimizer': optimizer_name,
                                'score': result.score,
                                'best_score': self.best_score,
                                'evaluations': result.n_evals,
                                'total_evaluations': self.total_evaluations,
                                'success': result.success,
                                'features': self.current_features,
                                'problem_type': self.current_problem_type
                            })
                            
                            # Update selection tracker
                            self._update_selection_tracker(result)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing results from {optimizer_name}: {e}")
                    
            # Check if we're done
            if self.total_evaluations >= max_evals:
                break
            
            # Check if we've converged
            if self._current_iteration > 1 and len(self.convergence_curve) > 1:
                prev_score = self.convergence_curve[-2][1]
                curr_score = self.convergence_curve[-1][1]
                improvement = prev_score - curr_score
                
                # If improvement is very small, we might have converged
                if improvement < 1e-8 * prev_score:
                    self.logger.info(f"Convergence detected after {self._current_iteration} iterations")
                    break
        
        # Record end time
        self.end_time = time.time()
        
        # Log final results
        self.logger.info(f"Optimization completed in {self.end_time - self.start_time:.2f} seconds")
        self.logger.info(f"Total evaluations: {self.total_evaluations}")
        self.logger.info(f"Best score: {self.best_score:.10f}")
        
        # Return best solution
        return self.best_solution

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
        """Reset the optimizer state."""
        # Save visualization state before reset
        save_algo_viz = self.algo_selection_viz if hasattr(self, 'algo_selection_viz') else None
        save_enable_algo_viz = self.enable_algo_viz if hasattr(self, 'enable_algo_viz') else False
        save_visualize_algo_selection = self.visualize_algorithm_selection if hasattr(self, 'visualize_algorithm_selection') else False
        
        # Save selection history if requested
        if self.selection_file:
            if hasattr(self, 'selection_history') and self.selection_history:
                try:
                    with open(self.selection_file, 'w') as f:
                        json.dump(self.selection_history, f, indent=2)
                    self.logger.info(f"Saved selection history to {self.selection_file}")
                except Exception as e:
                    self.logger.warning(f"Could not save selection history to {self.selection_file}: {e}")
                    
        # Reset optimization state
        self.total_evaluations = 0
        self.best_solution = None
        self.best_score = float('inf')
        self._current_iteration = 0
        
        # Reset current problem data
        self.current_features = None
        self.current_problem_type = None
        
        # Reset optimizers
        for optimizer in self.optimizers.values():
            optimizer.reset()
        
        # Reset optimization history but keep the selection history
        self.optimization_history = []
        
        # Restore visualization state
        self.algo_selection_viz = save_algo_viz
        self.enable_algo_viz = save_enable_algo_viz
        self.visualize_algorithm_selection = save_visualize_algo_selection

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

    def enable_live_visualization(self, save_path: Optional[str] = None, max_data_points: int = 1000, auto_show: bool = True, headless: bool = False):
        """
        Enable live visualization of the optimization process.
        
        Args:
            save_path: Optional path to save visualization files
            max_data_points: Maximum number of data points to store per optimizer
            auto_show: Whether to automatically show the plot when monitoring starts
            headless: Whether to run in headless mode (no display, save plots only)
        """
        from ..visualization.live_visualization import LiveOptimizationMonitor
        self.live_viz_monitor = LiveOptimizationMonitor(
            max_data_points=max_data_points, 
            auto_show=auto_show,
            headless=headless
        )
        self.live_viz_monitor.start_monitoring()
        self.enable_viz = True
        self.save_viz_path = save_path
        
        # Initialize algorithm selection visualization if available
        if ALGORITHM_VIZ_AVAILABLE:
            self.algo_selection_viz = AlgorithmSelectionVisualizer(save_dir=save_path)
            self.enable_algo_viz = True
            self.visualize_algorithm_selection = True
            self.logger.info("Algorithm selection visualization enabled")
        
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
            self.live_viz_monitor = None
            self.enable_viz = False
            self.logger.info("Live optimization visualization disabled")
            
        # Generate algorithm selection visualizations if enabled
        if self.enable_algo_viz and self.algo_selection_viz and save_results:
            if not results_path and self.save_viz_path:
                results_path = self.save_viz_path
                
            # Create algorithm selection visualizations
            self.algo_selection_viz.plot_selection_frequency(save=True)
            self.algo_selection_viz.plot_selection_timeline(save=True)
            self.algo_selection_viz.plot_problem_distribution(save=True)
            self.algo_selection_viz.plot_performance_comparison(save=True)
            self.algo_selection_viz.plot_phase_selection(save=True)
            self.algo_selection_viz.create_summary_dashboard(save=True)
            
            self.logger.info("Algorithm selection visualizations saved")
            
        self.enable_algo_viz = False
        self.algo_selection_viz = None

    def visualize_algorithm_selection(self, 
                                      save_dir: str = 'results/algorithm_selection', 
                                      plot_types: List[str] = None,
                                      interactive: bool = True,
                                      title_prefix: str = "Algorithm Selection Analysis") -> Dict[str, str]:
        """
        Generate visualizations for algorithm selection patterns.
        
        Args:
            save_dir: Directory to save visualization files
            plot_types: List of plot types to generate. If None, generates all plots.
                        Options: ['frequency', 'timeline', 'problem', 'performance', 'phase', 'dashboard']
            interactive: Whether to generate interactive visualizations
            title_prefix: Prefix for plot titles
            
        Returns:
            Dictionary with paths to generated visualization files
        """
        if not hasattr(self, 'algo_selection_viz') or not self.algo_selection_viz:
            self.logger.warning("Algorithm selection visualization not available. No visualizations generated.")
            return {"error": "Algorithm selection visualization not available"}
        
        if not hasattr(self, 'selection_history') or not self.selection_history:
            self.logger.warning("No algorithm selection history available. No visualizations generated.")
            return {"error": "No algorithm selection history available"}
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup default plot types if not specified
        if plot_types is None:
            plot_types = ['frequency', 'timeline', 'problem', 'performance', 'phase', 'dashboard']
        
        visualization_files = {}
        
        # Generate static plots
        if 'frequency' in plot_types:
            self.algo_selection_viz.plot_selection_frequency(
                title=f"{title_prefix}: Selection Frequency",
                save=True
            )
            visualization_files['frequency'] = os.path.join(save_dir, "algorithm_selection_frequency.png")
        
        if 'timeline' in plot_types:
            self.algo_selection_viz.plot_selection_timeline(
                title=f"{title_prefix}: Selection Timeline",
                save=True
            )
            visualization_files['timeline'] = os.path.join(save_dir, "algorithm_selection_timeline.png")
        
        if 'problem' in plot_types:
            self.algo_selection_viz.plot_problem_distribution(
                title=f"{title_prefix}: Problem Distribution",
                save=True
            )
            visualization_files['problem'] = os.path.join(save_dir, "algorithm_selection_by_problem.png")
        
        if 'performance' in plot_types:
            self.algo_selection_viz.plot_performance_comparison(
                title=f"{title_prefix}: Performance Comparison",
                save=True
            )
            visualization_files['performance'] = os.path.join(save_dir, "optimizer_performance_comparison.png")
        
        if 'phase' in plot_types:
            self.algo_selection_viz.plot_phase_selection(
                title=f"{title_prefix}: Phase Selection",
                save=True
            )
            visualization_files['phase'] = os.path.join(save_dir, "algorithm_selection_by_phase.png")
        
        if 'dashboard' in plot_types:
            self.algo_selection_viz.create_summary_dashboard(
                title=f"{title_prefix}: Summary Dashboard",
                save=True
            )
            visualization_files['dashboard'] = os.path.join(save_dir, "algorithm_selection_dashboard.png")
        
        # Generate interactive visualizations if requested
        if interactive:
            try:
                import plotly
                
                self.algo_selection_viz.interactive_selection_timeline(
                    title=f"{title_prefix}: Interactive Selection Timeline",
                    save=True
                )
                visualization_files['interactive_timeline'] = os.path.join(save_dir, "interactive_algorithm_timeline.html")
                
                self.algo_selection_viz.interactive_dashboard(
                    title=f"{title_prefix}: Interactive Dashboard",
                    save=True
                )
                visualization_files['interactive_dashboard'] = os.path.join(save_dir, "interactive_dashboard.html")
                
            except (ImportError, Exception) as e:
                self.logger.warning(f"Could not generate interactive visualizations: {e}")
        
        self.logger.info(f"Generated {len(visualization_files)} algorithm selection visualizations in {save_dir}")
        return visualization_files

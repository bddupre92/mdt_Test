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
    def __init__(
        self,
        dim,
        bounds,
        optimizers=None,
        n_parallel=2,
        budget_per_iteration=50,
        history_db_path=None,
        history_table=None,
        use_ml_selection=False,
        max_retrain_interval=10,
        visualize_selection=True,
        verbose=False,
        default_max_evals=None,
        use_selection_tracker=True,
        selection_tracker_weights=None,
        early_stopping=False,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-6,
    ):
        """Initialize Meta-Optimizer.
        
        Args:
            dim: Problem dimensionality
            bounds: List of (lower, upper) bounds for each dimension
            optimizers: Dictionary of optimizer instances keyed by name
            n_parallel: Number of parallel optimization runs
            budget_per_iteration: Budget per iteration
            history_db_path: Path to history database
            history_table: Name of history table
            use_ml_selection: Whether to use ML for algorithm selection
            max_retrain_interval: Maximum interval between model retraining
            visualize_selection: Whether to visualize algorithm selection
            verbose: Verbosity level
            default_max_evals: Default maximum evaluations cap
            use_selection_tracker: Whether to use selection tracker
            selection_tracker_weights: Selection tracker weights
            early_stopping: Whether to use early stopping
            early_stopping_patience: Early stopping patience
            early_stopping_min_delta: Early stopping minimum delta
        """
        self.dim = dim
        self.bounds = bounds
        self.optimizers = optimizers or {}
        
        # Debug: Count and log received optimizers
        optimizers_count = len(self.optimizers) if self.optimizers else 0
        print(f"DEBUG: Received {optimizers_count} optimizers: {', '.join(self.optimizers.keys() if self.optimizers else [])}")
        
        # Set verbosity for all optimizers
        if self.optimizers:
            for name, optimizer in self.optimizers.items():
                if hasattr(optimizer, "verbose") and verbose:
                    optimizer.verbose = verbose
            
        # Setup logger
        self.logger = logging.getLogger("MetaOptimizer")
        self.verbose = verbose
        
        # Set log level based on verbosity
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            
        # Debug: Log optimizer count again
        if hasattr(self, 'optimizers'):
            self.logger.info(f"After setup: {len(self.optimizers)} optimizers available: {', '.join(self.optimizers.keys() if self.optimizers else [])}")
        else:
            self.logger.warning("After setup: No optimizers attribute exists")
            
        # Store optimizers reference
        self._original_optimizers = dict(self.optimizers)
        
        # Log initialization parameters
        self.logger.info(f"Initializing MetaOptimizer with dim={dim}, n_parallel={n_parallel}")
        
        # Store parameters
        self.dim = dim
        self.bounds = bounds
        self.n_parallel = n_parallel
        self.budget_per_iteration = budget_per_iteration
        self.default_max_evals = default_max_evals
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.use_ml_selection = use_ml_selection
        self.use_selection_tracker = use_selection_tracker
        self.max_retrain_interval = max_retrain_interval
        self.visualize_selection = visualize_selection
        
        # Initialize optimization history
        self.history = OptimizationHistory(history_db_path)
        
        # Initialize selection tracker
        self.selection_tracker = SelectionTracker(selection_tracker_weights) if use_selection_tracker else None
        
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
        
        # Load optimization history and selection data
        if hasattr(self, '_load_data'):
            self._load_data()
        else:
            # Placeholder for loading data when method is missing
            self.logger.info("No _load_data method available, initializing empty history")
            self.history = OptimizationHistory(history_db_path)
            self.selection_tracker = SelectionTracker(selection_tracker_weights) if use_selection_tracker else None
        
        # Initialize ML model for selections
        self.ml_model = None
        self.feature_scaler = None
        if use_ml_selection:
            self.train_selection_model()
        
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
        # Debug optimizers state before selection
        if hasattr(self, 'optimizers'):
            self.logger.info(f"_select_optimizer: {len(self.optimizers)} optimizers available: {', '.join(self.optimizers.keys() if self.optimizers else [])}")
        else:
            self.logger.warning("_select_optimizer: No optimizers attribute exists")
            
        # Create default optimizers if none are available
        if not self.optimizers:
            # Create optimizers using OptimizerFactory if available
            try:
                from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory
                bounds = self.bounds
                dim = self.dim
                
                self.logger.info(f"Creating default optimizers with dim={dim} and bounds={bounds}")
                
                factory = OptimizerFactory()
                default_optimizers = {
                    'DE': factory.create_optimizer('differential_evolution', dim=dim, bounds=bounds),
                    'ES': factory.create_optimizer('evolution_strategy', dim=dim, bounds=bounds)
                }
                
                # Store the created optimizers
                self.optimizers = default_optimizers
                self.logger.info(f"Created {len(self.optimizers)} default optimizers: {', '.join(self.optimizers.keys())}")
                
                # Return the first optimizer
                if self.optimizers:
                    return [list(self.optimizers.keys())[0]]
                    
            except (ImportError, Exception) as e:
                self.logger.warning(f"Failed to create default optimizers: {str(e)}")
                
            # No optimizers available, add a fallback algorithm
            self.logger.warning("No optimizers available, adding fallback differential evolution algorithm")
            try:
                # Try to import DE from scipy as a fallback
                from scipy.optimize import differential_evolution
                
                # Create wrapper class instead of simple function
                class FallbackDEOptimizer:
                    def __init__(self):
                        self.name = "fallback_de"
                        self.evaluations = 0
                        self.reset()
                        
                    def reset(self):
                        self.evaluations = 0
                        
                    def optimize(self, func, max_evals):
                        # Use bounds from MetaOptimizer
                        try:
                            # For scipy's differential_evolution, bounds must be a sequence of (min, max) pairs
                            if hasattr(self.meta_optimizer, 'bounds') and self.meta_optimizer.bounds:
                                # Check if we have nested bounds like [((0, 1), (0, 1))]
                                if isinstance(self.meta_optimizer.bounds[0], tuple) and len(self.meta_optimizer.bounds[0]) == 2:
                                    # Check if the bounds are nested tuples
                                    if isinstance(self.meta_optimizer.bounds[0][0], tuple):
                                        self.meta_optimizer.logger.warning(f"Nested bounds detected. Flattening bounds.")
                                        # Flatten nested bounds
                                        bounds = []
                                        for b in self.meta_optimizer.bounds:
                                            if isinstance(b[0], tuple):
                                                for inner_b in b:
                                                    bounds.append(inner_b)
                                            else:
                                                bounds.append(b)
                                    else:
                                        # Already in the correct format [(min1, max1), (min2, max2), ...]
                                        bounds = self.meta_optimizer.bounds
                                else:
                                    # Unexpected format, set default bounds
                                    self.meta_optimizer.logger.warning(f"Unexpected bounds format. Using default bounds.")
                                    bounds = [(0, 1)] * self.meta_optimizer.dim
                            else:
                                # No bounds, use default
                                self.meta_optimizer.logger.warning(f"No bounds found. Using default bounds.")
                                bounds = [(0, 1)] * self.meta_optimizer.dim
                                
                            # Verify bounds length matches dimensions
                            if len(bounds) != self.meta_optimizer.dim:
                                self.meta_optimizer.logger.warning(f"Bounds mismatch: have {len(bounds)} bounds for {self.meta_optimizer.dim} dimensions. Adjusting bounds.")
                                if len(bounds) > 0:
                                    # Use the first bound for all dimensions
                                    bounds = [bounds[0]] * self.meta_optimizer.dim
                                else:
                                    # Use default bounds
                                    bounds = [(0, 1)] * self.meta_optimizer.dim
                                
                            self.meta_optimizer.logger.debug(f"Using DE bounds: {bounds}")
                            
                            # Run scipy's differential_evolution with proper bounds
                            result = differential_evolution(
                                func, 
                                bounds=bounds, 
                                maxiter=max(10, max_evals//10), 
                                popsize=10,
                                updating='deferred',
                                workers=1  # Single worker for deterministic behavior
                            )
                            self.evaluations = result.nfev
                            return result.x, result.fun
                        except Exception as e:
                            self.meta_optimizer.logger.error(f"Fallback DE optimizer failed: {str(e)}")
                            # Return a zero vector as a fallback solution
                            return np.zeros(self.meta_optimizer.dim), float('inf')
                
                # Create instance and store self reference
                fallback_de = FallbackDEOptimizer()
                fallback_de.meta_optimizer = self
                
                # Add to optimizers
                self.optimizers = {"fallback_de": fallback_de}
                return ["fallback_de"]
            except ImportError:
                self.logger.error("Could not import scipy.optimize.differential_evolution for fallback")
                # Create a dummy optimizer for fallback
                class RandomSearchOptimizer:
                    def __init__(self):
                        self.evaluations = 0
                        self.reset()
                        
                    def reset(self):
                        self.evaluations = 0
                        
                    def optimize(self, func, max_evals):
                        # Use bounds from MetaOptimizer
                        bounds = self.meta_optimizer.bounds
                        best_x = None
                        best_y = float('inf')
                        
                        # Generate random points
                        for i in range(max_evals):
                            # Generate random point
                            x = np.array([np.random.uniform(low, high) for low, high in bounds])
                            
                            # Evaluate
                            y = func(x)
                            
                            # Update best
                            if y < best_y:
                                best_y = y
                                best_x = x
                        
                        self.evaluations = max_evals
                        return best_x, best_y
                
                # Create instance and store self reference
                random_search = RandomSearchOptimizer()
                random_search.meta_optimizer = self
                
                # Add to optimizers dictionary
                self.optimizers = {"random_search": random_search}
                return ["random_search"]
        
        if self.current_features is None:
            return list(np.random.choice(
                list(self.optimizers.keys()),
                size=min(self.n_parallel, len(self.optimizers)),
                replace=False
            ))
        
        # Calculate exploration rate (decreases over time)
        exploration_rate = self.min_exploration_rate + (1.0 - self.min_exploration_rate) * (
            self.exploration_decay ** self._current_iteration
        )
        
        # Number of optimizers to select
        n_select = min(self.n_parallel, len(self.optimizers))
        n_exploit = max(1, int(n_select * (1.0 - exploration_rate)))
        n_explore = n_select - n_exploit
        
        self.logger.info(f"Selecting {n_exploit} optimizers to exploit and {n_explore} to explore")
        
        # Track candidates with their scores
        candidates = {opt_name: 0.0 for opt_name in self.optimizers.keys()}
        
        # 1. Feature-based heuristic selection
        self._update_feature_based_scores(candidates)
        
        # 2. History-based selection (if we have historical data)
        if hasattr(self, 'history') and self.history and len(self.history.records) > 0:
            self._update_history_based_scores(candidates)
            
        # 3. Selection tracker-based selection (if available)
        if (hasattr(self, 'selection_tracker') and self.selection_tracker and 
            self.current_problem_type):
            self._update_tracker_based_scores(candidates)
        
        # Select top-scoring optimizers for exploitation
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        exploited_optimizers = [name for name, _ in sorted_candidates[:n_exploit]]
        
        # Select remaining optimizers randomly for exploration
        remaining_optimizers = [name for name in self.optimizers.keys() 
                               if name not in exploited_optimizers]
        
        explored_optimizers = []
        if remaining_optimizers and n_explore > 0:
            explored_optimizers = list(np.random.choice(
                remaining_optimizers,
                size=min(n_explore, len(remaining_optimizers)),
                replace=False
            ))
        
        # Combine and return selected optimizers
        selected_optimizers = exploited_optimizers + explored_optimizers
        self.logger.info(f"Selected optimizers: {selected_optimizers}")
        
        return selected_optimizers
    
    def _update_feature_based_scores(self, candidates: Dict[str, float]) -> None:
        """
        Update candidate scores based on problem features.
        
        Args:
            candidates: Dictionary of optimizer names to scores to update
        """
        # Try ML-based selection first if we have enough data
        if hasattr(self, 'ml_model') and self.ml_model is not None:
            try:
                ml_scores = self._get_ml_model_scores()
                if ml_scores:
                    for opt_name, score in ml_scores.items():
                        if opt_name in candidates:
                            # Blend with existing score (70% ML, 30% existing)
                            candidates[opt_name] = 0.7 * score + 0.3 * candidates.get(opt_name, 0.0)
                    return  # ML model scores applied successfully
            except Exception as e:
                self.logger.warning(f"Error applying ML model for selection: {e}")
        
        # Feature-specific heuristics based on optimizer strengths
        features = self.current_features
        
        # Extract key features
        modality = features.get('modality', 1)
        ruggedness = features.get('ruggedness', 0)
        convexity = features.get('convexity', 0)
        dimension = features.get('dimension', 2)
        gradient_variance = features.get('gradient_variance', 0)
        basin_ratio = features.get('basin_ratio', 0)
        separability = features.get('separability', 0)
        
        # New features if available
        noise = features.get('noise_estimation', 0)
        periodicity = features.get('periodicity', 0)
        neutral_regions = features.get('neutral_regions', 0)
        evolvability = features.get('evolvability', 0.5)
        curvature = features.get('curvature', 0)
        pca_ratio = features.get('pca_variance_ratio', 0.5)
        
        # Apply heuristics for each optimizer type
        for opt_name in candidates.keys():
            score = 0.0
            
            # Evolutionary Strategy works well for:
            # - High dimensionality
            # - Moderate to high ruggedness
            # - Low to moderate modality
            if 'ES' in opt_name:
                score += 0.5 * min(1.0, dimension / 10)  # Higher score for higher dimensions
                score += 0.3 * ruggedness  # Better for rugged landscapes
                score += 0.2 * (1 - modality / 10)  # Better for fewer local optima
                score += 0.2 * basin_ratio  # Does well with distinct basins
                # New feature relationships
                score += 0.2 * noise  # Good at handling noise
                score += 0.1 * (1 - neutral_regions)  # Better with clear gradient information
                
            # Differential Evolution works well for:
            # - Moderate dimensionality
            # - High modality problems
            # - Good separability
            elif 'DE' in opt_name:
                score += 0.3 * min(1.0, dimension / 5)  # Good up to moderate dimensions
                score += 0.4 * (modality / 10)  # Better for multimodal problems
                score += 0.3 * separability  # Good for separable problems
                score += 0.2 * convexity  # Does well with convex problems
                # New feature relationships
                score += 0.3 * pca_ratio  # Good when problem has principal component structure
                score += 0.2 * evolvability  # Performs well when incremental improvements help
                
            # Grey Wolf works well for:
            # - Low to moderate dimensionality
            # - Moderate ruggedness
            # - Good gradient information
            elif 'GWO' in opt_name:
                score += 0.5 * max(0, 1 - dimension / 10)  # Better for lower dimensions
                score += 0.3 * (1 - gradient_variance)  # Better with consistent gradients
                score += 0.2 * (1 - ruggedness)  # Better for smoother landscapes
                # New feature relationships
                score += 0.3 * evolvability  # Good when local improvements help
                score += 0.2 * (1 - noise)  # Less effective with high noise
                
            # Ant Colony works well for:
            # - Discrete-like problems
            # - High ruggedness
            # - Many local optima
            elif 'ACO' in opt_name:
                score += 0.5 * ruggedness  # Great for rugged landscapes
                score += 0.3 * (modality / 10)  # Good for many local optima
                score += 0.2 * (1 - separability)  # Better for non-separable problems
                # New feature relationships
                score += 0.4 * periodicity  # Excellent for problems with periodic patterns
                score += 0.2 * neutral_regions  # Good at navigating neutral regions
                
            # Add score component for adaptive algorithms
            if 'Adaptive' in opt_name:
                score += 0.1  # Small bonus for adaptive variants
                
            candidates[opt_name] += score
    
    def _get_ml_model_scores(self) -> Dict[str, float]:
        """
        Get algorithm scores from trained ML model.
        
        Returns:
            Dictionary mapping optimizer names to predicted scores
        """
        if not hasattr(self, 'ml_model') or self.ml_model is None:
            return {}
            
        if not self.current_features:
            return {}
            
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector()
            if feature_vector is None:
                return {}
                
            # Get model predictions
            scores = self.ml_model.predict_proba(feature_vector.reshape(1, -1))[0]
            
            # Map scores to optimizer names
            result = {}
            for i, opt_name in enumerate(self.ml_model.classes_):
                if opt_name in self.optimizers:
                    result[opt_name] = float(scores[i])
            
            return result
        except Exception as e:
            self.logger.warning(f"Error getting ML model scores: {e}")
            return {}
    
    def _prepare_feature_vector(self) -> np.ndarray:
        """
        Prepare feature vector for ML model.
        
        Returns:
            Numpy array with scaled features
        """
        if not self.current_features:
            return None
            
        # Define standard feature set (must match what model was trained on)
        standard_features = [
            'dimension', 'modality', 'ruggedness', 'convexity', 
            'gradient_variance', 'separability', 'basin_ratio',
            'noise_estimation', 'periodicity', 'evolvability', 
            'neutral_regions', 'curvature', 'pca_variance_ratio'
        ]
        
        # Extract features
        feature_values = []
        for feature in standard_features:
            if feature in self.current_features:
                feature_values.append(self.current_features[feature])
            else:
                # Use sensible defaults for missing features
                if feature == 'dimension':
                    feature_values.append(float(self.dim))
                else:
                    feature_values.append(0.5)  # Default middle value
        
        return np.array(feature_values)
    
    def train_selection_model(self) -> None:
        """
        Train ML model for algorithm selection using historical data.
        
        This builds a simple classifier to predict which optimizer is most likely
        to succeed based on problem features.
        """
        # Check if we have enough historical data
        if not hasattr(self, 'history') or not self.history or len(self.history.records) < 10:
            self.logger.warning("Not enough history data to train selection model")
            return
            
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Prepare training data
            X = []  # Features
            y = []  # Optimizer labels
            
            for record in self.history.records:
                if 'features' not in record or 'optimizer' not in record:
                    continue
                    
                # Prepare feature vector
                features = record['features']
                if not features:
                    continue
                    
                feature_vector = []
                for feature in ['dimension', 'modality', 'ruggedness', 'convexity', 
                              'gradient_variance', 'separability', 'basin_ratio']:
                    feature_vector.append(features.get(feature, 0.5))
                    
                # Add newer features if available
                for feature in ['noise_estimation', 'periodicity', 'evolvability', 
                              'neutral_regions', 'curvature', 'pca_variance_ratio']:
                    feature_vector.append(features.get(feature, 0.5))
                
                X.append(feature_vector)
                y.append(record['optimizer'])
            
            if len(X) < 10 or len(set(y)) < 2:
                self.logger.warning("Insufficient diverse data to train selection model")
                return
                
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_scaled, y)
            
            # Save model and scaler
            self.ml_model = model
            self.feature_scaler = scaler
            
            self.logger.info(f"Trained selection model with {len(X)} samples")
            
            # Evaluate feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = ['dimension', 'modality', 'ruggedness', 'convexity', 
                              'gradient_variance', 'separability', 'basin_ratio',
                              'noise_estimation', 'periodicity', 'evolvability', 
                              'neutral_regions', 'curvature', 'pca_variance_ratio']
                
                self.logger.debug("Feature importances:")
                for name, importance in zip(feature_names, importances):
                    self.logger.debug(f"  {name}: {importance:.4f}")
        
        except Exception as e:
            self.logger.error(f"Error training selection model: {e}")
            self.ml_model = None

    def _update_history_based_scores(self, candidates: Dict[str, float]) -> None:
        """
        Update candidate scores based on optimization history.
        
        Args:
            candidates: Dictionary of optimizer names to scores to update
        """
        # Find similar problems in history
        similar_problems = self.history.find_similar_problems(
            self.current_features, 
            k=min(10, len(self.history.records))
        )
        
        if not similar_problems:
            return
            
        # Weight by similarity score
        for similarity, record in similar_problems:
            optimizer = record['optimizer']
            if optimizer in candidates:
                # Apply weights for performance and similarity
                performance_weight = 1.0 if record['success'] else 0.5
                candidates[optimizer] += similarity * performance_weight
    
    def _update_tracker_based_scores(self, candidates: Dict[str, float]) -> None:
        """
        Update candidate scores based on selection tracker data.
        
        Args:
            candidates: Dictionary of optimizer names to scores to update
        """
        # Get feature correlations for this problem type
        correlations = self.selection_tracker.get_feature_correlations(self.current_problem_type)
        
        if not correlations:
            return
            
        # Calculate feature-weighted scores for each optimizer
        for opt_name, feat_corrs in correlations.items():
            if opt_name not in candidates:
                continue
                
            score = 0.0
            for feat, corr in feat_corrs.items():
                if feat in self.current_features:
                    # Weight the feature by its correlation with success
                    feature_value = self.current_features[feat]
                    score += feature_value * corr
                    
            candidates[opt_name] += score

    def _run_single_optimizer(self, optimizer_name: str, objective_func: Callable, max_evals: int) -> OptimizationResult:
        """Run a single optimizer and return its result."""
        optimizer = self.optimizers[optimizer_name]
        
        # Handle both object and function interfaces
        try:
            # Reset the optimizer if it's an object with reset method
            if hasattr(optimizer, 'reset'):
                optimizer.reset()
            
            # Different optimizers have different interfaces
            if hasattr(optimizer, 'optimize'):
                # Object with optimize method
                solution, score = optimizer.optimize(objective_func, max_evals)
                n_evals = getattr(optimizer, 'evaluations', max_evals)
            else:
                # Function that takes (func, bounds, max_evals)
                solution, score, n_evals = optimizer(objective_func, self.bounds, max_evals)
            
            success = score < float('inf')
            return OptimizationResult(
                optimizer_name=optimizer_name,
                solution=solution,
                score=score,
                n_evals=n_evals,
                success=success
            )
        except Exception as e:
            self.logger.error(f"Optimizer {optimizer_name} failed: {str(e)}")
            return OptimizationResult(
                optimizer_name=optimizer_name,
                solution=np.zeros(self.dim),  # Use zeros of correct dimensionality
                score=float('inf'),
                n_evals=0,
                success=False
            )

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
                max_evaluations: Optional[int] = None,
                context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run optimization with all configured optimizers.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum number of function evaluations
            max_evaluations: Alias for max_evals (for compatibility)
            context: Optional context information
            
        Returns:
            Best solution found (numpy array)
        """
        # Use max_evaluations as an alias for max_evals if provided
        if max_evaluations is not None and max_evals is None:
            max_evals = max_evaluations
            
        # Use default max evaluations if not specified
        max_evals = max_evals or self.default_max_evals
        
        # Set initial parameters
        self.reset()
        
        # Handle benchmark function objects that aren't callable
        if hasattr(objective_func, 'evaluate'):
            # Wrap the evaluate method in a callable function
            self.logger.info(f"Wrapping benchmark function {getattr(objective_func, 'name', 'unknown')}")
            original_func = objective_func
            objective_func = lambda x: original_func.evaluate(x)
            
            # Set problem dimensionality if available
            if hasattr(original_func, 'dims') and original_func.dims != self.dim:
                self.logger.info(f"Updating dimensionality from {self.dim} to {original_func.dims}")
                self.dim = original_func.dims
                
            # Set bounds if available
            if hasattr(original_func, 'bounds') and len(original_func.bounds) == 2:
                low, high = original_func.bounds
                self.bounds = [(low, high)] * self.dim
                self.logger.info(f"Using bounds from benchmark function: {self.bounds[0]}")
        
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
                
        # Main optimization loop with progress bar
        with tqdm(total=max_evals, desc=f"Meta Optimization ({self.current_problem_type or 'unknown'})", 
                  unit="evals", position=0, leave=True, 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                  dynamic_ncols=True) as pbar:
            pbar.n = self.total_evaluations
            pbar.set_postfix_str(f"Best: {self.best_score:.2e} | Iter: {self._current_iteration}")
            pbar.refresh()
            
            # Store progress bar for updates
            self._progress_bar = pbar
            
            while self.total_evaluations < max_evals:
                self._current_iteration += 1
                
                # Calculate remaining evaluations
                remaining_evals = max_evals - self.total_evaluations
                if remaining_evals <= 0:
                    continue
                    
                # Select optimizer to use for this iteration based on problem features
                selected_optimizers = self._select_optimizer(context or {})
                self.most_recent_selected_optimizers = selected_optimizers
                
                # Check if we need to select at least one
                if not selected_optimizers:
                    # Fallback: select a random optimizer
                    selected_optimizers = [np.random.choice(list(self.optimizers.keys()))]
                self.logger.debug(f"Selected optimizers: {selected_optimizers}")
                
                # Calculate budget for this iteration
                iteration_budget = min(self.budget_per_iteration, remaining_evals)
                
                # Run each selected optimizer for a portion of the budget
                per_optimizer_budget = iteration_budget // len(selected_optimizers)
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
                            objective_func,
                            per_optimizer_budget
                        )] = optimizer_name
                
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(optimizer_futures):
                        optimizer_name = optimizer_futures[future]
                        try:
                            result = future.result()
                            if result and result.solution is not None:
                                # Update best solution if this optimizer found a better one
                                if result.score < self.best_score:
                                    self.best_score = result.score
                                    self.best_solution = result.solution.copy()
                                    self.logger.info(
                                        f"New best solution from {optimizer_name}: {self.best_score:.10f}"
                                    )
                                    # Update progress bar with new best score
                                    pbar.set_postfix_str(f"Best: {self.best_score:.2e} | Iter: {self._current_iteration} | Opt: {optimizer_name}")
                                    
                                # Track evaluations and update progress
                                self.total_evaluations += result.n_evals
                                pbar.n = self.total_evaluations
                                pbar.refresh()
                                
                                # Update progress bar description
                                runtime = time.time() - self.start_time
                                evals_per_sec = self.total_evaluations / max(runtime, 0.001)
                                pbar.set_postfix({
                                    "best": f"{self.best_score:.2e}",
                                    "opt": optimizer_name.split('Optimizer')[0],
                                    "iter": self._current_iteration,
                                    "success": f"{result.success * 100:.0f}%",
                                    "e/s": f"{evals_per_sec:.1f}"
                                })
                                
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
                                if hasattr(self, 'selection_tracker') and self.selection_tracker:
                                    self.selection_tracker.record_selection(
                                        problem_type=self.current_problem_type,
                                        optimizer=optimizer_name,
                                        features=self.current_features,
                                        success=result.success,
                                        score=result.score
                                    )
                        except Exception as e:
                            self.logger.error(f"Error processing result from {optimizer_name}: {str(e)}")
                            continue
                
                # Check if we've converged
                if self._current_iteration > 1 and len(self.convergence_curve) > 1:
                    prev_score = self.convergence_curve[-2][1]
                    curr_score = self.convergence_curve[-1][1]
                    improvement = prev_score - curr_score
                    
                    # If improvement is very small, we might have converged
                    if improvement < 1e-8 * prev_score:
                        self.logger.info(f"Convergence detected after {self._current_iteration} iterations")
                        return self.best_solution
        
        # Record end time and cleanup
        self.end_time = time.time()
        runtime = self.end_time - self.start_time
        
        # Calculate success metrics
        success_count = sum(1 for h in self.optimization_history if h.get('success', False))
        total_runs = len(self.optimization_history)
        success_rate = (success_count / total_runs * 100) if total_runs > 0 else 0
        
        # Log final results
        self.logger.info("Optimization Summary:")
        self.logger.info(f"  Runtime: {runtime:.2f} seconds")
        self.logger.info(f"  Total evaluations: {self.total_evaluations}")
        self.logger.info(f"  Best score: {self.best_score:.10f}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        self.logger.info(f"  Iterations: {self._current_iteration}")
        
        # Update visualization if enabled
        if self.enable_viz and self.live_viz_monitor:
            self.live_viz_monitor.flush()
            if self.save_viz_path:
                self.live_viz_monitor.save(self.save_viz_path)
        
        # After optimization completes, update the ML selection model
        if self.use_ml_selection and self.optimization_history:
            try:
                # Only train if we have new data
                if len(self.history.records) > 0:
                    last_record_time = self.history.records[-1].get('timestamp', 0)
                    if time.time() - last_record_time < 60 * 60:  # Only if data newer than 1 hour
                        self.train_selection_model()
                        self.logger.info("Updated algorithm selection model after optimization")
            except Exception as e:
                self.logger.warning(f"Error updating selection model: {e}")
        
        # Return best solution
        return self.best_solution

    def run(self, objective_func: Optional[Callable] = None, max_evals: Optional[int] = None, record_history: bool = True, export_data: bool = False, export_format: str = 'json', export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the optimizer with the given objective function.
        
        Args:
            objective_func: Objective function to minimize
            max_evals: Maximum function evaluations (uses default if None)
            record_history: Whether to record optimization history
            export_data: Whether to export optimization data
            export_format: Format for data export ('json', 'csv', or 'both')
            export_path: Path to save exported data (uses 'results/optimization_data' if None)
            
        Returns:
            Dictionary with optimization results
        """
        # Use stored objective function if none provided
        if objective_func is None:
            if hasattr(self, 'objective_func'):
                objective_func = self.objective_func
            else:
                raise ValueError("No objective function provided or stored")
        
        # Store objective function
        self.objective_func = objective_func
        
        # Run optimization
        solution = self.optimize(objective_func, max_evals=max_evals)
        
        # Extract optimizer states
        optimizer_states = {}
        for name, optimizer in self.optimizers.items():
            if hasattr(optimizer, 'get_state'):
                optimizer_states[name] = optimizer.get_state()
        
        # Update selection strategy if needed
        if optimizer_states:
            self._update_selection_strategy(optimizer_states)
        
        # Prepare result dictionary
        result = {
            'solution': solution.tolist() if isinstance(solution, np.ndarray) else solution,
            'score': float(self.best_score) if hasattr(self, 'best_score') else None,
            'evaluations': self.total_evaluations,
            'iterations': self._current_iteration,
            'runtime': time.time() - self.start_time,
            'problem_type': self.current_problem_type,
            'best_optimizer': self.best_optimizer,
            'optimizer_states': {name: state.to_dict() for name, state in optimizer_states.items()},
            'success': hasattr(self, 'best_score') and self.best_score is not None
        }
        
        # Export data if requested
        if export_data:
            # Set default export path if none provided
            if export_path is None:
                export_path = os.path.join('results', 'optimization_data', 
                                          f"optimization_{time.strftime('%Y%m%d_%H%M%S')}")
            
            # Export data in the specified format
            export_file = self.export_data(
                filename=export_path,
                format=export_format,
                include_history=record_history,
                include_selections=True,
                include_parameters=True
            )
            
            # Add export path to result
            result['export_path'] = export_file
        
        return result

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
        if self.selection_tracker:
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
            try:
                if save_results and results_path:
                    # Make sure we're calling the correct method
                    self.live_viz_monitor.save_results(results_path)
                    
                if save_results and data_path:
                    self.live_viz_monitor.save_data(data_path)
            except Exception as e:
                self.logger.error(f"Error saving visualization results: {e}")
            
            # Always stop monitoring even if saving failed
            self.live_viz_monitor.stop_monitoring()
            self.live_viz_monitor = None
            self.enable_viz = False
            self.logger.info("Live optimization visualization disabled")
        
        # Generate algorithm selection visualizations if enabled
        if self.enable_algo_viz and self.algo_selection_viz and save_results:
            try:
                if not results_path and self.save_viz_path:
                    results_path = self.save_viz_path
                    
                # Create algorithm selection visualizations
                save_dir = os.path.dirname(os.path.abspath(results_path))
                self.algo_selection_viz.plot_selection_frequency(save=True, filename=os.path.join(save_dir, "algorithm_selection_frequency.png"))
                self.algo_selection_viz.plot_selection_timeline(save=True, filename=os.path.join(save_dir, "algorithm_selection_timeline.png"))
                self.algo_selection_viz.plot_problem_distribution(save=True, filename=os.path.join(save_dir, "algorithm_selection_by_problem.png"))
                self.algo_selection_viz.plot_performance_comparison(save=True, filename=os.path.join(save_dir, "optimizer_performance_comparison.png"))
                self.algo_selection_viz.create_summary_dashboard(save=True, filename=os.path.join(save_dir, "algorithm_selection_dashboard.png"))
                
                self.logger.info("Algorithm selection visualizations saved")
            except Exception as e:
                self.logger.error(f"Error saving algorithm selection visualizations: {e}")
                import traceback
                traceback.print_exc()
            
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

    def _process_optimizer_result(self, result: OptimizationResult) -> None:
        """Process result from an optimizer."""
        if not result or result.solution is None:
            return
            
        # Update best solution if better
        if result.score < self.best_score:
            self.best_score = result.score
            self.best_solution = result.solution.copy()
            self.logger.info(f"New best solution from {result.optimizer_name}: {self.best_score}")
            
        try:
            # Record selection in selection tracker
            if self.selection_tracker and self.current_features:
                self.selection_tracker.record_selection(
                    problem_type=self.current_problem_type,
                    optimizer=result.optimizer_name,
                    features=self.current_features,
                    success=bool(result.success),  # Convert to standard Python bool
                    score=float(result.score)      # Convert to standard Python float
                )
                
            # Record selection in algorithm selection visualizer
            if hasattr(self, 'enable_algo_viz') and self.enable_algo_viz and hasattr(self, 'algo_selection_viz') and self.algo_selection_viz:
                # Prepare context with additional information
                context = {
                    'function_name': self.current_problem_type,
                    'phase': 'optimization',
                    'features': {k: float(v) if isinstance(v, np.generic) else v 
                               for k, v in self.current_features.items()} if self.current_features else {},
                    'success': bool(result.success),
                    'evaluations': int(result.n_evals)
                }
                
                self.algo_selection_viz.record_selection(
                    iteration=self._current_iteration,
                    optimizer=result.optimizer_name,
                    problem_type=self.current_problem_type or "unknown",
                    score=float(result.score),
                    context=context
                )
                self.logger.info(f"Recorded selection of optimizer {result.optimizer_name} for iteration {self._current_iteration}")
        except Exception as e:
            self.logger.error(f"Error recording selection result from {result.optimizer_name}: {str(e)}")

    def export_data(self, 
                   filename: str, 
                   format: str = 'json', 
                   include_history: bool = True,
                   include_selections: bool = True,
                   include_parameters: bool = True) -> str:
        """
        Export optimization data to file in specified format.
        
        Args:
            filename: Path to save the data (without extension)
            format: Export format ('json', 'csv', or 'both')
            include_history: Whether to include full optimization history
            include_selections: Whether to include algorithm selections
            include_parameters: Whether to include parameter adaptation history
            
        Returns:
            Path to the exported file(s)
        """
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Prepare data for export
        export_data = self._prepare_export_data(
            include_history=include_history,
            include_selections=include_selections,
            include_parameters=include_parameters
        )
        
        # Export in the specified format
        if format.lower() == 'json' or format.lower() == 'both':
            self._export_to_json(f"{filename}.json", export_data)
            self.logger.info(f"Exported optimization data to {filename}.json")
        
        if format.lower() == 'csv' or format.lower() == 'both':
            self._export_to_csv(f"{filename}", export_data)
            self.logger.info(f"Exported optimization data to CSV files in {os.path.dirname(filename)}")
        
        return filename

    def _prepare_export_data(self, 
                           include_history: bool = True,
                           include_selections: bool = True,
                           include_parameters: bool = True) -> Dict[str, Any]:
        """
        Prepare data for export.
        
        Args:
            include_history: Whether to include full optimization history
            include_selections: Whether to include algorithm selections
            include_parameters: Whether to include parameter adaptation history
            
        Returns:
            Dictionary with data for export
        """
        # Basic information about the optimization run
        data = {
            'optimization_info': {
                'dimensions': self.dim,
                'bounds': self.bounds,
                'total_evaluations': self.total_evaluations,
                'best_score': float(self.best_score) if hasattr(self, 'best_score') and self.best_score is not None else None,
                'best_solution': self.best_solution.tolist() if hasattr(self, 'best_solution') and self.best_solution is not None else None,
                'runtime': time.time() - self.start_time if hasattr(self, 'start_time') and self.start_time is not None else 0,
                'iterations': self._current_iteration if hasattr(self, '_current_iteration') else 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'problem_type': self.current_problem_type if hasattr(self, 'current_problem_type') else "unknown"
            }
        }
        
        # Include optimization history if requested
        if include_history and hasattr(self, 'optimization_history'):
            data['optimization_history'] = []
            for entry in self.optimization_history:
                # Convert numpy arrays to lists for serialization
                history_entry = {}
                for key, value in entry.items():
                    if isinstance(value, np.ndarray):
                        history_entry[key] = value.tolist()
                    elif isinstance(value, np.generic):
                        history_entry[key] = value.item()
                    else:
                        history_entry[key] = value
                data['optimization_history'].append(history_entry)
        
        # Include algorithm selections if requested
        if include_selections and hasattr(self, 'selection_tracker') and self.selection_tracker:
            data['algorithm_selections'] = self.selection_tracker.get_history()
        
        # Include parameter adaptation history if requested
        if include_parameters and hasattr(self, 'optimizers'):
            data['parameter_history'] = {}
            for name, optimizer in self.optimizers.items():
                if hasattr(optimizer, 'param_history') and optimizer.param_history:
                    param_history = {}
                    for param, values in optimizer.param_history.items():
                        # Convert numpy values to Python types
                        if isinstance(values, list):
                            param_history[param] = [float(v) if isinstance(v, np.generic) else v for v in values]
                        else:
                            param_history[param] = float(values) if isinstance(values, np.generic) else values
                    data['parameter_history'][name] = param_history
        
        # Include problem features if available
        if hasattr(self, 'current_features') and self.current_features is not None:
            data['problem_features'] = {k: float(v) if isinstance(v, np.generic) else v 
                                      for k, v in self.current_features.items()}
        
        return data

    def _export_to_json(self, filename: str, data: Dict[str, Any]) -> None:
        """
        Export data to JSON file.
        
        Args:
            filename: Path to save the JSON file
            data: Data to export
        """
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_to_csv(self, base_filename: str, data: Dict[str, Any]) -> None:
        """
        Export data to CSV files.
        
        Args:
            base_filename: Base path for CSV files
            data: Data to export
        """
        import pandas as pd
        
        # Export optimization info
        pd.DataFrame([data['optimization_info']]).to_csv(f"{base_filename}_info.csv", index=False)
        
        # Export optimization history if available
        if 'optimization_history' in data and data['optimization_history']:
            pd.DataFrame(data['optimization_history']).to_csv(f"{base_filename}_history.csv", index=False)
        
        # Export algorithm selections if available
        if 'algorithm_selections' in data and data['algorithm_selections']:
            pd.DataFrame(data['algorithm_selections']).to_csv(f"{base_filename}_selections.csv", index=False)
        
        # Export parameter history if available
        if 'parameter_history' in data and data['parameter_history']:
            # Each optimizer gets its own CSV file
            for optimizer_name, params in data['parameter_history'].items():
                # Convert to DataFrame-friendly format
                param_data = {}
                max_length = 0
                for param_name, values in params.items():
                    if isinstance(values, list):
                        param_data[param_name] = values
                        max_length = max(max_length, len(values))
                    else:
                        param_data[param_name] = [values]
                        max_length = max(max_length, 1)
                
                # Ensure all lists have the same length
                for param_name, values in param_data.items():
                    if len(values) < max_length:
                        param_data[param_name] = values + [None] * (max_length - len(values))
                
                # Save to CSV
                if param_data:
                    pd.DataFrame(param_data).to_csv(f"{base_filename}_{optimizer_name}_params.csv", index=False)
        
        # Export problem features if available
        if 'problem_features' in data and data['problem_features']:
            pd.DataFrame([data['problem_features']]).to_csv(f"{base_filename}_features.csv", index=False)

    def import_data(self, 
                   filename: str,
                   restore_state: bool = True,
                   restore_optimizers: bool = True) -> Dict[str, Any]:
        """
        Import optimization data from a file.
        
        Args:
            filename: Path to the data file (JSON) or base path for CSV files
            restore_state: Whether to restore the meta-optimizer state
            restore_optimizers: Whether to restore individual optimizer states
            
        Returns:
            The imported data
        """
        # Determine file format and import data
        if filename.endswith('.json'):
            data = self._import_from_json(filename)
        else:
            # Assume CSV format
            data = self._import_from_csv(filename.rstrip('.csv'))
        
        # Restore meta-optimizer state if requested
        if restore_state:
            self._restore_meta_optimizer_state(data)
        
        # Restore individual optimizer states if requested
        if restore_optimizers:
            self._restore_optimizer_states(data)
        
        return data

    def _import_from_json(self, filename: str) -> Dict[str, Any]:
        """
        Import data from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            The imported data
        """
        import json
        
        # Make sure the file exists
        if not os.path.exists(filename):
            self.logger.error(f"File not found: {filename}")
            return {}
        
        # Load the data
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.logger.info(f"Imported data from {filename}")
            return data
        except Exception as e:
            self.logger.error(f"Error importing data from {filename}: {e}")
            return {}

    def _import_from_csv(self, base_filename: str) -> Dict[str, Any]:
        """
        Import data from CSV files.
        
        Args:
            base_filename: Base path for CSV files
            
        Returns:
            The imported data
        """
        import pandas as pd
        
        # Initialize data structure
        data = {
            "optimization_info": {},
            "optimization_history": [],
            "algorithm_selections": [],
            "parameter_history": {},
            "problem_features": {}
        }
        
        # Check if the directory exists
        base_dir = os.path.dirname(base_filename)
        if not os.path.exists(base_dir):
            self.logger.error(f"Directory not found: {base_dir}")
            return data
        
        # Import optimization info
        info_path = f"{base_filename}_info.csv"
        if os.path.exists(info_path):
            try:
                df_info = pd.read_csv(info_path)
                if not df_info.empty:
                    data["optimization_info"] = df_info.iloc[0].to_dict()
            except Exception as e:
                self.logger.error(f"Error importing optimization info: {e}")
        
        # Import optimization history
        history_path = f"{base_filename}_history.csv"
        if os.path.exists(history_path):
            try:
                df_history = pd.read_csv(history_path)
                data["optimization_history"] = df_history.to_dict('records')
            except Exception as e:
                self.logger.error(f"Error importing optimization history: {e}")
        
        # Import algorithm selections
        selections_path = f"{base_filename}_selections.csv"
        if os.path.exists(selections_path):
            try:
                df_selections = pd.read_csv(selections_path)
                data["algorithm_selections"] = df_selections.to_dict('records')
            except Exception as e:
                self.logger.error(f"Error importing algorithm selections: {e}")
        
        # Import problem features
        features_path = f"{base_filename}_features.csv"
        if os.path.exists(features_path):
            try:
                df_features = pd.read_csv(features_path)
                if not df_features.empty:
                    data["problem_features"] = df_features.iloc[0].to_dict()
            except Exception as e:
                self.logger.error(f"Error importing problem features: {e}")
        
        # Import parameter history for each optimizer
        base_name = os.path.basename(base_filename)
        try:
            optimizer_files = [f for f in os.listdir(base_dir) 
                             if f.startswith(base_name) and f.endswith("_params.csv")]
            
            for opt_file in optimizer_files:
                # Extract optimizer name from filename
                optimizer_name = opt_file.replace(base_name + "_", "").replace("_params.csv", "")
                
                # Read parameter data
                param_path = os.path.join(base_dir, opt_file)
                df_params = pd.read_csv(param_path)
                
                # Convert to dictionary structure
                params_dict = {}
                for column in df_params.columns:
                    values = df_params[column].dropna().tolist()
                    if len(values) == 1:
                        params_dict[column] = values[0]
                    else:
                        params_dict[column] = values
                
                data["parameter_history"][optimizer_name] = params_dict
        except Exception as e:
            self.logger.error(f"Error importing parameter history: {e}")
        
        self.logger.info(f"Imported data from CSV files in {base_dir}")
        return data

    def _restore_meta_optimizer_state(self, data: Dict[str, Any]) -> None:
        """
        Restore meta-optimizer state from imported data.
        
        Args:
            data: Imported data
        """
        # Extract basic information
        info = data.get("optimization_info", {})
        
        # Restore basic parameters
        if "dimensions" in info:
            self.dim = int(info["dimensions"])
        
        if "bounds" in info:
            bounds = info["bounds"]
            # Convert bounds if needed
            if not isinstance(bounds[0], tuple) and len(bounds) == self.dim * 2:
                self.bounds = [(bounds[i*2], bounds[i*2+1]) for i in range(self.dim)]
            else:
                self.bounds = bounds
        
        # Restore best solution and score
        if "best_solution" in info and "best_score" in info:
            self.best_solution = np.array(info["best_solution"])
            self.best_score = float(info["best_score"])
        
        # Restore problem features and type
        if "problem_features" in data:
            self.current_features = data["problem_features"]
            self.current_problem_type = info.get("problem_type", "unknown")
        
        # Restore optimization history
        if "optimization_history" in data:
            self.optimization_history = data["optimization_history"]
        
        # Restore algorithm selections
        if "algorithm_selections" in data and hasattr(self, "selection_tracker") and self.selection_tracker:
            self.selection_tracker.history = data["algorithm_selections"]
        
        self.logger.info("Restored meta-optimizer state from imported data")

    def _restore_optimizer_states(self, data: Dict[str, Any]) -> None:
        """
        Restore individual optimizer states from imported data.
        
        Args:
            data: Imported data
        """
        # Restore parameter history for each optimizer
        for opt_name, params in data.get("parameter_history", {}).items():
            if opt_name in self.optimizers:
                optimizer = self.optimizers[opt_name]
                
                # Restore parameters
                for param_name, value in params.items():
                    if hasattr(optimizer, param_name):
                        try:
                            setattr(optimizer, param_name, value)
                        except Exception as e:
                            self.logger.warning(f"Error setting {param_name} for {opt_name}: {e}")
                
                # Set parameter history
                if hasattr(optimizer, "param_history"):
                    optimizer.param_history = params
        
        self.logger.info("Restored optimizer states from imported data")

    def _random_search(self, objective_func, bounds, max_evals):
        """Fallback random search implementation"""
        self.logger.warning("Using fallback random search algorithm")
        best_x = None
        best_y = float('inf')
        
        # Generate random points
        for i in range(max_evals):
            # Generate random point
            x = np.array([np.random.uniform(low, high) for low, high in bounds])
            
            # Evaluate
            y = objective_func(x)
            
            # Update best
            if y < best_y:
                best_y = y
                best_x = x
                
        return best_x, best_y, max_evals

    def _log_evaluation_stats(self, optimizer, result, problem, final_eval_count):
        """Log stats about the optimizer's evaluation performance."""
        if self.verbose:
            logging.info(f"Optimizer {optimizer.name} finished with {final_eval_count} evaluations")
            logging.info(f"Best fitness found: {result[1]}")
            logging.info(f"Remaining budget: {self.budget_per_iteration - final_eval_count}")
    
    def get_selected_algorithm(self) -> str:
        """Return the name of the most recently selected algorithm(s).
        
        If multiple optimizers were used in parallel, returns a comma-separated list.
        If no optimizer was selected yet, returns 'unknown'.
        """
        if hasattr(self, 'most_recent_selected_optimizers') and self.most_recent_selected_optimizers:
            self.logger.debug(f"Selected optimizers: {self.most_recent_selected_optimizers}")
            if isinstance(self.most_recent_selected_optimizers, list) and len(self.most_recent_selected_optimizers) > 0:
                # This is the path that should be taken
                names = []
                for opt_name in self.most_recent_selected_optimizers:
                    # Handle string optimizer names
                    self.logger.debug(f"Optimizer name: {opt_name}, type: {type(opt_name)}")
                    if isinstance(opt_name, str):
                        names.append(opt_name)
                    elif hasattr(opt_name, 'name'):
                        names.append(opt_name.name)
                    else:
                        names.append(str(opt_name))
                
                result = ",".join(names)
                self.logger.debug(f"Returning joined optimizer names: {result}")
                return result
            else:
                # Fallback for non-list types
                self.logger.debug(f"most_recent_selected_optimizers is not a list: {type(self.most_recent_selected_optimizers)}")
                return str(self.most_recent_selected_optimizers)
        
        self.logger.debug("No optimizers selected yet, returning 'unknown'")
        return "unknown"

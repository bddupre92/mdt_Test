"""
meta_optimizer.py
---------------
Advanced meta-learning system for optimizer selection and adaptation.
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import warnings

class MetaOptimizer:
    def __init__(self, optimizers: Dict[str, Any], mode: str = 'bayesian'):
        """
        Initialize meta-optimizer
        
        Args:
            optimizers: Dictionary of optimizers to use
            mode: Selection mode ('bayesian' or 'bandit')
        """
        self.optimizers = optimizers
        self.mode = mode
        self.dim = next(iter(optimizers.values())).dim
        
        # Initialize performance history
        self.performance_history = pd.DataFrame(columns=[
            'optimizer',
            'problem_dim',
            'discrete_vars',
            'multimodal',
            'runtime',
            'score'
        ])
        
        # Initialize GP for Bayesian selection
        if mode == 'bayesian':
            kernel = 1.0 * RBF(
                length_scale=[1.0] * 3,
                length_scale_bounds=(1e-1, 1e3)
            )
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42
            )
            self.scaler = StandardScaler()
        
        # Performance tracking
        self.current_best = {name: float('inf') for name in optimizers}
        self.runtime_stats = {name: [] for name in optimizers}
        
    def select_optimizer(self, context: Dict[str, Any]) -> str:
        """
        Select best optimizer based on context and history.
        
        Args:
            context: Dictionary containing problem characteristics
                    (dimension, discrete/continuous, multimodality, etc.)
        """
        if self.mode == 'bayesian':
            return self._bayesian_selection(context)
        elif self.mode == 'bandit':
            return self._bandit_selection(context)
        else:  # rule-based
            return self._rule_based_selection(context)
    
    def _bayesian_selection(self, context: Dict[str, Any]) -> str:
        """Use Gaussian Process to predict best optimizer"""
        if len(self.performance_history) < len(self.optimizers) * 3:
            # Not enough data, use round-robin
            return list(self.optimizers.keys())[
                len(self.performance_history) % len(self.optimizers)
            ]
        
        try:
            # Prepare features for GP
            X = self.performance_history[[
                'problem_dim', 'discrete_vars', 'multimodal'
            ]].values
            y = -self.performance_history['score'].values  # negative for maximization
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Normalize scores
            y_mean = np.mean(y)
            y_std = np.std(y) if np.std(y) > 0 else 1.0
            y_norm = (y - y_mean) / y_std
            
            # Fit GP with normalized data and increased max_iter
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X_scaled, y_norm)
            
            # Predict performance for current context
            context_features = np.array([[
                context.get('dim', 2),
                context.get('discrete_vars', 0),
                context.get('multimodal', 0)
            ]])
            
            context_scaled = self.scaler.transform(context_features)
            
            # Get predictions for each optimizer
            predictions = []
            for name in self.optimizers:
                mean, std = self.gp.predict(context_scaled, return_std=True)
                # Use UCB acquisition with adaptive exploration
                beta = np.sqrt(2 * np.log(len(self.optimizers) * len(self.performance_history)))
                score = mean + beta * std
                predictions.append((name, score[0]))
            
            return max(predictions, key=lambda x: x[1])[0]
            
        except Exception as e:
            print(f"Warning: GP prediction failed ({str(e)}), falling back to round-robin")
            return list(self.optimizers.keys())[
                len(self.performance_history) % len(self.optimizers)
            ]
    
    def _bandit_selection(self, context: Dict[str, Any]) -> str:
        """Thompson sampling for optimizer selection"""
        if not self.performance_history.empty:
            # Calculate success rate for each optimizer
            success_rates = {}
            for name in self.optimizers:
                hist = self.performance_history[
                    self.performance_history['optimizer'] == name
                ]
                if len(hist) > 0:
                    # Beta distribution parameters
                    successes = sum(hist['score'] < hist['score'].mean())
                    failures = len(hist) - successes
                    # Sample from beta distribution
                    success_rates[name] = np.random.beta(successes + 1, failures + 1)
                else:
                    success_rates[name] = np.random.beta(1, 1)
            
            return max(success_rates.items(), key=lambda x: x[1])[0]
        
        return np.random.choice(list(self.optimizers.keys()))
    
    def _rule_based_selection(self, context: Dict[str, Any]) -> str:
        """Use domain knowledge rules to select optimizer"""
        dim = context['dim']
        is_discrete = context.get('discrete_vars', 0) > 0
        is_multimodal = context.get('multimodal', False)
        
        if is_discrete:
            return 'AntColonyOptimizer'  # Best for discrete problems
        elif is_multimodal and dim > 10:
            return 'EvolutionStrategy'  # Good for multimodal high-dim
        elif dim <= 10:
            return 'GreyWolfOptimizer'  # Efficient for lower dimensions
        else:
            return 'DifferentialEvolution'  # Robust general-purpose
    
    def update_performance(self, optimizer_name: str, problem_dim: int, discrete_vars: int, multimodal: int, runtime: float, score: float):
        """
        Update performance history for an optimizer.
        
        Args:
            optimizer_name: Name of the optimizer
            problem_dim: Problem dimension
            discrete_vars: Number of discrete variables
            multimodal: Whether the problem is multimodal
            runtime: Runtime in seconds
            score: Final objective value achieved
        """
        # Update performance history
        self.performance_history = pd.concat([
            self.performance_history,
            pd.DataFrame([{
                'optimizer': optimizer_name,
                'problem_dim': problem_dim,
                'discrete_vars': discrete_vars,
                'multimodal': multimodal,
                'runtime': runtime,
                'score': score
            }])
        ], ignore_index=True)
        
        # Update tracking
        self.current_best[optimizer_name] = min(
            self.current_best[optimizer_name],
            score
        )
        self.runtime_stats[optimizer_name].append(runtime)
    
    def optimize(self, objective_func: Callable, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Run optimization using meta-learning to select and adapt optimizers.
        
        Args:
            objective_func: Function to minimize
            context: Optional problem context
            
        Returns:
            Best solution found
        """
        if context is None:
            context = {}
            
        # Ensure context has required fields
        context = {
            'dim': context.get('dim', self.dim),
            'discrete_vars': context.get('discrete_vars', 0),
            'multimodal': context.get('multimodal', 0)
        }
        
        # Wrap objective function to ensure numpy array input/output
        def wrapped_objective(x):
            x = np.asarray(x).reshape(-1)  # Ensure 1D array
            return float(objective_func(x))  # Ensure scalar output
        
        # Select best optimizer for current context
        optimizer_name = self.select_optimizer(context)
        optimizer = self.optimizers[optimizer_name]
        
        # Run optimization
        start_time = time.time()
        try:
            solution = optimizer.optimize(wrapped_objective)
            solution = np.asarray(solution).reshape(-1)  # Ensure 1D array
            score = wrapped_objective(solution)
        except Exception as e:
            print(f"Optimization failed with {optimizer_name}: {str(e)}")
            # Try another optimizer
            remaining_optimizers = set(self.optimizers.keys()) - {optimizer_name}
            for name in remaining_optimizers:
                try:
                    optimizer = self.optimizers[name]
                    solution = optimizer.optimize(wrapped_objective)
                    solution = np.asarray(solution).reshape(-1)
                    score = wrapped_objective(solution)
                    optimizer_name = name  # Update selected optimizer
                    break
                except Exception as e2:
                    print(f"Backup optimization with {name} failed: {str(e2)}")
                    continue
            else:
                raise RuntimeError("All optimizers failed")
                
        runtime = time.time() - start_time
        
        # Update performance history
        self.update_performance(
            optimizer_name,
            context['dim'],
            context['discrete_vars'],
            context['multimodal'],
            runtime,
            score
        )
        
        return solution
    
    def get_optimizer_stats(self) -> Dict[str, Dict[str, float]]:
        """Return performance statistics for each optimizer"""
        stats = {}
        for name in self.optimizers:
            if self.runtime_stats[name]:
                stats[name] = {
                    'avg_runtime': np.mean(self.runtime_stats[name]),
                    'best_score': self.current_best[name],
                    'n_runs': len(self.runtime_stats[name])
                }
        return stats
    
    def get_convergence_curve(self) -> List[float]:
        """Get convergence curve for the last optimization run"""
        if hasattr(self, 'current_optimizer'):
            return self.current_optimizer.get_convergence_curve()
        return []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the meta-optimizer"""
        return {
            'performance_history': self.performance_history,
            'current_best': self.current_best,
            'runtime_stats': self.runtime_stats
        }

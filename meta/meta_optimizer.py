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
import logging

class MetaOptimizer:
    def __init__(self, optimizers: Dict[str, Any], mode: str = 'bayesian', gp_kwargs: Dict[str, Any] = None):
        """
        Initialize meta-optimizer
        
        Args:
            optimizers: Dictionary of optimizers to use
            mode: Selection mode ('bayesian' or 'bandit')
            gp_kwargs: Additional arguments for Gaussian Process
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
            'score',
            'iteration'
        ])
        
        # Initialize GP for Bayesian selection
        if mode == 'bayesian':
            gp_kwargs = gp_kwargs or {}
            kernel = 1.0 * RBF(
                length_scale=[1.0] * 3,
                length_scale_bounds=(1e-1, 1e3)
            )
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=gp_kwargs.get('alpha', 1e-6),
                normalize_y=gp_kwargs.get('normalize_y', True),
                n_restarts_optimizer=gp_kwargs.get('n_restarts_optimizer', 5),
                random_state=42
            )
            self.scaler = StandardScaler()
        
        # Performance tracking
        self.current_best = {name: float('inf') for name in optimizers}
        self.runtime_stats = {name: [] for name in optimizers}
        self._current_iteration = 0
        
        # Initialize history storage
        self.history = []
        self.diversity_history = []
        self.param_history = {
            'exploitation_ratio': [],
            'length_scale': []
        }
        
        # Performance history
        self._performance_history = pd.DataFrame(columns=['iteration', 'score', 'optimizer'])
        
    def get_performance_history(self) -> pd.DataFrame:
        """Get optimizer performance history"""
        return self._performance_history
        
    def reset(self):
        """Reset optimizer state"""
        self.history = []
        self.diversity_history = []
        self.param_history = {
            'exploitation_ratio': [],
            'length_scale': []
        }
        self._performance_history = pd.DataFrame(columns=['iteration', 'score', 'optimizer'])
        
    def _update_history(self, optimizer: str, context: Dict[str, Any], runtime: float, score: float):
        """Update performance history with new result"""
        new_record = pd.DataFrame({
            'optimizer': [optimizer],
            'problem_dim': [context.get('dim', self.dim)],
            'discrete_vars': [context.get('discrete_vars', 0)],
            'multimodal': [context.get('multimodal', 0)],
            'runtime': [runtime],
            'score': [score],
            'iteration': [self._current_iteration]
        })
        
        self.performance_history = pd.concat([
            self.performance_history,
            new_record
        ], ignore_index=True)
        
        self._current_iteration += 1
        
        # Update optimizer stats
        if score < self.current_best[optimizer]:
            self.current_best[optimizer] = score
        self.runtime_stats[optimizer].append(runtime)
    
    def _optimize(self,
                objective_func: Callable,
                max_evals: Optional[int] = None,
                record_history: bool = True,
                context: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, float]:
        """Internal optimization method"""
        if max_evals is None:
            max_evals = 1000
            
        if context is None:
            context = {}
            
        best_solution = None
        best_score = float('inf')
        self._current_iteration = 0
        n_evals = 0
        
        # Initialize tracking
        self.current_best = {name: float('inf') for name in self.optimizers}
        
        while n_evals < max_evals:
            # Select optimizer
            selected_optimizer = self._select_optimizer(context)
            optimizer = self.optimizers[selected_optimizer]
            
            # Reset optimizer state
            if hasattr(optimizer, 'reset'):
                optimizer.reset()
            
            # Calculate remaining evaluations
            remaining_evals = max_evals - n_evals
            
            try:
                # Run optimization
                start_time = time.time()
                solution = optimizer.optimize(
                    objective_func,
                    max_evals=remaining_evals,
                    record_history=record_history,
                    context=context
                )
                runtime = time.time() - start_time
                
                # Evaluate solution
                if solution is not None:  # Check if solution exists
                    score = objective_func(solution)
                    
                    # Update history
                    self._update_history(selected_optimizer, context, runtime, score)
                    
                    # Update performance history
                    self._performance_history = pd.concat([
                        self._performance_history,
                        pd.DataFrame([{
                            'iteration': len(self._performance_history),
                            'score': score,
                            'optimizer': selected_optimizer
                        }])
                    ], ignore_index=True)
                    
                    # Update tracking
                    self.current_best[selected_optimizer] = min(
                        self.current_best[selected_optimizer],
                        score
                    )
                    
                    # Update best solution
                    if score < best_score:
                        best_score = score
                        best_solution = solution.copy()
                        
                    # Early stopping if we found a good solution
                    if best_score < 1e-4:
                        break
                        
                # Update evaluations
                if hasattr(optimizer, 'n_evals'):
                    n_evals += optimizer.n_evals
                else:
                    n_evals = max_evals  # Conservative estimate
                    
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
            scores = np.array([objective_func(x) for x in X])
            best_idx = np.argmin(scores)
            best_solution = X[best_idx]
            best_score = scores[best_idx]
            
        return best_solution, best_score
    
    def optimize(self, objective_func: callable, context: Dict[str, Any] = None) -> np.ndarray:
        """
        Optimize using meta-learning framework.
        
        Args:
            objective_func: Function to optimize
            context: Problem context (dimension, discrete/continuous, etc.)
            
        Returns:
            Best solution found
        """
        context = context or {'dim': self.dim}
        best_solution, _ = self._optimize(objective_func, context=context)
        return best_solution
    
    def _select_optimizer(self, context: Dict[str, Any]) -> str:
        """Select optimizer based on context and history"""
        if self.mode == 'bayesian':
            return self._select_optimizer_bayesian(context)
        else:
            return self._select_optimizer_random()
            
    def _select_optimizer_random(self) -> str:
        """Random optimizer selection"""
        return np.random.choice(list(self.optimizers.keys()))
        
    def _select_optimizer_bayesian(self, context: Dict[str, Any]) -> str:
        """Bayesian optimizer selection"""
        if self._performance_history.empty:
            # Try each optimizer at least once
            return np.random.choice(list(self.optimizers.keys()))
        
        # Use best performing optimizer more frequently
        best_scores = {}
        for opt in self.optimizers:
            opt_history = self._performance_history[
                self._performance_history['optimizer'] == opt
            ]
            if not opt_history.empty:
                best_scores[opt] = min(opt_history['score'].min(), 1e6)
            else:
                best_scores[opt] = 1e6
        
        # Convert to probabilities (lower score = higher probability)
        total = sum(1.0 / score for score in best_scores.values())
        probs = {
            opt: (1.0 / score) / total 
            for opt, score in best_scores.items()
        }
        
        return np.random.choice(
            list(probs.keys()),
            p=list(probs.values())
        )

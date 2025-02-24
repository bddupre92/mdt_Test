"""
meta_optimizer.py
---------------
Advanced meta-learning system for optimizer selection and adaptation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

class MetaOptimizer:
    def __init__(self, optimizers: Dict[str, Any], mode: str = 'bayesian'):
        """
        Initialize meta-optimizer with multiple optimization strategies.
        
        Args:
            optimizers: Dictionary mapping optimizer names to their instances
            mode: Strategy for optimizer selection ('bayesian', 'bandit', or 'rule')
        """
        self.optimizers = optimizers
        self.mode = mode
        
        # Initialize performance history with explicit dtypes
        self.performance_history = pd.DataFrame({
            'optimizer': pd.Series(dtype='string'),
            'problem_dim': pd.Series(dtype='int32'),
            'discrete_vars': pd.Series(dtype='int32'),
            'multimodal': pd.Series(dtype='int32'),
            'runtime': pd.Series(dtype='float64'),
            'score': pd.Series(dtype='float64'),
            'timestamp': pd.Series(dtype='datetime64[ns]')
        })
        
        # Bayesian optimization components
        self.gp = GaussianProcessRegressor()
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
        
        # Prepare features for GP
        X = self.performance_history[[
            'problem_dim', 'discrete_vars', 'multimodal'
        ]].values
        y = -self.performance_history['score'].values  # negative for maximization
        
        # Fit GP
        self.gp.fit(self.scaler.fit_transform(X), y)
        
        # Predict performance for current context
        context_features = np.array([[
            context['dim'],
            context.get('discrete_vars', 0),
            context.get('multimodal', 0)
        ]])
        
        predictions = []
        for name in self.optimizers:
            mean, std = self.gp.predict(
                self.scaler.transform(context_features), 
                return_std=True
            )
            # Use UCB acquisition
            score = mean + 2 * std
            predictions.append((name, score[0]))
        
        return max(predictions, key=lambda x: x[1])[0]
    
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
    
    def update_performance(self, 
                         optimizer_name: str,
                         context: Dict[str, Any],
                         runtime: float,
                         score: float):
        """Log optimizer performance for future selection"""
        # Create new performance data with explicit dtypes
        new_data = pd.DataFrame([{
            'optimizer': str(optimizer_name),
            'problem_dim': int(context['dim']),
            'discrete_vars': int(context.get('discrete_vars', 0)),
            'multimodal': int(context.get('multimodal', 0)),
            'runtime': float(runtime),
            'score': float(score),
            'timestamp': pd.Timestamp.now()
        }])
        
        # Ensure both DataFrames have the same schema
        for col in self.performance_history.columns:
            if col not in new_data.columns:
                new_data[col] = pd.NA
        
        # Convert dtypes to match self.performance_history
        for col in new_data.columns:
            if col in self.performance_history.columns:
                new_data[col] = new_data[col].astype(self.performance_history[col].dtype)
        
        # Concatenate with schema validation
        self.performance_history = pd.concat(
            [self.performance_history, new_data],
            ignore_index=True,
            verify_integrity=True,
            copy=False
        )
        
        # Update running statistics
        self.current_best[optimizer_name] = min(
            self.current_best[optimizer_name],
            score
        )
        self.runtime_stats[optimizer_name].append(runtime)
    
    def get_optimizer_stats(self) -> Dict[str, Dict[str, float]]:
        """Return performance statistics for each optimizer"""
        stats = {}
        for name in self.optimizers:
            hist = self.performance_history[
                self.performance_history['optimizer'] == name
            ]
            if not hist.empty:
                stats[name] = {
                    'best_score': self.current_best[name],
                    'avg_runtime': np.mean(self.runtime_stats[name]),
                    'success_rate': sum(hist['score'] < hist['score'].mean()) / len(hist),
                    'runs': len(hist)
                }
        return stats

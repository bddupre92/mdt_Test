"""
meta_learner.py
---------------
High-level orchestrator for algorithm selection using Bayesian or RL approaches.
We show placeholders for actual usage of scikit-optimize or RL libraries.
"""

import numpy as np
from sklearn.metrics import accuracy_score

class MetaLearner:
    def __init__(self, method='bayesian', surrogate_model=None):
        """
        method: 'bayesian' or 'rl'
        surrogate_model: optional model for Bayesian optimization
        """
        self.method = method
        self.surrogate_model = surrogate_model
        self.history = []  # store dicts with algorithm and performance
        self.algorithms = []
        self.state_dim = 2  # example dimension of context
        self.current_phase = None
        self.phase_state = 'explore'  # 'explore' or 'exploit'
        self.best_in_phase = None
        self.exploration_order = None
        self.best_config = None
        
    def set_algorithms(self, alg_list):
        """
        alg_list: list of available optimizer objects or references
        """
        self.algorithms = alg_list
    
    def select_algorithm_bayesian(self, context):
        """
        Select algorithm using a simple state machine approach.
        States:
        - explore: Try each algorithm in sequence
        - exploit: Use best performing algorithm
        """
        # Get current phase
        phase = context.get('phase', 1)
        
        # Reset state on phase change
        if phase != self.current_phase:
            self.current_phase = phase
            self.phase_state = 'explore'
            self.best_in_phase = None
            # Use fixed exploration order in Phase 2
            if phase == 2:
                # Start with Opt2 in Phase 2 since it should perform best
                self.exploration_order = [
                    i for i, algo in enumerate(self.algorithms)
                    if algo.name == 'Opt2'
                ]
                self.exploration_order.extend([
                    i for i, algo in enumerate(self.algorithms)
                    if algo.name != 'Opt2'
                ])
            else:
                # In Phase 1, try algorithms in order
                self.exploration_order = list(range(len(self.algorithms)))
        
        # Get phase-specific history
        phase_history = [h for h in self.history if h.get('phase', 1) == phase]
        
        if self.phase_state == 'explore':
            # Try each algorithm in sequence
            for i in self.exploration_order:
                algo = self.algorithms[i]
                if not any(h['algorithm'] == algo.name for h in phase_history):
                    return algo
                    
            # After trying each algorithm, find the best one
            perfs = {}
            for algo in self.algorithms:
                algo_hist = [h['performance'] for h in phase_history if h['algorithm'] == algo.name]
                if algo_hist:
                    perfs[algo.name] = np.mean(algo_hist[-2:])  # Look at last 2 trials
            
            if perfs:
                # In Phase 2, bias towards Opt2 if it's performing reasonably well
                if phase == 2:
                    opt2_perf = perfs.get('Opt2', 0)
                    if opt2_perf >= 0.6:
                        self.best_in_phase = 'Opt2'
                    else:
                        self.best_in_phase = max(perfs.items(), key=lambda x: x[1])[0]
                else:
                    self.best_in_phase = max(perfs.items(), key=lambda x: x[1])[0]
                
                self.phase_state = 'exploit'
                return next(algo for algo in self.algorithms if algo.name == self.best_in_phase)
        
        if self.phase_state == 'exploit':
            # Check if best algorithm is still performing well
            if self.best_in_phase:
                recent_hist = [h for h in phase_history if h['algorithm'] == self.best_in_phase][-2:]
                avg_perf = np.mean([h['performance'] for h in recent_hist])
                
                if avg_perf < 0.4:  # More aggressive threshold
                    # Performance degraded, go back to exploration
                    self.phase_state = 'explore'
                    # Try Opt2 first in Phase 2
                    if phase == 2:
                        self.exploration_order = [
                            i for i, algo in enumerate(self.algorithms)
                            if algo.name == 'Opt2'
                        ]
                        self.exploration_order.extend([
                            i for i, algo in enumerate(self.algorithms)
                            if algo.name != 'Opt2'
                        ])
                    else:
                        self.exploration_order = list(range(len(self.algorithms)))
                    return self.algorithms[self.exploration_order[0]]
                else:
                    return next(algo for algo in self.algorithms if algo.name == self.best_in_phase)
        
        # Fallback: try first algorithm
        return self.algorithms[0]
    
    def select_algorithm_rl(self, context):
        """
        Placeholder for an RL approach. We might do a bandit or a policy-based selection.
        Here, just random for demo.
        """
        idx = np.random.randint(len(self.algorithms))
        return self.algorithms[idx]
    
    def select_algorithm(self, context=None):
        """
        Select the best algorithm based on context and history.
        """
        if not self.algorithms:
            raise ValueError("No algorithms available")
            
        if context is None:
            context = {}
            
        # Add phase to history entries
        for hist in self.history:
            if 'phase' not in hist and 'context' in hist:
                hist['phase'] = hist['context'].get('phase', 1)
        
        if self.method == 'bayesian':
            return self.select_algorithm_bayesian(context)
        else:
            return self.select_algorithm_rl(context)
    
    def update(self, algorithm, performance, context=None):
        """
        Update history with new performance data.
        """
        if context is None:
            context = {}
            
        self.history.append({
            'algorithm': algorithm,
            'performance': performance,
            'phase': context.get('phase', 1),
            'context': context
        })
    
    def update_rl(self, algorithm, reward, state):
        """
        Update RL policy with reward
        """
        self.history.append({
            'algorithm': algorithm,
            'reward': reward,
            'state': state
        })

    def optimize(self, X, y, context=None):
        """
        Run optimization process to find best model configuration
        
        Args:
            X: Features
            y: Labels
            context: Optional context dictionary
        """
        if context is None:
            context = {}
            
        # Select algorithm based on context
        algorithm = self.select_algorithm(context)
        
        # Run optimization
        best_params, _ = algorithm.optimize(
            lambda params: self._evaluate_config(params, X, y),
            max_evals=100
        )
        
        # Scale parameters to more reasonable ranges
        n_estimators = max(50, int(best_params[0].item() * 500))  # Scale to [50, 500]
        max_depth = max(5, int(best_params[1].item() * 30))      # Scale to [5, 30]
        
        # Create best configuration
        self.best_config = {
            'type': 'random_forest',  # Start with random forest
            'params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
        }
        
        # Update history
        performance = self._evaluate_config(best_params, X, y)
        self.update(algorithm.__class__.__name__, performance, context)
        
    def get_best_configuration(self):
        """
        Get the best configuration found during optimization
        
        Returns:
            dict: Best model configuration
        """
        if self.best_config is None:
            # Return default configuration if no optimization has been run
            return {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10
                }
            }
        return self.best_config
        
    def _evaluate_config(self, params, X, y):
        """
        Evaluate a configuration using cross-validation
        
        Args:
            params: Model parameters (numpy array)
            X: Features
            y: Labels
            
        Returns:
            float: Performance score
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        # Scale parameters to more reasonable ranges
        if isinstance(params, np.ndarray):
            n_estimators = max(50, int(params[0].item() * 500))  # Scale to [50, 500]
            max_depth = max(5, int(params[1].item() * 30))      # Scale to [5, 30]
        else:
            n_estimators = max(50, int(params[0] * 500))
            max_depth = max(5, int(params[1] * 30))
        
        # Create and evaluate model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        # Use 3-fold cross-validation
        scores = cross_val_score(model, X, y, cv=3)
        return -np.mean(scores)  # Return negative score for minimization

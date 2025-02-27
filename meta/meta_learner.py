"""
meta_learner.py
-------------
Meta-learner implementation that combines multiple optimization algorithms
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from models.model_factory import ModelFactory
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

logger = logging.getLogger(__name__)

class MetaLearner:
    def __init__(self, method='bayesian', surrogate_model=None, selection_strategy=None,
                 exploration_factor=0.2, history_weight=0.7):
        """Initialize MetaLearner with specified method."""
        self.algorithms = []
        self.method = method
        self.selection_strategy = selection_strategy or method
        self.exploration_factor = max(0.3, exploration_factor)
        self.history_weight = history_weight
        self.best_config = None
        self.best_score = float('-inf')
        self.feature_names = None
        self.X_train = None
        self.y_train = None
        self.phase_scores = {}  # Track scores per phase
        
        # Initialize method-specific components
        if method == 'bayesian':
            self.gp_model = surrogate_model or GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF([1.0]),
                normalize_y=True,
                alpha=0.1
            )
        elif method == 'rl':
            self.policy_network = nn.Sequential(
                nn.Linear(3, 32),  # State dimension matches test
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, len(self.algorithms) or 3)  # Default to 3 actions
            )
            self.optimizer = torch.optim.Adam(self.policy_network.parameters())
        
        logger.info(
            f"Initializing MetaLearner - Method: {method}, "
            f"Strategy: {self.selection_strategy}, "
            f"Exploration: {self.exploration_factor:.3f}"
        )
        
        # Enhanced parameter ranges
        self.param_ranges = {
            'n_estimators': (100, 1000),
            'max_depth': (5, 30),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', None]
        }
        
        logger.debug(f"Parameter ranges: {self.param_ranges}")
        
        # Performance tracking
        self.history = []
        self.eval_history = []
        self.algorithm_scores = {}
        self.param_importance = {}
    
    def set_algorithms(self, algorithms: List[Any]) -> None:
        """Set optimization algorithms to use"""
        logger.info(f"Setting algorithms: {algorithms}")
        self.algorithms = algorithms
    
    def select_algorithm_bayesian(self, context: Dict[str, Any]) -> Any:
        """Select algorithm using Bayesian optimization."""
        if not self.algorithms:
            raise ValueError("No algorithms available")
        
        # Convert context to features
        phase = float(context.get('phase', 0))
        context_feature = np.array([[phase]])
        logger.debug(f"Context: phase={phase}")
        
        # Get predictions for each algorithm
        scores = []
        uncertainties = []
        
        # Use phase-specific scores
        phase_scores = self.phase_scores.get(phase, {})
        
        # Calculate scores and uncertainties
        for algo in self.algorithms:
            if algo.name in phase_scores and phase_scores[algo.name]:
                # Use only performance from current phase
                perf = phase_scores[algo.name][-5:]  # Last 5 performances
                mean_perf = np.mean(perf)
                std_perf = np.std(perf) if len(perf) > 1 else 1.0
                scores.append(mean_perf)
                uncertainties.append(std_perf)
                logger.debug(f"{algo.name} phase {phase}: mean={mean_perf:.3f}, std={std_perf:.3f}")
            else:
                # No performance in current phase - start fresh
                scores.append(0)
                uncertainties.append(2.0)
                logger.debug(f"{algo.name}: no data in phase {phase}")
        
        # UCB acquisition with phase-aware exploration
        phase_steps = sum(len(scores) for scores in phase_scores.values())
        
        if phase not in self.phase_scores:
            # New phase - high exploration
            exploration = 2.0
            logger.info(f"New phase {phase} detected, setting high exploration")
        else:
            if phase_steps == 0:
                exploration = 2.0  # High exploration for new phase
            else:
                # Exponential decay with minimum exploration
                decay_rate = 0.3  # Slower decay
                exploration = max(0.2, 2.0 * np.exp(-decay_rate * phase_steps))
            logger.debug(f"Within phase {phase}, steps={phase_steps}, exploration={exploration:.3f}")
        
        # Calculate acquisition scores with Thompson sampling
        noise_scale = 0.3 if phase_steps < 3 else 0.1  # More noise early in phase
        noise = np.random.normal(0, noise_scale, len(scores))
        acquisition_scores = []
        for i, (score, uncertainty) in enumerate(zip(scores, uncertainties)):
            acq = score + exploration * uncertainty + noise[i]
            acquisition_scores.append(acq)
            logger.debug(f"{self.algorithms[i].name} acquisition: score={score:.3f} + {exploration:.3f}*{uncertainty:.3f} + {noise[i]:.3f} = {acq:.3f}")
        
        best_idx = np.argmax(acquisition_scores)
        selected_algo = self.algorithms[best_idx]
        logger.info(f"Selected {selected_algo.name} for phase {phase} (score={acquisition_scores[best_idx]:.3f})")
        return selected_algo
    
    def select_algorithm_rl(self, state: torch.Tensor) -> Any:
        """Select algorithm using reinforcement learning."""
        if not self.algorithms:
            raise ValueError("No algorithms available")
            
        # Normalize state
        state_mean = state.mean()
        state_std = state.std() + 1e-8
        normalized_state = (state - state_mean) / state_std
        
        with torch.no_grad():
            logits = self.policy_network(normalized_state)
            # Apply temperature scaling for better exploration
            temperature = max(0.5, 1.0 - len(self.eval_history) / 1000)
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            # Ensure valid probabilities
            probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
            action = torch.multinomial(probs, 1).item()
            
        return self.algorithms[action % len(self.algorithms)]
    
    def select_algorithm(self, context: Dict[str, Any]) -> Any:
        """Select algorithm based on specified method."""
        if self.method == 'bayesian':
            return self.select_algorithm_bayesian(context)
        elif self.method == 'rl':
            state = torch.tensor([
                context.get('dim', 0),
                hash(context.get('complexity', '')) % 100
            ], dtype=torch.float32)
            return self.select_algorithm_rl(state)
        else:
            if not self.algorithms:
                raise ValueError("No algorithms available")
            
            logger.debug(f"Selecting algorithm - Context: {context}")
            
            if self.selection_strategy == 'random':
                return np.random.choice(self.algorithms)
                
            # Calculate scores for each algorithm
            scores = []
            for idx, algo in enumerate(self.algorithms):
                history = self.algorithm_scores.get(algo.name, [])
                if not history:
                    scores.append(float('inf'))  # Encourage exploration
                    continue
                    
                mean_score = np.mean(history)
                if self.selection_strategy == 'bayesian':
                    uncertainty = np.std(history) if len(history) > 1 else 1.0
                    score = mean_score + self.exploration_factor * uncertainty
                else:  # UCB
                    n_total = sum(len(s) for s in self.algorithm_scores.values())
                    n_algo = len(history)
                    score = mean_score + self.exploration_factor * np.sqrt(2 * np.log(n_total) / n_algo)
                scores.append(score)
                
            logger.debug(f"Algorithm scores: {scores}")
            
            return self.algorithms[np.argmax(scores)]
        
    def update_rl(self, algorithm_name: str, reward: float, state: torch.Tensor) -> None:
        """Update RL policy based on received reward."""
        if self.method != 'rl':
            return
            
        # Convert reward to tensor
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        
        # Get action probabilities
        action_probs = self.policy_network(state)
        
        # Calculate loss (policy gradient)
        algo_idx = next(i for i, algo in enumerate(self.algorithms) 
                       if algo.name == algorithm_name)
        log_prob = torch.log(action_probs[algo_idx])
        loss = -log_prob * reward_tensor
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update history
        self.history.append({
            'algorithm': algorithm_name,
            'performance': float(reward),
            'iteration': len(self.history),
            'state': state.numpy().tolist()
        })
        
        # Update algorithm scores
        if algorithm_name not in self.algorithm_scores:
            self.algorithm_scores[algorithm_name] = []
        self.algorithm_scores[algorithm_name].append(float(reward))
    
    def update(self, algorithm_name: str, performance: float, context: Optional[Dict[str, Any]] = None) -> None:
        """Update algorithm performance history."""
        if algorithm_name not in self.algorithm_scores:
            self.algorithm_scores[algorithm_name] = []
        self.algorithm_scores[algorithm_name].append(performance)
        
        # Update phase-specific scores
        phase = context.get('phase', 0) if context else 0
        if phase not in self.phase_scores:
            self.phase_scores[phase] = {}
        if algorithm_name not in self.phase_scores[phase]:
            self.phase_scores[phase][algorithm_name] = []
        self.phase_scores[phase][algorithm_name].append(performance)
        
        # Update history with context
        history_entry = {
            'algorithm': algorithm_name,
            'performance': performance,
            'iteration': len(self.history)
        }
        if context:
            history_entry.update(context)
        self.history.append(history_entry)
        
        # Log performance stats
        phase_perf = self.phase_scores[phase][algorithm_name]
        phase_mean = np.mean(phase_perf)
        phase_std = np.std(phase_perf) if len(phase_perf) > 1 else 0
        logger.info(f"{algorithm_name} performance in phase {phase}: current={performance:.3f}, mean={phase_mean:.3f}, std={phase_std:.3f}")
        
        # Update eval history for backward compatibility
        self.eval_history.append({
            'algorithm': algorithm_name,
            'score': performance
        })
        
        if performance > self.best_score:
            self.best_score = performance
            logger.info(f"New best score: {self.best_score:.3f} from {algorithm_name}")
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                context: Optional[Dict] = None,
                n_iter: int = 50,
                patience: int = 10,
                progress_callback: Optional[Callable[[int, float], None]] = None,
                feature_names: Optional[List[str]] = None) -> Tuple[Dict[str, Any], float]:
        """Run optimization process.
        
        Args:
            X: Training features
            y: Training labels
            context: Optional context information
            n_iter: Number of optimization iterations
            patience: Early stopping patience
            progress_callback: Optional callback function to report progress
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (best_configuration, best_score)
        """
        context = context or {}
        best_score = float('-inf')
        best_config = None
        no_improve = 0
        
        # Store feature names
        self.feature_names = feature_names
        self.X_train = X
        self.y_train = y
        
        for i in range(n_iter):
            # Select algorithm
            algo = self.select_algorithm(context)
            
            try:
                # Run optimization step
                config = algo.suggest()
                score = algo.evaluate(config)
                
                # Update algorithm state
                algo.update(config, score)
                
                # Update meta-learner
                self.update(algo.name, score, context)
                
                # Check for improvement
                if score > best_score:
                    best_score = score
                    best_config = config
                    no_improve = 0
                    self.best_config = best_config
                    self.best_score = best_score
                else:
                    no_improve += 1
                
                # Early stopping
                if no_improve >= patience:
                    logger.info(f"Early stopping after {i+1} iterations")
                    break
                    
                # Report progress
                if progress_callback:
                    progress_callback(i, best_score)
                    
            except Exception as e:
                logger.error(f"Optimization error: {str(e)}")
                continue
        
        return best_config, best_score
    
    def _evaluate_config(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate a configuration using cross-validation
        
        Args:
            params: Model configuration to evaluate
            X: Training features
            y: Training labels
            
        Returns:
            Mean validation score
        """
        logger.debug(f"Evaluating configuration: {params}")
        
        try:
            factory = ModelFactory()
            model = factory.create_model(params)
            model.fit(X, y, feature_names=self.feature_names)
            return model.score(X, y)
        except Exception as e:
            logger.error(f"Error evaluating configuration: {str(e)}")
            return float('-inf')
    
    def _update_param_importance(self):
        """Calculate parameter importance based on evaluation history"""
        if not self.eval_history:
            return
            
        logger.debug("Updating parameter importance")
        
        # Convert history to numpy arrays
        scores = np.array([h['score'] for h in self.eval_history])
        
        # Calculate importance for each parameter
        for param in self.param_ranges.keys():
            values = np.array([h['params'][param] for h in self.eval_history])
            correlation = np.corrcoef(values, scores)[0, 1]
            self.param_importance[param] = abs(correlation)
    
    def get_best_configuration(self) -> Dict[str, Any]:
        """Get best configuration found during optimization"""
        logger.info("Getting best configuration")
        return self.best_config
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get detailed optimization statistics"""
        if not self.eval_history:
            return {}
            
        logger.info("Getting optimization statistics")
        
        scores = [h['score'] for h in self.eval_history]
        iterations = [h['iteration'] for h in self.eval_history]
        
        # Calculate convergence metrics
        convergence_rate = (max(scores) - scores[0]) / len(scores)
        plateau_threshold = 0.001
        plateau_count = sum(1 for i in range(1, len(scores))
                          if abs(scores[i] - scores[i-1]) < plateau_threshold)
        
        # Calculate exploration metrics
        param_ranges = {}
        for param in self.param_ranges.keys():
            values = [h['params'][param] for h in self.eval_history]
            if isinstance(values[0], (int, float)):
                param_ranges[param] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
            else:
                # For categorical parameters
                unique_values = set(values)
                param_ranges[param] = {
                    'unique_values': list(unique_values),
                    'most_common': max(set(values), key=values.count)
                }
        
        # Calculate performance improvement
        initial_window = scores[:5]
        final_window = scores[-5:]
        improvement = (np.mean(final_window) - np.mean(initial_window)) / np.mean(initial_window) * 100
        
        return {
            'best_score': self.best_score,
            'evaluations': len(self.eval_history),
            'param_importance': self.param_importance,
            'convergence': [h['score'] for h in self.eval_history],
            'convergence_rate': convergence_rate,
            'plateau_percentage': (plateau_count / len(scores)) * 100,
            'param_ranges': param_ranges,
            'performance_improvement': improvement,
            'final_performance': {
                'mean': np.mean(final_window),
                'std': np.std(final_window),
                'stability': 1 - (np.std(final_window) / np.mean(final_window))
            },
            'exploration_coverage': {
                param: (ranges['max'] - ranges['min']) / 
                      (self.param_ranges[param][1] - self.param_ranges[param][0])
                for param, ranges in param_ranges.items()
                if isinstance(ranges, dict) and 'min' in ranges
            }
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using best model configuration"""
        if self.best_config is None:
            raise ValueError("Must run optimize() before making predictions")
            
        logger.info("Making predictions")
        
        model = ModelFactory().create_model(self.best_config)
        model.fit(self.X_train, self.y_train)
        return model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using best model configuration"""
        if self.best_config is None:
            raise ValueError("Must run optimize() before making predictions")
            
        logger.info("Getting prediction probabilities")
        
        model = ModelFactory().create_model(self.best_config)
        model.fit(self.X_train, self.y_train)
        return model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from best model"""
        if self.best_config is None:
            raise ValueError("Must run optimize() before getting feature importance")
            
        logger.info("Getting feature importance")
        
        model = ModelFactory().create_model(self.best_config)
        model.fit(self.X_train, self.y_train)
        return model.feature_importances_

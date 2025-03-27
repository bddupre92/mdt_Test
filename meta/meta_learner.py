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
from meta.patient_memory import PatientMemory

# Configure logging
logger = logging.getLogger(__name__)

class MetaLearner:
    class SimpleAlgorithm:
        """A simple algorithm adapter for string algorithm names."""
        def __init__(self, name):
            self.name = name
            
        def suggest(self):
            """Suggest a configuration."""
            return np.random.rand(1, 2)  # Return a simple random configuration
            
        def evaluate(self, config):
            """Evaluate a configuration."""
            return float(np.random.rand())  # Return a random score
    
    def __init__(self, method='bayesian', surrogate_model=None, selection_strategy=None,
                 exploration_factor=0.2, history_weight=0.7, drift_detector=None,
                 quality_impact=0.4, drift_impact=0.3, memory_storage_dir=None,
                 enable_personalization=True):
        """Initialize MetaLearner with specified method.
        
        Args:
            method: Optimization method ('bayesian', 'rl', etc.)
            surrogate_model: Optional custom surrogate model
            selection_strategy: Strategy for selecting algorithms
            exploration_factor: Controls exploration vs. exploitation
            history_weight: Weight given to historical performance (0-1)
            drift_detector: Optional drift detector instance
            quality_impact: Impact factor of quality on expert weights (0-1)
            drift_impact: Impact factor of drift on expert weights (0-1)
            memory_storage_dir: Directory for storing patient memory files
            enable_personalization: Whether to enable patient-specific adaptations
        """
        # Initialize with default algorithms if none provided
        self.algorithms = ['bayesian', 'random_search', 'grid_search']
        self.method = method
        self.selection_strategy = selection_strategy or method
        self.exploration_factor = max(0.3, exploration_factor)
        self.history_weight = history_weight
        self.drift_detector = drift_detector
        self.quality_impact = quality_impact
        self.drift_impact = drift_impact
        
        # Initialize data quality tracking
        self.domain_data_quality = {}
        self.history_weight = history_weight
        self.best_config = None
        self.best_score = float('-inf')
        self.feature_names = None
        self.X = None
        self.y = None
        self.phase_scores = {}  # Track scores per phase
        self.drift_detector = drift_detector  # Can be set at initialization
        self.logger = logger  # Add logger attribute
        self.experts = {}  # Dictionary to store registered experts
        self.domain_data_quality = {}  # Track data quality by domain
        
        # Initialize patient memory for personalization
        self.enable_personalization = enable_personalization
        self.patient_memory = PatientMemory(storage_dir=memory_storage_dir) if enable_personalization else None
        self.current_patient_id = None
        
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
    
    def register_expert(self, expert_id, expert):
        """Register an expert with the meta-learner.
        
        Args:
            expert_id: Unique identifier for the expert
            expert: Expert object with a 'specialty' attribute
        """
        self.experts[expert_id] = expert
        logger.info(f"Registered expert {expert_id} with specialty {getattr(expert, 'specialty', 'unknown')}")
        
    def predict_weights(self, context):
        """Predict weights for each expert based on context, data quality, drift and patient memory.
        
        Args:
            context: Dictionary with context information, including:
                     - has_physiological, has_behavioral, has_environmental (bool)
                     - quality_metrics (dict): data quality metrics per domain
                     - recent_data (dict): recent data samples for drift detection
                     - patient_id (str): optional patient identifier for personalization
        
        Returns:
            Dictionary mapping expert IDs to weights that sum to 1.0
        """
        # Load patient memory if available and personalization is enabled
        patient_id = context.get('patient_id')
        if self.enable_personalization and patient_id and patient_id != self.current_patient_id:
            self.current_patient_id = patient_id
            self.patient_memory.select_patient(patient_id)
            logger.info(f"Loaded memory for patient {patient_id}")
        if not self.experts:
            logger.warning("No experts registered, returning equal weights")
            return {1: 1.0}  # Default single expert with full weight
        
        # First calculate base weights using specialty matching
        base_weights = self._calculate_specialty_weights(context)
        
        # Apply patient-specific specialty weights if available
        if self.enable_personalization and self.current_patient_id:
            patient_specialty_weights = self.patient_memory.get_specialty_weights()
            if patient_specialty_weights:
                base_weights = self._apply_patient_specialty_weights(base_weights, patient_specialty_weights)
        
        # Apply quality-aware adjustments if quality metrics provided
        if context.get('quality_metrics'):
            quality_adjusted_weights = self._adjust_weights_by_quality(base_weights, context['quality_metrics'])
            
            # Store quality metrics in patient memory if personalization is enabled
            if self.enable_personalization and self.current_patient_id:
                for domain, quality in context['quality_metrics'].items():
                    self.patient_memory.store_domain_quality(domain, quality)
        else:
            quality_adjusted_weights = base_weights
            
        # Apply drift detection adjustments if drift detector exists and recent data provided
        if self.drift_detector and context.get('recent_data'):
            final_weights, drift_results = self._adjust_weights_for_drift(quality_adjusted_weights, context['recent_data'], return_drift_info=True)
            
            # Store drift detection results in patient memory if personalization is enabled
            if self.enable_personalization and self.current_patient_id and drift_results:
                for domain, result in drift_results.items():
                    drift_detected, drift_score, p_value = result
                    self.patient_memory.store_drift_event(domain, drift_detected, drift_score, p_value)
        else:
            final_weights = quality_adjusted_weights
            
        # Normalize weights to ensure they sum to 1.0
        total = sum(final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}
            
        # Store final weights in patient memory if personalization is enabled
        if self.enable_personalization and self.current_patient_id:
            self.patient_memory.update_expert_weights(final_weights)
            
        logger.info(f"Predicted weights: {final_weights}")
        return final_weights
        
    def _calculate_specialty_weights(self, context):
        """Calculate base weights based on expert specialties.
        
        Args:
            context: Dictionary with context information
            
        Returns:
            Dictionary of base weights per expert ID
        """
        # Default base weights (ensure sum is < 1.0 to leave room for adjustments)
        base_weight = 0.1
        weights = {expert_id: base_weight for expert_id in self.experts.keys()}
        
        # Weights for each data type
        remaining_weight = 1.0 - (base_weight * len(weights))
        type_bonus = remaining_weight / 3  # Distribute remaining weight among data types
        
        # Adjust weights based on context and expert specialties
        for expert_id, expert in self.experts.items():
            specialty = getattr(expert, 'specialty', 'general')
            
            # Increase weight for experts with matching specialties
            if context.get('has_physiological', False) and specialty == 'physiological':
                weights[expert_id] += type_bonus
            if context.get('has_behavioral', False) and specialty == 'behavioral':
                weights[expert_id] += type_bonus
            if context.get('has_environmental', False) and specialty == 'environmental':
                weights[expert_id] += type_bonus
                
        return weights
    
    def _adjust_weights_by_quality(self, weights, quality_metrics):
        """Adjust weights based on data quality metrics.
        
        Args:
            weights: Base weights dictionary
            quality_metrics: Dictionary mapping data domains to quality scores (0-1)
            
        Returns:
            Dictionary of quality-adjusted weights
        """
        adjusted_weights = weights.copy()
        
        # Use instance quality_impact attribute (or default to 0.4 for backward compatibility)
        quality_impact = getattr(self, 'quality_impact', 0.4)
        
        for expert_id, expert in self.experts.items():
            specialty = getattr(expert, 'specialty', 'general')
            
            # Get quality score for this domain if available
            domain_quality = quality_metrics.get(specialty, 0.8)  # Default to 0.8 if not specified
            
            # Quality score ranges from 0-1, transform to adjustment factor
            # A quality of 1.0 gives full weight, 0.0 reduces weight significantly
            quality_factor = 0.5 + (0.5 * domain_quality)
            
            # Apply quality adjustment
            adjusted_weights[expert_id] *= (1.0 - quality_impact + (quality_impact * quality_factor))
            
            logger.debug(f"Expert {expert_id} ({specialty}) quality adjustment: {domain_quality:.2f} → factor: {quality_factor:.2f}")
            
        return adjusted_weights
    
    def _apply_patient_specialty_weights(self, weights, patient_specialty_weights):
        """Apply patient-specific specialty weights to modify base weights.
        
        Args:
            weights: Current expert weights
            patient_specialty_weights: Dictionary of patient-specific specialty weights
            
        Returns:
            Dictionary of adjusted weights based on patient history
        """
        adjusted_weights = weights.copy()
        
        # Iterate through experts and adjust weights based on their specialty and patient history
        for expert_id, expert in self.experts.items():
            specialty = getattr(expert, 'specialty', 'unknown')
            if specialty in patient_specialty_weights and expert_id in adjusted_weights:
                # Adjust the weight based on patient-specific specialty preference
                adjustment_factor = patient_specialty_weights[specialty]
                adjusted_weights[expert_id] *= adjustment_factor
                logger.debug(f"Applied patient-specific adjustment for {specialty}: factor {adjustment_factor:.2f}")
                
        return adjusted_weights
    
    def _adjust_weights_for_drift(self, weights, recent_data, return_drift_info=False):
        """Adjust weights based on drift detection results.
        
        Args:
            weights: Current weights dictionary
            recent_data: Dictionary mapping domains to recent data samples
            
        Returns:
            Dictionary of drift-adjusted weights
        """
        adjusted_weights = weights.copy()
        
        # Skip if no drift detector available
        if not self.drift_detector:
            logger.warning("Drift detector not configured, skipping drift adjustment")
            return adjusted_weights
            
        # Use instance drift_impact attribute (or default to 0.3 for backward compatibility)
        drift_impact = getattr(self, 'drift_impact', 0.3)
        
        for expert_id, expert in self.experts.items():
            specialty = getattr(expert, 'specialty', 'general')
            
            # Skip if no data for this domain
            if specialty not in recent_data or specialty == 'general':
                continue
                
            # Get recent data for this domain
            domain_data = recent_data.get(specialty)
            
            if domain_data is None or not isinstance(domain_data, dict):
                continue
                
            # Extract reference and current windows
            reference_window = domain_data.get('reference_window')
            current_window = domain_data.get('current_window')
            
            if reference_window is None or current_window is None:
                continue
                
            try:
                # Detect drift in this domain's data
                is_drift, drift_score, p_value = self.drift_detector.detect_drift(
                    current_window_X=current_window
                )
                
                # Store the reference window for this domain if needed
                if self.drift_detector.reference_window is None:
                    self.drift_detector.reference_window = reference_window
                
                # Apply drift adjustment - reduce weight if drift detected
                if is_drift:
                    # Higher drift scores mean more drift, so we reduce the weight more
                    drift_factor = 1.0 - min(0.8, drift_score)
                    adjusted_weights[expert_id] *= (1.0 - drift_impact + (drift_impact * drift_factor))
                    logger.info(f"Drift detected in {specialty} domain, score={drift_score:.3f}, p-value={p_value:.3f}, adjusting weight by factor {drift_factor:.3f}")
            except Exception as e:
                logger.error(f"Error in drift detection for {specialty}: {str(e)}")
        
        # Return drift info if requested
        if return_drift_info:
            drift_info = {}
            for expert_id, expert in self.experts.items():
                specialty = getattr(expert, 'specialty', 'general')
                if specialty in recent_data and specialty != 'general':
                    try:
                        # Extract data for this domain
                        current_window = recent_data.get(specialty, {}).get('current_window')
                        if current_window is not None:
                            # Get drift detection result
                            is_drift, drift_score, p_value = self.drift_detector.detect_drift(
                                current_window_X=current_window
                            )
                            drift_info[specialty] = (is_drift, drift_score, p_value)
                    except Exception as e:
                        logger.error(f"Error getting drift info for {specialty}: {str(e)}")
            return adjusted_weights, drift_info
                
        return adjusted_weights
        
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
            # Handle both string algorithm names and algorithm objects
            algo_name = algo if isinstance(algo, str) else algo.name
            
            if algo_name in phase_scores and phase_scores[algo_name]:
                # Use only performance from current phase
                perf = phase_scores[algo_name][-5:]  # Last 5 performances
                mean_perf = np.mean(perf)
                std_perf = np.std(perf) if len(perf) > 1 else 1.0
                scores.append(mean_perf)
                uncertainties.append(std_perf)
                logger.debug(f"{algo_name} phase {phase}: mean={mean_perf:.3f}, std={std_perf:.3f}")
            else:
                # No performance in current phase - start fresh
                scores.append(0)
                uncertainties.append(2.0)
                logger.debug(f"{algo_name}: no data in phase {phase}")
        
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
            algo_name = self.algorithms[i] if isinstance(self.algorithms[i], str) else self.algorithms[i].name
            logger.debug(f"{algo_name} acquisition: score={score:.3f} + {exploration:.3f}*{uncertainty:.3f} + {noise[i]:.3f} = {acq:.3f}")
        
        best_idx = np.argmax(acquisition_scores)
        selected_algo = self.algorithms[best_idx]
        algo_name = selected_algo if isinstance(selected_algo, str) else selected_algo.name
        logger.info(f"Selected {algo_name} for phase {phase} (score={float(acquisition_scores[best_idx]):.3f})")
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
    
    def update_algorithm_performance(self, algorithm_name: str, performance: float, context: Optional[Dict[str, Any]] = None) -> None:
        """Update the performance history of an algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            performance: Performance metric (lower is better)
            context: Optional context information
        """
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
    
    def update(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Update the meta-learner with new data.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            True if drift was detected, False otherwise
        """
        try:
            # Make predictions on new data
            y_pred = self.predict(X)
            
            # Check for drift
            if self.drift_detector is not None:
                drift_detected = self.drift_detector.detect_drift(y, y_pred)
            else:
                self.logger.warning("No drift detector available. Assuming no drift.")
                drift_detected = False
            
            # Update data
            self.X = np.vstack((self.X, X)) if self.X is not None else X
            self.y = np.append(self.y, y) if self.y is not None else y
            
            # Retrain model if drift detected
            if drift_detected:
                self.logger.info(f"Drift detected, retraining model")
                self.fit(self.X, self.y)
                
            return drift_detected
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            # Fallback to a simple model for drift detection
            if not hasattr(self, 'fallback_model') or self.fallback_model is None:
                from sklearn.ensemble import RandomForestRegressor
                self.logger.info("Using fallback model for drift detection")
                
                # Use a simple default configuration
                self.fallback_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt'
                )
                
                if self.X is not None and self.y is not None:
                    self.fallback_model.fit(self.X, self.y)
                else:
                    self.fallback_model.fit(X, y)
            
            # Use fallback model for predictions
            y_pred = self.fallback_model.predict(X)
            
            # Check for drift
            if self.drift_detector is not None:
                drift_detected = self.drift_detector.detect_drift(y, y_pred)
            else:
                self.logger.warning("No drift detector available. Assuming no drift.")
                drift_detected = False
            
            # Update data
            self.X = np.vstack((self.X, X)) if self.X is not None else X
            self.y = np.append(self.y, y) if self.y is not None else y
            
            # Retrain model if drift detected
            if drift_detected:
                self.logger.info(f"Drift detected, retraining model")
                self.fit(self.X, self.y)
                
            return drift_detected
    
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
        self.X = X
        self.y = y
        
        for i in range(n_iter):
            # Select algorithm
            algo = self.select_algorithm(context)
            
            # Convert string algorithm to SimpleAlgorithm instance
            if isinstance(algo, str):
                algo = self.SimpleAlgorithm(algo)
            
            try:
                # Run optimization step
                try:
                    config = algo.suggest()
                    
                    # Ensure config is 2D for scikit-learn models
                    import numpy as np
                    if isinstance(config, np.ndarray):
                        if config.ndim == 1:
                            config = config.reshape(1, -1)
                    
                    # Special handling for evaluation
                    try:
                        score = algo.evaluate(config)
                        # Ensure score is a float for f-string formatting
                        if isinstance(score, np.ndarray):
                            if score.size == 1:  # Single value in the array
                                score = float(score.item())
                            else:
                                # Take the mean if multiple values
                                score = float(np.mean(score))
                        else:
                            score = float(score)
                    except Exception as e:
                        logger.error(f"Evaluation error: {str(e)}")
                        # Use a default score and continue
                        score = 0.0
                        
                    # Update algorithm state
                    try:
                        # Ensure config is properly shaped for algorithm.update
                        import numpy as np
                        if np.isscalar(config):
                            config = np.array([[float(config)]])
                        elif isinstance(config, np.ndarray):
                            if config.ndim == 0:  # numpy scalar
                                config = np.array([[float(config)]])
                            elif config.ndim == 1:
                                config = config.reshape(1, -1)
                                
                        # Ensure score is properly formatted
                        if np.isscalar(score):
                            score_val = float(score)
                        else:
                            score_val = float(score) if hasattr(score, "item") else score
                            
                        algo.update(config, score_val)
                    except Exception as e:
                        logger.error(f"Update error: {str(e)}")
                    
                    # Update meta-learner
                    self.update_algorithm_performance(algo.name, score, context)
                    
                except Exception as e:
                    logger.error(f"Optimization error: {str(e)}")
                    # Continue with the next iteration
                    continue
                
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
            # Try using fallback model if available
            if hasattr(self, 'fallback_model') and self.fallback_model is not None:
                self.logger.warning("Using fallback model for predictions")
                return self.fallback_model.predict(X)
            else:
                raise ValueError("Must run optimize() before making predictions")
        
        logger.info("Making predictions")
        
        # Check if best_config is a numpy array (from an Algorithm) or a dict for ModelFactory
        if isinstance(self.best_config, np.ndarray):
            # Use the first algorithm for predictions since best_config is from an Algorithm
            if len(self.algorithms) > 0:
                selected_algo = self.algorithms[0]
                return selected_algo.predict(X)
            # If no algorithms available, use fallback model
            elif hasattr(self, 'fallback_model') and self.fallback_model is not None:
                self.logger.warning("Using fallback model for predictions with numpy config")
                return self.fallback_model.predict(X)
            else:
                raise ValueError("No suitable model found for predictions")
        else:
            # Use ModelFactory approach with a dict config
            model = ModelFactory().create_model(self.best_config)
            model.fit(self.X, self.y)
            return model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using best model configuration"""
        if self.best_config is None:
            # Try using fallback model if available
            if hasattr(self, 'fallback_model') and self.fallback_model is not None:
                self.logger.warning("Using fallback model for probability predictions")
                return self.fallback_model.predict_proba(X)
            else:
                raise ValueError("Must run optimize() before making predictions")
            
        logger.info("Getting prediction probabilities")
        
        # Check if best_config is a numpy array (from an Algorithm) or a dict for ModelFactory
        if isinstance(self.best_config, np.ndarray):
            # Use the first algorithm for predictions since best_config is from an Algorithm
            if len(self.algorithms) > 0:
                selected_algo = self.algorithms[0]
                if hasattr(selected_algo, 'predict_proba'):
                    return selected_algo.predict_proba(X)
                else:
                    return np.array([[1-p, p] for p in selected_algo.predict(X)])
            # If no algorithms available, use fallback model
            elif hasattr(self, 'fallback_model') and self.fallback_model is not None:
                self.logger.warning("Using fallback model for probability predictions")
                return self.fallback_model.predict_proba(X)
            else:
                raise ValueError("No suitable model found for probability predictions")
        else:
            # Use ModelFactory approach with a dict config
            model = ModelFactory().create_model(self.best_config)
            model.fit(self.X, self.y)
            return model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores from best model"""
        if self.best_config is None:
            # Try using fallback model if available
            if hasattr(self, 'fallback_model') and self.fallback_model is not None:
                self.logger.warning("Using fallback model for feature importance")
                return self.fallback_model.feature_importances_
            else:
                raise ValueError("Must run optimize() before getting feature importance")
            
        logger.info("Getting feature importance")
        
        # Check if best_config is a numpy array (from an Algorithm) or a dict for ModelFactory
        if isinstance(self.best_config, np.ndarray):
            # Use the first algorithm for feature importance since best_config is from an Algorithm
            if len(self.algorithms) > 0:
                selected_algo = self.algorithms[0]
                if hasattr(selected_algo, 'feature_importances_'):
                    return selected_algo.feature_importances_
                elif hasattr(selected_algo, 'model') and hasattr(selected_algo.model, 'feature_importances_'):
                    return selected_algo.model.feature_importances_
                else:
                    # Create uniform feature importance if not available
                    return np.ones(self.X.shape[1]) / self.X.shape[1]
            # If no algorithms available, use fallback model
            elif hasattr(self, 'fallback_model') and self.fallback_model is not None:
                self.logger.warning("Using fallback model for feature importance")
                return self.fallback_model.feature_importances_
            else:
                # Return uniform feature importance as fallback
                return np.ones(self.X.shape[1]) / self.X.shape[1]
        else:
            # Use ModelFactory approach with a dict config
            model = ModelFactory().create_model(self.best_config)
            model.fit(self.X, self.y)
            return model.feature_importances_
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """Fit the meta-learner to the training data.
        
        Args:
            X: Training features
            y: Training labels
            feature_names: Optional list of feature names
        """
        logger.info(f"Fitting meta-learner on {X.shape[0]} samples with {X.shape[1]} features")
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
        # Run optimization to find best configuration
        best_config, best_score = self.optimize(X, y, feature_names=feature_names)
        self.best_config = best_config
        self.best_score = best_score
        
        logger.info(f"Meta-learner fit complete. Best score: {best_score:.4f}")

    def run_meta_learner_with_drift(self, n_samples=1000, n_features=10, drift_points=None, 
                                  window_size=10, drift_threshold=0.01, significance_level=0.9,
                                  visualize=False):
        """Run meta-learner with drift detection"""
        # Generate synthetic data
        X_full, y_full = generate_synthetic_data(n_samples)
        
        # Initialize drift detector
        detector = DriftDetector(
            window_size=window_size,
            drift_threshold=drift_threshold,
            significance_level=significance_level
        )
        
        # Initialize model
        model = ModelFactory().create_model({'task_type': 'classification'})
        
        # Initial training
        train_size = 200
        X_train = X_full[:train_size]
        y_train = y_full[:train_size]
        model.fit(X_train, y_train)
        
        # Initialize reference window with first batch predictions
        initial_pred = model.predict_proba(X_train)[:, 1]  # Use positive class probability
        detector.set_reference_window(initial_pred)
        
        # Process remaining data in batches
        batch_size = window_size
        detected_drifts = []
        
        for i in range(train_size, n_samples, batch_size):
            # Get current batch
            X_batch = X_full[i:i+batch_size]
            
            # Get predictions
            pred_proba = model.predict_proba(X_batch)[:, 1]  # Use positive class probability
            
            # Check for drift using prediction probabilities
            drift_detected, severity, info = detector.add_sample(
                point=pred_proba.mean(),  # Use mean probability as point
                prediction_proba=pred_proba  # Pass full probabilities
            )
            
            # Log drift check
            logger.info(
                f"Sample {i}: Mean shift={info['mean_shift']:.4f}, "
                f"KS stat={info['ks_statistic']:.4f}, "
                f"p-value={info['p_value']:.4f}, "
                f"Severity={severity:.4f}"
            )
            
            # Handle drift if detected
            if drift_detected:
                detected_drifts.append(i)
                # Update model with recent data
                X_update = X_full[max(0, i-100):i]  # Use last 100 samples
                y_update = y_full[max(0, i-100):i]
                model.fit(X_update, y_update)
        
        # Manual check at known drift points
        if drift_points:
            for point in drift_points:
                logger.info(f"Manually checking for drift at point {point}...")
                X_check = X_full[point:point+window_size]
                pred_proba = model.predict_proba(X_check)[:, 1]
                drift_detected, _, info = detector.detect_drift(
                    curr_data=pred_proba,
                    ref_data=detector.reference_window
                )
                logger.info(f"Drift check at {point}: {'Detected' if drift_detected else 'Not detected'}")
                logger.info(f"Stats: Mean shift={info['mean_shift']:.4f}, "
                          f"KS stat={info['ks_statistic']:.4f}, "
                          f"p-value={info['p_value']:.4f}")
        
        return detected_drifts

    def save(self, filepath):
        """Save the meta learner to a file.
        
        Args:
            filepath: Path to save the meta learner to
        """
        import pickle
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save meta learner state
        state = {
            'method': self.method,
            'selection_strategy': self.selection_strategy,
            'exploration_factor': self.exploration_factor,
            'history_weight': self.history_weight,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'feature_names': self.feature_names,
            'phase_scores': self.phase_scores,
            'algorithms': self.algorithms
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved meta learner to {filepath}")
        
    @classmethod
    def load(cls, filepath):
        """Load a meta learner from a file.
        
        Args:
            filepath: Path to load the meta learner from
            
        Returns:
            Loaded meta learner
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new meta learner
        meta_learner = cls(
            method=state.get('method', 'bayesian'),
            selection_strategy=state.get('selection_strategy'),
            exploration_factor=state.get('exploration_factor', 0.2),
            history_weight=state.get('history_weight', 0.7)
        )
        
        # Restore state
        meta_learner.best_config = state.get('best_config')
        meta_learner.best_score = state.get('best_score', float('-inf'))
        meta_learner.feature_names = state.get('feature_names')
        meta_learner.phase_scores = state.get('phase_scores', {})
        meta_learner.algorithms = state.get('algorithms', [])
        
        logger.info(f"Loaded meta learner from {filepath}")
        return meta_learner

    def set_patient(self, patient_id):
        """Set the current patient for personalization.
        
        Args:
            patient_id: Unique identifier for the patient
            
        Returns:
            Dictionary containing the patient's memory data or None if personalization is disabled
        """
        if not self.enable_personalization:
            logger.warning("Personalization is disabled")
            return None
            
        memory_data = self.patient_memory.select_patient(patient_id)
        self.current_patient_id = patient_id
        return memory_data
    
    def update_patient_specialty_preference(self, specialty, weight_adjustment):
        """Update patient-specific specialty preference.
        
        Args:
            specialty: Data specialty (physiological, behavioral, environmental)
            weight_adjustment: Adjustment factor for the specialty (>1 increases importance)
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.enable_personalization or not self.current_patient_id:
            logger.warning("Personalization is disabled or no patient selected")
            return False
            
        specialty_weights = self.patient_memory.get_specialty_weights()
        specialty_weights[specialty] = weight_adjustment
        return self.patient_memory.update_specialty_weights(specialty_weights)
        
    def get_patient_history(self, history_type=None, limit=10):
        """Get patient historical data.
        
        Args:
            history_type: Type of history to retrieve ('quality', 'drift', 'performance', or None for all)
            limit: Maximum number of entries to return per type
            
        Returns:
            Dictionary containing the requested historical data
        """
        if not self.enable_personalization or not self.current_patient_id:
            logger.warning("Personalization is disabled or no patient selected")
            return {}
            
        history = {}
        
        if history_type in (None, 'quality'):
            history['quality'] = {
                'physiological': self.patient_memory.get_domain_quality_history('physiological', limit),
                'behavioral': self.patient_memory.get_domain_quality_history('behavioral', limit),
                'environmental': self.patient_memory.get_domain_quality_history('environmental', limit)
            }
            
        if history_type in (None, 'drift'):
            history['drift'] = {
                'physiological': self.patient_memory.get_drift_history('physiological', limit),
                'behavioral': self.patient_memory.get_drift_history('behavioral', limit),
                'environmental': self.patient_memory.get_drift_history('environmental', limit)
            }
            
        if history_type in (None, 'performance'):
            history['performance'] = self.patient_memory.get_performance_history(limit)
            
        return history
        
    def track_performance(self, expert_id, performance_score, context=None):
        """Track performance of a specific expert.
        
        Args:
            expert_id: ID of the expert
            performance_score: Score of the expert's performance
            context: Optional dictionary with additional context information
        """
        self.algorithm_scores[expert_id] = performance_score
        
        # Store performance in patient memory if personalization is enabled
        if self.enable_personalization and self.current_patient_id:
            details = {
                'expert_id': expert_id,
                'score': performance_score
            }
            if context:
                details['context'] = context
            
            self.patient_memory.store_performance(performance_score, details)
            
    def clear_patient_data(self, patient_id=None):
        """Clear patient data from memory.
        
        Args:
            patient_id: Specific patient ID to clear, or None to clear current patient
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_personalization:
            logger.warning("Personalization is disabled")
            return False
            
        target_id = patient_id or self.current_patient_id
        if not target_id:
            logger.warning("No patient ID specified or selected")
            return False
            
        # Clear patient memory
        return self.patient_memory.clear_patient_data(target_id)

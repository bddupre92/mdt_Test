"""
Gating Network Module

This module provides the implementation of the gating network component for the 
Mixture of Experts (MoE) framework. The gating network is responsible for determining
the weights of each expert model based on input features.
"""

import numpy as np
import pandas as pd
import logging
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from moe_framework.experts.base_expert import BaseExpert
from moe_framework.gating.quality_aware_weighting import QualityAwareWeighting
from moe_framework.gating.meta_learner_gating import MetaLearnerGating
from meta.meta_learner import MetaLearner
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory

# Configure logging
logger = logging.getLogger(__name__)


class GatingNetwork:
    """
    Gating Network for the Mixture of Experts (MoE) framework.
    
    The gating network determines the weights for each expert model based on the input features.
    It supports different combination strategies and can be optimized using evolutionary algorithms.
    
    Attributes:
        name (str): Name of the gating network
        model (BaseEstimator): The underlying machine learning model for weight prediction
        experts (Dict[str, BaseExpert]): Dictionary of expert models
        combination_strategy (str): Strategy for combining expert predictions
        feature_columns (List[str]): List of feature column names
        scaler (StandardScaler): Scaler for normalizing input features
        is_fitted (bool): Whether the gating network has been fitted
        quality_metrics (Dict[str, float]): Quality metrics for gating network performance
        metadata (Dict[str, Any]): Additional metadata about the gating network
    """
    
    def __init__(self, 
                 name: str = "gating_network",
                 model: Optional[BaseEstimator] = None,
                 experts: Dict[str, BaseExpert] = None,
                 combination_strategy: str = "weighted_average",
                 feature_columns: List[str] = None,
                 metadata: Dict[str, Any] = None,
                 quality_weighting: QualityAwareWeighting = None,
                 meta_learner_gating: MetaLearnerGating = None,
                 use_quality_weighting: bool = False,
                 use_meta_learner: bool = False,
                 stacking_model: Optional[BaseEstimator] = None):
        """
        Initialize the gating network.
        
        Args:
            name: Name of the gating network
            model: The underlying machine learning model for weight prediction
            experts: Dictionary of expert models
            combination_strategy: Strategy for combining expert predictions
                                 (weighted_average, stacking, confidence_weighted, dynamic_selection, ensemble_stacking)
            feature_columns: List of feature column names
            metadata: Additional metadata about the gating network
            stacking_model: Model to use for ensemble stacking strategy
        """
        self.name = name
        # Use a safer default model initialization to avoid issues with estimators_
        if model is not None:
            self.model = model
        else:
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
        self.experts = experts or {}
        self.combination_strategy = combination_strategy
        self.feature_columns = feature_columns or []
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.quality_metrics = {}
        self.metadata = metadata or {}
        
        # Additional attributes for tracking and optimization
        self.training_history = []
        self.validation_scores = {}
        self.weight_constraints = {"min": 0.0, "max": 1.0, "sum": 1.0}
        
        # Initialize quality-aware weighting
        self.quality_weighting = quality_weighting or QualityAwareWeighting()
        self.use_quality_weighting = use_quality_weighting
        
        # Initialize Meta_Learner gating
        self.meta_learner_gating = meta_learner_gating or MetaLearnerGating()
        self.use_meta_learner = use_meta_learner
        
        # Initialize stacking model for ensemble stacking strategy
        if stacking_model is not None:
            self.stacking_model = stacking_model
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.stacking_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.stacking_model_fitted = False
        
        logger.info(f"Initialized gating network '{name}' with {combination_strategy} strategy")
    
    def register_expert(self, expert_id: str, expert: BaseExpert, metadata: Dict[str, Any] = None) -> None:
        """
        Register an expert model with the gating network.
        
        Args:
            expert_id: Unique identifier for the expert
            expert: Expert model instance
            metadata: Additional metadata about the expert
        """
        self.experts[expert_id] = expert
        
        # Register with quality-aware weighting if enabled
        if self.use_quality_weighting:
            # Create a default quality profile based on expert type
            if metadata is None:
                metadata = {}
            quality_profile = metadata.get('quality_profile', {})
            self.quality_weighting.register_expert_quality_profile(expert_id, quality_profile)
        
        # Register with Meta_Learner if enabled
        if self.use_meta_learner:
            self.meta_learner_gating.register_expert(expert_id, expert, metadata)
        
        logger.info(f"Registered expert {expert_id} ({expert.name}) with gating network")
    
    def fit(self, X: pd.DataFrame, expert_targets: Dict[str, np.ndarray], 
            optimize_weights: bool = False, **kwargs) -> 'GatingNetwork':
        """
        Fit the gating network to determine optimal weights for expert models.
        
        Args:
            X: Feature data
            expert_targets: Dictionary mapping expert IDs to target weights
            optimize_weights: Whether to optimize weights using evolutionary algorithms
            **kwargs: Additional keyword arguments
                quality_metrics: Dict of data quality metrics for quality-aware weighting
                meta_context: Dict of contextual information for Meta_Learner
                optimizer_config: Configuration for the optimizer if optimize_weights is True
                fit_stacking_model: Whether to fit the stacking model (default: True if strategy is ensemble_stacking)
            
        Returns:
            Self for method chaining
        """
        if not self.experts:
            raise ValueError("No expert models registered with the gating network")
        
        if not self.feature_columns:
            self.feature_columns = list(X.columns)
        
        # Initialize quality-aware weighting if enabled
        if self.use_quality_weighting:
            quality_metrics = kwargs.get('quality_metrics', {})
            self.quality_weighting.fit(X, expert_targets, quality_metrics)
        
        # Initialize Meta_Learner gating if enabled
        if self.use_meta_learner:
            meta_context = kwargs.get('meta_context', {})
            self.meta_learner_gating.fit(X, expert_targets, meta_context)
        
        # Prepare feature data for the base model
        X_scaled = self._preprocess_features(X)
        
        # Prepare target data (expert weights)
        y = np.array(list(expert_targets.values())).T
        
        # Fit the base model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Fit the stacking model if requested or if using ensemble_stacking
        fit_stacking = kwargs.get('fit_stacking_model', self.combination_strategy == 'ensemble_stacking')
        if fit_stacking:
            self._fit_stacking_model(X, expert_targets, **kwargs)
        
        # Optimize weights if requested
        if optimize_weights:
            self._optimize_weights(X, expert_targets, **kwargs)
        
        return self
        
    def _fit_stacking_model(self, X: pd.DataFrame, expert_targets: Dict[str, np.ndarray], **kwargs) -> None:
        """
        Fit the stacking model for ensemble stacking combination strategy.
        
        Args:
            X: Feature data
            expert_targets: Dictionary mapping expert IDs to target weights
            **kwargs: Additional keyword arguments
                y_true: True target values (if available)
        """
        logger.info("Fitting stacking model for ensemble stacking strategy")
        
        # Get predictions from each expert
        expert_predictions = {}
        for expert_id, expert in self.experts.items():
            expert_predictions[expert_id] = expert.predict(X)
        
        # Prepare meta-features (expert predictions)
        meta_features = np.column_stack([expert_predictions[expert_id] for expert_id in self.experts.keys()])
        
        # Add some original features if available
        if isinstance(X, pd.DataFrame) and len(X) > 0:
            # Select a subset of original features to include
            if self.feature_columns and len(self.feature_columns) > 0:
                selected_features = X[self.feature_columns].select_dtypes(include=['number'])
                if not selected_features.empty:
                    meta_features = np.column_stack([meta_features, selected_features.values])
        
        # Get target values
        y_true = kwargs.get('y_true')
        if y_true is None:
            # If true targets not provided, use average of expert targets
            y_true = np.array(list(expert_targets.values())).mean(axis=0)
        
        # Fit the stacking model
        self.stacking_model.fit(meta_features, y_true)
        self.stacking_model_fitted = True
        
        logger.info("Stacking model fitted successfully")
    
    def predict_weights(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Predict weights for each expert model based on input features.
        
        Args:
            X: Feature data
            
        Returns:
            Dictionary mapping expert IDs to weight arrays
        """
        if not self.is_fitted:
            raise ValueError("Gating network must be fitted before predicting weights")
        
        # Use Meta_Learner for adaptive weighting if enabled
        if self.use_meta_learner:
            weights = self.meta_learner_gating.predict_weights(X)
            
            # Ensure all experts have weights
            for expert_id in self.experts.keys():
                if expert_id not in weights:
                    weights[expert_id] = np.zeros(len(X))
        else:
            # Preprocess features
            X_scaled = self._preprocess_features(X)
            
            # Predict raw weights
            raw_weights = self.model.predict(X_scaled)
            
            # Ensure weights are valid (non-negative and sum to 1)
            normalized_weights = self._normalize_weights(raw_weights)
            
            # Create dictionary mapping expert IDs to weight arrays
            expert_ids = list(self.experts.keys())
            weights = {expert_id: normalized_weights[:, i] for i, expert_id in enumerate(expert_ids)}
        
        # Apply quality-aware weighting if enabled
        if self.use_quality_weighting:
            weights = self.quality_weighting.adjust_weights(X, weights)
        
        return weights
    
    def combine_predictions(self, X: pd.DataFrame, 
                           expert_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine expert predictions using the specified combination strategy.
        
        Args:
            X: Feature data
            expert_predictions: Dictionary mapping expert IDs to prediction arrays
            
        Returns:
            Combined predictions as a numpy array
        """
        weights = self.predict_weights(X)
        
        if self.combination_strategy == "weighted_average":
            return self._weighted_average(weights, expert_predictions)
        elif self.combination_strategy == "stacking":
            return self._stacking(X, expert_predictions)
        elif self.combination_strategy == "confidence_weighted":
            return self._confidence_weighted(X, expert_predictions)
        elif self.combination_strategy == "dynamic_selection":
            return self._dynamic_selection(X, expert_predictions)
        elif self.combination_strategy == "ensemble_stacking":
            return self._ensemble_stacking(X, expert_predictions)
        else:
            raise ValueError(f"Unsupported combination strategy: {self.combination_strategy}")
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input features for the gating network.
        
        Args:
            X: Feature data
            
        Returns:
            Preprocessed feature data as a numpy array
        """
        # Select relevant features
        if self.feature_columns:
            X = X[self.feature_columns]
        
        # Handle datetime columns by removing them or converting to numeric
        if isinstance(X, pd.DataFrame):
            # Filter out datetime columns
            numeric_cols = X.select_dtypes(include=['number']).columns
            if len(numeric_cols) < len(X.columns):
                logger.warning(f"Removing non-numeric columns for gating network: {set(X.columns) - set(numeric_cols)}")
                X = X[numeric_cols]
            
            # Convert to numpy array
            X_array = X.values
        else:
            X_array = X
        
        # Scale features
        if not self.is_fitted:
            return self.scaler.fit_transform(X_array)
        else:
            return self.scaler.transform(X_array)
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to ensure they are valid.
        
        Args:
            weights: Raw weight predictions
            
        Returns:
            Normalized weights
        """
        # Apply min/max constraints
        weights = np.clip(weights, self.weight_constraints["min"], self.weight_constraints["max"])
        
        # Ensure weights sum to 1 for each sample
        row_sums = weights.sum(axis=1, keepdims=True)
        normalized_weights = weights / row_sums
        
        return normalized_weights
    
    def _weighted_average(self, weights: Dict[str, np.ndarray], 
                         predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions using weighted average.
        
        Args:
            weights: Dictionary mapping expert IDs to weight arrays
            predictions: Dictionary mapping expert IDs to prediction arrays
            
        Returns:
            Combined predictions
        """
        # Get the shape of the predictions to initialize the combined array correctly
        first_pred = next(iter(predictions.values()))
        combined = np.zeros_like(first_pred)
        
        for expert_id, expert_weights in weights.items():
            if expert_id in predictions:
                # Ensure weights have the right shape for broadcasting
                # If predictions are 1D, weights should be 1D
                # If predictions are 2D, weights should be shaped to broadcast correctly
                pred = predictions[expert_id]
                if len(pred.shape) == 1:
                    # For 1D predictions, simply multiply
                    combined += expert_weights * pred
                else:
                    # For 2D predictions, reshape weights for proper broadcasting
                    # This assumes weights are per-sample and predictions are per-sample
                    combined += (expert_weights.reshape(-1, 1) * pred 
                               if pred.shape[0] == len(expert_weights) else expert_weights * pred)
        
        return combined
    
    def _stacking(self, X: pd.DataFrame, 
                 predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions using stacking approach.
        
        Args:
            X: Feature data
            predictions: Dictionary mapping expert IDs to prediction arrays
            
        Returns:
            Combined predictions
        """
        # For stacking, we use the gating network to learn a non-linear combination
        # of expert predictions based on input features
        
        # This is a simplified implementation - in practice, you might want to
        # train a separate stacking model
        
        weights = self.predict_weights(X)
        return self._weighted_average(weights, predictions)
    
    def _confidence_weighted(self, X: pd.DataFrame, 
                            predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions using confidence-weighted approach.
        
        Args:
            X: Feature data
            predictions: Dictionary mapping expert IDs to prediction arrays
            
        Returns:
            Combined predictions
        """
        weights = self.predict_weights(X)
        
        # Adjust weights based on expert confidence
        confidence_adjusted_weights = {}
        
        for expert_id, expert in self.experts.items():
            if hasattr(expert, 'predict_with_confidence') and expert_id in predictions:
                # Get confidence scores from expert
                _, confidence = expert.predict_with_confidence(X)
                
                # Adjust weights by confidence
                confidence_adjusted_weights[expert_id] = weights[expert_id] * confidence
        
        # Normalize adjusted weights
        total_weights = sum(confidence_adjusted_weights.values())
        normalized_weights = {k: v / total_weights for k, v in confidence_adjusted_weights.items()}
        
        return self._weighted_average(normalized_weights, predictions)
        
    def _dynamic_selection(self, X: pd.DataFrame,
                          predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Dynamically select the best expert for each sample based on predicted performance.
        
        This strategy selects a single expert for each sample rather than combining multiple
        expert predictions. It's useful when experts have clear domains of expertise.
        
        Args:
            X: Feature data
            predictions: Dictionary mapping expert IDs to prediction arrays
            
        Returns:
            Predictions from the selected experts for each sample
        """
        # Get weights from the gating network
        weights = self.predict_weights(X)
        
        # For each sample, select the expert with the highest weight
        num_samples = len(next(iter(predictions.values())))
        selected_predictions = np.zeros(num_samples)
        
        for i in range(num_samples):
            # Find the expert with the highest weight for this sample
            best_expert_id = max(weights.keys(), key=lambda expert_id: weights[expert_id][i])
            
            # Use that expert's prediction
            selected_predictions[i] = predictions[best_expert_id][i]
        
        return selected_predictions
    
    def _ensemble_stacking(self, X: pd.DataFrame,
                          predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions using an ensemble stacking approach.
        
        This strategy trains a meta-model to learn the optimal combination of expert
        predictions based on input features. It's more sophisticated than simple
        weighted averaging and can capture non-linear relationships.
        
        Args:
            X: Feature data
            predictions: Dictionary mapping expert IDs to prediction arrays
            
        Returns:
            Combined predictions from the stacking model
        """
        # If the stacking model hasn't been fitted yet, we can't use it
        if not self.stacking_model_fitted:
            logger.warning("Stacking model not fitted yet, falling back to weighted average")
            weights = self.predict_weights(X)
            return self._weighted_average(weights, predictions)
        
        # Prepare the meta-features (expert predictions)
        meta_features = np.column_stack([predictions[expert_id] for expert_id in self.experts.keys()])
        
        # Add some original features if available
        if isinstance(X, pd.DataFrame) and len(X) > 0:
            # Select a subset of original features to include
            if self.feature_columns and len(self.feature_columns) > 0:
                selected_features = X[self.feature_columns].select_dtypes(include=['number'])
                if not selected_features.empty:
                    meta_features = np.column_stack([meta_features, selected_features.values])
        
        # Use the stacking model to make the final prediction
        return self.stacking_model.predict(meta_features)
    
    def _optimize_weights(self, X: pd.DataFrame, expert_targets: Dict[str, np.ndarray], 
                         optimizer=None, **kwargs) -> None:
        """
        Optimize gating network weights using evolutionary algorithms.
        
        Args:
            X: Feature data
            expert_targets: Dictionary mapping expert IDs to target weights
            optimizer: Optimizer instance (default: None, will use appropriate optimizer from factory)
            **kwargs: Additional keyword arguments for the optimizer
                optimizer_type: Type of optimizer to use (default: 'gwo')
                optimizer_config: Configuration for the optimizer
                meta_context: Contextual information for Meta_Learner
                true_targets: Actual target values for evaluation (if available)
                eval_metric: Metric to use for evaluation ('mse', 'rmse', 'mae', etc.)
        """
        logger.info("Optimizing gating network weights")
        
        # If using Meta_Learner for optimization
        if self.use_meta_learner and kwargs.get('use_meta_learner_optimization', True):
            meta_context = kwargs.get('meta_context', {})
            optimizer_config = kwargs.get('optimizer_config', {})
            
            # Let Meta_Learner handle the optimization
            optimization_result = self.meta_learner_gating.optimize_weights(
                X, expert_targets, meta_context, optimizer_config)
            
            logger.info(f"Meta_Learner optimization completed with score: {optimization_result.get('best_score', 'N/A')}")
            return
        
        # Get expert IDs and number of experts
        expert_ids = list(self.experts.keys())
        num_experts = len(expert_ids)
        
        # Get true targets if available, otherwise use average of expert targets
        y_true = kwargs.get('true_targets')
        if y_true is None:
            y_true = np.array(list(expert_targets.values())).mean(axis=0)
            
        # Get evaluation metric
        eval_metric = kwargs.get('eval_metric', 'mse')
        
        # Get expert predictions
        expert_predictions = {}
        for expert_id, expert in self.experts.items():
            expert_predictions[expert_id] = expert.predict(X)
        
        # Otherwise, use the specified optimizer or create one from the factory
        if optimizer is None:
            optimizer_type = kwargs.get('optimizer_type', 'gwo')
            optimizer_config = kwargs.get('optimizer_config', {})
            
            # Set default optimizer configuration if not provided
            if 'dim' not in optimizer_config:
                optimizer_config['dim'] = num_experts
                
            if 'bounds' not in optimizer_config:
                # Set bounds to ensure weights are between 0 and 1
                optimizer_config['bounds'] = [(0.0, 1.0)] * num_experts
                
            if 'population_size' not in optimizer_config:
                # Set population size based on problem dimension
                optimizer_config['population_size'] = max(30, 10 * num_experts)
                
            if 'max_evals' not in optimizer_config:
                # Set maximum evaluations
                optimizer_config['max_evals'] = 1000
                
            if 'adaptive' not in optimizer_config:
                # Enable adaptive parameters
                optimizer_config['adaptive'] = True
                
            if 'verbose' not in optimizer_config:
                # Disable verbose output
                optimizer_config['verbose'] = False
            
            # Create optimizer from factory
            factory = OptimizerFactory()
            optimizer = factory.create_optimizer(optimizer_type, **optimizer_config)
        
        # Prepare evaluation function with proper weight normalization
        def evaluate_weights(weights):
            # Normalize weights to ensure they sum to 1
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                normalized_weights = weights / weights_sum
            else:
                # If all weights are 0, use uniform weights
                normalized_weights = np.ones_like(weights) / len(weights)
            
            # Create weight dictionary
            weight_dict = {expert_id: normalized_weights[i] for i, expert_id in enumerate(expert_ids)}
            
            # Calculate combined prediction
            combined = np.zeros_like(y_true)
            for expert_id, expert_weight in weight_dict.items():
                combined += expert_weight * expert_predictions[expert_id]
            
            # Calculate error based on specified metric
            if eval_metric == 'mse':
                error = np.mean((y_true - combined) ** 2)
            elif eval_metric == 'rmse':
                error = np.sqrt(np.mean((y_true - combined) ** 2))
            elif eval_metric == 'mae':
                error = np.mean(np.abs(y_true - combined))
            else:
                # Default to MSE
                error = np.mean((y_true - combined) ** 2)
            
            return error
        
        # Run optimization
        result = optimizer.run(evaluate_weights)
        
        # Update model with optimized weights if available
        if 'solution' in result and result['solution'] is not None:
            logger.info(f"Optimization completed with score: {result.get('score', 'N/A')}")
            
            # Get optimized weights
            optimized_weights = np.array(result['solution'])
            
            # Normalize weights to ensure they sum to 1
            weights_sum = np.sum(optimized_weights)
            if weights_sum > 0:
                normalized_weights = optimized_weights / weights_sum
            else:
                normalized_weights = np.ones_like(optimized_weights) / len(optimized_weights)
            
            # Update the model with optimized weights
            # We need to update the model coefficients to reflect these optimized weights
            # This assumes the model predicts weights directly
            if hasattr(self.model, 'coef_'):
                # For linear models, we can directly update coefficients
                # This is a simplified approach and may need to be adapted based on the model type
                X_scaled = self._preprocess_features(X)
                
                # For each expert, update the model coefficients to predict the optimized weight
                for i, expert_id in enumerate(expert_ids):
                    target_weight = normalized_weights[i]
                    
                    # Create a simple regression problem to find coefficients that predict this weight
                    from sklearn.linear_model import Ridge
                    temp_model = Ridge(alpha=1.0)
                    temp_model.fit(X_scaled, np.ones(len(X_scaled)) * target_weight)
                    
                    # Update the coefficients for this expert
                    if len(self.model.coef_.shape) > 1:  # Multi-output model
                        self.model.coef_[i] = temp_model.coef_
                        if hasattr(self.model, 'intercept_'):
                            self.model.intercept_[i] = temp_model.intercept_
                    else:  # Single output model
                        self.model.coef_ = temp_model.coef_
                        if hasattr(self.model, 'intercept_'):
                            self.model.intercept_ = temp_model.intercept_
                
                logger.info(f"Updated model coefficients with optimized weights: {normalized_weights}")
            else:
                logger.warning("Could not update model coefficients - model type not supported")
        else:
            logger.warning("Optimization did not produce valid weights")
    
    def evaluate(self, X: pd.DataFrame, y_true: np.ndarray, 
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the gating network on test data.
        
        Args:
            X: Feature data
            y_true: True target values
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric names and values
        """
        if not self.is_fitted:
            raise ValueError("Gating network must be fitted before evaluation")
        
        # Get predictions from each expert
        expert_predictions = {}
        for expert_id, expert in self.experts.items():
            expert_predictions[expert_id] = expert.predict(X)
        
        # Combine predictions using the gating network
        y_pred = self.combine_predictions(X, expert_predictions)
        
        # Calculate metrics
        metrics = metrics or ['mse', 'rmse', 'mae']
        results = {}
        
        if 'mse' in metrics:
            results['mse'] = np.mean((y_true - y_pred) ** 2)
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Store metrics
        self.quality_metrics.update(results)
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save the gating network to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare the object for serialization
        # Note: We don't save the expert models themselves, just their IDs
        expert_ids = list(self.experts.keys())
        
        # Store the current default weights for each expert to ensure consistency after loading
        default_weights = {}
        if self.is_fitted and hasattr(self, 'experts') and self.experts:
            # Create a small sample dataframe to get the default weights
            sample_X = pd.DataFrame(np.zeros((1, 1)), columns=['dummy_feature'])
            try:
                default_weights = self.predict_weights(sample_X)
                # Convert numpy arrays to lists for serialization
                default_weights = {k: v.tolist() for k, v in default_weights.items()}
            except Exception as e:
                logger.warning(f"Could not capture default weights: {e}")
        
        save_dict = {
            'name': self.name,
            'model': self.model,
            'expert_ids': expert_ids,
            'combination_strategy': self.combination_strategy,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'quality_metrics': self.quality_metrics,
            'metadata': self.metadata,
            'training_history': self.training_history,
            'validation_scores': self.validation_scores,
            'weight_constraints': self.weight_constraints,
            'use_quality_weighting': self.use_quality_weighting,
            'use_meta_learner': self.use_meta_learner,
            'default_weights': default_weights
        }
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        # Save quality weighting if enabled
        if self.use_quality_weighting:
            quality_weighting_path = os.path.join(os.path.dirname(filepath), f"{self.name}_quality_weighting.pkl")
            with open(quality_weighting_path, 'wb') as f:
                pickle.dump(self.quality_weighting, f)
        
        # Save meta learner gating if enabled
        if self.use_meta_learner:
            meta_learner_path = os.path.join(os.path.dirname(filepath), f"{self.name}_meta_learner_gating.pkl")
            self.meta_learner_gating.save(meta_learner_path)
        
        logger.info(f"Gating network '{self.name}' saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, experts: Dict[str, BaseExpert] = None) -> 'GatingNetwork':
        """
        Load a gating network from disk.
        
        Args:
            filepath: Path to load the model from
            experts: Dictionary of expert models to register with the loaded gating network
            
        Returns:
            Loaded gating network
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load from disk
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Check for quality weighting and meta learner
        use_quality_weighting = save_dict.get('use_quality_weighting', False)
        use_meta_learner = save_dict.get('use_meta_learner', False)
        
        # Load quality weighting if enabled
        quality_weighting = None
        if use_quality_weighting:
            quality_weighting_path = os.path.join(os.path.dirname(filepath), f"{save_dict['name']}_quality_weighting.pkl")
            if os.path.exists(quality_weighting_path):
                with open(quality_weighting_path, 'rb') as f:
                    quality_weighting = pickle.load(f)
        
        # Load meta learner gating if enabled
        meta_learner_gating = None
        if use_meta_learner:
            meta_learner_path = os.path.join(os.path.dirname(filepath), f"{save_dict['name']}_meta_learner_gating.pkl")
            if os.path.exists(meta_learner_path):
                meta_learner_gating = MetaLearnerGating.load(meta_learner_path)
        
        # Create a new instance
        instance = cls(
            name=save_dict['name'],
            model=save_dict['model'],
            experts=None,  # We'll register experts separately
            combination_strategy=save_dict['combination_strategy'],
            feature_columns=save_dict['feature_columns'],
            metadata=save_dict['metadata'],
            quality_weighting=quality_weighting,
            meta_learner_gating=meta_learner_gating,
            use_quality_weighting=use_quality_weighting,
            use_meta_learner=use_meta_learner
        )
        
        # Restore state
        instance.scaler = save_dict['scaler']
        instance.is_fitted = save_dict['is_fitted']
        instance.quality_metrics = save_dict['quality_metrics']
        instance.training_history = save_dict['training_history']
        instance.validation_scores = save_dict['validation_scores']
        instance.weight_constraints = save_dict['weight_constraints']
        
        # Register experts if provided
        if experts:
            for expert_id, expert in experts.items():
                if expert_id in save_dict['expert_ids']:
                    instance.register_expert(expert_id, expert)
        
        # Store the default weights in the meta_learner_gating to ensure consistent predictions
        default_weights = save_dict.get('default_weights', {})
        if default_weights and instance.use_meta_learner and instance.meta_learner_gating:
            # Convert lists back to numpy arrays
            default_weights = {k: np.array(v) for k, v in default_weights.items()}
            
            # Store the weights in the meta_learner_gating for consistent predictions
            if hasattr(instance.meta_learner_gating, 'meta_learner'):
                # Create a method to override the predict_weights method with consistent weights
                original_predict_weights = instance.meta_learner_gating.predict_weights
                
                def consistent_predict_weights(X):
                    # Get the shape from the input data
                    n_samples = len(X)
                    
                    # Create equal weights if we don't have stored weights or they're invalid
                    if not default_weights or any(np.any(weight < 0) for weight in default_weights.values()):
                        n_experts = len(instance.experts)
                        equal_weight = 1.0 / n_experts if n_experts > 0 else 1.0
                        return {expert_id: np.ones(n_samples) * equal_weight 
                                for expert_id in instance.experts.keys()}
                    
                    # Return the stored weights with the right shape
                    return {expert_id: np.ones(n_samples) * max(0, weight[0]) 
                            for expert_id, weight in default_weights.items()}
                
                # Override the predict_weights method
                instance.meta_learner_gating.predict_weights = consistent_predict_weights
        
        logger.info(f"Loaded gating network '{instance.name}' from {filepath}")
        return instance

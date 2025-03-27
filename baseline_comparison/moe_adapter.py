"""MoE Baseline Adapter

This module provides an adapter to integrate the Mixture of Experts (MoE) framework
with the baseline comparison framework, allowing direct performance comparison
between MoE and other algorithm selection approaches.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union

# Import baseline comparison components
from baseline_comparison.comparison_runner import BaselineComparison

# Import MoE components
from moe_framework.workflow.moe_pipeline import MoEPipeline
from moe_framework.validation.time_series_validation import TimeSeriesValidator
from moe_framework.interfaces.base import PatientContext
from moe_framework.integration.event_system import MoEEventTypes, EventListener

# Import MoE metrics
from baseline_comparison.moe_metrics import MoEMetricsCalculator

logger = logging.getLogger(__name__)


class MoEBaselineAdapter:
    """
    Adapter class that allows the MoE framework to be used within the baseline
    comparison framework for standardized performance evaluation.
    
    This adapter translates between the interfaces of the MoE framework and the
    baseline comparison framework, allowing MoE to be compared directly against
    other algorithm selection approaches.
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        model_path: Optional[str] = None,
        verbose: bool = False,
        metrics_output_dir: str = "results/moe_metrics"
    ):
        """
        Initialize the MoE adapter.
        
        Args:
            config: Configuration dictionary for the MoE pipeline
            model_path: Optional path to a pre-trained MoE model to load
            verbose: Whether to display detailed logs
            metrics_output_dir: Directory to save metrics and visualizations
        """
        self.config = config or {}
        self.verbose = verbose
        self.model_path = model_path
        
        # Initialize the MoE pipeline
        self.moe_pipeline = MoEPipeline(config=self.config, verbose=self.verbose)
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            try:
                self.moe_pipeline.load_from_checkpoint(model_path)
                logger.info(f"Loaded MoE model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load MoE model from {model_path}: {e}")
        
        # Track selected algorithms for reporting
        self.selected_algorithms = []
        
        # Initialize TimeSeriesValidator for consistent validation
        self.time_series_validator = TimeSeriesValidator(
            time_column=self.config.get('time_column', 'timestamp'),
            patient_column=self.config.get('patient_column', 'patient_id'),
            gap_size=self.config.get('gap_size', 0)
        )
        
        # Initialize MoE metrics calculator
        self.metrics_calculator = MoEMetricsCalculator(output_dir=metrics_output_dir)
        
        # Track data for metrics computation
        self.last_predictions = None
        self.last_true_values = None
        self.last_expert_contributions = {}
        self.last_expert_errors = {}
        self.last_confidence_scores = None
        self.last_timestamps = None
        self.last_patient_ids = None
        self.last_computed_metrics = None
        
        # Register event listener to track which experts are being used
        self.moe_pipeline.event_manager.register_listener(
            EventListener(
                event_type=MoEEventTypes.EXPERT_SELECTED,
                callback=self._track_expert_selection
            )
        )
        
        if self.verbose:
            logger.info("Initialized MoEBaselineAdapter")
    
    def _track_expert_selection(self, event):
        """Track which experts are selected during prediction."""
        if hasattr(event, 'data') and 'expert_name' in event.data:
            self.selected_algorithms.append(event.data['expert_name'])
    
    def get_available_algorithms(self) -> List[str]:
        """
        Get the list of available algorithms (experts) in the MoE framework.
        
        Returns:
            List of algorithm names
        """
        # Return names of available experts
        return list(self.moe_pipeline.experts.keys())
    
    def set_available_algorithms(self, algorithms: List[str]) -> None:
        """
        Set the list of available algorithms (experts).
        
        Args:
            algorithms: List of algorithm names to use
        """
        # Filter experts to only include the specified ones
        # This is a no-op if the expert doesn't exist
        current_experts = set(self.moe_pipeline.experts.keys())
        for expert_name in list(self.moe_pipeline.experts.keys()):
            if expert_name not in algorithms and expert_name in current_experts:
                logger.info(f"Removing expert {expert_name} as it's not in the allowed algorithms list")
                self.moe_pipeline.experts.pop(expert_name)
        
        if self.verbose:
            logger.info(f"Set available algorithms to: {algorithms}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the MoE pipeline on the provided data.
        
        Args:
            X: Features DataFrame
            y: Target values Series
        """
        # Reset tracking of selected algorithms
        self.selected_algorithms = []
        
        # Combine X and y for MoE training
        data = X.copy()
        target_column = self.config.get('target_column', 'target')
        data[target_column] = y
        
        # Train the MoE pipeline
        self.moe_pipeline.train(data)
        
        if self.verbose:
            logger.info("MoE pipeline trained successfully")
    
    def predict(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Generate predictions using the MoE pipeline.
        
        Args:
            X: Features DataFrame
            y: Optional ground truth values for tracking metrics
            
        Returns:
            Array of predictions
        """
        # Reset tracking of selected algorithms
        self.selected_algorithms = []
        
        # Make predictions using the MoE pipeline
        predictions = self.moe_pipeline.predict(X)
        
        # Store information for metrics computation
        self.last_predictions = predictions
        
        # If ground truth is provided, store it
        if y is not None:
            self.last_true_values = y.values
        
        # Get expert contributions
        expert_weights = self.moe_pipeline.get_expert_weights_batch(X)
        if expert_weights:
            self.last_expert_contributions = expert_weights
        
        # Get individual expert predictions and errors if ground truth is available
        if hasattr(self.moe_pipeline, 'get_expert_predictions') and y is not None:
            expert_predictions = self.moe_pipeline.get_expert_predictions(X)
            self.last_expert_errors = {}
            
            for expert_name, expert_preds in expert_predictions.items():
                self.last_expert_errors[expert_name] = (y.values - expert_preds).tolist()
        
        # Get confidence scores
        if hasattr(self.moe_pipeline, 'get_prediction_confidence'):
            self.last_confidence_scores = self.moe_pipeline.get_prediction_confidence(X)
        else:
            # Default confidence based on expert agreement
            self.last_confidence_scores = np.ones_like(predictions)  # Default confidence of 1.0
        
        # Store timestamp and patient information if available in the data
        if 'timestamp' in X.columns:
            self.last_timestamps = X['timestamp'].values
        elif self.config.get('time_column') in X.columns:
            self.last_timestamps = X[self.config.get('time_column')].values
            
        patient_col = self.config.get('patient_column', 'patient_id')
        if patient_col in X.columns:
            self.last_patient_ids = X[patient_col].values
        
        return predictions
    
    def compute_metrics(self, name: str = "moe_metrics") -> Dict[str, Any]:
        """
        Compute comprehensive MoE-specific metrics based on the latest predictions.
        
        Args:
            name: Name for saved metrics and visualizations
            
        Returns:
            Dictionary of computed metrics
        """
        if self.last_predictions is None or self.last_true_values is None:
            logger.warning("Cannot compute metrics: no predictions or true values available")
            return {}
            
        # Compute all available metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            predictions=self.last_predictions,
            actual_values=self.last_true_values,
            expert_contributions=self.last_expert_contributions,
            confidence_scores=self.last_confidence_scores,
            expert_errors=self.last_expert_errors,
            timestamps=self.last_timestamps,
            patient_ids=self.last_patient_ids
        )
        
        # Save metrics and generate visualizations
        self.metrics_calculator.save_metrics(metrics, name)
        self.metrics_calculator.visualize_metrics(metrics, name)
        
        # Store the computed metrics for future reference
        self.last_computed_metrics = metrics
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summarized performance report of the MoE model.
        
        Returns:
            Dictionary containing key performance indicators
        """
        summary = {}
        
        # Check if we have metrics available
        if hasattr(self, 'last_computed_metrics') and self.last_computed_metrics:
            metrics = self.last_computed_metrics
        else:
            # Compute metrics if not already done
            metrics = self.compute_metrics()
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Extract key performance indicators
        if "standard" in metrics:
            summary.update(metrics["standard"])
        
        # Add MoE-specific summary metrics
        if "expert_contribution" in metrics:
            expert_contrib = metrics["expert_contribution"]
            summary["expert_distribution"] = expert_contrib.get("expert_dominance_percentage", {})
            summary["expert_diversity"] = expert_contrib.get("normalized_entropy", 0)
        
        if "confidence" in metrics:
            summary["mean_confidence"] = metrics["confidence"].get("mean_confidence", 0)
            summary["confidence_calibration"] = 1.0 - metrics["confidence"].get("expected_calibration_error", 0)
        
        if "gating_network" in metrics:
            gating_metrics = metrics["gating_network"]
            summary["optimal_selection_rate"] = gating_metrics.get("optimal_expert_selection_rate", 0)
            summary["mean_regret"] = gating_metrics.get("mean_regret", 0)
        
        if "personalization" in metrics:
            pers_metrics = metrics["personalization"]
            summary["per_patient_variability"] = pers_metrics.get("patient_rmse_std", 0)
        
        return summary
    
    def select_algorithm(self, problem_instance: Any) -> str:
        """
        Select the best algorithm (expert) for the given problem instance.
        
        This method is required for compatibility with the baseline comparison framework.
        For MoE, this will return the expert with the highest weight from the gating network.
        
        Args:
            problem_instance: The problem to solve
            
        Returns:
            Name of the selected algorithm
        """
        # Convert problem instance to DataFrame if needed
        if not isinstance(problem_instance, pd.DataFrame):
            # Attempt to convert to DataFrame
            try:
                features = problem_instance.get_features()
                problem_df = pd.DataFrame(features)
            except (AttributeError, TypeError):
                # Fall back to running a prediction and returning the most used expert
                logger.warning("Could not convert problem instance to DataFrame. Using default expert selection.")
                if not self.selected_algorithms:
                    # If no algorithms have been selected yet, return the first available
                    available_algos = self.get_available_algorithms()
                    return available_algos[0] if available_algos else "unknown"
                # Return the most frequently selected algorithm
                from collections import Counter
                return Counter(self.selected_algorithms).most_common(1)[0][0]
        else:
            problem_df = problem_instance
            
        # Get expert weights from the gating network
        context = PatientContext()  # Create empty context
        weights = self.moe_pipeline.get_expert_weights(problem_df, context)
        
        # Return the expert with the highest weight
        if weights:
            # Sort experts by weight (descending)
            sorted_experts = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            selected_expert = sorted_experts[0][0]
            
            # Record the selection
            self.selected_algorithms.append(selected_expert)
            
            return selected_expert
        else:
            # Fallback if no weights are available
            available_algos = self.get_available_algorithms()
            return available_algos[0] if available_algos else "unknown"
    
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.Series, 
                      n_splits: int = 5, 
                      method: str = 'patient_aware') -> Dict[str, Any]:
        """
        Perform time-series aware cross-validation using the MoE pipeline.
        
        Args:
            X: Features DataFrame
            y: Target values Series
            n_splits: Number of CV splits
            method: Validation method ('rolling_window', 'patient_aware', or 'expanding_window')
            
        Returns:
            Dictionary with validation scores
        """
        # Combine X and y for validation
        data = X.copy()
        target_column = self.config.get('target_column', 'target')
        data[target_column] = y
        
        # Get features list (excluding target)
        features = [col for col in data.columns if col != target_column]
        
        # Get scoring metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        scoring_functions = {
            'mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
            'r2': lambda y_true, y_pred: r2_score(y_true, y_pred)
        }
        
        # Perform cross-validation
        return self.time_series_validator.get_validation_scores(
            data=data,
            model=self.moe_pipeline,
            features=features,
            target=target_column,
            method=method,
            n_splits=n_splits,
            scoring_functions=scoring_functions
        )
    
    def run_optimizer(self, problem_instance: Any, max_evaluations: int = 1000) -> Dict[str, Any]:
        """
        Run the MoE pipeline on a problem instance.
        
        This method is required for compatibility with the baseline comparison framework.
        
        Args:
            problem_instance: The problem to solve
            max_evaluations: Maximum number of evaluations (ignored for MoE)
            
        Returns:
            Dictionary with results
        """
        # Reset tracking of selected algorithms
        self.selected_algorithms = []
        
        # Convert problem instance to DataFrame if needed
        if not isinstance(problem_instance, pd.DataFrame):
            # Attempt to convert to DataFrame
            try:
                features = problem_instance.get_features()
                X = pd.DataFrame(features)
                y = problem_instance.get_target() if hasattr(problem_instance, 'get_target') else None
            except (AttributeError, TypeError):
                logger.error("Cannot run MoE optimizer: problem instance could not be converted to DataFrame")
                return {
                    "best_fitness": float('inf'),
                    "evaluations": 0,
                    "convergence_data": [],
                    "selected_algorithm": "none"
                }
        else:
            X = problem_instance
            # Extract target column if available
            target_column = self.config.get('target_column', 'target')
            if target_column in X.columns:
                y = X[target_column]
                X = X.drop(columns=[target_column])
            else:
                y = None
                
        # If we have target values, train the model
        if y is not None:
            self.fit(X, y)
            
        # Generate predictions
        predictions = self.predict(X)
        
        # Calculate fitness (error) if target values are available
        if y is not None:
            from sklearn.metrics import mean_squared_error
            best_fitness = mean_squared_error(y, predictions)
        else:
            # No way to evaluate fitness without target values
            best_fitness = 0.0
            
        # Get the most selected algorithm
        from collections import Counter
        selected_algorithm = Counter(self.selected_algorithms).most_common(1)[0][0] if self.selected_algorithms else "unknown"
            
        return {
            "best_fitness": best_fitness,
            "evaluations": len(X),  # Each row counts as one evaluation
            "convergence_data": [],  # MoE doesn't track convergence in the same way
            "selected_algorithm": selected_algorithm
        }
    
    def save_model(self, path: str) -> None:
        """
        Save the MoE model to a file.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the MoE pipeline checkpoint
        self.moe_pipeline.save_checkpoint(path)
        
        if self.verbose:
            logger.info(f"Saved MoE model to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load the MoE model from a file.
        
        Args:
            path: Path to load the model from
        """
        if os.path.exists(path):
            try:
                # Load the MoE pipeline from checkpoint
                self.moe_pipeline.load_from_checkpoint(path)
                logger.info(f"Loaded MoE model from {path}")
            except Exception as e:
                logger.warning(f"Could not load MoE model from {path}: {e}")

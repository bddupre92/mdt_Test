"""
Physiological Expert Module

This module implements the PhysiologicalExpert class, which specializes in analyzing
and making predictions based on physiological data such as heart rate, blood pressure,
and other vital signs related to migraine prediction.
"""

import logging
import numpy as np
import pandas as pd
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from moe_framework.experts.base_expert import BaseExpert
from data.domain_specific_preprocessing import PhysiologicalSignalProcessor
from moe_framework.experts.optimizer_adapters import DifferentialEvolutionAdapter

# Configure logging
logger = logging.getLogger(__name__)


class PhysiologicalExpert(BaseExpert):
    """
    Expert model specializing in physiological data for migraine prediction.
    
    This expert focuses on physiological signals such as heart rate, blood pressure,
    and other vital signs to predict migraine occurrence or severity. It uses
    Differential Evolution for hyperparameter optimization.
    
    Attributes:
        vital_cols (List[str]): List of vital sign column names
        patient_id_col (str): Name of the patient ID column
        timestamp_col (str): Name of the timestamp column
        normalize_vitals (bool): Whether to normalize vital signs
        extract_variability (bool): Whether to extract variability features
        signal_processor (PhysiologicalSignalProcessor): Preprocessing for physiological signals
        scaler (StandardScaler): Scaler for normalizing features
    """
    
    def __init__(self, 
                 vital_cols: List[str],
                 patient_id_col: str = 'patient_id',
                 timestamp_col: str = 'date',
                 normalize_vitals: bool = True,
                 extract_variability: bool = True,
                 model: Optional[Any] = None,
                 name: str = "PhysiologicalExpert",
                 metadata: Dict[str, Any] = None):
        """
        Initialize the physiological expert.
        
        Args:
            vital_cols: List of vital sign column names
            patient_id_col: Name of the patient ID column
            timestamp_col: Name of the timestamp column
            normalize_vitals: Whether to normalize vital signs
            extract_variability: Whether to extract variability features
            model: The underlying machine learning model (default: RandomForestRegressor)
            name: Name of the expert model
            metadata: Additional metadata about the expert
        """
        # Initialize with default model if none provided
        if model is None:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        # Initialize base class
        super().__init__(
            name=name,
            model=model,
            metadata=metadata or {}
        )
        
        # Store physiological-specific parameters
        self.vital_cols = vital_cols
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.normalize_vitals = normalize_vitals
        self.extract_variability = extract_variability
        
        # Initialize preprocessing components
        self.signal_processor = PhysiologicalSignalProcessor(
            vital_cols=vital_cols,
            patient_id_col=patient_id_col,
            timestamp_col=timestamp_col,
            calculate_variability=extract_variability,
            calculate_trends=normalize_vitals  # Using normalize_vitals as calculate_trends
        )
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Initialize optimizer
        self.optimizer = None
        
        # Update metadata
        self.metadata.update({
            'vital_cols': vital_cols,
            'patient_id_col': patient_id_col,
            'timestamp_col': timestamp_col,
            'normalize_vitals': normalize_vitals,
            'extract_variability': extract_variability,
            'domain': 'physiological'
        })
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data using the physiological signal processor.
        
        Args:
            data: Input data containing physiological signals
            
        Returns:
            Preprocessed data with extracted physiological features
        """
        # Apply the physiological signal processor
        processed_data = self.signal_processor.fit_transform(data.copy())
        
        # Log the preprocessing results
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        logger.info(f"Added features: {set(processed_data.columns) - set(data.columns)}")
        
        return processed_data
    
    def extract_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract features for the physiological expert model.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Tuple of (feature_matrix, feature_column_names)
        """
        # Identify physiological feature columns
        feature_cols = []
        
        # Include original vital columns
        feature_cols.extend(self.vital_cols)
        
        # Include normalized vital columns if available
        norm_cols = [f"{col}_normalized" for col in self.vital_cols]
        feature_cols.extend([col for col in norm_cols if col in data.columns])
        
        # Include variability features if available
        var_cols = [col for col in data.columns if "_variability" in col or "_trend" in col]
        feature_cols.extend(var_cols)
        
        # Include anomaly features if available
        anomaly_cols = [col for col in data.columns if "_anomaly" in col or "_zscore" in col]
        feature_cols.extend(anomaly_cols)
        
        # Include rolling statistics if available
        rolling_cols = [col for col in data.columns if "_rolling_" in col and any(vital in col for vital in self.vital_cols)]
        feature_cols.extend(rolling_cols)
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Check if we have the necessary columns
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Extract feature matrix
        X = data[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        return X, feature_cols
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize_hyperparams: bool = False, **kwargs) -> 'PhysiologicalExpert':
        """
        Fit the physiological expert model to the data.
        
        Args:
            X: Feature data
            y: Target data
            optimize_hyperparams: Whether to optimize hyperparameters using DE
            **kwargs: Additional keyword arguments
            
        Returns:
            Self for method chaining
        """
        # Store target column name
        self.target_column = y.name
        
        # Preprocess data
        processed_data = self.preprocess_data(X.copy())
        
        # Extract features
        X_features, feature_cols = self.extract_features(processed_data)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_features),
            columns=feature_cols,
            index=X_features.index
        )
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            self._optimize_hyperparameters(X_scaled, y, **kwargs)
        
        # Fit the model
        logger.info(f"Fitting {self.name} model with {len(feature_cols)} features")
        # Convert to numpy array to avoid feature names warning
        self.model.fit(X_scaled.values, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        # Calculate feature importance
        self.calculate_feature_importance()
        
        # Log fitting results
        logger.info(f"{self.name} model fitted successfully")
        logger.info(f"Top 5 important features: {dict(sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions using the physiological expert model.
        
        Args:
            X: Feature data
            **kwargs: Additional keyword arguments
            
        Returns:
            Predictions as a numpy array
        """
        if not self.is_fitted:
            raise ValueError(f"Expert model '{self.name}' must be fitted before prediction.")
        
        # Preprocess data
        processed_data = self.preprocess_data(X.copy())
        
        # Extract features
        X_features, _ = self.extract_features(processed_data)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )
        
        # Generate predictions
        # Convert to numpy array to avoid feature names warning
        predictions = self.model.predict(X_scaled.values)
        
        return predictions
    
    def _calculate_confidence(self, X: pd.DataFrame, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        For random forest models, we use the standard deviation of predictions across trees
        as a measure of uncertainty, which we convert to confidence.
        
        Args:
            X: Feature data
            predictions: Model predictions
            **kwargs: Additional keyword arguments
            
        Returns:
            Confidence scores as a numpy array
        """
        # Preprocess data
        processed_data = self.preprocess_data(X.copy())
        
        # Extract features
        X_features, _ = self.extract_features(processed_data)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )
        
        # For random forest, use the standard deviation of predictions across trees
        if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
            # Get predictions from all trees - use numpy arrays to avoid feature names warning
            tree_preds = np.array([tree.predict(X_scaled.values) for tree in self.model.estimators_])
            # Calculate standard deviation across trees
            uncertainty = np.std(tree_preds, axis=0)
            # Convert uncertainty to confidence (higher uncertainty = lower confidence)
            max_uncertainty = np.max(uncertainty) if np.max(uncertainty) > 0 else 1.0
            confidence = 1.0 - (uncertainty / max_uncertainty)
            return confidence
        # For scikit-learn RandomForestRegressor with different attribute structure
        elif hasattr(self.model, 'estimator') and hasattr(self.model, 'n_estimators'):
            # Use a simpler confidence calculation based on model properties
            predictions = self.model.predict(X_scaled.values)
            # Calculate a proxy for confidence based on prediction values
            confidence = np.ones(len(predictions))  # Default high confidence
            return confidence
        
        # Default confidence calculation
        return super()._calculate_confidence(X, predictions, **kwargs)
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Optimize hyperparameters using Differential Evolution.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional keyword arguments
        """
        logger.info(f"Optimizing hyperparameters for {self.name} using Differential Evolution")
        
        # Define parameter bounds for RandomForestRegressor
        param_bounds = {
            'n_estimators': (50, 200),
            'max_depth': (5, 20),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10)
        }
        
        # Define fitness function (negative cross-validation score)
        def fitness_function(params):
            # Extract parameters
            n_estimators = int(params[0])
            max_depth = int(params[1])
            min_samples_split = int(params[2])
            min_samples_leaf = int(params[3])
            
            # Create model with these parameters
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            
            # Return negative mean score (DE minimizes)
            return -np.mean(cv_scores)
        
        # Initialize optimizer using the adapter
        self.optimizer = DifferentialEvolutionAdapter(
            fitness_function=fitness_function,
            bounds=[
                param_bounds['n_estimators'],
                param_bounds['max_depth'],
                param_bounds['min_samples_split'],
                param_bounds['min_samples_leaf']
            ],
            population_size=10,
            max_iterations=10,
            crossover_probability=0.7,
            differential_weight=0.8,
            random_seed=42
        )
        
        # Run optimization
        best_params, best_fitness = self.optimizer.optimize()
        
        # Update model with best parameters
        self.model = RandomForestRegressor(
            n_estimators=int(best_params[0]),
            max_depth=int(best_params[1]),
            min_samples_split=int(best_params[2]),
            min_samples_leaf=int(best_params[3]),
            random_state=42
        )
        
        # Log optimization results
        logger.info(f"Optimized hyperparameters: n_estimators={int(best_params[0])}, "
                   f"max_depth={int(best_params[1])}, min_samples_split={int(best_params[2])}, "
                   f"min_samples_leaf={int(best_params[3])}")
        logger.info(f"Best fitness (negative MSE): {best_fitness}")
        
        # Store optimization results in metadata
        self.metadata['optimization_results'] = {
            'best_params': {
                'n_estimators': int(best_params[0]),
                'max_depth': int(best_params[1]),
                'min_samples_split': int(best_params[2]),
                'min_samples_leaf': int(best_params[3])
            },
            'best_fitness': float(best_fitness),
            'optimizer': 'DifferentialEvolutionOptimizer'
        }
        
    def save(self, filepath: str) -> None:
        """
        Save the expert model to disk.
        
        Args:
            filepath: Path to save the model to
        """
        # Prepare the object for serialization
        save_dict = {
            'name': self.name,
            'model': self.model,
            'feature_importances': self.feature_importances,
            'quality_metrics': self.quality_metrics,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'metadata': self.metadata,
            'training_history': self.training_history,
            'validation_scores': self.validation_scores,
            # Save preprocessing components
            'scaler': self.scaler,
            'signal_processor': self.signal_processor
        }
        
        # Save to disk
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Expert model '{self.name}' saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PhysiologicalExpert':
        """
        Load a physiological expert model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded physiological expert model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load from disk
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Extract required parameters for initialization
        metadata = save_dict['metadata']
        vital_cols = metadata.get('vital_cols', [])
        patient_id_col = metadata.get('patient_id_col', 'patient_id')
        timestamp_col = metadata.get('timestamp_col', 'date')
        normalize_vitals = metadata.get('normalize_vitals', True)
        extract_variability = metadata.get('extract_variability', True)
        
        # Create a new instance
        instance = cls(
            vital_cols=vital_cols,
            patient_id_col=patient_id_col,
            timestamp_col=timestamp_col,
            normalize_vitals=normalize_vitals,
            extract_variability=extract_variability,
            model=save_dict['model'],
            name=save_dict['name'],
            metadata=metadata
        )
        
        # Restore state
        instance.feature_importances = save_dict['feature_importances']
        instance.quality_metrics = save_dict['quality_metrics']
        instance.is_fitted = save_dict['is_fitted']
        instance.feature_columns = save_dict.get('feature_columns', [])
        instance.target_column = save_dict.get('target_column', None)
        
        # Restore preprocessing components
        if 'scaler' in save_dict:
            instance.scaler = save_dict['scaler']
        if 'preprocessing_pipeline' in save_dict:
            instance.preprocessing_pipeline = save_dict['preprocessing_pipeline']
        if 'signal_processor' in save_dict:
            instance.signal_processor = save_dict['signal_processor']
        
        return instance

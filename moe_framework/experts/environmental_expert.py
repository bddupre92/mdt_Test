"""
Environmental Expert Module

This module implements the EnvironmentalExpert class, which specializes in analyzing
and making predictions based on environmental data such as weather, pollution,
and other environmental factors related to migraine triggers.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from moe_framework.experts.base_expert import BaseExpert
from data.domain_specific_preprocessing import EnvironmentalTriggerAnalyzer
from moe_framework.experts.optimizer_adapters import EvolutionStrategyAdapter

# Configure logging
logger = logging.getLogger(__name__)


class EnvironmentalExpert(BaseExpert):
    """
    Expert model specializing in environmental data for migraine prediction.
    
    This expert focuses on environmental factors such as weather conditions,
    air quality, and other environmental triggers to predict migraine occurrence
    or severity. It uses Evolution Strategy for hyperparameter optimization.
    
    Attributes:
        env_cols (List[str]): List of environmental factor column names
        location_col (str): Name of the location column
        timestamp_col (str): Name of the timestamp column
        include_weather (bool): Whether to include weather features
        include_pollution (bool): Whether to include pollution features
        trigger_analyzer (EnvironmentalTriggerAnalyzer): Preprocessing for environmental triggers
        scaler (StandardScaler): Scaler for normalizing features
    """
    
    def __init__(self, 
                 env_cols: List[str],
                 location_col: str = 'location',
                 timestamp_col: str = 'date',
                 include_weather: bool = True,
                 include_pollution: bool = True,
                 model: Optional[Any] = None,
                 name: str = "EnvironmentalExpert",
                 metadata: Dict[str, Any] = None):
        """
        Initialize the environmental expert.
        
        Args:
            env_cols: List of environmental factor column names
            location_col: Name of the location column
            timestamp_col: Name of the timestamp column
            include_weather: Whether to include weather features
            include_pollution: Whether to include pollution features
            model: The underlying machine learning model (default: GradientBoostingRegressor)
            name: Name of the expert model
            metadata: Additional metadata about the expert
        """
        # Initialize with default model if none provided
        if model is None:
            # Use HistGradientBoostingRegressor which handles NaN values natively
            model = HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        # Initialize base class
        super().__init__(
            name=name,
            model=model,
            metadata=metadata or {}
        )
        
        # Store environmental-specific parameters
        self.env_cols = env_cols
        self.location_col = location_col
        self.timestamp_col = timestamp_col
        self.include_weather = include_weather
        self.include_pollution = include_pollution
        
        # Initialize preprocessing components
        # Split env_cols into specific categories based on include flags
        weather_cols = env_cols if include_weather else []
        pollution_cols = env_cols if include_pollution else []
        
        self.trigger_analyzer = EnvironmentalTriggerAnalyzer(
            weather_cols=weather_cols,
            pollution_cols=pollution_cols,
            light_cols=[],  # Not used in current implementation
            noise_cols=[],  # Not used in current implementation
            timestamp_col=timestamp_col,
            location_col=location_col
        )
        
        # Initialize preprocessing pipeline with imputer and scaler
        self.preprocessing_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.scaler = self.preprocessing_pipeline.named_steps['scaler']
        
        # Initialize optimizer
        self.optimizer = None
        
        # Update metadata
        self.metadata.update({
            'env_cols': env_cols,
            'location_col': location_col,
            'timestamp_col': timestamp_col,
            'include_weather': include_weather,
            'include_pollution': include_pollution,
            'domain': 'environmental'
        })
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data using the environmental trigger analyzer.
        
        Args:
            data: Input data containing environmental factors
            
        Returns:
            Preprocessed data with extracted environmental features
        """
        # Apply the environmental trigger analyzer
        processed_data = self.trigger_analyzer.fit_transform(data.copy())
        
        # Log the preprocessing results
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        logger.info(f"Added features: {set(processed_data.columns) - set(data.columns)}")
        
        return processed_data
    
    def extract_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract features for the environmental expert model.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Tuple of (feature_matrix, feature_column_names)
        """
        # Identify environmental feature columns
        feature_cols = []
        
        # Include original environmental columns
        feature_cols.extend(self.env_cols)
        
        # Include weather features if available
        if self.include_weather:
            weather_cols = [col for col in data.columns if "weather_" in col or "temp_" in col or "humidity_" in col or "pressure_" in col]
            feature_cols.extend(weather_cols)
        
        # Include pollution features if available
        if self.include_pollution:
            pollution_cols = [col for col in data.columns if "pollution_" in col or "aqi_" in col or "pollen_" in col]
            feature_cols.extend(pollution_cols)
        
        # Include trigger features if available
        trigger_cols = [col for col in data.columns if "_trigger_score" in col or "_trigger_likelihood" in col]
        feature_cols.extend(trigger_cols)
        
        # Include seasonal and cyclic features if available
        seasonal_cols = [col for col in data.columns if "season_" in col or "month_" in col or "day_" in col]
        feature_cols.extend(seasonal_cols)
        
        # Include change rate features if available
        change_cols = [col for col in data.columns if "_change_rate" in col or "_delta" in col]
        feature_cols.extend(change_cols)
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Check if we have the necessary columns
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Extract feature matrix
        X = data[feature_cols].copy()
        
        # Missing values will be handled by the imputer in the pipeline
        # No need to manually handle them here
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        return X, feature_cols
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize_hyperparams: bool = False, **kwargs) -> 'EnvironmentalExpert':
        """
        Fit the environmental expert model to the data.
        
        Args:
            X: Feature data
            y: Target data
            optimize_hyperparams: Whether to optimize hyperparameters using ES
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
        
        # Use the pipeline to handle missing values and scale features
        X_scaled = pd.DataFrame(
            self.preprocessing_pipeline.fit_transform(X_features),
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
        Generate predictions using the environmental expert model.
        
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
        
        # Use the preprocessing pipeline to handle missing values and scale features
        X_scaled = pd.DataFrame(
            self.preprocessing_pipeline.transform(X_features),
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
        
        For gradient boosting models, we use the standard deviation of predictions across trees
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
        
        # Use the preprocessing pipeline to handle missing values and scale features
        X_scaled = pd.DataFrame(
            self.preprocessing_pipeline.transform(X_features),
            columns=X_features.columns,
            index=X_features.index
        )
        
        # For gradient boosting, use prediction variance as uncertainty
        if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
            # Get predictions from all estimators - use numpy arrays to avoid feature names warning
            stage_preds = np.array([estimator[0].predict(X_scaled.values) for estimator in self.model.estimators_])
            # Calculate standard deviation across stages
            uncertainty = np.std(stage_preds, axis=0)
            # Convert uncertainty to confidence (higher uncertainty = lower confidence)
            max_uncertainty = np.max(uncertainty) if np.max(uncertainty) > 0 else 1.0
            confidence = 1.0 - (uncertainty / max_uncertainty)
            return confidence
        elif hasattr(self.model, 'estimator') and hasattr(self.model, 'n_estimators'):
            # Use a simpler approach for newer scikit-learn versions
            predictions = self.model.predict(X_scaled.values)
            return np.ones(len(predictions))  # Default high confidence
        
        # Default confidence calculation
        return super()._calculate_confidence(X, predictions, **kwargs)
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Optimize hyperparameters using Evolution Strategy.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional keyword arguments
        """
        logger.info(f"Optimizing hyperparameters for {self.name} using Evolution Strategy")
        
        # Define parameter bounds for GradientBoostingRegressor
        param_bounds = {
            'n_estimators': (50, 200),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'min_samples_split': (2, 20)
        }
        
        # Define fitness function (negative cross-validation score)
        def fitness_function(params):
            # Extract parameters
            n_estimators = int(params[0])
            learning_rate = params[1]
            max_depth = int(params[2])
            min_samples_split = int(params[3])
            
            # Create model with these parameters
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            
            # Return negative mean score (ES minimizes)
            return -np.mean(cv_scores)
        
        # Initialize optimizer using the adapter
        self.optimizer = EvolutionStrategyAdapter(
            fitness_function=fitness_function,
            bounds=[
                param_bounds['n_estimators'],
                param_bounds['learning_rate'],
                param_bounds['max_depth'],
                param_bounds['min_samples_split']
            ],
            population_size=10,
            max_iterations=10,
            initial_step_size=0.3,
            adaptation_rate=0.2,
            random_seed=42
        )
        
        # Run optimization
        best_params, best_fitness = self.optimizer.optimize()
        
        # Update model with best parameters
        self.model = GradientBoostingRegressor(
            n_estimators=int(best_params[0]),
            learning_rate=best_params[1],
            max_depth=int(best_params[2]),
            min_samples_split=int(best_params[3]),
            random_state=42
        )
        
        # Log optimization results
        logger.info(f"Optimized hyperparameters: n_estimators={int(best_params[0])}, "
                   f"learning_rate={best_params[1]:.3f}, max_depth={int(best_params[2])}, "
                   f"min_samples_split={int(best_params[3])}")
        logger.info(f"Best fitness (negative MSE): {best_fitness}")
        
        # Store optimization results in metadata
        self.metadata['optimization_results'] = {
            'best_params': {
                'n_estimators': int(best_params[0]),
                'learning_rate': float(best_params[1]),
                'max_depth': int(best_params[2]),
                'min_samples_split': int(best_params[3])
            },
            'best_fitness': float(best_fitness),
            'optimizer': 'EvolutionStrategyOptimizer'
        }

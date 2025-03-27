"""
Behavioral Expert Module

This module implements the BehavioralExpert class, which specializes in analyzing
and making predictions based on behavioral data such as sleep patterns, activity levels,
stress, and other behavioral factors related to migraine triggers.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from moe_framework.experts.base_expert import BaseExpert
from moe_framework.experts.optimizer_adapters import AntColonyAdapter

# Configure logging
logger = logging.getLogger(__name__)


class BehavioralExpert(BaseExpert):
    """
    Expert model specializing in behavioral data for migraine prediction.
    
    This expert focuses on behavioral factors such as sleep patterns, activity levels,
    stress, and other lifestyle factors to predict migraine occurrence or severity.
    It uses Ant Colony Optimization for feature selection.
    
    Attributes:
        behavior_cols (List[str]): List of behavioral factor column names
        patient_id_col (str): Name of the patient ID column
        timestamp_col (str): Name of the timestamp column
        include_sleep (bool): Whether to include sleep features
        include_activity (bool): Whether to include activity features
        include_stress (bool): Whether to include stress features
        scaler (StandardScaler): Scaler for normalizing features
    """
    
    def __init__(self, 
                 behavior_cols: List[str],
                 patient_id_col: str = 'patient_id',
                 timestamp_col: str = 'date',
                 include_sleep: bool = True,
                 include_activity: bool = True,
                 include_stress: bool = True,
                 model: Optional[Any] = None,
                 name: str = "BehavioralExpert",
                 metadata: Dict[str, Any] = None):
        """
        Initialize the behavioral expert.
        
        Args:
            behavior_cols: List of behavioral factor column names
            patient_id_col: Name of the patient ID column
            timestamp_col: Name of the timestamp column
            include_sleep: Whether to include sleep features
            include_activity: Whether to include activity features
            include_stress: Whether to include stress features
            model: The underlying machine learning model (default: ExtraTreesRegressor)
            name: Name of the expert model
            metadata: Additional metadata about the expert
        """
        # Initialize with default model if none provided
        if model is None:
            model = ExtraTreesRegressor(
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
        
        # Store behavioral-specific parameters
        self.behavior_cols = behavior_cols
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.include_sleep = include_sleep
        self.include_activity = include_activity
        self.include_stress = include_stress
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Initialize optimizer
        self.optimizer = None
        self.selected_features = None
        
        # Update metadata
        self.metadata.update({
            'behavior_cols': behavior_cols,
            'patient_id_col': patient_id_col,
            'timestamp_col': timestamp_col,
            'include_sleep': include_sleep,
            'include_activity': include_activity,
            'include_stress': include_stress,
            'domain': 'behavioral'
        })
    
    def preprocess_data(self, data) -> pd.DataFrame:
        """
        Preprocess the input data for behavioral analysis.
        
        Args:
            data: Input data containing behavioral factors (pandas DataFrame or numpy array)
            
        Returns:
            Preprocessed data with extracted behavioral features
        """
        # Check if the input data is a pandas DataFrame or a numpy array
        if isinstance(data, np.ndarray):
            # For numpy arrays during optimization, just return the array as is
            # This is a simplified approach for use during optimization
            return data
        
        # For pandas DataFrames, apply the full preprocessing pipeline
        # Create a copy of the input data
        processed_data = data.copy()
        
        # Extract time-based features
        if isinstance(processed_data, pd.DataFrame) and self.timestamp_col in processed_data.columns:
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(processed_data[self.timestamp_col]):
                processed_data[self.timestamp_col] = pd.to_datetime(processed_data[self.timestamp_col])
            
            # Extract day of week, hour of day
            processed_data['day_of_week'] = processed_data[self.timestamp_col].dt.dayofweek
            processed_data['is_weekend'] = processed_data['day_of_week'].isin([5, 6]).astype(int)
            processed_data['hour_of_day'] = processed_data[self.timestamp_col].dt.hour
            
            # Create time of day categories
            conditions = [
                (processed_data['hour_of_day'] >= 5) & (processed_data['hour_of_day'] < 12),
                (processed_data['hour_of_day'] >= 12) & (processed_data['hour_of_day'] < 17),
                (processed_data['hour_of_day'] >= 17) & (processed_data['hour_of_day'] < 22),
                (processed_data['hour_of_day'] >= 22) | (processed_data['hour_of_day'] < 5)
            ]
            categories = ['morning', 'afternoon', 'evening', 'night']
            processed_data['time_of_day'] = np.select(conditions, categories, default='unknown')
            
            # One-hot encode time of day
            for category in categories:
                processed_data[f'time_of_day_{category}'] = (processed_data['time_of_day'] == category).astype(int)
        
        # Process sleep features
        if self.include_sleep:
            sleep_cols = [col for col in self.behavior_cols if 'sleep' in col.lower()]
            if sleep_cols:
                # Calculate sleep quality score if multiple sleep metrics are available
                if len(sleep_cols) > 1:
                    # Normalize sleep columns to 0-1 range
                    sleep_normalized = processed_data[sleep_cols].copy()
                    for col in sleep_cols:
                        if processed_data[col].max() > processed_data[col].min():
                            sleep_normalized[col] = (processed_data[col] - processed_data[col].min()) / (processed_data[col].max() - processed_data[col].min())
                    
                    # Calculate sleep quality score (average of normalized metrics)
                    processed_data['sleep_quality_score'] = sleep_normalized.mean(axis=1)
                
                # Calculate sleep consistency if timestamp is available
                if self.timestamp_col in processed_data.columns and self.patient_id_col in processed_data.columns:
                    # Group by patient
                    for patient_id, patient_data in processed_data.groupby(self.patient_id_col):
                        if 'sleep_duration' in processed_data.columns:
                            # Calculate rolling mean and std of sleep duration
                            sleep_mean = patient_data['sleep_duration'].rolling(window=7, min_periods=1).mean()
                            sleep_std = patient_data['sleep_duration'].rolling(window=7, min_periods=1).std()
                            
                            # Calculate sleep consistency (1 - normalized std)
                            sleep_consistency = 1 - (sleep_std / sleep_mean).fillna(0).clip(0, 1)
                            
                            # Add to processed data
                            processed_data.loc[patient_data.index, 'sleep_consistency'] = sleep_consistency
        
        # Process activity features
        if self.include_activity:
            activity_cols = [col for col in self.behavior_cols if 'activity' in col.lower() or 'exercise' in col.lower()]
            if activity_cols:
                # Calculate activity level score if multiple activity metrics are available
                if len(activity_cols) > 1:
                    # Normalize activity columns to 0-1 range
                    activity_normalized = processed_data[activity_cols].copy()
                    for col in activity_cols:
                        if processed_data[col].max() > processed_data[col].min():
                            activity_normalized[col] = (processed_data[col] - processed_data[col].min()) / (processed_data[col].max() - processed_data[col].min())
                    
                    # Calculate activity level score (average of normalized metrics)
                    processed_data['activity_level_score'] = activity_normalized.mean(axis=1)
                
                # Calculate activity consistency if timestamp is available
                if self.timestamp_col in processed_data.columns and self.patient_id_col in processed_data.columns:
                    for patient_id, patient_data in processed_data.groupby(self.patient_id_col):
                        if activity_cols:
                            # Use the first activity column for consistency calculation
                            activity_col = activity_cols[0]
                            
                            # Calculate rolling mean and std of activity
                            activity_mean = patient_data[activity_col].rolling(window=7, min_periods=1).mean()
                            activity_std = patient_data[activity_col].rolling(window=7, min_periods=1).std()
                            
                            # Calculate activity consistency (1 - normalized std)
                            activity_consistency = 1 - (activity_std / activity_mean).fillna(0).clip(0, 1)
                            
                            # Add to processed data
                            processed_data.loc[patient_data.index, 'activity_consistency'] = activity_consistency
        
        # Process stress features
        if self.include_stress:
            stress_cols = [col for col in self.behavior_cols if 'stress' in col.lower() or 'anxiety' in col.lower()]
            if stress_cols:
                # Calculate stress level score if multiple stress metrics are available
                if len(stress_cols) > 1:
                    # Normalize stress columns to 0-1 range
                    stress_normalized = processed_data[stress_cols].copy()
                    for col in stress_cols:
                        if processed_data[col].max() > processed_data[col].min():
                            stress_normalized[col] = (processed_data[col] - processed_data[col].min()) / (processed_data[col].max() - processed_data[col].min())
                    
                    # Calculate stress level score (average of normalized metrics)
                    processed_data['stress_level_score'] = stress_normalized.mean(axis=1)
        
        # Log the preprocessing results
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        logger.info(f"Added features: {set(processed_data.columns) - set(data.columns)}")
        
        return processed_data
    
    def extract_features(self, data) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for the behavioral expert model.
        
        Args:
            data: Preprocessed data (pandas DataFrame or numpy array)
            
        Returns:
            Tuple of (feature_matrix, feature_column_names)
        """
        # For numpy arrays (during optimization), return the array as is with default column names
        if isinstance(data, np.ndarray):
            # Create dummy feature column names based on array shape
            feature_cols = [f'feature_{i}' for i in range(data.shape[1])]
            return data, feature_cols
        
        # Identify behavioral feature columns
        feature_cols = []
        
        # Include original behavioral columns
        feature_cols.extend(self.behavior_cols)
        
        # Include sleep features if available
        if self.include_sleep:
            sleep_cols = [col for col in data.columns if 'sleep' in col.lower()]
            feature_cols.extend(sleep_cols)
        
        # Include activity features if available
        if self.include_activity:
            activity_cols = [col for col in data.columns if 'activity' in col.lower() or 'exercise' in col.lower()]
            feature_cols.extend(activity_cols)
        
        # Include stress features if available
        if self.include_stress:
            stress_cols = [col for col in data.columns if 'stress' in col.lower() or 'anxiety' in col.lower()]
            feature_cols.extend(stress_cols)
        
        # Include time-based features if available
        time_cols = ['day_of_week', 'is_weekend', 'hour_of_day']
        time_cols.extend([col for col in data.columns if 'time_of_day_' in col])
        feature_cols.extend([col for col in time_cols if col in data.columns])
        
        # Include derived features if available
        derived_cols = ['sleep_quality_score', 'sleep_consistency', 
                        'activity_level_score', 'activity_consistency',
                        'stress_level_score']
        feature_cols.extend([col for col in derived_cols if col in data.columns])
        
        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Check if we have the necessary columns
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Use selected features if available (from ACO feature selection)
        if self.selected_features is not None:
            feature_cols = [col for col in feature_cols if col in self.selected_features]
            logger.info(f"Using {len(feature_cols)} selected features from ACO")
        
        # Extract feature matrix
        X = data[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Store feature columns
        self.feature_columns = feature_cols
        
        return X, feature_cols
    
    def fit(self, X: pd.DataFrame, y: pd.Series, optimize_features: bool = False, **kwargs) -> 'BehavioralExpert':
        """
        Fit the behavioral expert model to the data.
        
        Args:
            X: Feature data
            y: Target data
            optimize_features: Whether to optimize feature selection using ACO
            **kwargs: Additional keyword arguments
            
        Returns:
            Self for method chaining
        """
        # Store target column name if available (pandas Series) or use a default name (numpy array)
        self.target_column = getattr(y, 'name', 'target') if hasattr(y, 'name') else 'target'
        
        # Preprocess data
        processed_data = self.preprocess_data(X.copy())
        
        # Optimize feature selection if requested
        if optimize_features:
            self.optimize_feature_selection(processed_data, y, **kwargs)
        
        # Extract features
        X_features, feature_cols = self.extract_features(processed_data)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(X_features)
        
        # Create DataFrame from scaled data, handling both pandas DataFrame and numpy array inputs
        if isinstance(X_features, pd.DataFrame):
            X_scaled = pd.DataFrame(
                scaled_data,
                columns=feature_cols,
                index=X_features.index
            )
        else:  # numpy array case
            X_scaled = pd.DataFrame(
                scaled_data,
                columns=feature_cols
            )
        
        # Fit the model
        logger.info(f"Fitting {self.name} model with {len(feature_cols)} features")
        
        # Validate model parameters before fitting
        self._validate_model_parameters()
        
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
    
    def _validate_model_parameters(self):
        """
        Validate and adjust model parameters to ensure they meet scikit-learn's requirements.
        This method is called before fitting the model to prevent parameter validation errors.
        """
        # Check and adjust max_features if it's a float
        if hasattr(self.model, 'max_features') and isinstance(self.model.max_features, float):
            # Ensure max_features is in the range (0.0, 1.0]
            if self.model.max_features > 1.0 or self.model.max_features <= 0.0:
                logger.warning(f"Adjusting max_features from {self.model.max_features} to a valid value in (0.0, 1.0]")
                self.model.max_features = min(1.0, max(0.01, self.model.max_features))
        
        # Ensure min_samples_split is at least 2
        if hasattr(self.model, 'min_samples_split'):
            if self.model.min_samples_split < 2:
                logger.warning(f"Adjusting min_samples_split from {self.model.min_samples_split} to 2")
                self.model.min_samples_split = 2
                
        # Ensure min_samples_leaf is at least 1
        if hasattr(self.model, 'min_samples_leaf'):
            if self.model.min_samples_leaf < 1:
                logger.warning(f"Adjusting min_samples_leaf from {self.model.min_samples_leaf} to 1")
                self.model.min_samples_leaf = 1
                
        # Validate bootstrap is boolean
        if hasattr(self.model, 'bootstrap') and not isinstance(self.model.bootstrap, bool):
            logger.warning(f"Converting bootstrap from {self.model.bootstrap} to boolean")
            self.model.bootstrap = bool(self.model.bootstrap)
    
    def predict(self, X, **kwargs) -> np.ndarray:
        """
        Generate predictions using the behavioral expert model.
        
        Args:
            X: Feature data (pandas DataFrame or numpy array)
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
        scaled_data = self.scaler.transform(X_features)
        
        # Handle differently based on input type
        if isinstance(X_features, pd.DataFrame):
            X_scaled = pd.DataFrame(
                scaled_data,
                columns=X_features.columns,
                index=X_features.index
            )
        else:  # numpy array case
            feature_cols = [f'feature_{i}' for i in range(scaled_data.shape[1])]
            X_scaled = pd.DataFrame(
                scaled_data,
                columns=feature_cols
            )
        
        # Generate predictions
        # Convert to numpy array to avoid feature names warning
        predictions = self.model.predict(X_scaled.values)
        
        return predictions
    
    def _calculate_confidence(self, X: pd.DataFrame, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        For extra trees models, we use the standard deviation of predictions across trees
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
        scaled_data = self.scaler.transform(X_features)
        
        # Handle differently based on input type
        if isinstance(X_features, pd.DataFrame):
            X_scaled = pd.DataFrame(
                scaled_data,
                columns=X_features.columns,
                index=X_features.index
            )
        else:  # numpy array case
            feature_cols = [f'feature_{i}' for i in range(scaled_data.shape[1])]
            X_scaled = pd.DataFrame(
                scaled_data,
                columns=feature_cols
            )
        
        # For extra trees, use the standard deviation of predictions across trees
        if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
            # Get predictions from all trees - use numpy arrays to avoid feature names warning
            tree_preds = np.array([tree.predict(X_scaled.values) for tree in self.model.estimators_])
            # Calculate standard deviation across trees
            uncertainty = np.std(tree_preds, axis=0)
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
    
    def optimize_feature_selection(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Optimize feature selection using Ant Colony Optimization.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional keyword arguments
        """
        logger.info(f"Optimizing feature selection for {self.name} using Ant Colony Optimization")
        
        # Extract all potential features
        _, all_features = self.extract_features(X)
        
        # Define fitness function for feature subset evaluation
        def fitness_function(feature_mask):
            # Convert binary mask to feature indices
            selected_indices = np.where(feature_mask == 1)[0]
            
            # Skip if no features are selected
            if len(selected_indices) == 0:
                return -np.inf
            
            # Get selected feature names
            selected_features = [all_features[i] for i in selected_indices]
            
            # Extract selected features
            X_selected = X[selected_features]
            
            # Scale features
            X_scaled = StandardScaler().fit_transform(X_selected)
            
            # Create model
            model = ExtraTreesRegressor(
                n_estimators=50,  # Use fewer estimators for faster evaluation
                max_depth=5,
                random_state=42
            )
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_scaled, y, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            
            # Calculate fitness (negative MSE with penalty for too many features)
            mse = -np.mean(cv_scores)
            n_features_penalty = 0.001 * len(selected_indices)  # Small penalty for each feature
            
            # Return negative fitness (ACO minimizes)
            return mse + n_features_penalty
        
        # Initialize optimizer using the adapter
        self.optimizer = AntColonyAdapter(
            fitness_function=fitness_function,
            bounds=[(0, 1) for _ in range(len(all_features))],  # Binary feature selection bounds
            population_size=10,  # Equivalent to n_ants
            max_iterations=20,
            alpha=1.0,  # Pheromone importance
            beta=2.0,   # Heuristic importance
            evaporation_rate=0.1,
            random_seed=42
        )
        
        # Run optimization
        best_mask, best_fitness = self.optimizer.optimize()
        
        # Convert binary mask to feature names
        selected_indices = np.where(best_mask == 1)[0]
        self.selected_features = [all_features[i] for i in selected_indices]
        
        # Log optimization results
        logger.info(f"Selected {len(self.selected_features)} features out of {len(all_features)}")
        logger.info(f"Selected features: {self.selected_features}")
        logger.info(f"Best fitness (MSE with penalty): {best_fitness}")
        
        # Store optimization results in metadata
        self.metadata['optimization_results'] = {
            'n_features_total': len(all_features),
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features,
            'best_fitness': float(best_fitness),
            'optimizer': 'AntColonyOptimizer'
        }

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
    
    def preprocess_data(self, X):
        """
        Preprocess behavioral data with robust error handling.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            X = X.copy()
            
            # Add missing behavioral columns with default values
            default_columns = {
                'sleep_duration': 7.0,  # hours
                'sleep_quality': 'medium',
                'activity_level': 'moderate',
                'stress_level': 'medium',
                'mood': 'neutral',
                'social_interaction': 'moderate'
            }
            
            for col, default_val in default_columns.items():
                if col not in X.columns:
                    logger.info(f"Adding missing column {col} with default value {default_val}")
                    X[col] = default_val
            
            # Handle categorical columns
            categorical_mappings = {
                'sleep_quality': {'poor': 0, 'medium': 0.5, 'good': 1.0, 'unknown': 0.5},
                'activity_level': {'sedentary': 0, 'light': 0.25, 'moderate': 0.5, 'vigorous': 1.0, 'unknown': 0.5},
                'stress_level': {'low': 0, 'medium': 0.5, 'high': 1.0, 'unknown': 0.5},
                'mood': {'negative': 0, 'neutral': 0.5, 'positive': 1.0, 'unknown': 0.5},
                'social_interaction': {'low': 0, 'moderate': 0.5, 'high': 1.0, 'unknown': 0.5}
            }
            
            for col, mapping in categorical_mappings.items():
                if col in X.columns:
                    try:
                        # Convert to lowercase and replace unknown values
                        X[col] = X[col].astype(str).str.lower()
                        X[col] = X[col].replace(['nan', 'none', 'null', ''], 'unknown')
                        
                        # Map categories to numeric values
                        X[col] = X[col].map(mapping)
                        
                        # Fill any remaining unknown values with median
                        median = X[col].median()
                        if pd.isna(median):
                            median = 0.5  # Default to middle value
                        X[col] = X[col].fillna(median)
                        
                        logger.info(f"Processed {col}: unique values = {X[col].unique()}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {col}: {str(e)}")
                        X[col] = mapping.get('unknown', 0.5)
            
            # Handle numeric columns
            numeric_columns = {
                'sleep_duration': (0, 24),  # hours
                'physical_activity_minutes': (0, 1440),  # minutes per day
                'screen_time': (0, 1440),  # minutes per day
                'meditation_minutes': (0, 480),  # minutes per day
                'water_intake': (0, 5000)  # ml per day
            }
            
            for col, (min_val, max_val) in numeric_columns.items():
                if col in X.columns:
                    try:
                        # Convert to numeric, coercing errors to NaN
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        
                        # Fill missing values with median
                        median = X[col].median()
                        if pd.isna(median):
                            median = (min_val + max_val) / 2  # Use middle of range
                        X[col] = X[col].fillna(median)
                        
                        # Clip to reasonable ranges
                        X[col] = X[col].clip(min_val, max_val)
                        
                        logger.info(f"Processed {col}: range [{X[col].min():.1f}, {X[col].max():.1f}]")
                        
                    except Exception as e:
                        logger.error(f"Error processing {col}: {str(e)}")
                        X[col] = (min_val + max_val) / 2
            
            # Extract time-based features if timestamp is available
            if 'timestamp' in X.columns:
                try:
                    time_features = self._extract_time_features(X['timestamp'])
                    X = pd.concat([X, time_features], axis=1)
                    X = X.drop('timestamp', axis=1)
                except Exception as e:
                    logger.error(f"Error extracting time features: {str(e)}")
            
            # Store feature columns for importance calculation
            self.feature_columns = list(X.columns)
            
            return X
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            return X
    
    def _extract_time_features(self, timestamp_series):
        """Extract time-based features from timestamp."""
        try:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(timestamp_series):
                timestamp_series = pd.to_datetime(timestamp_series)
            
            # Extract basic time features
            time_features = pd.DataFrame()
            time_features['hour'] = timestamp_series.dt.hour
            time_features['day_of_week'] = timestamp_series.dt.dayofweek
            
            # Create time of day categories
            def get_time_of_day(hour):
                if 5 <= hour < 12:
                    return 'morning'
                elif 12 <= hour < 17:
                    return 'afternoon'
                elif 17 <= hour < 22:
                    return 'evening'
                else:
                    return 'night'
            
            time_features['time_of_day'] = time_features['hour'].apply(get_time_of_day)
            
            # One-hot encode time of day
            time_of_day_dummies = pd.get_dummies(time_features['time_of_day'], prefix='time_of_day')
            time_features = pd.concat([time_features, time_of_day_dummies], axis=1)
            time_features = time_features.drop('time_of_day', axis=1)
            
            return time_features
            
        except Exception as e:
            logger.error(f"Error extracting time features: {str(e)}")
            return pd.DataFrame()
    
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
            activity_cols = [col for col in data.columns if 'activity' in col.lower()]
            feature_cols.extend(activity_cols)
        
        # Include stress features if available
        if self.include_stress:
            stress_cols = [col for col in data.columns if 'stress' in col.lower()]
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
    
    def optimize_feature_selection(self, X, y):
        """
        Optimize feature selection using Ant Colony Optimization.
        
        Args:
            X: Training features DataFrame
            y: Target values
            
        Returns:
            List of selected feature names
        """
        try:
            # Get all feature names
            feature_names = list(X.columns)
            n_features = len(feature_names)
            
            # Define bounds for binary feature selection
            bounds = [(0, 1) for _ in range(n_features)]
            
            # Define fitness function for feature subset evaluation
            def fitness_function(solution):
                try:
                    # Convert continuous ACO solution to binary
                    binary_solution = [1 if x > 0.5 else 0 for x in solution]
                    
                    # If no features selected, return worst score
                    if sum(binary_solution) == 0:
                        return float('inf')
                    
                    # Get selected feature indices and names
                    selected_indices = [i for i, x in enumerate(binary_solution) if x == 1]
                    selected_features = [feature_names[i] for i in selected_indices]
                    
                    # Create model with selected features
                    X_selected = X[selected_features]
                    
                    # Evaluate using cross-validation
                    model = self._create_model()
                    scores = cross_val_score(model, X_selected, y, cv=3, scoring='neg_mean_squared_error')
                    mse = -np.mean(scores)
                    
                    # Add penalty for number of features to encourage parsimony
                    n_selected = len(selected_features)
                    penalty = 0.1 * n_selected / n_features
                    
                    return mse + penalty
                    
                except Exception as e:
                    logger.error(f"Error in feature selection fitness function: {str(e)}")
                    return float('inf')
            
            # Initialize optimizer
            optimizer = AntColonyAdapter(
                fitness_function=fitness_function,
                bounds=bounds,
                population_size=20,  # number of ants
                max_iterations=20,
                alpha=1.0,  # pheromone importance
                beta=2.0,   # heuristic importance
                evaporation_rate=0.1
            )
            
            # Run optimization
            best_solution, best_fitness = optimizer.optimize()
            
            # Convert continuous solution to binary
            binary_solution = [1 if x > 0.5 else 0 for x in best_solution]
            
            # Get selected feature names
            selected_features = [name for name, selected in zip(feature_names, binary_solution) if selected]
            
            # Log results
            logger.info(f"Selected {len(selected_features)} features: {selected_features}")
            logger.info(f"Feature selection fitness score: {best_fitness}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection optimization: {str(e)}")
            # Return all features if optimization fails
            return list(X.columns)

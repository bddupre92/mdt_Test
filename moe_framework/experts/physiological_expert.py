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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    
    def __init__(self, model_type='neural_network', hidden_layers=None, activation='relu', input_dim=None, **kwargs):
        """
        Initialize the physiological expert.
        
        Args:
            model_type: Type of model to use ('neural_network', 'lstm', etc.)
            hidden_layers: List of hidden layer sizes
            activation: Activation function to use
            input_dim: Input dimension for the model
            **kwargs: Additional model parameters
        """
        super().__init__(name='physiological')
        
        self.model_type = model_type
        self.hidden_layers = hidden_layers or [64, 32]
        self.activation = activation
        self.input_dim = input_dim
        self.model_params = kwargs
        
        # Initialize vital signs columns
        self.vital_cols = kwargs.get('vital_cols', [
            'heart_rate', 'blood_pressure', 'respiratory_rate', 
            'temperature', 'oxygen_saturation', 'systolic', 'diastolic'
        ])
        
        self.patient_id_col = kwargs.get('patient_id_col', 'patient_id')
        self.timestamp_col = kwargs.get('timestamp_col', 'date')
        self.normalize_vitals = kwargs.get('normalize_vitals', True)
        self.extract_variability = kwargs.get('extract_variability', True)
        
        # Initialize the scaler
        self.scaler = StandardScaler()
        
        # Initialize signal processor
        try:
            self.signal_processor = PhysiologicalSignalProcessor(
                vital_cols=self.vital_cols,
                patient_id_col=self.patient_id_col,
                timestamp_col=self.timestamp_col
            )
        except Exception as e:
            logger.warning(f"Could not initialize PhysiologicalSignalProcessor: {str(e)}")
            self.signal_processor = StandardScaler()
        
        # Initialize the model
        self._initialize_model()
        
    def get_hyperparameter_space(self):
        """Get the hyperparameter space for optimization."""
        return {
            'hidden_layers': {
                'type': 'int',
                'bounds': [(16, 256), (8, 128)],  # Two layers with their bounds
                'value': self.hidden_layers
            },
            'learning_rate': {
                'type': 'float',
                'bounds': [(0.0001, 0.1)],
                'value': self.model_params.get('learning_rate', 0.001)
            },
            'batch_size': {
                'type': 'int',
                'bounds': [(16, 128)],
                'value': self.model_params.get('batch_size', 32)
            },
            'activation': {
                'type': 'categorical',
                'choices': ['relu', 'tanh', 'sigmoid'],
                'value': self.activation
            }
        }
        
    def _initialize_model(self):
        """Initialize the underlying model based on configuration."""
        try:
            if self.model_type == 'neural_network':
                from sklearn.neural_network import MLPRegressor
                
                # Ensure hidden_layers is valid
                if not self.hidden_layers or not all(layer > 0 for layer in self.hidden_layers):
                    logger.warning("Invalid hidden_layers detected, using default [64, 32]")
                    self.hidden_layers = [64, 32]
                
                # Create valid model parameters
                model_params = self.model_params.copy()

                # Fix batch_size type if present
                if 'batch_size' in model_params:
                    try:
                        model_params['batch_size'] = max(1, min(512, int(model_params['batch_size'])))
                    except Exception as e:
                        logger.warning(f"Invalid batch_size value: {model_params['batch_size']}, defaulting to 32")
                        model_params['batch_size'] = 32

                # Always use tuple for hidden_layer_sizes
                model_params['hidden_layer_sizes'] = tuple(self.hidden_layers)
                
                # Initialize the model
                self.model = MLPRegressor(
                    activation=self.activation,
                    **model_params
                )
                
                logger.info(f"Initialized MLPRegressor with hidden_layer_sizes={model_params['hidden_layer_sizes']}")
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Error initializing physiological expert: {str(e)}")
            self.model = None
            
    def prepare_data(self, data, features=None, target=None):
        """Prepare data for the expert model."""
        if features is None:
            # Use all numeric columns as features
            features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
        self.features = features
        self.target = target
        
        # Update input dimension if not set
        if self.input_dim is None:
            self.input_dim = len(features)
            self._initialize_model()
            
        # Preprocess the data using signal processor
        if not features:
            return data[features]
        
        try:
            processed_data = self.signal_processor.fit_transform(data[features])
            return processed_data
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return data[features]
            
    def train(self, X, y=None):
        """Train the expert model."""
        if self.model is None:
            raise ValueError("Model not initialized")
            
        try:
            # Preprocess the data
            X_processed = self.prepare_data(X, self.features, self.target)
            
            # Train the model
            self.model.fit(X_processed, y)
            self.is_trained = True
            
            # Calculate feature importance
            self.calculate_feature_importance()
            
            return True
        except Exception as e:
            logger.error(f"Error training physiological expert: {str(e)}")
            return False
            
    def predict(self, X):
        """Generate predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
            
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error in physiological expert prediction: {str(e)}")
            return None
    
    def preprocess_data(self, X):
        """
        Preprocess physiological data with robust error handling.
        
        Args:
            X: Input features DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            X = X.copy()
            
            # Handle blood pressure values
            if 'blood_pressure' in X.columns:
                try:
                    # Convert string BP values (e.g. "120/80") to separate columns
                    def extract_bp(value):
                        try:
                            if pd.isna(value):
                                return pd.Series({'systolic': np.nan, 'diastolic': np.nan})
                            if isinstance(value, str) and '/' in value:
                                parts = value.split('/')
                                if len(parts) == 2:
                                    try:
                                        systolic = float(parts[0].strip())
                                        diastolic = float(parts[1].strip())
                                        return pd.Series({'systolic': systolic, 'diastolic': diastolic})
                                    except ValueError:
                                        # Log the problematic value
                                        logger.warning(f"Could not convert blood pressure: '{value}'")
                                        return pd.Series({'systolic': np.nan, 'diastolic': np.nan})
                            elif isinstance(value, (int, float)):
                                # Assume single value is systolic
                                return pd.Series({'systolic': float(value), 'diastolic': np.nan})
                            else:
                                # Log the problematic value
                                logger.warning(f"Unexpected blood pressure format: '{value}' of type {type(value)}")
                                return pd.Series({'systolic': np.nan, 'diastolic': np.nan})
                        except Exception as e:
                            logger.warning(f"Error processing blood pressure value '{value}': {str(e)}")
                            return pd.Series({'systolic': np.nan, 'diastolic': np.nan})
                    
                    # Apply the extraction function, handling errors
                    bp_df = X['blood_pressure'].apply(extract_bp)
                    
                    # Drop the original column
                    X = X.drop('blood_pressure', axis=1)
                    
                    # Add the new columns
                    X = pd.concat([X, bp_df], axis=1)
                    
                    # Fill missing values with medians
                    for col in ['systolic', 'diastolic']:
                        median = X[col].median()
                        if pd.isna(median):
                            # Use typical values if no valid data
                            median = 120 if col == 'systolic' else 80
                        X[col] = X[col].fillna(median)
                        
                    # Clip to physiologically reasonable ranges
                    X['systolic'] = X['systolic'].clip(70, 200)
                    X['diastolic'] = X['diastolic'].clip(40, 130)
                    
                    logger.info(f"Processed blood pressure values: {len(bp_df)} rows - systolic range [{X['systolic'].min():.1f}, {X['systolic'].max():.1f}], diastolic range [{X['diastolic'].min():.1f}, {X['diastolic'].max():.1f}]")
                    
                except Exception as e:
                    logger.error(f"Error processing blood pressure: {str(e)}")
                    # Create default columns if processing fails
                    X['systolic'] = 120
                    X['diastolic'] = 80
            
            # Handle other vital signs
            vital_signs = ['heart_rate', 'respiratory_rate', 'temperature', 'oxygen_saturation']
            for col in vital_signs:
                if col in X.columns:
                    try:
                        # Convert to numeric, coercing errors to NaN
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        
                        # Fill missing values with median
                        median = X[col].median()
                        if pd.isna(median):
                            # Use typical values if no valid data
                            default_values = {
                                'heart_rate': 75,
                                'respiratory_rate': 16,
                                'temperature': 37,
                                'oxygen_saturation': 98
                            }
                            median = default_values.get(col, X[col].mean())
                        X[col] = X[col].fillna(median)
                        
                        # Clip to physiologically reasonable ranges
                        ranges = {
                            'heart_rate': (40, 200),
                            'respiratory_rate': (8, 40),
                            'temperature': (35, 42),
                            'oxygen_saturation': (70, 100)
                        }
                        if col in ranges:
                            X[col] = X[col].clip(*ranges[col])
                            
                        logger.info(f"Processed {col}: range [{X[col].min():.1f}, {X[col].max():.1f}]")
                        
                    except Exception as e:
                        logger.error(f"Error processing {col}: {str(e)}")
                        # Use typical value if processing fails
                        default_values = {
                            'heart_rate': 75,
                            'respiratory_rate': 16,
                            'temperature': 37,
                            'oxygen_saturation': 98
                        }
                        X[col] = default_values.get(col, 0)
            
            # Store feature columns for importance calculation
            self.feature_columns = list(X.columns)
            
            return X
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            # Return original data if preprocessing fails
            return X
    
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
    
    def optimize_hyperparameters(self, X, y):
        """
        Optimize model hyperparameters using evolutionary strategy.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Dictionary of optimized parameters
        """
        try:
            from ..experts.optimizer_adapters import DifferentialEvolutionAdapter
            
            # Define parameter bounds
            param_space = {
                'hidden_layer_sizes': {
                    'bounds': [(16, 256), (8, 128)],  # Two layers with their respective bounds
                },
                'learning_rate': {
                    'bounds': [(0.0001, 0.1)],
                },
                'max_iter': {
                    'bounds': [(100, 500)],
                },
                'alpha': {
                    'bounds': [(0.0001, 0.01)],
                }
            }
            
            # Extract bounds for optimizer
            bounds = []
            param_names = []
            
            for param_name, config in param_space.items():
                for bound in config['bounds']:
                    bounds.append(bound)
                    param_names.append(param_name)
            
            # Create fitness function for hyperparameter optimization
            def fitness_function(params):
                try:
                    # Create parameter dictionary
                    param_dict = {}
                    
                    # Group parameters by name
                    current_param = 0
                    for i, name in enumerate(param_names):
                        if name not in param_dict:
                            param_dict[name] = []
                        param_dict[name].append(params[i])
                    
                    # Process parameters
                    processed_params = {}
                    
                    # Handle special cases for neural network parameters
                    if 'hidden_layer_sizes' in param_dict:
                        # Ensure values are integers and at least 4
                        hidden_layers = [max(4, int(layer)) for layer in param_dict['hidden_layer_sizes']]
                        # Convert to tuple for MLPRegressor
                        processed_params['hidden_layer_sizes'] = tuple(hidden_layers)
                    
                    # Process other parameters
                    if 'learning_rate' in param_dict:
                        processed_params['learning_rate'] = float(param_dict['learning_rate'][0])
                    
                    if 'max_iter' in param_dict:
                        processed_params['max_iter'] = int(param_dict['max_iter'][0])
                        
                    if 'alpha' in param_dict:
                        processed_params['alpha'] = float(param_dict['alpha'][0])
                    
                    # Create a model with these parameters
                    from sklearn.neural_network import MLPRegressor
                    model = MLPRegressor(
                        activation=self.activation,
                        random_state=42,
                        **processed_params
                    )
                    
                    if 'batch_size' in param_dict:
                        try:
                            batch_size = param_dict['batch_size'][0] # get the float
                            processed_params['batch_size'] = max(1, min(512, int('batch_size')))
                        except:
                            processed_params['batch_size'] = 32
                    
                    # Use cross-validation to evaluate
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(
                        model, X, y, 
                        cv=3, 
                        scoring='neg_mean_squared_error',
                        error_score=float('nan')
                    )
                    
                    # Filter out NaNs
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    
                    if valid_scores:
                        return -np.mean(valid_scores)  # Return negative MSE for minimization
                    else:
                        logger.warning("All cross-validation scores were NaN")
                        return float('inf')  # Return worst possible score
                
                except Exception as e:
                    logger.error(f"Error in fitness function: {str(e)}")
                    return float('inf')  # Return worst possible score
            
            # Initialize optimizer
            optimizer = DifferentialEvolutionAdapter(
                fitness_function=fitness_function,
                bounds=bounds,
                population_size=15,
                max_iterations=30,
                crossover_probability=0.7,
                differential_weight=0.8,
                random_seed=42
            )
            
            # Run optimization
            best_params, best_fitness = optimizer.optimize()
            
            # Convert best parameters to dictionary
            best_param_dict = {}
            
            # Group parameters by name
            current_param = 0
            grouped_params = {}
            for i, name in enumerate(param_names):
                if name not in grouped_params:
                    grouped_params[name] = []
                grouped_params[name].append(best_params[i])
            
            # Process parameters for the model
            if 'hidden_layer_sizes' in grouped_params:
                # Ensure values are integers and at least 4
                hidden_layers = [max(4, int(layer)) for layer in grouped_params['hidden_layer_sizes']]
                best_param_dict['hidden_layer_sizes'] = tuple(hidden_layers)
                # Also update the expert's hidden_layers attribute
                self.hidden_layers = list(hidden_layers)
            
            if 'learning_rate' in grouped_params:
                best_param_dict['learning_rate'] = float(grouped_params['learning_rate'][0])
                
            if 'max_iter' in grouped_params:
                best_param_dict['max_iter'] = int(grouped_params['max_iter'][0])
                
            if 'alpha' in grouped_params:
                best_param_dict['alpha'] = float(grouped_params['alpha'][0])
            
            # Update model parameters and reinitialize
            self.model_params.update(best_param_dict)
            self._initialize_model()
            
            logger.info(f"Optimized hyperparameters: {best_param_dict}")
            return best_param_dict
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            
            # Return default parameters if optimization fails
            default_params = {
                'hidden_layer_sizes': (64, 32),
                'learning_rate': 0.001,
                'max_iter': 200,
                'alpha': 0.0001
            }
            
            # Update model with default parameters
            self.hidden_layers = [64, 32]
            self.model_params.update(default_params)
            self._initialize_model()
            
            logger.info(f"Using default parameters due to optimization failure: {default_params}")
            return default_params
    
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
            model_type=metadata.get('model_type', 'neural_network'),
            hidden_layers=metadata.get('hidden_layers', [64, 32]),
            activation=metadata.get('activation', 'relu'),
            input_dim=metadata.get('input_dim', None),
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
        if 'signal_processor' in save_dict:
            instance.signal_processor = save_dict['signal_processor']
        
        return instance

    def extract_variability_features(self, data):
        """
        Extract variability features from physiological time series data.
        
        Args:
            data: DataFrame containing physiological time series data
            
        Returns:
            DataFrame with added variability features
        """
        try:
            # Clone the input data
            data_with_variability = data.copy()
            
            # Identify numerical columns that might contain physiological signals
            numeric_cols = data.select_dtypes(include=['number']).columns
            potential_vital_cols = [col for col in numeric_cols if col in self.vital_cols or
                                  any(term in col.lower() for term in 
                                     ['heart', 'bp', 'pressure', 'temp', 'spo2', 'respiratory',
                                      'systolic', 'diastolic', 'pulse', 'oxygen'])]
            
            if not potential_vital_cols:
                logger.warning("No suitable columns found for variability extraction")
                return data
                
            # Calculate variability metrics for each potential vital sign
            for col in potential_vital_cols:
                # Skip if the column has too many NaN values
                if data[col].isna().sum() > 0.5 * len(data):
                    continue
                    
                try:
                    # Calculate basic statistics
                    data_with_variability[f"{col}_std"] = data[col].std()
                    data_with_variability[f"{col}_range"] = data[col].max() - data[col].min()
                    
                    # Calculate rolling statistics if we have enough data points
                    if len(data) >= 5:
                        # Rolling mean and std
                        rolling_window = min(5, len(data) // 2)
                        rolling_mean = data[col].rolling(window=rolling_window, min_periods=1).mean()
                        rolling_std = data[col].rolling(window=rolling_window, min_periods=1).std().fillna(0)
                        
                        data_with_variability[f"{col}_rolling_mean"] = rolling_mean
                        data_with_variability[f"{col}_rolling_std"] = rolling_std
                        
                        # Calculate trend (simple linear slope)
                        if len(data) >= 3:
                            import numpy as np
                            x = np.arange(len(data))
                            y = data[col].values
                            mask = ~np.isnan(y)
                            if np.sum(mask) >= 2:  # Need at least 2 points for linear regression
                                from scipy.stats import linregress
                                slope, _, _, _, _ = linregress(x[mask], y[mask])
                                data_with_variability[f"{col}_trend"] = slope
                    
                    # Calculate z-scores for anomaly detection
                    mean, std = data[col].mean(), data[col].std()
                    if std > 0:
                        data_with_variability[f"{col}_zscore"] = (data[col] - mean) / std
                
                except Exception as e:
                    logger.error(f"Error calculating variability for {col}: {str(e)}")
                
            logger.info(f"Added {len(data_with_variability.columns) - len(data.columns)} variability features")
            return data_with_variability
            
        except Exception as e:
            logger.error(f"Error extracting variability features: {str(e)}")
            return data

"""
Preprocessing Pipeline Module

This module provides a configurable, extensible pipeline for preprocessing data within the MoE framework.
It extends the existing preprocessing functionality while maintaining compatibility with the
evolutionary computation aspects of the framework.
"""

import pandas as pd
import numpy as np
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif


class PreprocessingOperation(ABC):
    """Abstract base class for all preprocessing operations."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the operation to the data."""
        pass
        
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply the operation to the data."""
        pass
        
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        pass
        
    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        pass
        
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform the data."""
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        return {}


class MissingValueHandler(PreprocessingOperation):
    """Handle missing values in the data."""
    
    def __init__(self, strategy: str = 'mean', fill_value: Optional[Any] = None, 
                 categorical_strategy: str = 'most_frequent', exclude_cols: List[str] = None):
        """Initialize the missing value handler.
        
        Args:
            strategy: Strategy for imputing missing values in numeric columns.
                     Options: 'mean', 'median', 'most_frequent', 'constant'
            fill_value: Value to use with the 'constant' strategy
            categorical_strategy: Strategy for imputing missing values in categorical columns.
                                 Options: 'most_frequent', 'constant'
            exclude_cols: Columns to exclude from imputation
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.categorical_strategy = categorical_strategy
        self.exclude_cols = exclude_cols or []
        self.fitted_imputers = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit imputers to the data."""
        # Clear previous fitted imputers
        self.fitted_imputers = {}
        
        # Get numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        
        # Filter out excluded columns
        numeric_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        categorical_cols = [col for col in categorical_cols if col not in self.exclude_cols]
        
        # Fit imputers for numeric columns
        for col in numeric_cols:
            if data[col].isna().any():
                if self.strategy == 'constant':
                    imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
                else:
                    imputer = SimpleImputer(strategy=self.strategy)
                imputer.fit(data[[col]])
                self.fitted_imputers[col] = imputer
        
        # Fit imputers for categorical columns
        for col in categorical_cols:
            if data[col].isna().any():
                if self.categorical_strategy == 'constant':
                    imputer = SimpleImputer(strategy=self.categorical_strategy, fill_value=self.fill_value)
                else:
                    imputer = SimpleImputer(strategy=self.categorical_strategy)
                # Convert to object type to handle categorical data
                imputer.fit(data[[col]].astype('object'))
                self.fitted_imputers[col] = imputer
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Impute missing values in the data."""
        result = data.copy()
        
        # Apply imputers
        for col, imputer in self.fitted_imputers.items():
            if col in result.columns:
                is_categorical = result[col].dtype.name not in ['int64', 'float64', 'int32', 'float32']
                
                if is_categorical:
                    # Convert to object type for categorical data
                    col_data = result[[col]].astype('object')
                    result[col] = imputer.transform(col_data).flatten()
                else:
                    result[col] = imputer.transform(result[[col]]).flatten()
        
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'strategy': self.strategy,
            'fill_value': self.fill_value,
            'categorical_strategy': self.categorical_strategy,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'strategy' in params:
            self.strategy = params['strategy']
        if 'fill_value' in params:
            self.fill_value = params['fill_value']
        if 'categorical_strategy' in params:
            self.categorical_strategy = params['categorical_strategy']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Count missing values before and after
        missing_before = data.isna().sum().sum()
        missing_after = transformed_data.isna().sum().sum()
        
        metrics['missing_values_before'] = int(missing_before)
        metrics['missing_values_after'] = int(missing_after)
        metrics['missing_values_handled'] = int(missing_before - missing_after)
        
        if missing_before > 0:
            metrics['missing_values_handled_pct'] = (missing_before - missing_after) / missing_before * 100
        else:
            metrics['missing_values_handled_pct'] = 100.0
            
        return metrics


class OutlierHandler(PreprocessingOperation):
    """Detect and handle outliers in the data."""
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0, 
                 strategy: str = 'winsorize', exclude_cols: List[str] = None):
        """Initialize the outlier handler.
        
        Args:
            method: Method for detecting outliers. Options: 'zscore', 'iqr'
            threshold: Threshold for outlier detection
            strategy: Strategy for handling outliers. Options: 'winsorize', 'remove'
            exclude_cols: Columns to exclude from outlier handling
        """
        self.method = method
        self.threshold = float(threshold)  # Ensure threshold is always a float
        self.strategy = strategy
        self.exclude_cols = exclude_cols or []
        self.outlier_stats = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the outlier detection method to the data."""
        # Clear previous outlier stats
        self.outlier_stats = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out excluded columns
        numeric_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        # Calculate outlier stats for each column
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            if self.method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                if std == 0:
                    continue
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
                
            elif self.method == 'iqr':
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower_bound = q1 - self.threshold * iqr
                upper_bound = q3 + self.threshold * iqr
                
            else:
                raise ValueError(f"Unknown outlier detection method: {self.method}")
                
            self.outlier_stats[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Detect and handle outliers in the data."""
        result = data.copy()
        outlier_mask = pd.Series(False, index=data.index)
        
        # Apply outlier handling for each column
        for col, stats in self.outlier_stats.items():
            if col not in result.columns:
                continue
                
            lower_bound = stats['lower_bound']
            upper_bound = stats['upper_bound']
            
            # Detect outliers
            col_outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers
            
            # Handle outliers
            if self.strategy == 'winsorize':
                # Cap values at the bounds
                result.loc[result[col] < lower_bound, col] = lower_bound
                result.loc[result[col] > upper_bound, col] = upper_bound
        
        # Remove outlier rows if strategy is 'remove'
        if self.strategy == 'remove':
            result = result[~outlier_mask]
            
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'strategy': self.strategy,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'method' in params:
            self.method = params['method']
        if 'threshold' in params:
            self.threshold = float(params['threshold'])  # Ensure threshold is always a float
        if 'strategy' in params:
            self.strategy = params['strategy']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Count outliers detected and handled
        outlier_count = 0
        outlier_handled = 0
        
        for col, stats in self.outlier_stats.items():
            if col not in data.columns:
                continue
                
            lower_bound = stats['lower_bound']
            upper_bound = stats['upper_bound']
            
            # Count outliers in original data
            col_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_count += col_outliers
            
            # Count outliers in transformed data
            if self.strategy == 'winsorize':
                col_outliers_after = ((transformed_data[col] < lower_bound) | 
                                     (transformed_data[col] > upper_bound)).sum()
                outlier_handled += (col_outliers - col_outliers_after)
            elif self.strategy == 'remove':
                # For remove strategy, all outliers are handled
                outlier_handled += col_outliers
        
        metrics['outliers_detected'] = int(outlier_count)
        metrics['outliers_handled'] = int(outlier_handled)
        
        if outlier_count > 0:
            metrics['outliers_handled_pct'] = outlier_handled / outlier_count * 100
        else:
            metrics['outliers_handled_pct'] = 100.0
            
        return metrics


class FeatureScaler(PreprocessingOperation):
    """Scale features in the data."""
    
    def __init__(self, method: str = 'minmax', feature_range: Tuple[float, float] = (0, 1), 
                 exclude_cols: List[str] = None):
        """Initialize the feature scaler.
        
        Args:
            method: Method for scaling features. Options: 'minmax', 'standard', 'robust'
            feature_range: Range for scaled features (only used with 'minmax')
            exclude_cols: Columns to exclude from scaling
        """
        self.method = method
        self.feature_range = feature_range
        self.exclude_cols = exclude_cols or []
        self.scalers = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit scalers to the data."""
        # Clear previous scalers
        self.scalers = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out excluded columns
        numeric_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        # Create and fit scalers for each column
        for col in numeric_cols:
            col_data = data[[col]].dropna()
            
            if len(col_data) == 0:
                continue
                
            if self.method == 'minmax':
                scaler = MinMaxScaler(feature_range=self.feature_range)
            elif self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")
                
            scaler.fit(col_data)
            self.scalers[col] = scaler
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Scale features in the data."""
        result = data.copy()
        
        # Apply scalers
        for col, scaler in self.scalers.items():
            if col in result.columns:
                # Create a mask for non-NaN values
                mask = ~result[col].isna()
                
                if mask.any():
                    # Apply scaler only to non-NaN values
                    result.loc[mask, col] = scaler.transform(result.loc[mask, [col]]).flatten()
        
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'feature_range': self.feature_range,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'method' in params:
            self.method = params['method']
        if 'feature_range' in params:
            self.feature_range = params['feature_range']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Calculate scaling metrics
        scaled_cols = 0
        mean_range_before = 0
        mean_range_after = 0
        
        for col in self.scalers.keys():
            if col in data.columns and col in transformed_data.columns:
                scaled_cols += 1
                
                # Calculate range before scaling
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    range_before = col_data.max() - col_data.min()
                    mean_range_before += range_before
                
                # Calculate range after scaling
                col_data_after = transformed_data[col].dropna()
                if len(col_data_after) > 0:
                    range_after = col_data_after.max() - col_data_after.min()
                    mean_range_after += range_after
        
        if scaled_cols > 0:
            metrics['scaled_columns'] = scaled_cols
            metrics['mean_range_before'] = mean_range_before / scaled_cols
            metrics['mean_range_after'] = mean_range_after / scaled_cols
            
        return metrics


class CategoryEncoder(PreprocessingOperation):
    """Encode categorical features in the data."""
    
    def __init__(self, method: str = 'label', exclude_cols: List[str] = None):
        """Initialize the category encoder.
        
        Args:
            method: Method for encoding categorical features. Options: 'label', 'onehot'
            exclude_cols: Columns to exclude from encoding
        """
        self.method = method
        self.exclude_cols = exclude_cols or []
        self.encoders = {}
        self.encoded_columns = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit encoders to the data."""
        # Clear previous encoders
        self.encoders = {}
        self.encoded_columns = {}
        
        # Get categorical columns
        categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        
        # Filter out excluded columns
        categorical_cols = [col for col in categorical_cols if col not in self.exclude_cols]
        
        # Create and fit encoders for each column
        for col in categorical_cols:
            col_data = data[[col]].astype('object').fillna('missing')
            
            if self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(col_data[col])
                self.encoders[col] = encoder
                
            elif self.method == 'onehot':
                # Handle both older and newer scikit-learn versions
                try:
                    # For newer scikit-learn versions
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                except TypeError:
                    # For older scikit-learn versions
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoder.fit(col_data)
                self.encoders[col] = encoder
                
                # Store the encoded column names
                categories = encoder.categories_[0]
                self.encoded_columns[col] = [f"{col}_{cat}" for cat in categories]
                
            else:
                raise ValueError(f"Unknown encoding method: {self.method}")
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Encode categorical features in the data."""
        result = data.copy()
        
        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in result.columns:
                # Fill missing values with 'missing'
                col_data = result[[col]].astype('object').fillna('missing')
                
                if self.method == 'label':
                    try:
                        result[col] = encoder.transform(col_data[col])
                    except ValueError:
                        # Handle unknown categories
                        result[col] = col_data[col].map(lambda x: 0 if x not in encoder.classes_ else encoder.transform([x])[0])
                        
                elif self.method == 'onehot':
                    encoded = encoder.transform(col_data)
                    
                    # Create DataFrame with encoded columns
                    encoded_df = pd.DataFrame(encoded, index=result.index, columns=self.encoded_columns[col])
                    
                    # Drop original column and add encoded columns
                    result = result.drop(columns=[col])
                    result = pd.concat([result, encoded_df], axis=1)
        
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'method' in params:
            self.method = params['method']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Count encoded columns
        encoded_cols = len(self.encoders)
        metrics['encoded_columns'] = encoded_cols
        
        if self.method == 'onehot':
            # Count new columns created
            new_cols = sum(len(cols) for cols in self.encoded_columns.values())
            metrics['new_columns_created'] = new_cols
            
        return metrics


class FeatureSelector(PreprocessingOperation):
    """Select features from the data."""
    
    def __init__(self, method: str = 'variance', threshold: float = 0.0, 
                 k: int = None, target_col: str = None, exclude_cols: List[str] = None,
                 use_evolutionary: bool = False, ec_algorithm: str = 'aco'):
        """Initialize the feature selector.
        
        Args:
            method: Method for selecting features. Options: 'variance', 'kbest', 'evolutionary'
            threshold: Threshold for variance-based selection
            k: Number of top features to select for k-best selection
            target_col: Target column for supervised feature selection
            exclude_cols: Columns to exclude from selection
            use_evolutionary: Whether to use evolutionary computation for feature selection
            ec_algorithm: Evolutionary computation algorithm to use. Options: 'aco', 'de', 'gwo'
        """
        self.method = method
        self.threshold = threshold
        self.k = k
        self.target_col = target_col
        self.exclude_cols = exclude_cols or []
        self.use_evolutionary = use_evolutionary
        self.ec_algorithm = ec_algorithm
        self.selected_features = []
        self.selector = None
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the feature selection method to the data."""
        # Clear previous selection
        self.selected_features = []
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out excluded columns and target column
        exclude = self.exclude_cols.copy()
        if self.target_col is not None:
            exclude.append(self.target_col)
        numeric_cols = [col for col in numeric_cols if col not in exclude]
        
        if len(numeric_cols) == 0:
            return
            
        # Create feature matrix
        X = data[numeric_cols].copy()
        
        # Handle missing values for feature selection
        X = X.fillna(X.mean())
        
        if self.method == 'variance':
            # Variance-based selection
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(X)
            
            # Get selected features
            mask = self.selector.get_support()
            self.selected_features = [col for i, col in enumerate(numeric_cols) if mask[i]]
            
        elif self.method == 'kbest' and self.target_col is not None and self.target_col in data.columns:
            # K-best selection
            if self.k is None or self.k > len(numeric_cols):
                self.k = len(numeric_cols)
                
            y = data[self.target_col].fillna(0)  # Handle missing values in target
            
            self.selector = SelectKBest(f_classif, k=self.k)
            self.selector.fit(X, y)
            
            # Get selected features
            mask = self.selector.get_support()
            self.selected_features = [col for i, col in enumerate(numeric_cols) if mask[i]]
            
        elif self.method == 'evolutionary' or self.use_evolutionary:
            # Evolutionary feature selection
            # This is a placeholder for integration with EC algorithms
            # In a real implementation, this would use the specified EC algorithm
            
            if self.ec_algorithm == 'aco':
                # Placeholder for Ant Colony Optimization
                # In a real implementation, this would use the ACO algorithm
                # For now, we'll just select random features as a placeholder
                import random
                random.seed(42)  # For reproducibility
                
                if self.k is None or self.k > len(numeric_cols):
                    self.k = len(numeric_cols) // 2  # Select half of features by default
                    
                self.selected_features = random.sample(numeric_cols, self.k)
                
            else:
                # Default to selecting all features
                self.selected_features = numeric_cols
        else:
            # Default to selecting all features
            self.selected_features = numeric_cols
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Select features from the data."""
        result = data.copy()
        
        # Get all columns that should be kept
        keep_cols = self.selected_features.copy()
        
        # Add excluded columns and target column
        keep_cols.extend([col for col in self.exclude_cols if col in data.columns])
        if self.target_col is not None and self.target_col in data.columns:
            keep_cols.append(self.target_col)
            
        # Add non-numeric columns
        non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        keep_cols.extend(non_numeric_cols)
        
        # Select only the columns to keep
        keep_cols = list(set(keep_cols))  # Remove duplicates
        keep_cols = [col for col in keep_cols if col in data.columns]  # Ensure all columns exist
        
        if len(keep_cols) > 0:
            result = result[keep_cols]
            
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'k': self.k,
            'target_col': self.target_col,
            'exclude_cols': self.exclude_cols,
            'use_evolutionary': self.use_evolutionary,
            'ec_algorithm': self.ec_algorithm
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'method' in params:
            self.method = params['method']
        if 'threshold' in params:
            self.threshold = params['threshold']
        if 'k' in params:
            self.k = params['k']
        if 'target_col' in params:
            self.target_col = params['target_col']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        if 'use_evolutionary' in params:
            self.use_evolutionary = params['use_evolutionary']
        if 'ec_algorithm' in params:
            self.ec_algorithm = params['ec_algorithm']
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Count features before and after selection
        features_before = len(data.columns)
        features_after = len(transformed_data.columns)
        features_removed = features_before - features_after
        
        metrics['features_before'] = features_before
        metrics['features_after'] = features_after
        metrics['features_removed'] = features_removed
        
        if features_before > 0:
            metrics['features_removed_pct'] = features_removed / features_before * 100
            
        return metrics


class TimeSeriesProcessor(PreprocessingOperation):
    """Process time series data."""
    
    def __init__(self, time_col: str, resample_freq: str = None, 
                 lag_features: List[int] = None, rolling_windows: List[int] = None,
                 exclude_cols: List[str] = None):
        """Initialize the time series processor.
        
        Args:
            time_col: Column containing timestamps
            resample_freq: Frequency for resampling. Options: 'D', 'H', '15min', etc.
            lag_features: List of lag periods to create features for
            rolling_windows: List of window sizes for rolling statistics
            exclude_cols: Columns to exclude from processing
        """
        self.time_col = time_col
        self.resample_freq = resample_freq
        self.lag_features = lag_features or []
        self.rolling_windows = rolling_windows or []
        self.exclude_cols = exclude_cols or []
        self.time_stats = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the time series processor to the data."""
        # Clear previous stats
        self.time_stats = {}
        
        # Check if time column exists
        if self.time_col not in data.columns:
            return
            
        # Convert time column to datetime if needed
        time_data = pd.to_datetime(data[self.time_col], errors='coerce')
        
        # Calculate time statistics
        self.time_stats['min_time'] = time_data.min()
        self.time_stats['max_time'] = time_data.max()
        
        if self.time_stats['min_time'] is not pd.NaT and self.time_stats['max_time'] is not pd.NaT:
            self.time_stats['time_range'] = (self.time_stats['max_time'] - self.time_stats['min_time']).total_seconds()
        else:
            self.time_stats['time_range'] = 0
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process time series data."""
        result = data.copy()
        
        # Check if time column exists
        if self.time_col not in result.columns:
            return result
            
        # Convert time column to datetime if needed
        result[self.time_col] = pd.to_datetime(result[self.time_col], errors='coerce')
        
        # Set time column as index for time series operations
        result = result.set_index(self.time_col)
        
        # Get numeric columns for feature creation
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out excluded columns
        numeric_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        # Resample if frequency is specified
        if self.resample_freq is not None:
            try:
                # Resample and aggregate
                result = result.resample(self.resample_freq).mean()
                
                # Forward fill missing values after resampling
                result = result.fillna(method='ffill')
            except Exception as e:
                # Resampling failed, revert to original data
                result = data.copy()
                result[self.time_col] = pd.to_datetime(result[self.time_col], errors='coerce')
                result = result.set_index(self.time_col)
        
        # Create lag features
        for lag in self.lag_features:
            if lag <= 0:
                continue
                
            for col in numeric_cols:
                lag_col_name = f"{col}_lag_{lag}"
                result[lag_col_name] = result[col].shift(lag)
        
        # Create rolling window features
        for window in self.rolling_windows:
            if window <= 1:
                continue
                
            for col in numeric_cols:
                # Rolling mean
                mean_col_name = f"{col}_roll_mean_{window}"
                result[mean_col_name] = result[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling standard deviation
                std_col_name = f"{col}_roll_std_{window}"
                result[std_col_name] = result[col].rolling(window=window, min_periods=1).std()
        
        # Extract time features
        result['hour'] = result.index.hour
        result['day'] = result.index.day
        result['month'] = result.index.month
        result['year'] = result.index.year
        result['dayofweek'] = result.index.dayofweek
        
        # Reset index to get time column back
        result = result.reset_index()
        
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'time_col': self.time_col,
            'resample_freq': self.resample_freq,
            'lag_features': self.lag_features,
            'rolling_windows': self.rolling_windows,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'time_col' in params:
            self.time_col = params['time_col']
        if 'resample_freq' in params:
            self.resample_freq = params['resample_freq']
        if 'lag_features' in params:
            self.lag_features = params['lag_features']
        if 'rolling_windows' in params:
            self.rolling_windows = params['rolling_windows']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Count features before and after transformation
        features_before = len(data.columns)
        features_after = len(transformed_data.columns)
        features_added = features_after - features_before
        
        metrics['features_before'] = features_before
        metrics['features_after'] = features_after
        metrics['features_added'] = features_added
        
        # Count missing values after transformation
        missing_values = transformed_data.isna().sum().sum()
        metrics['missing_values_after'] = int(missing_values)
        
        if features_after > 0:
            metrics['missing_values_pct'] = missing_values / (features_after * len(transformed_data)) * 100
            
        return metrics


class PreprocessingPipeline:
    """A configurable pipeline for preprocessing data.
    
    This class allows for chaining multiple preprocessing operations together
    and executing them in sequence. It provides functionality for saving and
    loading pipeline configurations, tracking quality metrics, and integrating
    with the evolutionary computation aspects of the MoE framework.
    """
    
    def __init__(self, operations: List[PreprocessingOperation] = None, name: str = 'default'):
        """Initialize the preprocessing pipeline.
        
        Args:
            operations: List of preprocessing operations to apply in sequence
            name: Name of the pipeline
        """
        self.operations = operations or []
        self.name = name
        self.quality_metrics = {}
        self.is_fitted = False
        
    def add_operation(self, operation: PreprocessingOperation) -> 'PreprocessingPipeline':
        """Add a preprocessing operation to the pipeline.
        
        Args:
            operation: The preprocessing operation to add
            
        Returns:
            The pipeline instance for method chaining
        """
        self.operations.append(operation)
        self.is_fitted = False
        return self
        
    def remove_operation(self, index: int) -> 'PreprocessingPipeline':
        """Remove a preprocessing operation from the pipeline.
        
        Args:
            index: The index of the operation to remove
            
        Returns:
            The pipeline instance for method chaining
        """
        if 0 <= index < len(self.operations):
            self.operations.pop(index)
            self.is_fitted = False
        return self
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'PreprocessingPipeline':
        """Fit all operations in the pipeline to the data.
        
        Args:
            data: The data to fit the pipeline to
            **kwargs: Additional arguments to pass to the operations
            
        Returns:
            The pipeline instance for method chaining
        """
        # Clear previous quality metrics
        self.quality_metrics = {}
        
        # Fit each operation in sequence
        current_data = data.copy()
        
        for i, operation in enumerate(self.operations):
            # Fit the operation
            operation.fit(current_data, **kwargs)
            
            # Transform the data for the next operation
            transformed_data = operation.transform(current_data, **kwargs)
            
            # Get quality metrics for this operation
            metrics = operation.get_quality_metrics(current_data, transformed_data)
            self.quality_metrics[f"step_{i}_{operation.__class__.__name__}"] = metrics
            
            # Update current data for next operation
            current_data = transformed_data
        
        self.is_fitted = True
        return self
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply all operations in the pipeline to the data.
        
        Args:
            data: The data to transform
            **kwargs: Additional arguments to pass to the operations
            
        Returns:
            The transformed data
        """
        current_data = data.copy()
        
        for operation in self.operations:
            current_data = operation.transform(current_data, **kwargs)
            
        return current_data
        
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform the data.
        
        Args:
            data: The data to fit and transform
            **kwargs: Additional arguments to pass to the operations
            
        Returns:
            The transformed data
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)
        
    def get_quality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get quality metrics for all operations in the pipeline.
        
        Returns:
            A dictionary of quality metrics for each operation
        """
        return self.quality_metrics
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the pipeline.
        
        Returns:
            A dictionary of pipeline parameters
        """
        params = {
            'name': self.name,
            'operations': []
        }
        
        for op in self.operations:
            op_params = {
                'type': op.__class__.__name__,
                'params': op.get_params()
            }
            params['operations'].append(op_params)
            
        return params
        
    def set_params(self, params: Dict[str, Any]) -> 'PreprocessingPipeline':
        """Set the parameters of the pipeline.
        
        Args:
            params: A dictionary of pipeline parameters
            
        Returns:
            The pipeline instance for method chaining
        """
        # Set pipeline name
        if 'name' in params:
            self.name = params['name']
            
        # Clear existing operations
        self.operations = []
        
        # Create and add operations
        for op_config in params.get('operations', []):
            op_type = op_config.get('type')
            op_params = op_config.get('params', {})
            
            # Create operation instance based on type
            if op_type == 'MissingValueHandler':
                operation = MissingValueHandler()
            elif op_type == 'OutlierHandler':
                operation = OutlierHandler()
            elif op_type == 'FeatureScaler':
                operation = FeatureScaler()
            elif op_type == 'CategoryEncoder':
                operation = CategoryEncoder()
            elif op_type == 'FeatureSelector':
                operation = FeatureSelector()
            elif op_type == 'TimeSeriesProcessor':
                # TimeSeriesProcessor requires a time_col parameter
                if 'time_col' in op_params:
                    operation = TimeSeriesProcessor(time_col=op_params['time_col'])
                else:
                    # Skip this operation if time_col is not provided
                    continue
            else:
                # Skip unknown operation types
                continue
                
            # Set operation parameters
            operation.set_params(op_params)
            
            # Add operation to pipeline
            self.add_operation(operation)
            
        self.is_fitted = False
        return self
        
    def save_config(self, filepath: str) -> None:
        """Save the pipeline configuration to a JSON file.
        
        Args:
            filepath: Path to save the configuration file
        """
        config = self.get_params()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save configuration to file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
            
    @classmethod
    def load_config(cls, filepath: str) -> 'PreprocessingPipeline':
        """Load a pipeline configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            A new pipeline instance with the loaded configuration
        """
        # Load configuration from file
        with open(filepath, 'r') as f:
            config = json.load(f)
            
        # Create pipeline instance
        pipeline = cls(name=config.get('name', 'default'))
        
        # Create and add operations
        for op_config in config.get('operations', []):
            op_type = op_config.get('type')
            op_params = op_config.get('params', {})
            
            # Create operation instance based on type
            if op_type == 'MissingValueHandler':
                operation = MissingValueHandler()
            elif op_type == 'OutlierHandler':
                operation = OutlierHandler()
            elif op_type == 'FeatureScaler':
                operation = FeatureScaler()
            elif op_type == 'CategoryEncoder':
                operation = CategoryEncoder()
            elif op_type == 'FeatureSelector':
                operation = FeatureSelector()
            elif op_type == 'TimeSeriesProcessor':
                # TimeSeriesProcessor requires a time_col parameter
                if 'time_col' in op_params:
                    operation = TimeSeriesProcessor(time_col=op_params['time_col'])
                else:
                    # Skip this operation if time_col is not provided
                    continue
            else:
                # Skip unknown operation types
                continue
                
            # Set operation parameters
            operation.set_params(op_params)
            
            # Add operation to pipeline
            pipeline.add_operation(operation)
            
        return pipeline
    
    def get_operation_by_index(self, index: int) -> Optional[PreprocessingOperation]:
        """Get an operation by its index in the pipeline.
        
        Args:
            index: The index of the operation
            
        Returns:
            The operation at the specified index, or None if index is out of range
        """
        if 0 <= index < len(self.operations):
            return self.operations[index]
        return None
    
    def get_operation_by_type(self, op_type: str) -> Optional[PreprocessingOperation]:
        """Get the first operation of the specified type in the pipeline.
        
        Args:
            op_type: The type of operation to find
            
        Returns:
            The first operation of the specified type, or None if not found
        """
        for op in self.operations:
            if op.__class__.__name__ == op_type:
                return op
        return None
    
    def update_operation(self, index: int, operation: PreprocessingOperation) -> 'PreprocessingPipeline':
        """Update an operation in the pipeline.
        
        Args:
            index: The index of the operation to update
            operation: The new operation
            
        Returns:
            The pipeline instance for method chaining
        """
        if 0 <= index < len(self.operations):
            self.operations[index] = operation
            self.is_fitted = False
        return self
    
    def clear(self) -> 'PreprocessingPipeline':
        """Clear all operations from the pipeline.
        
        Returns:
            The pipeline instance for method chaining
        """
        self.operations = []
        self.quality_metrics = {}
        self.is_fitted = False
        return self
    
    def __len__(self) -> int:
        """Get the number of operations in the pipeline.
        
        Returns:
            The number of operations
        """
        return len(self.operations)
    
    def __str__(self) -> str:
        """Get a string representation of the pipeline.
        
        Returns:
            A string representation
        """
        return f"PreprocessingPipeline(name='{self.name}', operations={len(self.operations)})"
    
    def summary(self) -> str:
        """Get a summary of the pipeline.
        
        Returns:
            A summary string
        """
        summary = f"PreprocessingPipeline: {self.name}\n"
        summary += f"Number of operations: {len(self.operations)}\n"
        summary += "Operations:\n"
        
        for i, op in enumerate(self.operations):
            summary += f"  {i}. {op.__class__.__name__}\n"
            
        if self.is_fitted:
            summary += "Status: Fitted\n"
        else:
            summary += "Status: Not fitted\n"
            
        return summary

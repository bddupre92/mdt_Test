"""
Universal Data Adapter for Migraine Prediction

This module provides functionality to ingest and adapt any migraine dataset
for use with the migraine prediction system. It handles:
- Dataset inspection and profiling
- Schema detection and adaptation
- Feature selection
- Data preprocessing
- Train/test splitting
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class UniversalDataAdapter:
    """
    Adapts any migraine dataset format to be compatible with the migraine prediction system.
    
    This class inspects incoming data, detects schema, maps features, and prepares 
    datasets for training and prediction in the migraine prediction system.
    """
    
    # Known essential features for migraine prediction
    CORE_FEATURES = [
        'heart_rate', 'temperature', 'barometric_pressure', 'humidity', 'stress_level',
        'sleep_hours', 'hydration_ml', 'caffeine_mg', 'alcohol_units'
    ]
    
    # Potential target columns
    TARGET_COLUMNS = ['migraine', 'migraine_occurred', 'has_migraine', 'migraine_event']
    
    # Date column candidates
    DATE_COLUMNS = ['date', 'timestamp', 'datetime', 'day', 'record_date']
    
    def __init__(self, data_dir=None, auto_feature_selection=True, verbose=False):
        """
        Initialize the adapter.
        
        Args:
            data_dir: Directory where data files are stored
            auto_feature_selection: Whether to automatically perform feature selection
            verbose: Whether to display detailed logs during processing
        """
        self.data_dir = data_dir or os.getcwd()
        self.auto_feature_selection = auto_feature_selection
        self.verbose = verbose
        self.schema = None
        self.feature_map = {}
        self.target_column = None
        self.date_column = None
        self.selected_features = []
        self.derived_features = {}
        
    def load_data(self, file_path, **kwargs):
        """
        Load data from a file or DataFrame.
        
        Args:
            file_path: Path to the CSV file or a pandas DataFrame
            kwargs: Additional arguments to pass to pandas read_csv
            
        Returns:
            Pandas DataFrame with loaded data
        """
        if isinstance(file_path, pd.DataFrame):
            data = file_path.copy()
            file_info = "DataFrame input"
        else:
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.data_dir, file_path)
            data = pd.read_csv(file_path, **kwargs) 
            file_info = f"File: {os.path.basename(file_path)}"
        
        if self.verbose:
            logger.info(f"Loaded data: {file_info}, Shape: {data.shape}")
            
        return data
    
    def inspect_dataset(self, data):
        """
        Inspect a dataset to understand its structure, datatypes, and identify key columns.
        
        Args:
            data: Pandas DataFrame to inspect
            
        Returns:
            Dictionary with dataset profile information
        """
        profile = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_columns': list(data.select_dtypes(include=['number']).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': [],
            'potential_target': None,
            'potential_date': None
        }
        
        # Try to identify datetime columns
        for col in data.columns:
            try:
                if data[col].dtype == 'object':
                    pd.to_datetime(data[col], errors='raise')
                    profile['datetime_columns'].append(col)
            except:
                pass
                
        # Try to identify target column
        for target in self.TARGET_COLUMNS:
            if target in data.columns:
                profile['potential_target'] = target
                break
                
        # Try to identify date column
        for date_col in self.DATE_COLUMNS:
            if date_col in data.columns or date_col in profile['datetime_columns']:
                profile['potential_date'] = date_col
                break
        
        return profile
    
    def detect_schema(self, data):
        """
        Automatically detect the schema of the dataset and map features.
        
        Args:
            data: Pandas DataFrame to analyze
            
        Returns:
            Dictionary with detected schema
        """
        profile = self.inspect_dataset(data)
        
        # Identify target column
        target_column = profile['potential_target']
        if target_column is None:
            # Try to find binary columns that might represent migraine occurrence
            binary_cols = []
            for col in profile['numeric_columns']:
                unique_values = data[col].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                    binary_cols.append(col)
            
            if binary_cols:
                if 'migraine' in binary_cols:
                    target_column = 'migraine'
                elif len(binary_cols) == 1:
                    target_column = binary_cols[0]
                else:
                    for col in binary_cols:
                        if any(kw in col.lower() for kw in ['migraine', 'headache', 'target']):
                            target_column = col
                            break
        
        # Identify date column
        date_column = profile['potential_date']
        if date_column is None and profile['datetime_columns']:
            date_column = profile['datetime_columns'][0]
        
        # Map core features to available columns
        feature_map = {}
        for core_feature in self.CORE_FEATURES:
            # Check for exact matches
            if core_feature in data.columns:
                feature_map[core_feature] = core_feature
            else:
                # Check for similar column names
                potential_matches = []
                for col in data.columns:
                    # Check if core feature name is a substring of column name (case insensitive)
                    if core_feature.lower() in col.lower():
                        potential_matches.append(col)
                
                if potential_matches:
                    feature_map[core_feature] = potential_matches[0]
        
        schema = {
            'target_column': target_column,
            'date_column': date_column,
            'feature_map': feature_map,
            'available_features': [col for col in profile['numeric_columns'] 
                                  if col != target_column and col != date_column],
            'missing_core_features': [f for f in self.CORE_FEATURES if f not in feature_map]
        }
        
        if self.verbose:
            logger.info(f"Detected schema: Target={schema['target_column']}, "
                      f"Date={schema['date_column']}, "
                      f"Mapped {len(schema['feature_map'])} core features, "
                      f"Missing {len(schema['missing_core_features'])} core features")
        
        self.schema = schema
        self.feature_map = feature_map
        self.target_column = target_column
        self.date_column = date_column
        
        return schema
    
    def auto_select_features(self, data, target_col=None, max_features=20, method='f_classif'):
        """
        Automatically select the most relevant features for predicting migraines.
        
        Args:
            data: Pandas DataFrame with features and target
            target_col: Target column name
            max_features: Maximum number of features to select
            method: Feature selection method ('f_classif' or 'mutual_info')
            
        Returns:
            List of selected feature names
        """
        if target_col is None:
            target_col = self.target_column
            
        if target_col is None or target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        # Get numeric columns only
        numeric_cols = data.select_dtypes(include=['number']).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Prepare data
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Fill missing values for feature selection
        X = X.fillna(X.mean())
        
        # Select features
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(max_features, len(feature_cols)))
        else:
            selector = SelectKBest(f_classif, k=min(max_features, len(feature_cols)))
            
        selector.fit(X, y)
        
        # Get feature scores
        scores = selector.scores_
        
        # Create a feature score dictionary
        feature_scores = dict(zip(feature_cols, scores))
        
        # Sort features by score
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected_features = [f[0] for f in sorted_features[:max_features]]
        
        if self.verbose:
            logger.info(f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}...")
            
        self.selected_features = selected_features
        return selected_features
    
    def add_derived_features(self, data):
        """
        Add derived features based on available columns in the dataset.
        
        Args:
            data: Pandas DataFrame to enhance
            
        Returns:
            Enhanced DataFrame with derived features
        """
        df = data.copy()
        derived = {}
        
        # Handle missing core features by creating substitutes
        if 'heart_rate' not in df.columns and 'hr_rest' in df.columns:
            df['heart_rate'] = df['hr_rest']
            derived['heart_rate'] = 'from hr_rest'
            
        if 'hydration_ml' not in df.columns:
            # Try to derive from other columns or set a default
            if 'water_intake_ml' in df.columns:
                df['hydration_ml'] = df['water_intake_ml']
                derived['hydration_ml'] = 'from water_intake_ml'
            else:
                # Use a default value with random variation
                np.random.seed(42)  # For reproducibility
                df['hydration_ml'] = 2000 + np.random.normal(0, 200, size=len(df))
                derived['hydration_ml'] = 'default with variation'
                
        # More derived features could be added here based on domain knowledge
        # For example:
        if 'sleep_hours' in df.columns and 'stress_level' in df.columns:
            df['sleep_stress_ratio'] = df['sleep_hours'] / (df['stress_level'] + 1)  # +1 to avoid div by zero
            derived['sleep_stress_ratio'] = 'sleep_hours / (stress_level + 1)'
            
        if 'barometric_pressure' in df.columns:
            # Calculate pressure changes (if date column exists and data is sorted)
            if self.date_column and self.date_column in df.columns:
                df = df.sort_values(by=self.date_column)
                df['pressure_delta'] = df['barometric_pressure'].diff()
                derived['pressure_delta'] = 'barometric_pressure daily change'
                
        if self.verbose and derived:
            logger.info(f"Added {len(derived)} derived features: {list(derived.keys())}")
            
        self.derived_features = derived
        return df
    
    def prepare_training_data(self, data, target_col=None, test_size=0.2, random_state=42):
        """
        Prepare data for training a migraine prediction model.
        
        Args:
            data: Pandas DataFrame with features and target
            target_col: Target column name
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with X_train, X_test, y_train, y_test, and feature names
        """
        if target_col is None:
            target_col = self.target_column
            
        if target_col is None or target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Detect schema if not already done
        if self.schema is None:
            self.detect_schema(data)
            
        # Add derived features
        data = self.add_derived_features(data)
        
        # Auto select features if enabled
        if self.auto_feature_selection and not self.selected_features:
            self.auto_select_features(data, target_col)
            
        if not self.selected_features:
            # Use all available numeric features if none selected
            self.selected_features = data.select_dtypes(include=['number']).columns.tolist()
            self.selected_features = [f for f in self.selected_features if f != target_col]
            
        # Prepare data
        feature_cols = self.selected_features
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if self.verbose:
            logger.info(f"Prepared training data with {len(feature_cols)} features")
            logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols
        }
    
    def prepare_prediction_data(self, data):
        """
        Prepare data for making predictions.
        
        Args:
            data: Pandas DataFrame with features
            
        Returns:
            DataFrame with prepared features
        """
        # Add derived features
        data = self.add_derived_features(data)
        
        # Get feature columns (either selected or all numeric)
        if self.selected_features:
            feature_cols = [f for f in self.selected_features if f in data.columns]
        else:
            feature_cols = data.select_dtypes(include=['number']).columns.tolist()
            if self.target_column and self.target_column in feature_cols:
                feature_cols.remove(self.target_column)
                
        # Prepare data
        X = data[feature_cols].copy()
        
        # Handle missing values - use mean or default values
        for col in X.columns:
            if X[col].isna().any():
                if col in self.feature_map:
                    # Use default value for core features
                    if col == 'heart_rate':
                        X[col] = X[col].fillna(70)
                    elif col == 'hydration_ml':
                        X[col] = X[col].fillna(2000)
                    elif col == 'stress_level':
                        X[col] = X[col].fillna(5)
                    elif col == 'sleep_hours':
                        X[col] = X[col].fillna(7)
                    else:
                        X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mean())
                    
        if self.verbose:
            logger.info(f"Prepared prediction data with {len(feature_cols)} features")
            
        return X
    
    def preprocess_data(self, data, scaler=None, fit_scaler=False):
        """
        Preprocess data by scaling numeric features.
        
        Args:
            data: Pandas DataFrame to preprocess
            scaler: Optional pre-fitted StandardScaler
            fit_scaler: Whether to fit the scaler on this data
            
        Returns:
            Preprocessed DataFrame and fitted scaler
        """
        if scaler is None and fit_scaler:
            scaler = StandardScaler()
            
        if scaler is not None:
            columns = data.columns
            if fit_scaler:
                data_scaled = scaler.fit_transform(data)
            else:
                data_scaled = scaler.transform(data)
            data = pd.DataFrame(data_scaled, columns=columns)
            
        return data, scaler
        
    def train_test_split_with_dates(self, data, date_col=None, test_ratio=0.2):
        """
        Split data into train and test sets while respecting time order.
        
        Args:
            data: Pandas DataFrame with features and target
            date_col: Date column name
            test_ratio: Fraction of data to use for testing
            
        Returns:
            Train and test DataFrames
        """
        if date_col is None:
            date_col = self.date_column
            
        if date_col is None or date_col not in data.columns:
            # Fall back to random split
            return train_test_split(data, test_size=test_ratio, random_state=42)
            
        # Ensure date column is datetime
        if data[date_col].dtype != 'datetime64[ns]':
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            
        # Sort by date
        sorted_data = data.sort_values(by=date_col)
        
        # Split
        train_size = int(len(sorted_data) * (1 - test_ratio))
        train_data = sorted_data.iloc[:train_size]
        test_data = sorted_data.iloc[train_size:]
        
        return train_data, test_data

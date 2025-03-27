"""
Time Series Validation Module

This module provides specialized cross-validation strategies for time series data
in the context of medical applications, where traditional k-fold cross-validation
may lead to data leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Generator, Union
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import logging

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """
    Specialized validation strategies for time series medical data.
    
    This class provides methods for properly validating models on time series data,
    ensuring that temporal dependencies are respected and data leakage is prevented.
    It includes multiple validation strategies tailored to different types of medical
    time series data.
    """
    
    def __init__(self, 
                 time_column: str = 'timestamp',
                 patient_column: Optional[str] = 'patient_id',
                 gap_size: int = 0,
                 min_train_size: Optional[int] = None,
                 test_size: Optional[Union[int, float]] = None):
        """
        Initialize the time series validator.
        
        Args:
            time_column: Name of the column containing time information
            patient_column: Name of the column containing patient identifiers (if None, all data is treated as from a single patient)
            gap_size: Number of samples to exclude between train and test sets to prevent leakage
            min_train_size: Minimum number of samples in the training set
            test_size: Size of the test set (int for absolute number, float for proportion)
        """
        self.time_column = time_column
        self.patient_column = patient_column
        self.gap_size = gap_size
        self.min_train_size = min_train_size
        self.test_size = test_size
        
        logger.info(f"Initialized TimeSeriesValidator with time_column={time_column}, "
                   f"patient_column={patient_column}, gap_size={gap_size}")
    
    def rolling_window_split(self, 
                            data: pd.DataFrame, 
                            n_splits: int = 5, 
                            initial_window: Optional[int] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for rolling window cross-validation.
        
        This approach expands the training window over time while keeping the test window fixed,
        which is suitable for forecasting tasks where the model needs to learn from an expanding history.
        
        Args:
            data: The dataset to split
            n_splits: Number of train/test splits
            initial_window: Initial size of the training window (if None, calculated based on data size)
            
        Returns:
            Generator yielding (train_index, test_index) for each split
        """
        # Sort data by time
        if self.time_column in data.columns:
            data = data.sort_values(by=self.time_column).reset_index(drop=True)
        
        # Use sklearn's TimeSeriesSplit with modifications
        tscv = TimeSeriesSplit(n_splits=n_splits, 
                              gap=self.gap_size, 
                              max_train_size=None, 
                              test_size=self.test_size)
        
        for train_index, test_index in tscv.split(data):
            # Ensure minimum training size if specified
            if self.min_train_size is not None and len(train_index) < self.min_train_size:
                continue
                
            yield train_index, test_index
    
    def patient_aware_split(self, 
                           data: pd.DataFrame, 
                           n_splits: int = 5) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for patient-aware cross-validation.
        
        This approach ensures that data from the same patient stays together in either
        the training or testing set, which prevents patient-specific data leakage.
        
        Args:
            data: The dataset to split
            n_splits: Number of train/test splits
            
        Returns:
            Generator yielding (train_index, test_index) for each split
        """
        if self.patient_column not in data.columns:
            logger.warning(f"Patient column '{self.patient_column}' not found. Falling back to standard rolling window split.")
            yield from self.rolling_window_split(data, n_splits)
            return
        
        # Get unique patients and sort by their earliest timestamp
        patients = data[self.patient_column].unique()
        
        # For each patient, get their earliest timestamp
        if self.time_column in data.columns:
            patient_first_time = {}
            for patient in patients:
                patient_data = data[data[self.patient_column] == patient]
                patient_first_time[patient] = patient_data[self.time_column].min()
            
            # Sort patients by their earliest timestamp
            patients = sorted(patients, key=lambda p: patient_first_time[p])
        
        # Calculate patients per fold
        patients_per_fold = max(1, len(patients) // n_splits)
        
        # Create folds
        for i in range(n_splits):
            if i < n_splits - 1:
                test_patients = patients[i * patients_per_fold:(i + 1) * patients_per_fold]
                train_patients = [p for p in patients if p not in test_patients]
            else:
                # Last fold gets remaining patients
                test_patients = patients[i * patients_per_fold:]
                train_patients = [p for p in patients if p not in test_patients]
            
            # Create indices
            train_index = data[data[self.patient_column].isin(train_patients)].index.values
            test_index = data[data[self.patient_column].isin(test_patients)].index.values
            
            if len(train_index) > 0 and len(test_index) > 0:
                yield train_index, test_index
    
    def expanding_window_split(self, 
                              data: pd.DataFrame, 
                              n_splits: int = 5, 
                              test_window_size: Union[int, float] = 0.2) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for expanding window cross-validation.
        
        This approach gradually increases both training and test windows, which is
        useful for evaluating model performance over an expanding time horizon.
        
        Args:
            data: The dataset to split
            n_splits: Number of train/test splits
            test_window_size: Size of the test window (int for absolute size, float for proportion)
            
        Returns:
            Generator yielding (train_index, test_index) for each split
        """
        # Sort data by time
        if self.time_column in data.columns:
            data = data.sort_values(by=self.time_column).reset_index(drop=True)
        
        # Calculate test window size if it's a proportion
        if isinstance(test_window_size, float):
            test_window_size = int(len(data) * test_window_size)
        
        # Calculate initial training set size
        initial_train_size = max(self.min_train_size or 1, int(len(data) // (n_splits + 1)))
        
        # Calculate increment for expanding window
        data_for_splits = len(data) - initial_train_size
        increment = max(1, data_for_splits // n_splits)
        
        for i in range(n_splits):
            if i < n_splits - 1:
                train_end = initial_train_size + i * increment
                test_start = train_end + self.gap_size
                test_end = min(test_start + test_window_size, len(data))
            else:
                # Last split uses all remaining data
                train_end = len(data) - test_window_size - self.gap_size
                test_start = train_end + self.gap_size
                test_end = len(data)
            
            if test_start >= len(data) or train_end <= 0 or test_end <= test_start:
                continue
                
            train_index = np.arange(0, train_end)
            test_index = np.arange(test_start, test_end)
            
            yield train_index, test_index
    
    def get_validation_scores(self, 
                             data: pd.DataFrame, 
                             model: Any, 
                             features: List[str], 
                             target: str, 
                             method: str = 'rolling_window', 
                             n_splits: int = 5, 
                             scoring_functions: Dict[str, callable] = None) -> Dict[str, Any]:
        """
        Perform validation using the specified method and return performance scores.
        
        Args:
            data: The dataset to validate on
            model: The model to validate
            features: List of feature column names
            target: Name of the target column
            method: Validation method ('rolling_window', 'patient_aware', or 'expanding_window')
            n_splits: Number of validation splits
            scoring_functions: Dictionary mapping score names to scoring functions
            
        Returns:
            Dictionary containing validation scores
        """
        if scoring_functions is None:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            scoring_functions = {
                'mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                'r2': lambda y_true, y_pred: r2_score(y_true, y_pred)
            }
        
        # Select the appropriate validation method
        if method == 'rolling_window':
            split_method = self.rolling_window_split
        elif method == 'patient_aware':
            split_method = self.patient_aware_split
        elif method == 'expanding_window':
            split_method = self.expanding_window_split
        else:
            raise ValueError(f"Unknown validation method: {method}")
        
        scores = {name: [] for name in scoring_functions.keys()}
        fold_info = []
        
        # Perform validation
        for fold, (train_idx, test_idx) in enumerate(split_method(data, n_splits)):
            X_train = data.iloc[train_idx][features]
            y_train = data.iloc[train_idx][target]
            X_test = data.iloc[test_idx][features]
            y_test = data.iloc[test_idx][target]
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate scores
                fold_scores = {}
                for name, scoring_func in scoring_functions.items():
                    score = scoring_func(y_test, y_pred)
                    scores[name].append(score)
                    fold_scores[name] = score
                
                # Record fold information
                fold_info.append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_start': data.iloc[train_idx[0]].get(self.time_column, train_idx[0]),
                    'train_end': data.iloc[train_idx[-1]].get(self.time_column, train_idx[-1]),
                    'test_start': data.iloc[test_idx[0]].get(self.time_column, test_idx[0]),
                    'test_end': data.iloc[test_idx[-1]].get(self.time_column, test_idx[-1]),
                    'scores': fold_scores
                })
                
                logger.info(f"Fold {fold+1}/{n_splits} - "
                           f"Train: {len(train_idx)}, Test: {len(test_idx)}, "
                           f"RMSE: {fold_scores.get('rmse', 'N/A'):.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold+1}/{n_splits}: {str(e)}")
                continue
        
        # Calculate aggregate scores
        result = {
            'method': method,
            'n_splits': n_splits,
            'gap_size': self.gap_size,
            'folds': fold_info,
            'scores': {}
        }
        
        for name in scoring_functions.keys():
            if scores[name]:
                result['scores'][name] = {
                    'mean': np.mean(scores[name]),
                    'std': np.std(scores[name]),
                    'min': np.min(scores[name]),
                    'max': np.max(scores[name]),
                    'values': scores[name]
                }
        
        return result

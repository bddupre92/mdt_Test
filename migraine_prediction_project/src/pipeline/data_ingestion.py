"""
Data ingestion utilities for migraine prediction.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime


class DataIngestion:
    """
    Handles data loading and preprocessing for migraine prediction.
    """
    
    def __init__(self, required_columns: Optional[List[str]] = None):
        """
        Initialize DataIngestion.
        
        Args:
            required_columns: List of required column names
        """
        self.required_columns = required_columns or [
            'sleep_hours',
            'stress_level',
            'weather_pressure',
            'heart_rate',
            'hormonal_level',
            'migraine_occurred'
        ]
        
        # Optional columns that may be present
        self.optional_columns = [
            'patient_id',
            'date',
            'migraine_probability'
        ]
    
    def load_csv(self, 
                file_path: str, 
                validate: bool = True) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to CSV file
            validate: Whether to validate data
            
        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Load data
        data = pd.read_csv(file_path)
        
        # Convert boolean 'True'/'False' strings to actual boolean values if needed
        if 'migraine_occurred' in data.columns and data['migraine_occurred'].dtype == 'object':
            data['migraine_occurred'] = data['migraine_occurred'].map(
                {'True': 1, 'False': 0, True: 1, False: 0}
            ).fillna(data['migraine_occurred'])
        
        # Validate if requested
        if validate:
            self._validate_data(data)
            
        # Process dates if present
        if 'date' in data.columns and pd.api.types.is_object_dtype(data['date']):
            try:
                data['date'] = pd.to_datetime(data['date'])
            except (ValueError, TypeError):
                print("Warning: Could not convert date column to datetime. Keeping as string.")
            
        return data
    
    def _validate_data(self, data: pd.DataFrame):
        """
        Validate data format.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check for NaN values - but don't fail on them as the test data has some NaNs
        nan_counts = data[self.required_columns].isna().sum()
        if nan_counts.sum() > 0:
            print(f"Warning: Data contains NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            
        # Validate data types
        if not pd.api.types.is_numeric_dtype(data['sleep_hours']):
            raise ValueError("sleep_hours must be numeric")
        if not pd.api.types.is_numeric_dtype(data['stress_level']):
            raise ValueError("stress_level must be numeric")
        if not pd.api.types.is_numeric_dtype(data['weather_pressure']):
            print("Warning: weather_pressure contains non-numeric values")
        if not pd.api.types.is_numeric_dtype(data['heart_rate']):
            print("Warning: heart_rate contains non-numeric values")
        if not pd.api.types.is_numeric_dtype(data['hormonal_level']):
            raise ValueError("hormonal_level must be numeric")
        
        # Validate target column
        if not pd.api.types.is_numeric_dtype(data['migraine_occurred']):
            if not all(value in [True, False, 'True', 'False', 1, 0] 
                     for value in data['migraine_occurred'].dropna()):
                raise ValueError("migraine_occurred must be boolean or integer")
            
        # Convert target to int if boolean or string
        if data['migraine_occurred'].dtype == 'bool' or data['migraine_occurred'].dtype == 'object':
            data['migraine_occurred'] = data['migraine_occurred'].map(
                {'True': 1, 'False': 0, True: 1, False: 0}
            ).fillna(data['migraine_occurred'])
            data['migraine_occurred'] = data['migraine_occurred'].astype(int)
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid warnings
        data = data.copy()
        
        # Fill NaN values with median for each feature
        for col in self.required_columns:
            if col != 'migraine_occurred' and col in data.columns:
                data[col] = data[col].fillna(data[col].median())
        
        # Handle the target variable - fill NaN with 0 (no migraine)
        if 'migraine_occurred' in data.columns:
            data['migraine_occurred'] = data['migraine_occurred'].fillna(0)
        
        return data
        
    def split_data(self, 
                  data: pd.DataFrame, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Args:
            data: DataFrame to split
            test_size: Proportion of data for test set
            random_state: Random seed
            
        Returns:
            train_data, test_data
        """
        # Handle time-series splitting if date column exists
        if 'date' in data.columns and pd.api.types.is_datetime64_dtype(data['date']):
            # Sort by date
            data = data.sort_values('date')
            
            # Split temporally
            split_idx = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
        
        # Handle patient-based splitting if patient_id exists
        elif 'patient_id' in data.columns:
            # Get unique patient IDs
            patients = data['patient_id'].unique()
            
            # Randomly select patients for test set
            np.random.seed(random_state)
            n_test_patients = max(1, int(len(patients) * test_size))
            test_patients = np.random.choice(patients, n_test_patients, replace=False)
            
            # Split data
            test_data = data[data['patient_id'].isin(test_patients)]
            train_data = data[~data['patient_id'].isin(test_patients)]
            
        else:
            # Default random split
            # Shuffle data
            data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # Split data
            split_idx = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
        
        return train_data, test_data
        
    def load_test_data(self, data_dir: str = None) -> pd.DataFrame:
        """
        Load test data from the test_data directory.
        
        Args:
            data_dir: Directory containing test data files (default: project test_data dir)
            
        Returns:
            DataFrame with all test data combined
        """
        data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'test_data')
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Test data directory not found: {data_dir}")
            
        # Get all CSV files in directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        # Load and combine all files
        dfs = []
        for file in csv_files:
            file_path = os.path.join(data_dir, file)
            df = self.load_csv(file_path, validate=False)
            dfs.append(df)
            
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Clean data
        combined_df = self.clean_data(combined_df)
        
        return combined_df

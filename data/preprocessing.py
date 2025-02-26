"""
preprocessing.py
--------------
Data preprocessing utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(data: pd.DataFrame, strategy_numeric='mean', scale_method='minmax', exclude_cols=None) -> pd.DataFrame:
    """
    Preprocess data for model training/prediction
    
    Args:
        data: Raw data
        strategy_numeric: Strategy for numeric imputation ('mean', 'median', etc.)
        scale_method: Method for scaling ('minmax' or 'standard')
        exclude_cols: List of columns to exclude from scaling
        
    Returns:
        Preprocessed data
    """
    df = data.copy()
    
    if exclude_cols is None:
        exclude_cols = ['migraine_occurred', 'severity', 'patient_id', 'migraine_probability', 'date']
    
    # Handle timestamps
    timestamp_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in timestamp_cols:
        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df = df.drop(columns=[col])
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if numeric_cols:
        imputer = SimpleImputer(strategy=strategy_numeric)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Handle categorical variables
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col not in exclude_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Scale numeric features
    if scale_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        if scale_method == 'minmax':
            df[numeric_cols] = np.clip(df[numeric_cols], 0, 1)
    
    return df

def encode_features(data: pd.DataFrame, encoders: dict = None) -> tuple:
    """
    Encode features using provided or new encoders
    
    Args:
        data: Data to encode
        encoders: Optional dict of existing encoders
        
    Returns:
        (encoded_data, encoders)
    """
    df = data.copy()
    if encoders is None:
        encoders = {}
    
    # Handle timestamps
    timestamp_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in timestamp_cols:
        df[f"{col}_hour"] = df[col].dt.hour.astype(float)
        df[f"{col}_day"] = df[col].dt.day.astype(float)
        df[f"{col}_month"] = df[col].dt.month.astype(float)
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek.astype(float)
        df = df.drop(columns=[col])
    
    # Handle categorical variables
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col not in encoders:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))
        else:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    # Handle numeric variables
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if 'scaler' not in encoders:
        encoders['scaler'] = StandardScaler()
        df[num_cols] = encoders['scaler'].fit_transform(df[num_cols])
    else:
        df[num_cols] = encoders['scaler'].transform(df[num_cols])
    
    return df, encoders

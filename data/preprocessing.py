"""
preprocessing.py
----------------
Contains helper functions for data cleaning, imputation,
and transformations.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def impute_missing(df, strategy_numeric='mean'):
    """
    Imputes missing values in numeric columns of df using the specified strategy.
    Categorical columns could be handled separately if present.
    
    :param df: pandas DataFrame
    :param strategy_numeric: 'mean', 'median', etc.
    :return: DataFrame with missing values imputed
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy=strategy_numeric)
    
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def scale_data(df, method='minmax', exclude_cols=None):
    """
    Scales numeric columns with either Min-Max or Standard Scaler,
    excluding certain columns if needed (e.g., target columns).
    
    :param df: DataFrame to scale
    :param method: 'minmax' or 'standard'
    :param exclude_cols: list of columns to exclude from scaling
    :return: DataFrame with scaled columns
    """
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess_data(df, strategy_numeric='mean', scale_method='minmax', exclude_cols=None):
    """
    High-level preprocessing: imputes missing numeric, then scales numeric data.
    """
    df = impute_missing(df, strategy_numeric=strategy_numeric)
    df = scale_data(df, method=scale_method, exclude_cols=exclude_cols)
    return df

"""
statistical.py
-------------
Statistical tests for drift detection
"""

import numpy as np
from scipy import stats
import pandas as pd

def chi2_drift_test(old_data, new_data, feature=None, bins=10, alpha=0.05):
    """
    Compare distributions using Chi-square test
    
    Args:
        old_data: Reference dataset
        new_data: Current dataset
        feature: Feature to test (optional)
        bins: Number of bins for continuous data
        alpha: Significance level
        
    Returns:
        bool: True if drift detected
    """
    # Convert to numpy arrays if needed
    if isinstance(old_data, pd.DataFrame) and feature is not None:
        old_data = old_data[feature].values
    if isinstance(new_data, pd.DataFrame) and feature is not None:
        new_data = new_data[feature].values
        
    old_data = np.asarray(old_data)
    new_data = np.asarray(new_data)
    
    # Remove NaN values
    if np.issubdtype(old_data.dtype, np.number):
        old_data = old_data[~np.isnan(old_data)]
        new_data = new_data[~np.isnan(new_data)]
    else:
        # For non-numeric data, filter out None values
        old_data = old_data[old_data != None]
        new_data = new_data[new_data != None]
    
    # Check if data is categorical or continuous
    if np.issubdtype(old_data.dtype, np.number) and bins > 0:
        # For continuous data, bin it first
        # Determine bin edges based on combined data
        all_data = np.concatenate([old_data, new_data])
        bin_edges = np.linspace(np.min(all_data), np.max(all_data), bins + 1)
        
        # Count occurrences in each bin
        old_counts, _ = np.histogram(old_data, bins=bin_edges)
        new_counts, _ = np.histogram(new_data, bins=bin_edges)
        
        # Create contingency table
        contingency = np.vstack([old_counts, new_counts])
    else:
        # For categorical data, create contingency table
        # Convert to pandas Series if needed
        if not hasattr(old_data, 'value_counts'):
            old_data = pd.Series(old_data)
        if not hasattr(new_data, 'value_counts'):
            new_data = pd.Series(new_data)
            
        # Get value counts
        old_counts = old_data.value_counts()
        new_counts = new_data.value_counts()
        
        # Get all unique categories
        all_categories = list(set(old_counts.index) | set(new_counts.index))
        
        # Create contingency table with zeros for missing categories
        contingency = np.zeros((2, len(all_categories)))
        
        for i, cat in enumerate(all_categories):
            contingency[0, i] = old_counts.get(cat, 0)
            contingency[1, i] = new_counts.get(cat, 0)
    
    # Check minimum cell count
    if np.any(contingency < 5):
        # Use Fisher's exact test for small counts
        _, p_value = stats.fisher_exact(contingency)
    else:
        # Use Chi-square test
        _, p_value, _, _ = stats.chi2_contingency(contingency)
        
    return p_value < alpha  # Return True if drift detected

def ks_drift_test(old_data, new_data, feature=None, alpha=0.05):
    """
    Compare continuous distributions using KS test
    
    Args:
        old_data: Reference dataset
        new_data: Current dataset
        feature: Feature to test (optional)
        alpha: Significance level
        
    Returns:
        bool: True if drift detected
    """
    # Convert to numpy arrays if needed
    if isinstance(old_data, pd.DataFrame) and feature is not None:
        old_data = old_data[feature].values
    if isinstance(new_data, pd.DataFrame) and feature is not None:
        new_data = new_data[feature].values
    
    old_data = np.asarray(old_data)
    new_data = np.asarray(new_data)
    
    # Remove NaN values
    old_data = old_data[~np.isnan(old_data)]
    new_data = new_data[~np.isnan(new_data)]
    
    # Perform KS test
    statistic, p_value = stats.ks_2samp(old_data, new_data)
    
    return p_value < alpha  # Return True if drift detected

def detect_drift(old_data, new_data, feature=None, method='auto', alpha=0.05):
    """
    Detect drift using appropriate statistical test
    
    Args:
        old_data: Reference dataset
        new_data: Current dataset
        feature: Feature to test
        method: Test method ('auto', 'chi2', or 'ks')
        alpha: Significance level
        
    Returns:
        bool: True if drift detected
    """
    if method == 'auto':
        # Check if data is categorical
        if isinstance(old_data, pd.Series) and old_data.dtype == 'category':
            method = 'chi2'
        elif isinstance(old_data, pd.DataFrame) and feature and old_data[feature].dtype == 'category':
            method = 'chi2'
        else:
            method = 'ks'
    
    if method == 'chi2':
        return chi2_drift_test(old_data, new_data, feature, alpha=alpha)
    elif method == 'ks':
        return ks_drift_test(old_data, new_data, feature, alpha)
    else:
        raise ValueError(f"Unknown drift detection method: {method}")

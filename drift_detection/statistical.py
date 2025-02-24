"""
statistical.py
--------------
Implements statistical drift detection using Kolmogorov-Smirnov (KS)
and Chi-square tests.
"""

import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

def ks_drift_test(old_data, new_data, feature):
    """
    Compare distributions of 'feature' in old_data vs new_data 
    using KS test. Return p-value. 
    Lower p-value => more likely drift.
    """
    old_values = old_data[feature].dropna()
    new_values = new_data[feature].dropna()
    if len(old_values) < 2 or len(new_values) < 2:
        return 1.0  # not enough data to conclude drift
    stat, p = ks_2samp(old_values, new_values)
    return p

def chi2_drift_test(old_data, new_data, feature):
    """
    Compares categorical distributions with Chi-square test.
    Return p-value. 
    """
    old_counts = old_data[feature].value_counts()
    new_counts = new_data[feature].value_counts()
    
    # Combine categories
    all_cats = list(set(old_counts.index).union(set(new_counts.index)))
    old_arr = np.array([old_counts.get(cat, 0) for cat in all_cats])
    new_arr = np.array([new_counts.get(cat, 0) for cat in all_cats])
    
    if old_arr.sum() < 1 or new_arr.sum() < 1:
        return 1.0
    
    obs = np.stack([old_arr, new_arr], axis=0)
    chi2, p, dof, expected = chi2_contingency(obs)
    return p

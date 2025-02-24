"""
domain_knowledge.py
-------------------
Functions to incorporate domain-based rules or feature engineering
in migraine data (e.g. identifying 'poor sleep' flags, etc.).
"""

import pandas as pd

def add_migraine_features(df):
    """
    Adds domain-inspired features. For example:
    - poor_sleep if sleep_hours < 5
    - big_pressure_drop if day's pressure is significantly lower than previous day's
    - high_stress if stress_level > 7
    etc.
    
    :param df: DataFrame with relevant columns
    :return: modified DataFrame with new features
    """
    df = df.copy()
    
    df['poor_sleep'] = (df['sleep_hours'] < 5).astype(int)
    
    # Pressure difference from previous day
    df['pressure_diff'] = df['weather_pressure'].diff().fillna(0)
    df['big_pressure_drop'] = (df['pressure_diff'] < -3).astype(int)
    
    df['high_stress'] = (df['stress_level'] > 7).astype(int)
    
    # Potential combined trigger count
    df['trigger_count'] = df['poor_sleep'] + df['big_pressure_drop'] + df['high_stress']
    
    return df

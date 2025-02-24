"""
generate_synthetic.py
---------------------
Generates a synthetic dataset simulating migraine events
alongside triggers like sleep hours, weather, etc.
"""

import numpy as np
import pandas as pd

def generate_synthetic_data(num_days=180, random_seed=42):
    """
    Generate a synthetic dataset for migraine prediction.
    
    :param num_days: Number of days of data to generate
    :param random_seed: Random seed for reproducibility
    :return: A pandas DataFrame indexed by date, with columns:
             ['sleep_hours', 'weather_pressure', 'weather_temp', 'heart_rate',
              'stress_level', 'migraine_occurred', 'severity', ...]
    """
    rng = np.random.default_rng(random_seed)
    
    dates = pd.date_range(start="2025-01-01", periods=num_days, freq='D')
    df = pd.DataFrame(index=dates)
    
    # Generate base features
    df['sleep_hours'] = rng.normal(loc=7.0, scale=1.5, size=num_days).clip(0, 12)
    df['weather_pressure'] = 1013 + 5*np.sin(np.linspace(0, 6*np.pi, num_days)) 
    df['weather_pressure'] += rng.normal(0, 2, size=num_days)
    df['weather_temp'] = 15 + 10*np.sin(np.linspace(0, 2*np.pi, num_days))
    df['weather_temp'] += rng.normal(0, 3, size=num_days)
    df['heart_rate'] = rng.normal(70, 5, size=num_days).clip(40, 120)
    df['stress_level'] = rng.normal(5, 2, size=num_days).clip(0, 10)
    
    # Probability of migraine influenced by sleep & stress & weather changes
    prob = 0.05 + 0.02*(7 - df['sleep_hours']).clip(0) + 0.01*(df['stress_level'] - 5).clip(0) 
    # if big pressure drop from previous day
    pressure_drop = df['weather_pressure'].diff().fillna(0).abs()
    prob += 0.01 * (pressure_drop > 3)
    
    # Convert prob to [0,1]
    prob = np.clip(prob, 0, 0.9)
    
    df['migraine_occurred'] = (rng.random(num_days) < prob).astype(int)
    
    # For severity, if migraine occurred, random scale 1-10
    severity = np.zeros(num_days, dtype=int)
    severity[df['migraine_occurred'] == 1] = rng.integers(1, 10, size=(df['migraine_occurred'] == 1).sum())
    df['severity'] = severity
    
    # Introduce missing values (5% random)
    for col in df.columns:
        mask = rng.random(num_days) < 0.05
        df.loc[mask, col] = np.nan
    
    return df

def main():
    # Quick test
    data = generate_synthetic_data(180)
    print("Sample synthetic data:\n", data.head(10))

if __name__ == "__main__":
    main()

"""
Generate sample drift detection data.
"""
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import os

def generate_drift_data(days=30, samples_per_day=48):
    """Generate sample drift detection data."""
    np.random.seed(42)
    
    # Generate timestamps
    end_date = datetime.utcnow()
    timestamps = [(end_date - timedelta(days=i)).timestamp() for i in range(days)]
    timestamps = sorted(timestamps)
    
    # Generate severity scores with some drift points
    severities = []
    drift_points = []
    
    for i in range(len(timestamps)):
        # Add some drift points
        if i in [5, 12, 20]:
            severity = np.random.uniform(1.9, 2.5)  # Above threshold 1.8
            drift_points.append(i)
        else:
            severity = np.random.uniform(0.5, 1.7)
        severities.append(severity)
    
    # Generate feature-specific drift counts
    features = ['age', 'blood_pressure', 'heart_rate', 'temperature', 'oxygen_level']
    feature_drifts = {
        feature: np.random.randint(1, 10) for feature in features
    }
    
    data = {
        "timestamps": timestamps,
        "severities": severities,
        "drift_points": drift_points,
        "feature_drifts": feature_drifts,
        "window_size": 50,
        "drift_threshold": 1.8,
        "min_interval": 40
    }
    
    # Save to file
    os.makedirs('results/drift_detection', exist_ok=True)
    with open('results/drift_detection/data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Drift detection data generated successfully!")

if __name__ == "__main__":
    generate_drift_data()

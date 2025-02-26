"""
test_full_pipeline.py
--------------------
Full pipeline test including meta-optimization, model selection, and prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from meta.meta_learner import MetaLearner
from optimizers.aco import AntColonyOptimizer
from drift_detection.detector import DriftDetector
from models.base_model import BaseModel
from models.model_factory import ModelFactory

def generate_synthetic_migraine_data(n_samples=1000):
    """Generate synthetic migraine data with concept drift"""
    np.random.seed(42)
    
    # First half: normal conditions
    n_half = n_samples // 2
    weather_temp = np.random.normal(20, 5, n_half)
    weather_pressure = np.random.normal(1013, 5, n_half)
    stress_level = np.random.uniform(0, 10, n_half)
    sleep_hours = np.random.normal(7, 1, n_half)
    screen_time = np.random.normal(6, 2, n_half)
    
    # Second half: changed conditions (concept drift)
    weather_temp_2 = np.random.normal(25, 5, n_samples - n_half)  # Higher temps
    weather_pressure_2 = np.random.normal(1018, 5, n_samples - n_half)  # Higher pressure
    stress_level_2 = np.random.uniform(2, 12, n_samples - n_half)  # Higher stress
    sleep_hours_2 = np.random.normal(6, 1, n_samples - n_half)  # Less sleep
    screen_time_2 = np.random.normal(8, 2, n_samples - n_half)  # More screen time
    
    # Combine data
    weather_temp = np.concatenate([weather_temp, weather_temp_2])
    weather_pressure = np.concatenate([weather_pressure, weather_pressure_2])
    stress_level = np.concatenate([stress_level, stress_level_2])
    sleep_hours = np.concatenate([sleep_hours, sleep_hours_2])
    screen_time = np.concatenate([screen_time, screen_time_2])
    
    # Create migraine probability function
    def get_migraine_prob(temp, pressure, stress, sleep, screen):
        return (
            0.3 * (temp - 20) / 5 +      # Higher temp increases probability
            0.2 * (pressure - 1013) / 5 + # Pressure changes matter
            0.25 * stress / 10 +          # Higher stress increases probability
            -0.15 * (sleep - 7) +         # Less sleep increases probability
            0.1 * screen / 6              # More screen time increases probability
        )
    
    # Calculate probabilities
    migraine_prob = get_migraine_prob(
        weather_temp, weather_pressure, stress_level, sleep_hours, screen_time
    )
    
    # Normalize to [0,1] range
    migraine_prob = (migraine_prob - migraine_prob.min()) / (migraine_prob.max() - migraine_prob.min())
    
    # Generate labels (1 for migraine, 0 for no migraine)
    labels = (migraine_prob > 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'temperature': weather_temp,
        'pressure': weather_pressure,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'screen_time': screen_time,
        'target': labels
    })
    
    return data

def plot_performance(history, predictions=None, drift_points=None):
    """Plot training history and drift detection results"""
    plt.figure(figsize=(12, 6))
    
    # Plot training metrics if available
    if history and hasattr(history, 'history'):
        plt.subplot(2, 1, 1)
        for metric in history.history:
            plt.plot(history.history[metric], label=metric)
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
    
    # Plot predictions and drift points if available
    if predictions is not None:
        plt.subplot(2, 1, 2)
        plt.plot(predictions, label='Prediction Probability')
        if drift_points:
            for point in drift_points:
                plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
        plt.title('Model Predictions with Drift Points')
        plt.xlabel('Sample')
        plt.ylabel('Probability')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('pipeline_performance.png')
    plt.close()

def main():
    """Run full pipeline test"""
    print("Starting full pipeline test...")
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_migraine_data(n_samples=1000)
    print("Data shape:", data.shape)
    print("\nFeature statistics:")
    print(data.describe())
    
    # 2. Split data and scale features
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Initialize components
    print("\n2. Initializing components...")
    meta_learner = MetaLearner()
    optimizer = AntColonyOptimizer(dim=2, bounds=[(0, 1), (0, 1)])
    meta_learner.set_algorithms([optimizer])
    
    drift_detector = DriftDetector(window_size=50)
    model_factory = ModelFactory()
    
    # 4. Meta-optimization
    print("\n3. Running meta-optimization...")
    context = {'phase': 1}  # Initial context for meta-learner
    meta_learner.optimize(X_train_scaled, y_train, context=context)
    best_model_config = meta_learner.get_best_configuration()
    print("\nBest model configuration:")
    print(best_model_config)
    
    # 5. Train model with best configuration
    print("\n4. Training model with best configuration...")
    model = model_factory.create_model(best_model_config)
    history = model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate performance
    print("\n5. Evaluating performance...")
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # 7. Test drift detection on prediction probabilities
    print("\n6. Testing drift detection...")
    test_probs = model.predict_proba(X_test_scaled)[:, 1]  # Get positive class probabilities
    
    drift_detector.set_reference_window(test_probs[:50])
    drift_points = []
    n_drifts = 0
    
    for i in range(50, len(test_probs)):
        drift_detector.add_sample(float(test_probs[i]))  # Convert to float
        is_drift, severity = drift_detector.detect_drift()
        if is_drift:
            n_drifts += 1
            drift_points.append(i)
            print(f"Drift detected at sample {i} with severity {severity:.2f}")
    
    print(f"Total drifts detected: {n_drifts}")
    
    # 8. Plot results
    print("\n7. Plotting results...")
    plot_performance(history, test_probs, drift_points)
    print("Performance plot saved as 'pipeline_performance.png'")

if __name__ == "__main__":
    main()

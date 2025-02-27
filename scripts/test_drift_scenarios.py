"""
test_drift_scenarios.py
----------------------
Script to test drift detection with various realistic scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
from drift_detection.detector import DriftDetector

def generate_data_with_drifts(n_samples=1000, noise_level=0.2):
    """Generate data with multiple types of drifts"""
    t = np.linspace(0, 10, n_samples)
    
    # Base signal
    signal = np.zeros(n_samples)
    
    # Add sudden drift at t=3
    signal[t > 3] += 2.0
    
    # Add gradual drift from t=5 to t=7
    mask = (t >= 5) & (t <= 7)
    signal[mask] += (t[mask] - 5) * 1.5
    
    # Add seasonal component
    signal += 0.5 * np.sin(2 * np.pi * t)
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    data = signal + noise
    
    return t, data, signal

def plot_results(t, data, signal, drift_points, severities, trends):
    """Plot the data and drift detection results"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Data and detected drifts
    plt.subplot(3, 1, 1)
    plt.plot(t, data, 'b-', alpha=0.5, label='Data with noise')
    plt.plot(t, signal, 'k--', label='True signal')
    
    # Plot drift points
    for idx, severity in zip(drift_points, severities):
        plt.axvline(x=t[idx], color='r', linestyle='--', alpha=0.5)
        plt.text(t[idx], plt.ylim()[1], f's={severity:.2f}', 
                rotation=90, verticalalignment='top')
    
    plt.title('Data with Detected Drift Points')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Drift Severities
    plt.subplot(3, 1, 2)
    plt.plot(t[50:], severities, 'r-', label='Drift Severity')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    plt.title('Drift Severity Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Trends
    plt.subplot(3, 1, 3)
    plt.plot(t[50:], trends, 'g-', label='Trend')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title('Trend Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('drift_detection_results.png')
    plt.close()

def main():
    # Generate data
    np.random.seed(42)
    t, data, signal = generate_data_with_drifts(n_samples=1000, noise_level=0.2)
    
    # Initialize detector
    detector = DriftDetector(window_size=50)
    
    # Initialize reference window
    detector.set_reference_window(data[:50])
    
    # Track drift detection results
    drift_points = []
    severities = []
    trends = []
    
    # Process data
    print("Processing data...")
    for i in range(50, len(data)):
        detector.add_sample(data[i])
        is_drift, severity, info = detector.detect_drift()
        trend = info.get('trend', 0.0)
        
        severities.append(severity)
        trends.append(trend)
        
        if is_drift:
            print(f"Drift detected at t={t[i]:.2f} with severity {severity:.2f}")
            drift_points.append(i)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(t, data, signal, drift_points, severities, trends)
    print("Results saved to 'drift_detection_results.png'")
    
    # Print summary
    print("\nSummary:")
    print(f"Number of drifts detected: {len(drift_points)}")
    print(f"Average severity: {np.mean(severities):.2f}")
    print(f"Average trend: {np.mean(trends):.2f}")

if __name__ == "__main__":
    main()

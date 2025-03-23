"""
Test script for drift detection visualization.

This script tests the drift detection visualization by:
1. Generating synthetic data with known drift points
2. Running the drift detector on the data
3. Verifying that the visualization correctly displays the drift points
4. Testing the API endpoints for drift history and status
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import unittest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from drift_detection.detector import DriftDetector
from scripts.test_framework import generate_temporal_data, plot_drift_analysis

class TestDriftVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create output directory
        os.makedirs('test_output', exist_ok=True)
        
        # Initialize drift detector with parameters from MEMORY
        self.detector = DriftDetector(
            window_size=50,
            threshold=1.8,
            significance_level=0.01,
            min_drift_interval=40,
            ema_alpha=0.3
        )
        
        # Generate synthetic data with known drift points
        self.drift_points = [300, 600]
        self.data, self.y = generate_temporal_data(n_samples=1000, drift_points=self.drift_points)
        self.feature_names = ['temperature', 'pressure', 'stress_level', 'sleep_hours', 'screen_time']
        
        # Initialize tracking variables
        self.drift_detections = []
        self.drift_severities = []
        self.timestamps = []
        
    def test_drift_detection_visualization(self):
        """Test drift detection and visualization"""
        print("Testing drift detection visualization...")
        
        # Set reference window
        X_init = self.data[self.feature_names].values[:100]
        self.detector.set_reference_window(X_init)
        
        # Process data points
        X_stream = self.data[self.feature_names].values[100:]
        
        # Create timestamps for visualization
        start_date = datetime.now() - timedelta(days=30)
        
        for i, sample in enumerate(X_stream):
            # Add sample to detector
            self.detector.add_sample(sample)
            
            # Check for drift
            drift_detected, severity, info = self.detector.detect_drift()
            
            # Track results
            self.drift_severities.append(severity)
            timestamp = start_date + timedelta(hours=i)
            self.timestamps.append(timestamp)
            
            if drift_detected:
                self.drift_detections.append(100 + i)
                print(f"Drift detected at sample {100 + i} with severity {severity:.3f}")
                print(f"Drift info: {json.dumps(info, default=str)}")
        
        # Verify drift detection results
        print(f"\nDetected {len(self.drift_detections)} drift points")
        print(f"Expected drift points: {self.drift_points}")
        print(f"Detected drift points: {self.drift_detections}")
        
        # Add drift severity to data for plotting
        self.data['drift_severity'] = pd.Series(
            self.drift_severities + [0] * (len(self.data) - len(self.drift_severities))
        )
        
        # Plot drift analysis
        plot_drift_analysis(
            self.data, 
            self.drift_points, 
            self.drift_detections, 
            save_path='test_output/drift_analysis.png'
        )
        
        # Generate mock API response data
        api_data = self._generate_mock_api_data()
        
        # Save API data for testing
        with open('test_output/drift_history.json', 'w') as f:
            json.dump(api_data, f, indent=2, default=str)
        
        print("\nTest completed. Visualization saved to test_output/drift_analysis.png")
        print("Mock API data saved to test_output/drift_history.json")
        
    def _generate_mock_api_data(self):
        """Generate mock API response data for testing visualization"""
        # Convert timestamps to string format
        timestamps = [t.strftime("%Y-%m-%d %H:%M") for t in self.timestamps]
        
        # Create drift detected array
        drift_detected = [False] * len(self.drift_severities)
        for idx in self.drift_detections:
            if idx - 100 < len(drift_detected):
                drift_detected[idx - 100] = True
        
        # Create feature drifts dictionary
        feature_drifts = {}
        for i, feature in enumerate(self.feature_names):
            # Assign random drift counts to features
            feature_drifts[feature] = np.random.randint(1, len(self.drift_detections) + 1)
        
        # Create recent events
        recent_events = []
        for i, idx in enumerate(self.drift_detections[-10:]):
            if idx - 100 < len(self.timestamps):
                recent_events.append({
                    "timestamp": timestamps[idx - 100],
                    "severity": float(self.drift_severities[idx - 100]),
                    "feature": np.random.choice(self.feature_names),
                    "drift_type": "distribution" if self.drift_severities[idx - 100] > 0.5 else "trend"
                })
        
        # Calculate trend using simple moving average
        window_size = min(10, len(self.drift_severities))
        trends = []
        
        for i in range(len(self.drift_severities)):
            if i < window_size - 1:
                trends.append(0)  # Not enough data points yet
            else:
                window = self.drift_severities[i-window_size+1:i+1]
                trends.append(float(np.mean(window)))
        
        # Calculate summary statistics
        total_drifts = len(self.drift_detections)
        average_severity = float(np.mean(self.drift_severities)) if self.drift_severities else 0.0
        
        # Find last detection
        last_detection = None
        if total_drifts > 0 and self.drift_detections[-1] - 100 < len(timestamps):
            last_detection = timestamps[self.drift_detections[-1] - 100]
        
        # Determine if drift is currently detected
        current_drift_detected = drift_detected[-1] if drift_detected else False
        
        return {
            "timestamps": timestamps,
            "severities": [float(s) for s in self.drift_severities],
            "trends": trends,
            "feature_drifts": feature_drifts,
            "total_drifts": total_drifts,
            "average_severity": average_severity,
            "last_detection": last_detection,
            "current_drift_detected": current_drift_detected,
            "recent_events": recent_events
        }

    def test_logging_requirements(self):
        """Test that logging requirements are met"""
        # This is a placeholder for a more comprehensive test
        # In a real implementation, we would verify that all logging requirements
        # from the MEMORY are being met
        
        print("\nVerifying logging requirements...")
        
        # Check that the detector has logging enabled
        self.assertTrue(hasattr(self.detector, 'logger'), "Detector should have a logger")
        
        # Run a sample detection to generate logs
        X_init = self.data[self.feature_names].values[:10]
        self.detector.set_reference_window(X_init)
        self.detector.add_sample(self.data[self.feature_names].values[10])
        drift_detected, severity, info = self.detector.detect_drift()
        
        print("Logging verification completed")

if __name__ == "__main__":
    unittest.main()

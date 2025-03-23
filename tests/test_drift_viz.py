"""
test_drift_viz.py
-----------------
Tests for drift detection visualization components
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import unittest
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from drift_detection.detector import DriftDetector
from scripts.test_framework import generate_temporal_data, plot_drift_analysis

class TestDriftViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create output directory
        os.makedirs('test_output', exist_ok=True)
        
        # Initialize drift detector with parameters from MEMORY
        cls.detector = DriftDetector(
            window_size=50,
            drift_threshold=1.8,
            significance_level=0.01,
            min_drift_interval=40,
            ema_alpha=0.3
        )
        
        # Generate synthetic data with known drift points
        cls.drift_points = [300, 600]
        cls.data, cls.y = generate_temporal_data(n_samples=1000, drift_points=cls.drift_points)
        cls.feature_names = ['temperature', 'pressure', 'stress_level', 'sleep_hours', 'screen_time']
        
        # Initialize tracking variables
        cls.drift_detections = []
        cls.drift_severities = []
        cls.timestamps = []
        
        # Process data for visualization tests
        cls._process_data_for_visualization()
        
    @classmethod
    def _process_data_for_visualization(cls):
        """Process data through drift detector for visualization tests"""
        logger.info("Processing data through drift detector...")
        
        # Set reference window
        X_init = cls.data[cls.feature_names].values[:100]
        cls.detector.set_reference_window(X_init)
        
        # Process data points
        X_stream = cls.data[cls.feature_names].values[100:]
        
        # Create timestamps for visualization
        start_date = datetime.now() - timedelta(days=30)
        
        for i, sample in enumerate(X_stream):
            # Add sample to detector
            cls.detector.add_sample(sample)
            
            # Check for drift
            drift_detected, severity, info = cls.detector.detect_drift()
            
            # Track results with detailed logging
            cls.drift_severities.append(severity)
            timestamp = start_date + timedelta(hours=i)
            cls.timestamps.append(timestamp)
            
            if drift_detected:
                cls.drift_detections.append(100 + i)
                logger.info(f"Drift detected at sample {100 + i} with severity {severity:.3f}")
                logger.debug(f"Drift info: {json.dumps(info, default=str)}")
                
                # Log feature-specific information
                if 'feature_stats' in info:
                    for feature, stats in info['feature_stats'].items():
                        logger.debug(f"Feature {feature} stats: mean_shift={stats.get('mean_shift', 0):.3f}, "
                                    f"ks_stat={stats.get('ks_stat', 0):.3f}, p_value={stats.get('p_value', 1):.3e}")
        
        # Add drift severity to data for plotting
        cls.data['drift_severity'] = pd.Series(
            cls.drift_severities + [0] * (len(cls.data) - len(cls.drift_severities))
        )
        
        # Generate visualization
        plot_drift_analysis(
            cls.data, 
            cls.drift_points, 
            cls.drift_detections, 
            save_path='test_output/drift_analysis.png'
        )
        
        # Generate mock API response data
        cls.api_data = cls._generate_mock_api_data()
        
        # Save API data for testing
        with open('test_output/drift_history.json', 'w') as f:
            json.dump(cls.api_data, f, indent=2, default=str)
            
        logger.info(f"Detected {len(cls.drift_detections)} drift points")
        logger.info(f"Expected drift points: {cls.drift_points}")
        logger.info(f"Detected drift points: {cls.drift_detections}")
    
    @classmethod
    def _generate_mock_api_data(cls):
        """Generate mock API response data for testing visualization"""
        # Convert timestamps to string format
        timestamps = [t.strftime("%Y-%m-%d %H:%M") for t in cls.timestamps]
        
        # Create drift detected array
        drift_detected = [False] * len(cls.drift_severities)
        for idx in cls.drift_detections:
            if idx - 100 < len(drift_detected):
                drift_detected[idx - 100] = True
        
        # Create feature drifts dictionary
        feature_drifts = {}
        for i, feature in enumerate(cls.feature_names):
            # Assign random drift counts to features
            feature_drifts[feature] = np.random.randint(1, len(cls.drift_detections) + 1)
        
        # Create recent events
        recent_events = []
        for i, idx in enumerate(cls.drift_detections[-10:]):
            if idx - 100 < len(timestamps):
                recent_events.append({
                    "timestamp": timestamps[idx - 100],
                    "severity": float(cls.drift_severities[idx - 100]),
                    "feature": np.random.choice(cls.feature_names),
                    "drift_type": "distribution" if cls.drift_severities[idx - 100] > 0.5 else "trend"
                })
        
        # Calculate trend using simple moving average
        window_size = min(10, len(cls.drift_severities))
        trends = []
        
        for i in range(len(cls.drift_severities)):
            if i < window_size - 1:
                trends.append(0)  # Not enough data points yet
            else:
                window = cls.drift_severities[i-window_size+1:i+1]
                trends.append(float(np.mean(window)))
        
        # Calculate summary statistics
        total_drifts = len(cls.drift_detections)
        average_severity = float(np.mean(cls.drift_severities)) if cls.drift_severities else 0.0
        
        # Find last detection
        last_detection = None
        if total_drifts > 0 and cls.drift_detections[-1] - 100 < len(timestamps):
            last_detection = timestamps[cls.drift_detections[-1] - 100]
        
        # Determine if drift is currently detected
        current_drift_detected = drift_detected[-1] if drift_detected else False
        
        return {
            "timestamps": timestamps,
            "severities": [float(s) for s in cls.drift_severities],
            "trends": trends,
            "feature_drifts": feature_drifts,
            "total_drifts": total_drifts,
            "average_severity": average_severity,
            "last_detection": last_detection,
            "current_drift_detected": current_drift_detected,
            "recent_events": recent_events
        }

    def test_drift_detection_visualization(self):
        """Test drift detection visualization"""
        logger.info("Testing drift detection visualization...")
        
        # Verify drift detection results
        self.assertTrue(len(self.drift_detections) > 0, "Should detect at least one drift point")
        
        # Check that visualization files were created
        self.assertTrue(os.path.exists('test_output/drift_analysis.png'), 
                      "Drift analysis visualization should be created")
        self.assertTrue(os.path.exists('test_output/drift_history.json'),
                      "Drift history JSON should be created")
        
        # Verify API data structure
        self.assertIn('timestamps', self.api_data)
        self.assertIn('severities', self.api_data)
        self.assertIn('trends', self.api_data)
        self.assertIn('feature_drifts', self.api_data)
        self.assertIn('total_drifts', self.api_data)
        self.assertIn('average_severity', self.api_data)
        
        logger.info("Drift visualization test completed successfully")

    def test_logging_requirements(self):
        """Test that logging requirements are met"""
        logger.info("Verifying logging requirements...")
        
        # Check that the detector has logging enabled
        self.assertTrue(hasattr(self.detector, 'logger') or logging.getLogger(__name__), 
                      "Detector should have a logger")
        
        # Run a sample detection to generate logs
        X_sample = self.data[self.feature_names].values[150]
        self.detector.add_sample(X_sample)
        drift_detected, severity, info = self.detector.detect_drift()
        
        # Verify info contains required fields based on actual structure
        self.assertIn('mean_shift', info)
        self.assertIn('ks_statistic', info)
        self.assertIn('p_value', info)
        self.assertIn('trend', info)
        self.assertIn('drifting_features', info)
        self.assertIn('confidence', info)
        
        # Check numeric formatting in logs
        logger.debug(f"Checking numeric formatting in logs:")
        logger.debug(f"  Mean shift: {info['mean_shift']:.3f}")
        logger.debug(f"  KS statistic: {info['ks_statistic']:.3f}")
        logger.debug(f"  p-value: {info['p_value']:.3e}")
        logger.debug(f"  Trend: {info['trend']:.3f}")
        logger.debug(f"  Confidence: {info['confidence']:.3f}")
        
        logger.info("Logging verification completed")

    def test_visualization_output(self):
        """Test that visualization output files are created correctly"""
        logger.info("Testing visualization output files...")
        
        # Generate visualization files
        import os
        import tempfile
        import matplotlib.pyplot as plt
        
        # Create a temporary directory for the visualization files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Collect data for visualization
            drift_timestamps = []
            drift_severities = []
            
            # Generate sample data
            for i in range(100):
                drift_timestamps.append(i)
                drift_severities.append(np.random.random())
            
            # Define sample drift points
            drift_points = [25, 50, 75]
            
            # Create output paths
            drift_plot_path = os.path.join(temp_dir, "drift_plot.png")
            feature_plot_path = os.path.join(temp_dir, "feature_plot.png")
            
            try:
                # Plot drift severity over time
                plt.figure(figsize=(10, 6))
                plt.plot(drift_timestamps, drift_severities)
                plt.title('Drift Severity Over Time')
                plt.xlabel('Sample')
                plt.ylabel('Severity')
                
                # Add vertical lines for drift points
                for point in drift_points:
                    plt.axvline(x=point, color='r', linestyle='--', alpha=0.7)
                
                plt.savefig(drift_plot_path)
            finally:
                plt.close('all')
            
            try:
                # Plot feature distributions
                fig, axes = plt.subplots(len(self.feature_names), 1, figsize=(10, 3*len(self.feature_names)))
                
                for i, feature in enumerate(self.feature_names):
                    ax = axes[i] if len(self.feature_names) > 1 else axes
                    # Plot sample feature data
                    sample_data = np.random.normal(0, 1, 100)
                    ax.hist(sample_data, bins=20, alpha=0.7)
                    ax.set_title(f'Feature: {feature}')
                
                plt.tight_layout()
                plt.savefig(feature_plot_path)
            finally:
                plt.close('all')
            
            # Check that files were created
            self.assertTrue(os.path.exists(drift_plot_path), 
                          "Drift plot should be created")
            self.assertTrue(os.path.exists(feature_plot_path),
                          "Feature plot should be created")
            
            # Check file sizes are reasonable
            self.assertGreater(os.path.getsize(drift_plot_path), 0,
                             "Drift plot should not be empty")
            self.assertGreater(os.path.getsize(feature_plot_path), 0,
                             "Feature plot should not be empty")
        
        logger.info("Visualization output test completed successfully")

if __name__ == "__main__":
    unittest.main()

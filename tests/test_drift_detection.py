import numpy as np
from drift_detection.drift_detector import DriftDetector

def test_detect_drift():
    """Test that the detect_drift method returns the expected tuple format."""
    window_size = 10
    threshold = 0.01
    alpha = 0.3
    
    # Create a DriftDetector instance
    detector = DriftDetector(window_size, threshold, alpha)
    
    # Generate test data
    np.random.seed(42)
    ref_data = np.random.randn(100)
    curr_data = np.random.randn(100) + 0.5  # Adding shift to create drift
    
    # Call detect_drift
    result = detector.detect_drift(curr_data, ref_data)
    
    # Check that result is a tuple with 3 elements
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    # Check types of elements in the tuple
    drift_detected, severity, info_dict = result
    assert isinstance(drift_detected, bool)
    assert isinstance(severity, (int, float)) 
    assert isinstance(info_dict, dict)
    
    # Print the result for inspection
    print(f"Drift detected: {drift_detected}")
    print(f"Severity: {severity}")
    print(f"Info dict: {info_dict}")
    
    return result

if __name__ == "__main__":
    # Run the test
    test_detect_drift()
    print("All tests passed!")

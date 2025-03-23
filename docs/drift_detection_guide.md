# Drift Detection Guide

This document provides a comprehensive guide to the drift detection capabilities in the Meta-Optimizer framework. Drift detection is a critical component that allows the system to identify when the underlying data distribution changes, enabling adaptive responses such as retraining models or adjusting optimization parameters.

## Overview

The drift detection system in this framework is designed to:

1. Monitor incoming data for distribution changes
2. Quantify the severity of detected drift
3. Trigger appropriate adaptation mechanisms
4. Provide explainable insights about the nature of the drift

## Key Components

### DriftDetector

The core component for drift detection is the `DriftDetector` class located in `drift_detection/drift_detector.py`. This class implements statistical methods to compare current data with a reference distribution.

```python
from drift_detection.drift_detector import DriftDetector

# Initialize the detector with appropriate parameters
detector = DriftDetector(
    window_size=10,     # Number of samples to consider
    threshold=0.01,     # Sensitivity threshold for drift
    alpha=0.3,          # Significance level
    min_interval=40     # Minimum samples between drift detections
)

# Detect drift between current and reference data
drift_detected, severity, info_dict = detector.detect_drift(current_data, reference_data)

# The returned values provide:
# - drift_detected: Boolean indicating if drift was detected
# - severity: A float value quantifying the magnitude of drift (0.0 to 1.0)
# - info_dict: Additional information about the drift, including:
#   - mean_shift: Change in distribution mean
#   - ks_statistic: Kolmogorov-Smirnov test statistic
#   - p_value: Statistical significance of the drift
#   - trend: Direction of drift over time
```

### Integration with Meta-Learning

The framework integrates drift detection with meta-learning through the `MetaLearner` class, which can automatically detect and adapt to drift:

```python
from meta.meta_learner import MetaLearner
from drift_detection.drift_detector import DriftDetector

# Create drift detector
detector = DriftDetector(window_size=10, threshold=0.01)

# Initialize meta-learner with drift detection
meta_learner = MetaLearner(
    method="bayesian",
    strategy="bandit",
    exploration=0.3,
    drift_detector=detector
)

# Train initially
meta_learner.fit(X_initial, y_initial)

# As new data arrives, update with drift detection
meta_learner.update(X_new, y_new)  # Will automatically detect drift and retrain if needed
```

## Command-Line Interface

The framework provides several command-line options for drift detection:

```bash
# Run standalone drift detection on synthetic data
python main.py --drift --drift-window 10 --drift-threshold 0.01 --visualize

# Run meta-learner with drift detection
python main.py --run-meta-learner-with-drift --drift-window 10 --drift-threshold 0.01 --visualize

# Explain detected drift
python main.py --explain-drift --drift-window 10 --drift-threshold 0.01
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--drift` | Run drift detection on synthetic data | - |
| `--drift-window` | Window size for drift detection | `10` |
| `--drift-threshold` | Threshold for drift detection | `0.01` |
| `--drift-significance` | Significance level (alpha) | `0.3` |
| `--run-meta-learner-with-drift` | Run meta-learner with drift detection | - |
| `--explain-drift` | Generate explanations for detected drift | - |

## Visualizations

When using the `--visualize` flag, the framework generates several plots:

1. **Drift Detection Timeline**: Shows when drift was detected over time
2. **Severity Timeline**: Displays the severity of drift over time
3. **True vs Predicted Values**: Compares model predictions before and after drift
4. **Feature Distribution Changes**: Visualizes how feature distributions changed

## Common Use Cases

### Continuous Monitoring

```python
# Initialize components
detector = DriftDetector(window_size=10, threshold=0.01)
model = initialize_model()

# Reference data (initial training data)
reference_data = get_initial_data()

# Continuous monitoring loop
while True:
    current_data = get_new_data()
    drift_detected, severity, info = detector.detect_drift(current_data, reference_data)
    
    if drift_detected:
        print(f"Drift detected with severity {severity}")
        # Take action (retrain model, adjust parameters, etc.)
        model = retrain_model(current_data)
        # Update reference data
        reference_data = current_data
```

### Adaptive Meta-Learning

The framework's meta-learning system automatically adapts to drift:

```python
meta_learner = MetaLearner(
    method="bayesian",
    drift_detector=DriftDetector(window_size=10, threshold=0.01)
)

# Initial training
meta_learner.fit(X_train, y_train)

# As environment changes (possibly with drift)
for X_new, y_new in new_data_stream:
    # Will detect drift and adapt automatically
    meta_learner.update(X_new, y_new)
    
    # Make predictions with the adapted model
    predictions = meta_learner.predict(X_test)
```

## Troubleshooting

### Common Issues

1. **False Positives**: If drift is detected too frequently:
   - Increase the `threshold` parameter
   - Increase the `min_interval` parameter

2. **False Negatives**: If drift is not detected when it should be:
   - Decrease the `threshold` parameter
   - Increase the `window_size` to consider more samples

3. **Handling Multi-dimensional Data**:
   - The drift detector can handle both scalar values and multi-dimensional arrays
   - For high-dimensional data, consider applying dimensionality reduction first

### Logging

The framework includes detailed logging for drift detection:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now drift detection operations will log detailed information
detector = DriftDetector(window_size=10, threshold=0.01)
```

## Examples

See the `examples` directory for complete usage examples:

- `examples/drift_detection_example.py`: Basic usage of drift detection
- `examples/adaptive_meta_learning.py`: Meta-learning with drift adaptation
- `examples/drift_visualization.py`: Generating visualizations of drift

## API Reference

For a complete API reference, see the class documentation:

- `DriftDetector`: Main class for drift detection
- `MetaLearner`: Integrates with drift detection for adaptive learning

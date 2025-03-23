"""
Metrics package for validation and benchmarking.

This package provides metrics for evaluating various aspects of the system:
- Clinical metrics: Patient-oriented and clinical relevance metrics
- Prediction metrics: Accuracy and performance metrics for prediction models
- Performance metrics: System performance and resource utilization metrics
"""

from .clinical_metrics import (
    symptom_tracking_accuracy,
    patient_reported_utility,
    clinical_correlation,
    intervention_efficacy
)

from .prediction_metrics import (
    binary_prediction_metrics,
    regression_metrics,
    confidence_calibration,
    forecast_reliability,
    drift_detection_metrics
)

from .performance_metrics import (
    computational_efficiency_metrics,
    memory_usage_metrics,
    scalability_metrics,
    latency_measurements,
    throughput_analysis
)

__version__ = '0.1.0' 
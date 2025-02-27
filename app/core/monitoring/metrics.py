"""
Prometheus metrics for the MDT API.
"""
from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Any

class MetricsCollector:
    def __init__(self):
        """Initialize metrics collectors."""
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_latency = Histogram(
            'http_request_duration_seconds',
            'HTTP request latency',
            ['method', 'endpoint']
        )
        
        # ML metrics
        self.prediction_count = Counter(
            'ml_predictions_total',
            'Total number of predictions made'
        )
        
        self.prediction_latency = Histogram(
            'ml_prediction_duration_seconds',
            'Time spent making predictions'
        )
        
        self.drift_score = Gauge(
            'ml_drift_score',
            'Current drift score'
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy'
        )
        
        # Resource metrics
        self.memory_usage = Gauge(
            'process_memory_bytes',
            'Current memory usage'
        )
        
        self.cpu_usage = Gauge(
            'process_cpu_seconds_total',
            'Total CPU time spent'
        )
        
    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ) -> None:
        """Record HTTP request metrics."""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.request_latency.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
    def record_prediction(
        self,
        duration: float,
        drift_score: float = None,
        accuracy: float = None
    ) -> None:
        """Record prediction metrics."""
        self.prediction_count.inc()
        self.prediction_latency.observe(duration)
        
        if drift_score is not None:
            self.drift_score.set(drift_score)
            
        if accuracy is not None:
            self.model_accuracy.set(accuracy)
            
    def update_resource_usage(
        self,
        memory_bytes: float,
        cpu_seconds: float
    ) -> None:
        """Update resource usage metrics."""
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_seconds)

metrics_collector = MetricsCollector()

"""
Base Validation Framework for Migraine Digital Twin System.

This module provides the core abstractions and utilities for validating the 
Migraine Digital Twin system components. It defines base classes and interfaces
for test cases, metrics, and validation workflows.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Enumeration of validation levels for test organization."""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"

class ComponentLayer(Enum):
    """Enumeration of architectural layers for test organization."""
    CORE = "core"
    ADAPTATION = "adaptation"
    APPLICATION = "application"

@dataclass
class ValidationContext:
    """Context information for validation execution."""
    level: ValidationLevel
    layer: ComponentLayer
    component_name: str
    parameters: Dict[str, Any]
    description: str

class ValidationResult:
    """Container for validation test results."""
    
    def __init__(self, context: ValidationContext):
        self.context = context
        self.passed: bool = False
        self.metrics: Dict[str, float] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.details: Dict[str, Any] = {}
        self.execution_time: float = 0.0
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a metric measurement to the results."""
        self.metrics[name] = value
    
    def add_error(self, error: str) -> None:
        """Add an error message to the results."""
        self.errors.append(error)
        logger.error(f"{self.context.component_name}: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message to the results."""
        self.warnings.append(warning)
        logger.warning(f"{self.context.component_name}: {warning}")
    
    def set_passed(self, passed: bool) -> None:
        """Set the overall pass/fail status."""
        self.passed = passed

class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    def __init__(self, context: ValidationContext):
        self.context = context
        self.result = ValidationResult(context)
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """Execute validation and return results."""
        pass
    
    def _validate_with_timeout(self, timeout_seconds: float = 30.0) -> ValidationResult:
        """Execute validation with timeout protection."""
        import time
        start_time = time.time()
        
        try:
            self.result = self.validate()
        except Exception as e:
            self.result.add_error(f"Validation failed with error: {str(e)}")
            self.result.set_passed(False)
        
        self.result.execution_time = time.time() - start_time
        return self.result

class ValidationSuite:
    """Container for organizing and executing multiple validators."""
    
    def __init__(self, name: str):
        self.name = name
        self.validators: List[BaseValidator] = []
        self.results: List[ValidationResult] = []
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the suite."""
        self.validators.append(validator)
    
    def run(self, parallel: bool = False) -> List[ValidationResult]:
        """Execute all validators in the suite."""
        logger.info(f"Running validation suite: {self.name}")
        
        if parallel:
            # TODO: Implement parallel execution
            pass
        
        for validator in self.validators:
            result = validator._validate_with_timeout()
            self.results.append(result)
        
        self._summarize_results()
        return self.results
    
    def _summarize_results(self) -> None:
        """Generate and log summary of validation results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        logger.info(f"""
        Validation Suite: {self.name}
        Total Tests: {total}
        Passed: {passed}
        Failed: {total - passed}
        """)

class MetricValidator(BaseValidator):
    """Base class for validators that check metric thresholds."""
    
    def __init__(self, context: ValidationContext, 
                 metric_thresholds: Dict[str, float]):
        super().__init__(context)
        self.metric_thresholds = metric_thresholds
    
    def _check_metrics(self, computed_metrics: Dict[str, float]) -> bool:
        """Check if computed metrics meet thresholds."""
        all_passed = True
        
        for metric_name, threshold in self.metric_thresholds.items():
            if metric_name not in computed_metrics:
                self.result.add_warning(f"Metric {metric_name} not computed")
                continue
                
            value = computed_metrics[metric_name]
            passed = value >= threshold
            
            if not passed:
                self.result.add_error(
                    f"Metric {metric_name} failed: {value} < {threshold}"
                )
                all_passed = False
            
            self.result.add_metric(metric_name, value)
        
        return all_passed

class DataValidator(BaseValidator):
    """Base class for validators that check data properties."""
    
    def __init__(self, context: ValidationContext,
                 required_fields: List[str],
                 value_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__(context)
        self.required_fields = required_fields
        self.value_ranges = value_ranges or {}
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data structure and values."""
        all_passed = True
        
        # Check required fields
        for field in self.required_fields:
            if field not in data:
                self.result.add_error(f"Required field missing: {field}")
                all_passed = False
                continue
        
        # Check value ranges
        for field, (min_val, max_val) in self.value_ranges.items():
            if field not in data:
                continue
                
            value = data[field]
            if not isinstance(value, (int, float)):
                continue
                
            if not min_val <= value <= max_val:
                self.result.add_error(
                    f"Field {field} value {value} outside range [{min_val}, {max_val}]"
                )
                all_passed = False
        
        return all_passed

# Utility functions for common validation tasks
def validate_array_properties(
    arr: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[np.dtype] = None,
    value_range: Optional[Tuple[float, float]] = None
) -> List[str]:
    """Validate numpy array properties."""
    errors = []
    
    if expected_shape and arr.shape != expected_shape:
        errors.append(f"Invalid shape: expected {expected_shape}, got {arr.shape}")
    
    if expected_dtype and arr.dtype != expected_dtype:
        errors.append(f"Invalid dtype: expected {expected_dtype}, got {arr.dtype}")
    
    if value_range:
        min_val, max_val = value_range
        if np.any(arr < min_val) or np.any(arr > max_val):
            errors.append(
                f"Values outside range [{min_val}, {max_val}]: "
                f"min={np.min(arr)}, max={np.max(arr)}"
            )
    
    return errors

def validate_timestamp_sequence(
    timestamps: np.ndarray,
    expected_frequency: Optional[float] = None,
    allow_gaps: bool = False
) -> List[str]:
    """Validate a sequence of timestamps."""
    errors = []
    
    if not np.all(np.diff(timestamps) > 0):
        errors.append("Timestamps are not strictly increasing")
    
    if expected_frequency:
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        
        if not np.isclose(mean_interval, expected_frequency, rtol=0.1):
            errors.append(
                f"Invalid sampling frequency: expected {expected_frequency}, "
                f"got {1/mean_interval}"
            )
        
        if not allow_gaps:
            max_interval = np.max(intervals)
            if max_interval > expected_frequency * 1.5:
                errors.append(f"Found gaps in timestamp sequence: max gap = {max_interval}")
    
    return errors 
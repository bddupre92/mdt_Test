"""
Validation module for the MoE Framework.

This package provides specialized validation strategies for time-series
medical data, ensuring proper handling of temporal dependencies and
patient-specific considerations.
"""

from moe_framework.validation.time_series_validation import (
    TimeSeriesValidator,
)

__all__ = [
    'TimeSeriesValidator',
]

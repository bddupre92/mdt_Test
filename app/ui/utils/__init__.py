"""
UI Utilities Package

This package provides utility functions for the benchmark dashboard.
"""

from .formatting import (
    format_scientific,
    format_decimal,
    format_percentage,
    format_time,
    format_optimizer_results,
    format_selection_patterns,
    highlight_best_value,
    highlight_meta_optimizer
)

__all__ = [
    'format_scientific',
    'format_decimal',
    'format_percentage',
    'format_time',
    'format_optimizer_results',
    'format_selection_patterns',
    'highlight_best_value',
    'highlight_meta_optimizer'
] 
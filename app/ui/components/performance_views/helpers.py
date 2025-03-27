"""
Helper functions for performance analysis dashboard components.

This module provides common utility functions used across various
performance dashboard components.
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def safe_get_metric(data: Any, key_path: str, default: Any = None) -> Any:
    """
    Safely access a metric from nested data regardless of whether it's a dictionary or object.
    
    Supports dot notation for nested access (e.g., "metrics.rmse" or "metrics.nested.value").
    
    Args:
        data: Dictionary or object containing metrics
        key_path: The key path to access (can use dot notation for nested access)
        default: Default value to return if the key is not found
        
    Returns:
        The value associated with the key path or the default value
    """
    if data is None:
        return default
    
    # Handle nested keys with dot notation
    if '.' in key_path:
        parts = key_path.split('.')
        current = data
        
        for part in parts:
            # For each part of the path, access it appropriately based on type
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
                
            # If we hit None in the middle of the path, return the default
            if current is None:
                return default
                
        return current
    
    # Handle direct key access
    if isinstance(data, dict):
        return data.get(key_path, default)
    
    return getattr(data, key_path, default)


def safe_has_metric(data: Any, key_path: str) -> bool:
    """
    Safely check if a metric exists regardless of whether it's in a dictionary or object.
    
    Supports dot notation for nested access (e.g., "metrics.rmse" or "metrics.nested.value").
    
    Args:
        data: Dictionary or object containing metrics
        key_path: The key path to check (can use dot notation for nested access)
        
    Returns:
        True if the key exists, False otherwise
    """
    if data is None:
        return False
    
    # Handle nested keys with dot notation
    if '.' in key_path:
        parts = key_path.split('.')
        current = data
        
        # For all parts except the last, navigate down the tree
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    return False
                current = current.get(part)
            else:
                if not hasattr(current, part):
                    return False
                current = getattr(current, part)
            
            if current is None:
                return False
        
        # Check if the last part exists
        last_part = parts[-1]
        if isinstance(current, dict):
            return last_part in current
        return hasattr(current, last_part)
    
    # Handle direct key access
    if isinstance(data, dict):
        return key_path in data
    
    return hasattr(data, key_path)

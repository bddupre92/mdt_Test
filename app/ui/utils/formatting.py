"""
Formatting Utilities

This module provides utility functions for consistent data formatting.
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

def format_scientific(value: float, precision: int = 4) -> str:
    """Format a number in scientific notation.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return "N/A"
    return f"{value:.{precision}e}"

def format_decimal(value: float, precision: int = 4) -> str:
    """Format a number with fixed decimal places.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if not isinstance(value, (int, float)) or np.isnan(value):
        return "N/A"
    return f"{value:.{precision}f}"

def format_percentage(value: float, precision: int = 2) -> str:
    """Format a value as a percentage.
    
    Args:
        value: Value to format (0-1)
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        percentage = float(value) * 100
        return f"{percentage:.{precision}f}%"
    except (ValueError, TypeError):
        return str(value)

def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if not isinstance(seconds, (int, float)) or np.isnan(seconds):
        return "N/A"
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def format_optimizer_results(results: Dict[str, Any]) -> pd.DataFrame:
    """Format optimizer results for display.
    
    Args:
        results: Dictionary containing optimizer results
        
    Returns:
        Formatted DataFrame
    """
    if not results or "optimizers" not in results:
        return pd.DataFrame()
    
    data = []
    for opt_name, opt_data in results["optimizers"].items():
        data.append({
            "Optimizer": opt_name,
            "Best Fitness": format_scientific(opt_data.get("best_fitness")),
            "Runtime": format_time(opt_data.get("runtime")),
            "Evaluations": format_decimal(opt_data.get("evaluations"), 0)
        })
    
    return pd.DataFrame(data)

def format_selection_patterns(selections: Dict[str, int]) -> pd.DataFrame:
    """Format optimizer selection patterns for display.
    
    Args:
        selections: Dictionary mapping optimizer names to selection counts
        
    Returns:
        Formatted DataFrame
    """
    if not selections:
        return pd.DataFrame()
    
    total = sum(selections.values())
    data = []
    
    for opt_name, count in selections.items():
        percentage = (count / total) * 100
        data.append({
            "Optimizer": opt_name,
            "Count": count,
            "Percentage": format_percentage(percentage)
        })
    
    df = pd.DataFrame(data)
    return df.sort_values("Count", ascending=False).reset_index(drop=True)

def highlight_best_value(df: pd.DataFrame, column: str, minimize: bool = True) -> pd.DataFrame:
    """Highlight the best value in a DataFrame column.
    
    Args:
        df: Input DataFrame
        column: Column name to highlight
        minimize: Whether lower values are better
        
    Returns:
        Styled DataFrame
    """
    if df.empty or column not in df.columns:
        return df
    
    def highlight(val):
        try:
            val = float(val)
            if minimize:
                is_best = val == df[column].min()
            else:
                is_best = val == df[column].max()
            return 'background-color: rgba(0, 255, 0, 0.2)' if is_best else ''
        except:
            return ''
    
    return df.style.applymap(highlight, subset=[column])

def highlight_meta_optimizer(df: pd.DataFrame) -> pd.DataFrame:
    """Highlight the meta-optimizer row in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Styled DataFrame
    """
    def highlight(row):
        return ['background-color: rgba(0, 255, 0, 0.2)' if row["Optimizer"] == "meta" 
                else '' for _ in row]
    
    return df.style.apply(highlight, axis=1)

def format_timestamp(timestamp: str = None) -> str:
    """Format a timestamp string.
    
    Args:
        timestamp: Optional timestamp string to format. If None, current time is used.
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    elif isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp  # Return original if parsing fails
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def format_number(value: float, precision: int = 6) -> str:
    """Format a number with specified precision.
    
    Args:
        value: Number to format
        precision: Number of decimal places
        
    Returns:
        Formatted number string
    """
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value)

def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    try:
        seconds = float(seconds)
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    except (ValueError, TypeError):
        return str(seconds) 
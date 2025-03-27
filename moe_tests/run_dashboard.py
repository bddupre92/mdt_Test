#!/usr/bin/env python
"""
Dashboard Runner Script

This script runs the benchmark dashboard with error suppression for known PyTorch issues.
It intercepts and suppresses the specific RuntimeError related to PyTorch custom classes
that occurs during Streamlit's file watching process.
"""

import os
import sys
import warnings
import logging
from contextlib import contextmanager

# Configure logging to suppress specific warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('streamlit')
logger.setLevel(logging.ERROR)

# Create a context manager to redirect stderr during problematic operations
@contextmanager
def suppress_torch_errors():
    """Suppress specific PyTorch errors during execution."""
    original_stderr = sys.stderr
    
    class FilteredStderr:
        def __init__(self):
            self.filtered_messages = [
                "Tried to instantiate class '__path__._path'",
                "no running event loop"
            ]
        
        def write(self, message):
            # Only write to stderr if the message doesn't contain filtered content
            if not any(filtered in message for filtered in self.filtered_messages):
                original_stderr.write(message)
        
        def flush(self):
            original_stderr.flush()
    
    try:
        sys.stderr = FilteredStderr()
        yield
    finally:
        sys.stderr = original_stderr

if __name__ == "__main__":
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")
    
    # Run the dashboard with error suppression
    with suppress_torch_errors():
        # Change to the project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Import and run streamlit programmatically
        import streamlit.web.bootstrap as bootstrap
        
        # Run the performance analysis dashboard
        bootstrap.run("/Users/blair.dupre/Documents/migrineDT/mdt_Test/app/ui/performance_analysis_dashboard.py", "", [], {})

"""
MetaOptimizer - Bridge Module

This module provides direct access to the MetaOptimizer class from the meta_optimizer package.
It's used to simplify imports in other parts of the codebase.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import MetaOptimizer from the actual implementation
from meta_optimizer.meta.meta_optimizer import MetaOptimizer

# Re-export the MetaOptimizer class
__all__ = ["MetaOptimizer"] 
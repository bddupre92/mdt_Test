#!/usr/bin/env python
"""
Quick test script to verify meta_optimizer package installation and functionality.
"""
import sys
import numpy as np
from meta_optimizer import __version__
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory
from meta_optimizer.visualization.optimizer_analysis import OptimizerAnalyzer

def main():
    """Run basic tests to verify package functionality."""
    print(f"Python version: {sys.version}")
    print(f"Meta-Optimizer version: {__version__}")
    
    # Test optimizer creation
    print("\nTesting optimizer creation...")
    factory = OptimizerFactory()
    optimizers = factory.get_available_optimizers()
    print(f"Available optimizers: {optimizers}")
    
    # Create a simple test optimizer
    if "DE" in optimizers:
        print("\nCreating DE optimizer...")
        optimizer = factory.create_optimizer("DE")
        print(f"Optimizer created: {optimizer.__class__.__name__}")
    
    print("\nMeta-Optimizer package is working correctly!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

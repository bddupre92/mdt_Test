#!/usr/bin/env python3
"""
Simplified test for Meta-Optimizer Framework

This script tests that the package is properly installed and can be imported.
"""

import numpy as np
from meta_optimizer import __version__
from meta_optimizer.benchmark.test_functions import ClassicalTestFunctions
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory


def main():
    """Run a simple test of the package."""
    print("Meta-Optimizer Framework Installation Test")
    print("------------------------------------------")
    print(f"Package version: {__version__}")
    
    # Test the benchmark functions
    print("\nTesting benchmark functions...")
    test_point = np.array([1.0, 2.0, 3.0])
    sphere_value = ClassicalTestFunctions.sphere(test_point)
    print(f"Sphere function value at {test_point}: {sphere_value}")
    
    # Test optimizer factory
    print("\nTesting optimizer factory...")
    factory = OptimizerFactory()
    available_optimizers = factory.get_available_optimizers()
    print(f"Available optimizers: {available_optimizers}")
    
    if available_optimizers:
        opt_name = available_optimizers[0]
        print(f"\nCreating optimizer: {opt_name}")
        optimizer = factory.create_optimizer(opt_name, dim=3, bounds=[(-10, 10)] * 3)
        print(f"Optimizer created: {optimizer.__class__.__name__}")
    
    print("\nMeta-Optimizer package test completed successfully!")


if __name__ == "__main__":
    main()

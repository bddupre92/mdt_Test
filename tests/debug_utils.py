#!/usr/bin/env python3
"""
Debugging utilities for the baseline comparison framework

This module provides debugging functions to help diagnose issues
with the baseline comparison framework implementation.
"""

import sys
import inspect
import logging
import importlib
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path if needed
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def check_imports():
    """
    Check that all required modules can be imported
    """
    modules_to_check = [
        "baseline_comparison",
        "baseline_comparison.comparison_runner",
        "baseline_comparison.baseline_algorithms.satzilla_inspired",
        "baseline_comparison.visualization",
        "baseline_comparison.benchmark_utils"
    ]
    
    print("\nChecking imports:")
    print("================")
    
    all_imports_ok = True
    
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ Successfully imported {module_name}")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            all_imports_ok = False
    
    # Try to import meta_optimizer if it exists
    try:
        from meta_optimizer import MetaOptimizer
        print(f"✅ Successfully imported meta_optimizer.MetaOptimizer")
    except ImportError as e:
        print(f"⚠️ Note: Could not import meta_optimizer.MetaOptimizer: {e}")
        print("   This may be expected if you're using a mock instead.")
    
    return all_imports_ok

def inspect_benchmark_functions():
    """
    Inspect the available benchmark functions
    """
    try:
        from baseline_comparison.benchmark_utils import get_benchmark_function
        
        print("\nInspecting benchmark functions:")
        print("==============================")
        
        # Test a few common benchmark functions
        test_dims = 2
        test_functions = ["sphere", "rosenbrock", "ackley", "rastrigin"]
        
        for func_name in test_functions:
            try:
                func = get_benchmark_function(func_name, test_dims)
                print(f"\nFunction: {func_name}")
                print(f"  Name: {func.name}")
                print(f"  Dimensions: {func.dims}")
                
                # Test evaluation
                x = np.zeros(test_dims)
                y = func.evaluate(x)
                print(f"  f({x}) = {y}")
                
                # Get bounds if available
                if hasattr(func, "bounds"):
                    print(f"  Bounds: {func.bounds}")
                
                # Get optimum if available
                if hasattr(func, "optimum"):
                    print(f"  Known optimum: {func.optimum}")
                
            except Exception as e:
                print(f"❌ Error with function '{func_name}': {e}")
        
    except ImportError as e:
        print(f"❌ Could not inspect benchmark functions: {e}")

def test_satzilla_selector():
    """
    Test the SATzilla-inspired selector
    """
    try:
        from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
        from baseline_comparison.benchmark_utils import get_benchmark_function
        
        print("\nTesting SATzilla-inspired selector:")
        print("==================================")
        
        # Create a selector
        selector = SatzillaInspiredSelector()
        print(f"✅ Created SatzillaInspiredSelector instance")
        
        # Create a test problem
        test_problem = get_benchmark_function("sphere", 2)
        print(f"✅ Created test problem: {test_problem.name}")
        
        # Test feature extraction if available
        if hasattr(selector, "extract_features"):
            features = selector.extract_features(test_problem)
            print(f"✅ Extracted features: {list(features.keys())}")
        else:
            print("⚠️ Selector doesn't have extract_features method")
        
        # Test algorithm selection if available
        if hasattr(selector, "select_algorithm"):
            algorithm = selector.select_algorithm(test_problem)
            print(f"✅ Selected algorithm: {algorithm}")
        else:
            print("⚠️ Selector doesn't have select_algorithm method")
            
    except Exception as e:
        print(f"❌ Error testing SatzillaInspiredSelector: {e}")
        import traceback
        traceback.print_exc()

def test_comparison_runner():
    """
    Test the comparison runner
    """
    try:
        from baseline_comparison.comparison_runner import BaselineComparison
        from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
        from baseline_comparison.benchmark_utils import get_benchmark_function
        
        # Try to import MetaOptimizer or create a mock
        try:
            from meta_optimizer import MetaOptimizer
        except ImportError:
            # Create a mock MetaOptimizer for testing
            class MetaOptimizer:
                def __init__(self, *args, **kwargs):
                    self.name = "MockMetaOptimizer"
                    
                def optimize(self, problem, *args, **kwargs):
                    # Simple random optimization for testing
                    best_x = np.random.uniform(-5, 5, problem.dims)
                    best_y = problem.evaluate(best_x)
                    return best_x, best_y, 100
        
        print("\nTesting comparison runner:")
        print("=========================")
        
        # Create components
        selector = SatzillaInspiredSelector()
        meta_optimizer = MetaOptimizer()
        
        # Create comparison runner
        comparison = BaselineComparison(
            baseline_selector=selector,
            meta_optimizer=meta_optimizer,
            max_evaluations=100,  # Small number for quick testing
            num_trials=1
        )
        print(f"✅ Created BaselineComparison instance")
        
        # Create a test problem
        test_problems = [get_benchmark_function("sphere", 2)]
        print(f"✅ Created test problem: {test_problems[0].name}")
        
        # Try to run a very simple comparison
        try:
            print("Running a simple comparison test...")
            results = comparison.run_comparison(test_problems)
            print(f"✅ Comparison ran successfully")
            print(f"  Results: {results}")
        except Exception as e:
            print(f"❌ Error running comparison: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error testing comparison runner: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Run all debugging checks
    """
    print("Running debugging utilities for baseline comparison framework")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check system path
    print("\nSystem path:")
    for path in sys.path:
        print(f"  {path}")
    
    # Check imports
    imports_ok = check_imports()
    
    if imports_ok:
        # Run further checks
        inspect_benchmark_functions()
        test_satzilla_selector()
        test_comparison_runner()
    else:
        print("\n❌ Import checks failed. Please fix import issues before continuing.")
    
    print("\nDebugging complete!")

if __name__ == "__main__":
    main() 
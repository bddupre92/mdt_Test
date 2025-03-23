"""
Standalone test runner for causal inference tests.

This script runs the causal inference tests without requiring pytest's conftest.py.
It adds the project root to the system path to facilitate imports.
"""

import unittest
import sys
import os

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import the test class
from tests.theory.test_causal_inference import TestCausalInferenceAnalyzer

def run_tests():
    """Run the causal inference tests."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCausalInferenceAnalyzer)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())

if __name__ == '__main__':
    run_tests() 
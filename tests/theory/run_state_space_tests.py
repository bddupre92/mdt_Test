"""
Standalone test runner for state space models.

This script runs the tests for state space models without relying on pytest's conftest.py.
"""

import unittest
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the test case
from tests.theory.test_state_space_models import TestStateSpaceModeler

if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStateSpaceModeler)
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 
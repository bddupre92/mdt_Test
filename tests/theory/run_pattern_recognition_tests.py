"""
Standalone test runner for pattern recognition tests.

This script runs the tests for pattern recognition without relying on pytest's conftest.py.
"""

import unittest
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the test cases
from tests.theory.test_pattern_recognition import (
    # Feature extraction tests
    TestTimeDomainFeatures,
    TestFrequencyDomainFeatures,
    TestStatisticalFeatures,
    TestPhysiologicalFeatures,
    # Pattern classification tests
    TestBinaryClassifier,
    TestEnsembleClassifier,
    TestProbabilisticClassifier
)

def create_test_suite():
    """Create a test suite containing all pattern recognition tests."""
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_cases = [
        # Feature extraction tests
        TestTimeDomainFeatures,
        TestFrequencyDomainFeatures,
        TestStatisticalFeatures,
        TestPhysiologicalFeatures,
        # Pattern classification tests
        TestBinaryClassifier,
        TestEnsembleClassifier,
        TestProbabilisticClassifier
    ]
    
    for test_case in test_cases:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_case))
    
    return suite

if __name__ == "__main__":
    # Create and run the test suite
    suite = create_test_suite()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 
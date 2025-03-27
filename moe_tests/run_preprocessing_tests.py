#!/usr/bin/env python
"""
Test Runner for Domain-Specific Preprocessing

This script runs the tests for the domain-specific preprocessing operations
to validate their functionality.
"""

import unittest
import sys
from tests.test_domain_specific_preprocessing import (
    TestMedicationNormalizer,
    TestEnvironmentalTriggerAnalyzer,
    TestAdvancedFeatureEngineer
)


def run_tests():
    """Run all domain-specific preprocessing tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMedicationNormalizer))
    test_suite.addTest(unittest.makeSuite(TestEnvironmentalTriggerAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestAdvancedFeatureEngineer))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())

"""Run all feature extraction tests.

This script runs all the unit tests for the feature extraction components:
- TimeDomainFeatures
- FrequencyDomainFeatures
- StatisticalFeatures
- PhysiologicalFeatures
"""

import unittest
import sys

# Import test modules
from tests.theory.test_time_domain_features import TestTimeDomainFeatures
from tests.theory.test_frequency_domain_features import TestFrequencyDomainFeatures
from tests.theory.test_statistical_features import TestStatisticalFeatures
from tests.theory.test_physiological_features import TestPhysiologicalFeatures

def run_tests():
    """Run all feature extraction tests."""
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases using the loader
    test_suite.addTests(loader.loadTestsFromTestCase(TestTimeDomainFeatures))
    test_suite.addTests(loader.loadTestsFromTestCase(TestFrequencyDomainFeatures))
    test_suite.addTests(loader.loadTestsFromTestCase(TestStatisticalFeatures))
    test_suite.addTests(loader.loadTestsFromTestCase(TestPhysiologicalFeatures))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests()) 
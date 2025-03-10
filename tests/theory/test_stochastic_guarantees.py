"""
Tests for the Stochastic Guarantees components.

This module contains tests for the theoretical components related to
stochastic guarantees and probabilistic performance bounds for optimization algorithms.
"""

import unittest
import numpy as np
from typing import Dict, List, Any

from core.theory.algorithm_analysis import StochasticGuaranteeAnalyzer


class TestStochasticGuaranteeAnalyzer(unittest.TestCase):
    """Tests for the StochasticGuaranteeAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.de_analyzer = StochasticGuaranteeAnalyzer("DE", "Test DE stochastic analyzer")
        self.pso_analyzer = StochasticGuaranteeAnalyzer("PSO", "Test PSO stochastic analyzer")
        self.gd_analyzer = StochasticGuaranteeAnalyzer("GD", "Test GD stochastic analyzer")
    
    def test_initialization(self):
        """Test proper initialization of StochasticGuaranteeAnalyzer."""
        # Test that the analyzers are properly initialized
        self.assertEqual(self.de_analyzer.algorithm_type, "DE")
        self.assertEqual(self.de_analyzer.name, "StochasticAnalyzer_DE")
        self.assertEqual(self.de_analyzer.description, "Test DE stochastic analyzer")
        
        # Test that stochastic properties are populated
        self.assertIsNotNone(self.de_analyzer.stochastic_properties)
        self.assertIn('stochastic_nature', self.de_analyzer.stochastic_properties)
        self.assertIn('convergence_probability', self.de_analyzer.stochastic_properties)
        self.assertIn('performance_distribution', self.de_analyzer.stochastic_properties)
        
        # Verify stochastic nature differences between algorithms
        self.assertIn('highly stochastic', self.de_analyzer.stochastic_properties['stochastic_nature'])
        self.assertIn('deterministic', self.gd_analyzer.stochastic_properties['stochastic_nature'])
    
    def test_analyze_stochastic(self):
        """Test stochastic analysis functionality."""
        # Analyze DE's stochastic properties
        analysis = self.de_analyzer.analyze({})
        
        # Check that the analysis contains the expected keys
        self.assertIn('algorithm_type', analysis)
        self.assertIn('stochastic_properties', analysis)
        self.assertIn('theoretical_guarantees', analysis)
        
        # Check theoretical guarantees
        guarantees = analysis['theoretical_guarantees']
        self.assertIn('global_convergence', guarantees)
        self.assertIn('local_convergence', guarantees)
        self.assertIn('iteration_bounds', guarantees)
        self.assertIn('progress_rate', guarantees)
        self.assertIn('failure_probability', guarantees)
        
        # GD should have deterministic guarantees for convex functions
        gd_analysis = self.gd_analyzer.analyze({})
        self.assertTrue(gd_analysis['theoretical_guarantees']['global_convergence']['guaranteed'])
        
        # DE should not have deterministic global guarantees
        self.assertFalse(analysis['theoretical_guarantees']['global_convergence']['guaranteed'])
    
    def test_parameter_effects(self):
        """Test parameter effects on stochastic guarantees."""
        # Test DE with parameters
        de_params = {
            'population_size': 50,
            'crossover_rate': 0.7,
            'scaling_factor': 0.5
        }
        
        analysis = self.de_analyzer.analyze(de_params)
        
        # Check parameter effects analysis
        self.assertIn('parameter_effects', analysis)
        effects = analysis['parameter_effects']
        
        # Check that parameter effects are analyzed
        self.assertIn('population_size', effects)
        self.assertIn('crossover_rate', effects)
        self.assertIn('scaling_factor', effects)
        
        # Each parameter should have effects on multiple aspects
        for param in ['population_size', 'crossover_rate', 'scaling_factor']:
            param_effects = effects[param]
            self.assertGreaterEqual(len(param_effects), 2, f"{param} should affect at least 2 stochastic properties")
    
    def test_compare_algorithms_unimodal(self):
        """Test stochastic comparison on unimodal problems."""
        # Define algorithms to compare
        algorithms = [
            {'type': 'DE'}, 
            {'type': 'PSO'}, 
            {'type': 'GD'}
        ]
        
        # Define a unimodal, smooth problem
        unimodal_problem = {
            'modality': 'unimodal',
            'landscape_smoothness': 'smooth',
            'dimension': 5,
            'target_precision': 1e-6
        }
        
        # Compare algorithms
        comparison = self.de_analyzer.compare_algorithms(algorithms, unimodal_problem)
        
        # Check basic comparison structure
        self.assertIn('problem_characteristics', comparison)
        self.assertIn('algorithm_comparisons', comparison)
        self.assertIn('stochastic_ranking', comparison)
        self.assertIn('confidence_levels', comparison)
        self.assertIn('expected_iterations_comparison', comparison)
        self.assertIn('robustness_comparison', comparison)
        self.assertIn('summary', comparison)
        
        # Check detailed metrics for each algorithm
        for algo in comparison['algorithm_comparisons']:
            metrics = algo['performance_metrics']
            self.assertIn('confidence_level', metrics)
            self.assertIn('expected_iterations', metrics)
            self.assertIn('probability_of_success', metrics)
            self.assertIn('robustness_score', metrics)
            self.assertIn('overall_score', metrics)
        
        # For unimodal, smooth problems, GD should have good stochastic guarantees
        gd_ranking = comparison['stochastic_ranking']['GD']
        self.assertLessEqual(gd_ranking, 2, "GD should rank well for unimodal, smooth problems")
    
    def test_compare_algorithms_multimodal(self):
        """Test stochastic comparison on multimodal problems."""
        # Define algorithms to compare
        algorithms = [
            {'type': 'DE'}, 
            {'type': 'PSO'}, 
            {'type': 'GD'}
        ]
        
        # Define a multimodal, rugged problem
        multimodal_problem = {
            'modality': 'highly multimodal',
            'landscape_smoothness': 'rugged',
            'dimension': 20,
            'target_precision': 1e-4
        }
        
        # Compare algorithms
        comparison = self.pso_analyzer.compare_algorithms(algorithms, multimodal_problem)
        
        # For multimodal, rugged problems, population-based methods should have better probability of success
        de_prob = comparison['algorithm_comparisons'][0]['performance_metrics']['probability_of_success']
        gd_prob = comparison['algorithm_comparisons'][2]['performance_metrics']['probability_of_success']
        
        self.assertGreater(de_prob, gd_prob, 
                      "DE should have higher probability of success than GD for multimodal, rugged problems")
        
        # Check expected iterations - should be higher for complex problems
        for algo, iterations in comparison['expected_iterations_comparison'].items():
            self.assertGreater(iterations, 100, f"{algo} should require many iterations for complex problems")
    
    def test_comparison_summary(self):
        """Test comparison summary generation."""
        # Define algorithms to compare
        algorithms = [
            {'type': 'DE'}, 
            {'type': 'GD'}
        ]
        
        # Define a unimodal problem
        unimodal_problem = {
            'modality': 'unimodal',
            'landscape_smoothness': 'smooth',
            'dimension': 5
        }
        
        # Compare algorithms
        comparison = self.de_analyzer.compare_algorithms(algorithms, unimodal_problem)
        
        # Check that summary is generated and mentions the problem characteristics
        summary = comparison['summary']
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 100)
        self.assertIn('unimodal', summary)
        
        # Check that it mentions algorithms being compared
        self.assertIn('DE', summary)
        self.assertIn('GD', summary)
    
    def test_formal_definition(self):
        """Test formal definition retrieval."""
        definition = self.de_analyzer.get_formal_definition()
        
        # Check that the definition is substantial and contains key terms
        self.assertGreater(len(definition), 100)
        self.assertIn("Stochastic Guarantee", definition)
        self.assertIn("Probabilistic Convergence", definition)
        self.assertIn("Confidence Bounds", definition)
        self.assertIn("Failure Probability", definition)


if __name__ == '__main__':
    unittest.main() 
"""
Tests for the Algorithm Analysis theoretical components.

This module contains tests for the theoretical components related to algorithm
analysis, including convergence analysis, landscape theory, No Free Lunch theorem
applications, and stochastic guarantees.
"""

import unittest
import numpy as np
from typing import Dict, List, Any

from core.theory.algorithm_analysis import ConvergenceAnalyzer


class TestConvergenceAnalyzer(unittest.TestCase):
    """Tests for the ConvergenceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.de_analyzer = ConvergenceAnalyzer("DE", "Test DE convergence analyzer")
        self.pso_analyzer = ConvergenceAnalyzer("PSO", "Test PSO convergence analyzer")
        self.gd_analyzer = ConvergenceAnalyzer("GD", "Test GD convergence analyzer")
    
    def test_initialization(self):
        """Test proper initialization of ConvergenceAnalyzer."""
        # Test that the analyzers are properly initialized
        self.assertEqual(self.de_analyzer.algorithm_type, "DE")
        self.assertEqual(self.de_analyzer.name, "ConvergenceAnalyzer_DE")
        self.assertEqual(self.de_analyzer.description, "Test DE convergence analyzer")
        
        # Test convergence type determination
        self.assertEqual(self.de_analyzer.convergence_type, "probabilistic")
        self.assertEqual(self.gd_analyzer.convergence_type, "local")
        
        # Test that convergence conditions are non-empty
        self.assertTrue(len(self.de_analyzer.convergence_conditions) > 0)
        self.assertTrue(len(self.pso_analyzer.convergence_conditions) > 0)
        self.assertTrue(len(self.gd_analyzer.convergence_conditions) > 0)
        
        # Test that convergence rate properties are populated
        self.assertIsNotNone(self.de_analyzer.convergence_rate.get('order'))
        self.assertIsNotNone(self.pso_analyzer.convergence_rate.get('asymptotic_complexity'))
        self.assertIsNotNone(self.gd_analyzer.convergence_rate.get('dimension_dependency'))
    
    def test_analyze_de(self):
        """Test analysis of DE algorithm."""
        de_params = {
            'type': 'DE',
            'crossover_rate': 0.7,
            'scaling_factor': 0.5,
            'population_size': 50,
            'strategy': 'DE/rand/1/bin'
        }
        
        analysis = self.de_analyzer.analyze(de_params)
        
        # Check that the analysis contains the expected keys
        self.assertIn('algorithm_type', analysis)
        self.assertIn('convergence_type', analysis)
        self.assertIn('convergence_type_description', analysis)
        self.assertIn('convergence_rate', analysis)
        self.assertIn('convergence_conditions', analysis)
        self.assertIn('parameter_impacts', analysis)
        self.assertIn('overall_assessment', analysis)
        
        # Check parameter impacts
        self.assertIn('crossover_rate', analysis['parameter_impacts'])
        self.assertIn('scaling_factor', analysis['parameter_impacts'])
        
        # Check overall assessment
        self.assertIn('expected_convergence', analysis['overall_assessment'])
        self.assertIn('reliability', analysis['overall_assessment'])
        self.assertIn('efficiency', analysis['overall_assessment'])
        self.assertIn('robustness', analysis['overall_assessment'])
        self.assertIn('limitations', analysis['overall_assessment'])
        
        # Check that the analysis is consistent with expectations
        self.assertEqual(analysis['algorithm_type'], 'DE')
        self.assertEqual(analysis['convergence_type'], 'probabilistic')
        self.assertIn('with high probability', analysis['overall_assessment']['expected_convergence'])
    
    def test_analyze_parameter_impacts(self):
        """Test parameter impact analysis."""
        # Test DE with poor parameters
        de_poor_params = {
            'crossover_rate': 0.05,  # Too low
            'scaling_factor': 1.5,   # Too high
        }
        
        impacts = self.de_analyzer._analyze_parameter_impacts(de_poor_params)
        self.assertIn('Too low', impacts['crossover_rate'])
        self.assertIn('Too high', impacts['scaling_factor'])
        
        # Test DE with good parameters
        de_good_params = {
            'crossover_rate': 0.7,    # Good value
            'scaling_factor': 0.5,    # Good value
        }
        
        impacts = self.de_analyzer._analyze_parameter_impacts(de_good_params)
        self.assertIn('Appropriate', impacts['crossover_rate'])
        self.assertIn('Appropriate', impacts['scaling_factor'])
        
        # Test PSO parameters
        pso_params = {
            'inertia_weight': 0.7,
            'cognitive_param': 1.5,
            'social_param': 1.5,
        }
        
        impacts = self.pso_analyzer._analyze_parameter_impacts(pso_params)
        self.assertIn('Appropriate', impacts['inertia_weight'])
        self.assertIn('Appropriate', impacts['cognitive_social_sum'])
    
    def test_compare_algorithms(self):
        """Test algorithm comparison functionality."""
        # Define algorithms to compare
        algorithms = [
            {
                'type': 'DE',
                'crossover_rate': 0.7,
                'scaling_factor': 0.5,
                'population_size': 50
            },
            {
                'type': 'PSO',
                'inertia_weight': 0.7,
                'cognitive_param': 1.5,
                'social_param': 1.5,
                'swarm_size': 40
            },
            {
                'type': 'GD',
                'learning_rate': 0.01,
                'momentum': 0.9
            }
        ]
        
        # Define problems to test with
        unimodal_problem = {
            'modality': 'unimodal',
            'dimension': 5,
            'landscape_smoothness': 'smooth'
        }
        
        multimodal_problem = {
            'modality': 'multimodal',
            'dimension': 20,
            'landscape_smoothness': 'rugged'
        }
        
        # Compare algorithms on unimodal problem
        unimodal_comparison = self.de_analyzer.compare_algorithms(algorithms, unimodal_problem)
        
        # Check that the comparison contains the expected keys
        self.assertIn('problem_characteristics', unimodal_comparison)
        self.assertIn('algorithm_comparisons', unimodal_comparison)
        self.assertIn('ranking', unimodal_comparison)
        self.assertIn('recommended_algorithm', unimodal_comparison)
        self.assertIn('recommendation_reason', unimodal_comparison)
        
        # For unimodal, smooth problem, GD should be highly ranked
        gd_rank = unimodal_comparison['ranking'].get('GD')
        self.assertIsNotNone(gd_rank)
        self.assertLessEqual(gd_rank, 2)  # GD should be ranked 1st or 2nd
        
        # Compare algorithms on multimodal problem
        multimodal_comparison = self.de_analyzer.compare_algorithms(algorithms, multimodal_problem)
        
        # For multimodal, rugged problem, DE or PSO should be recommended
        recommended = multimodal_comparison['recommended_algorithm']
        self.assertIn(recommended.upper(), ['DE', 'PSO'])
        
        # Check that the reasoning mentions the multimodal nature
        self.assertIn('multimodal', multimodal_comparison['recommendation_reason'].lower())
    
    def test_formal_definition(self):
        """Test formal definition retrieval."""
        de_definition = self.de_analyzer.get_formal_definition()
        gd_definition = self.gd_analyzer.get_formal_definition()
        
        # Check that the definitions are non-empty and algorithm-specific
        self.assertIn('Differential Evolution', de_definition)
        self.assertIn('probabilistically', de_definition)
        
        self.assertIn('Gradient Descent', gd_definition)
        self.assertIn('strongly convex', gd_definition)


if __name__ == '__main__':
    unittest.main() 
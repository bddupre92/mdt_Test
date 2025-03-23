"""
Tests for the No Free Lunch theorem components.

This module contains tests for the theoretical components related to
the No Free Lunch theorems and their implications for optimization algorithms.
"""

import unittest
import numpy as np
from typing import Dict, List, Any

from core.theory.algorithm_analysis import NoFreeLunchAnalyzer


class TestNoFreeLunchAnalyzer(unittest.TestCase):
    """Tests for the NoFreeLunchAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.de_analyzer = NoFreeLunchAnalyzer("DE", "Test DE NFL analyzer")
        self.pso_analyzer = NoFreeLunchAnalyzer("PSO", "Test PSO NFL analyzer")
        self.gd_analyzer = NoFreeLunchAnalyzer("GD", "Test GD NFL analyzer")
        self.random_analyzer = NoFreeLunchAnalyzer("RANDOM", "Test Random NFL analyzer")
    
    def test_initialization(self):
        """Test proper initialization of NoFreeLunchAnalyzer."""
        # Test that the analyzers are properly initialized
        self.assertEqual(self.de_analyzer.algorithm_type, "DE")
        self.assertEqual(self.de_analyzer.name, "NFLAnalyzer_DE")
        self.assertEqual(self.de_analyzer.description, "Test DE NFL analyzer")
        
        # Test that algorithm bias is populated
        self.assertIsNotNone(self.de_analyzer.algorithm_bias)
        self.assertIn('bias_type', self.de_analyzer.algorithm_bias)
        self.assertIn('favored_problem_classes', self.de_analyzer.algorithm_bias)
        self.assertIn('disfavored_problem_classes', self.de_analyzer.algorithm_bias)
        
        # Check different algorithm types have different biases
        self.assertNotEqual(
            self.de_analyzer.algorithm_bias['bias_type'],
            self.gd_analyzer.algorithm_bias['bias_type']
        )
        
        # Check random search has no bias
        self.assertEqual(self.random_analyzer.algorithm_bias['bias_type'], 'none')
    
    def test_analyze_nfl(self):
        """Test NFL analysis functionality."""
        # Analyze DE's NFL implications
        analysis = self.de_analyzer.analyze({})
        
        # Check that the analysis contains the expected keys
        self.assertIn('algorithm_type', analysis)
        self.assertIn('algorithm_bias', analysis)
        self.assertIn('nfl_principles', analysis)
        self.assertIn('nfl_implications', analysis)
        
        # Check that NFL implications are populated
        implications = analysis['nfl_implications']
        self.assertTrue(len(implications) > 0)
        
        # All implications should have principle and implication
        for implication in implications:
            self.assertIn('principle', implication)
            self.assertIn('implication', implication)
            
        # Check for algorithm-specific implications
        algname_in_implications = False
        for implication in implications:
            if 'DE' in implication['implication']:
                algname_in_implications = True
                break
        self.assertTrue(algname_in_implications, "No algorithm-specific implications found")
    
    def test_parameter_influence(self):
        """Test parameter influence analysis on NFL implications."""
        # Analyze with parameters
        de_params = {
            'crossover_rate': 0.9,
            'population_size': 10
        }
        
        analysis = self.de_analyzer.analyze(de_params)
        
        # Check parameter influence analysis
        self.assertIn('parameter_influence', analysis)
        influences = analysis['parameter_influence']
        
        # Check that both parameters are analyzed
        self.assertIn('crossover_rate', influences)
        self.assertIn('population_size', influences)
        
        # Different parameter values should yield different influences
        small_pop_influence = influences['population_size']
        
        large_pop_analysis = self.de_analyzer.analyze({
            'crossover_rate': 0.9,
            'population_size': 100
        })
        large_pop_influence = large_pop_analysis['parameter_influence']['population_size']
        
        self.assertNotEqual(small_pop_influence, large_pop_influence)
    
    def test_compare_algorithms(self):
        """Test NFL-based algorithm comparison."""
        # Define algorithms to compare
        algorithms = [
            {'type': 'DE'}, 
            {'type': 'PSO'}, 
            {'type': 'GD'}, 
            {'type': 'RANDOM'}
        ]
        
        # Define a specific problem class
        problem = {
            'modality': 'highly multimodal',
            'landscape_smoothness': 'rugged',
            'separability': 'nonseparable',
            'dimension': 30
        }
        
        # Compare algorithms
        comparison = self.de_analyzer.compare_algorithms(algorithms, problem)
        
        # Check basic comparison structure
        self.assertIn('problem_characteristics', comparison)
        self.assertIn('problem_class', comparison)
        self.assertIn('algorithm_comparisons', comparison)
        self.assertIn('theoretical_free_lunch', comparison)
        self.assertIn('nfl_implied_ranking', comparison)
        self.assertIn('meta_optimization_potential', comparison)
        
        # Check problem class determination
        problem_class = comparison['problem_class']
        self.assertEqual(problem_class['modality'], 'highly multimodal')
        self.assertEqual(problem_class['smoothness'], 'rugged')
        
        # Check NFL alignment scores are calculated
        for algo_comparison in comparison['algorithm_comparisons']:
            self.assertIn('alignment_score', algo_comparison)
        
        # For multimodal, rugged problem, population-based methods should have better alignment
        de_comparison = next(a for a in comparison['algorithm_comparisons'] if a['algorithm'] == 'DE')
        gd_comparison = next(a for a in comparison['algorithm_comparisons'] if a['algorithm'] == 'GD')
        random_comparison = next(a for a in comparison['algorithm_comparisons'] if a['algorithm'] == 'RANDOM')
        
        # DE should have better NFL alignment than GD for this problem
        self.assertGreater(
            de_comparison['alignment_score'],
            gd_comparison['alignment_score']
        )
        
        # Random should have neutral alignment (around 5)
        self.assertAlmostEqual(
            random_comparison['alignment_score'],
            5.0,
            delta=1.0
        )
    
    def test_free_lunch_detection(self):
        """Test the detection of theoretical free lunches."""
        # Define algorithms with very different biases
        algorithms = [
            {'type': 'GD'},
            {'type': 'RANDOM'}
        ]
        
        # Define a problem class that strongly aligns with GD's bias
        unimodal_problem = {
            'modality': 'unimodal',
            'landscape_smoothness': 'smooth',
            'separability': 'separable',
            'dimension': 5
        }
        
        # Compare algorithms
        comparison = self.gd_analyzer.compare_algorithms(algorithms, unimodal_problem)
        
        # Check free lunch detection
        free_lunch = comparison['theoretical_free_lunch']
        self.assertTrue(free_lunch['exists'])
        self.assertEqual(free_lunch['algorithm'], 'GD')
    
    def test_meta_optimization_potential(self):
        """Test the assessment of meta-optimization potential."""
        # Define diverse algorithms
        algorithms = [
            {'type': 'DE'}, 
            {'type': 'PSO'}, 
            {'type': 'GD'}
        ]
        
        # Problem with clear algorithmic preferences
        multimodal_problem = {
            'modality': 'highly multimodal',
            'landscape_smoothness': 'rugged',
            'dimension': 30
        }
        
        # Compare algorithms
        comparison = self.de_analyzer.compare_algorithms(algorithms, multimodal_problem)
        
        # Check meta-optimization potential
        potential = comparison['meta_optimization_potential']
        self.assertIn('level', potential)
        self.assertIn('explanation', potential)
        
        # For diverse algorithms on a problem with clear preferences, 
        # meta-optimization potential should be moderate to high
        self.assertIn(potential['level'], ['Moderate', 'High'])
    
    def test_formal_definition(self):
        """Test formal definition retrieval."""
        definition = self.de_analyzer.get_formal_definition()
        
        # Check that the definition is substantial and contains key terms
        self.assertGreater(len(definition), 100)
        self.assertIn("No Free Lunch", definition)
        self.assertIn("∑", definition)
        self.assertIn("all possible functions", definition.lower())


if __name__ == '__main__':
    unittest.main() 
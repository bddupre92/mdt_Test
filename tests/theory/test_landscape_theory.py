"""
Tests for the Landscape Theory components.

This module contains tests for the theoretical components related to
landscape theory, including landscape analysis and algorithm-landscape interactions.
"""

import unittest
import numpy as np
from typing import Dict, List, Any

from core.theory.algorithm_analysis import LandscapeAnalyzer


class TestLandscapeAnalyzer(unittest.TestCase):
    """Tests for the LandscapeAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.de_analyzer = LandscapeAnalyzer("DE", "Test DE landscape analyzer")
        self.pso_analyzer = LandscapeAnalyzer("PSO", "Test PSO landscape analyzer")
        self.gd_analyzer = LandscapeAnalyzer("GD", "Test GD landscape analyzer")
    
    def test_initialization(self):
        """Test proper initialization of LandscapeAnalyzer."""
        # Test that the analyzers are properly initialized
        self.assertEqual(self.de_analyzer.algorithm_type, "DE")
        self.assertEqual(self.de_analyzer.name, "LandscapeAnalyzer_DE")
        self.assertEqual(self.de_analyzer.description, "Test DE landscape analyzer")
        
        # Test that algorithm-landscape interactions are populated
        self.assertIsNotNone(self.de_analyzer.algorithm_landscape_interaction)
        self.assertIn('modality', self.de_analyzer.algorithm_landscape_interaction)
        self.assertIn('ruggedness', self.de_analyzer.algorithm_landscape_interaction)
        
        # Check the interaction details
        modality_interaction = self.de_analyzer.algorithm_landscape_interaction['modality']
        self.assertIn('strength', modality_interaction)
        self.assertIn('weakness', modality_interaction)
        self.assertIn('notes', modality_interaction)
    
    def test_analyze_landscape(self):
        """Test landscape analysis functionality."""
        # Analyze DE's landscape interactions
        analysis = self.de_analyzer.analyze({})
        
        # Check that the analysis contains the expected keys
        self.assertIn('algorithm_type', analysis)
        self.assertIn('landscape_interactions', analysis)
        self.assertIn('theoretical_insights', analysis)
        
        # Check that theoretical insights are populated
        insights = analysis['theoretical_insights']
        self.assertTrue(len(insights) > 0)
        
        # Check algorithm-specific insights
        de_specific_insight = False
        for insight in insights:
            if "DE's" in insight or "Differential" in insight:
                de_specific_insight = True
                break
        self.assertTrue(de_specific_insight, "No DE-specific insights found")
    
    def test_compare_algorithms_unimodal(self):
        """Test algorithm comparison on unimodal landscapes."""
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
            'separability': 'separable',
            'dimension': 5
        }
        
        # Compare algorithms
        comparison = self.de_analyzer.compare_algorithms(algorithms, unimodal_problem)
        
        # Check basic comparison structure
        self.assertIn('problem_characteristics', comparison)
        self.assertIn('landscape_properties', comparison)
        self.assertIn('algorithm_comparisons', comparison)
        self.assertIn('theoretical_ranking', comparison)
        self.assertIn('recommended_algorithm', comparison)
        
        # Check landscape properties analysis
        properties = comparison['landscape_properties']
        self.assertIn('modality', properties)
        self.assertEqual(properties['modality']['property'], 'unimodal')
        
        # For unimodal, smooth problems, gradient-based methods should be highly ranked
        ranking = comparison['theoretical_ranking']
        self.assertIn('GD', ranking)
        
        # Verify GD has a good rank for unimodal problems
        gd_rank = ranking['GD']
        self.assertLessEqual(gd_rank, 2, "GD should be highly ranked for unimodal, smooth problems")
    
    def test_compare_algorithms_multimodal(self):
        """Test algorithm comparison on multimodal landscapes."""
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
            'separability': 'nonseparable',
            'dimension': 20
        }
        
        # Compare algorithms
        comparison = self.pso_analyzer.compare_algorithms(algorithms, multimodal_problem)
        
        # Check basic comparison structure
        self.assertIn('recommended_algorithm', comparison)
        
        # For multimodal, rugged problems, population-based methods should be recommended
        recommended = comparison['recommended_algorithm']
        self.assertIn(recommended, ['DE', 'PSO'], 
                    "Population-based method should be recommended for multimodal, rugged problems")
        
        # Check recommendation reason
        reason = comparison['recommendation_reason']
        self.assertIsNotNone(reason)
        
        # GD should be ranked lower for multimodal problems
        ranking = comparison['theoretical_ranking']
        gd_rank = ranking['GD']
        de_rank = ranking['DE']
        self.assertGreater(gd_rank, de_rank, 
                         "GD should be ranked lower than DE for multimodal problems")
    
    def test_landscape_property_descriptions(self):
        """Test landscape property description generation."""
        # Test modality descriptions
        unimodal_desc = self.de_analyzer._get_modality_description('unimodal')
        multimodal_desc = self.de_analyzer._get_modality_description('multimodal')
        
        self.assertIn("one optimum", unimodal_desc.lower())
        self.assertIn("multiple local", multimodal_desc.lower())
        
        # Test ruggedness descriptions
        smooth_desc = self.de_analyzer._get_ruggedness_description('low')
        rugged_desc = self.de_analyzer._get_ruggedness_description('high')
        
        self.assertIn("smooth", smooth_desc.lower())
        self.assertIn("rough", rugged_desc.lower())
        
        # Test theoretical impacts
        unimodal_impact = self.de_analyzer._get_modality_impact('unimodal')
        multimodal_impact = self.de_analyzer._get_modality_impact('multimodal')
        
        self.assertIn("local search", unimodal_impact.lower())
        self.assertIn("global search", multimodal_impact.lower())
    
    def test_formal_definition(self):
        """Test formal definition retrieval."""
        definition = self.de_analyzer.get_formal_definition()
        
        # Check that the definition is substantial and contains key terms
        self.assertGreater(len(definition), 100)
        self.assertIn("Landscape analysis", definition)
        self.assertIn("Modality", definition)
        self.assertIn("Ruggedness", definition)
        self.assertIn("Separability", definition)


if __name__ == '__main__':
    unittest.main() 
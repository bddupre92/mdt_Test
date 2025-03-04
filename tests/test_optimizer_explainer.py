"""
test_optimizer_explainer.py
-------------------------
Unit tests for the optimizer explainer
"""

import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from optimizers.optimizer_factory import OptimizerFactory
from explainability.optimizer_explainer import OptimizerExplainer
from optimizers.base_optimizer import OptimizerState


class TestOptimizerExplainer(unittest.TestCase):
    """Test cases for OptimizerExplainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = OptimizerFactory()
        self.optimizer = self.factory.create_optimizer('differential_evolution', dim=5, bounds=[(-5, 5)] * 5)
        
        # Define a simple test function
        def sphere(x):
            return np.sum(x**2)
        
        # Run optimization with a small number of evaluations
        self.optimizer.run(sphere, max_evals=50)
        
        # Create explainer
        self.explainer = OptimizerExplainer(self.optimizer)
    
    def test_initialization(self):
        """Test initialization of optimizer explainer"""
        self.assertEqual(self.explainer.name, "optimizer")
        self.assertEqual(self.explainer.model, self.optimizer)
        self.assertIsNotNone(self.explainer.supported_plot_types)
        self.assertGreater(len(self.explainer.supported_plot_types), 0)
    
    def test_explain(self):
        """Test explanation generation"""
        explanation = self.explainer.explain()
        
        # Check that explanation contains expected keys
        self.assertIn('optimizer_type', explanation)
        self.assertIn('dimensions', explanation)
        self.assertIn('population_size', explanation)
        self.assertIn('evaluations', explanation)
        self.assertIn('iterations', explanation)
        self.assertIn('best_score', explanation)
        self.assertIn('execution_time', explanation)
        self.assertIn('convergence_curve', explanation)
        
        # Check that optimizer type is correct
        self.assertEqual(explanation['optimizer_type'], 'DifferentialEvolutionOptimizer')
        
        # Check that dimensions are correct
        self.assertEqual(explanation['dimensions'], 5)
        
        # Check that convergence curve is not empty
        self.assertGreater(len(explanation['convergence_curve']), 0)
    
    def test_parameter_sensitivity(self):
        """Test parameter sensitivity calculation"""
        explanation = self.explainer.explain()
        
        # Check that feature importance is calculated
        self.assertIn('feature_importance', explanation)
        
        # If parameter history exists, feature importance should not be empty
        if self.optimizer.parameter_history:
            self.assertGreater(len(explanation['feature_importance']), 0)
    
    def test_plot_convergence(self):
        """Test convergence plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate convergence plot
        fig = self.explainer.plot('convergence')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_parameter_adaptation(self):
        """Test parameter adaptation plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate parameter adaptation plot
        fig = self.explainer.plot('parameter_adaptation')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_diversity(self):
        """Test diversity plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate diversity plot
        fig = self.explainer.plot('diversity')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_landscape_analysis(self):
        """Test landscape analysis plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate landscape analysis plot
        fig = self.explainer.plot('landscape_analysis')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_decision_process(self):
        """Test decision process plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate decision process plot
        fig = self.explainer.plot('decision_process')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_exploration_exploitation(self):
        """Test exploration/exploitation plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate exploration/exploitation plot
        fig = self.explainer.plot('exploration_exploitation')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_gradient_estimation(self):
        """Test gradient estimation plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate gradient estimation plot
        fig = self.explainer.plot('gradient_estimation')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_plot_performance_profile(self):
        """Test performance profile plot generation"""
        # Generate explanation
        self.explainer.explain()
        
        # Generate performance profile plot
        fig = self.explainer.plot('performance_profile')
        
        # Check that figure is created
        self.assertIsNotNone(fig)
    
    def test_invalid_plot_type(self):
        """Test invalid plot type handling"""
        # Generate explanation
        self.explainer.explain()
        
        # Try to generate an invalid plot type
        with self.assertRaises(ValueError):
            self.explainer.plot('invalid_plot_type')
    
    def test_no_explanation(self):
        """Test plotting without explanation"""
        # Try to generate a plot without explanation
        with self.assertRaises(ValueError):
            self.explainer.plot('convergence')


if __name__ == '__main__':
    unittest.main()

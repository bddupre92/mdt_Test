"""
test_meta_learner.py
--------------------
Tests meta-learner's ability to select and adapt optimization algorithms.
"""

import unittest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from meta.meta_learner import MetaLearner
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategy
from optimizers.de import DifferentialEvolutionOptimizer
from benchmarking.test_functions import TEST_FUNCTIONS

class MockOptimizer:
    def __init__(self, name, performance_pattern):
        self.name = name
        self.performance_pattern = performance_pattern  # function: context -> performance
        self.calls = 0
    
    def optimize(self, func):
        self.calls += 1
        return None, self.performance_pattern(self.calls)

class TestMetaLearner(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.optimizers = {
            'ACO': AntColonyOptimizer(),
            'GWO': GreyWolfOptimizer(),
            'ES': EvolutionStrategy(),
            'DE': DifferentialEvolutionOptimizer()
        }
        
        # Create mock optimizers with different performance patterns
        self.mock_opts = [
            MockOptimizer("Opt1", lambda x: 0.5 + 0.1 * np.sin(x)),  # Cyclic
            MockOptimizer("Opt2", lambda x: 0.8 - 0.3 * np.exp(-x/5)),  # Improving
            MockOptimizer("Opt3", lambda x: 0.3 + 0.4 * (1 - np.exp(-x/10)))  # Degrading
        ]
    
    def test_bayesian_optimization(self):
        """Test Bayesian optimization for algorithm selection"""
        ml = MetaLearner(
            method='bayesian',
            surrogate_model=GaussianProcessRegressor(
                normalize_y=True,
                noise_level=0.1
            )
        )
        ml.set_algorithms(self.mock_opts)
        
        # Test multiple contexts
        contexts = [
            {'dim': 10, 'complexity': 'simple'},
            {'dim': 30, 'complexity': 'complex'},
            {'dim': 50, 'complexity': 'simple'}
        ]
        
        for context in contexts:
            algo = ml.select_algorithm_bayesian(context)
            self.assertIsNotNone(algo)
            
            # Simulate performance and update
            perf = algo.performance_pattern(algo.calls)
            ml.update(algo, perf, context)
        
        # Check learning progress
        self.assertGreater(len(ml.history), 0)
        if hasattr(ml, 'gp_model'):
            self.assertIsNotNone(ml.gp_model)
    
    def test_reinforcement_learning(self):
        """Test RL-based algorithm selection"""
        class SimplePolicy(nn.Module):
            def __init__(self, input_dim, num_actions):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_actions)
                )
            
            def forward(self, x):
                return torch.softmax(self.network(x), dim=-1)
        
        ml = MetaLearner(
            method='rl',
            policy_model=SimplePolicy(input_dim=3, num_actions=len(self.mock_opts))
        )
        ml.set_algorithms(self.mock_opts)
        
        # Test learning loop
        for _ in range(10):
            state = torch.randn(3)  # Mock state
            algo = ml.select_algorithm_rl(state)
            reward = algo.performance_pattern(algo.calls)
            ml.update_rl(algo, reward, state)
        
        self.assertGreater(len(ml.history), 0)
    
    def test_adaptive_selection(self):
        """Test meta-learner's ability to adapt to changing conditions"""
        ml = MetaLearner(method='bayesian')
        ml.set_algorithms(self.mock_opts)
        
        # Phase 1: First optimizer should perform best
        for _ in range(5):
            algo = ml.select_algorithm({'phase': 1})
            perf = algo.performance_pattern(algo.calls)
            ml.update(algo, perf, {'phase': 1})
        
        # Count selections in first phase
        phase1_counts = {opt.name: 0 for opt in self.mock_opts}
        for hist in ml.history[-5:]:
            phase1_counts[hist['algorithm'].name] += 1
        
        # Phase 2: Second optimizer should perform best
        for _ in range(5):
            algo = ml.select_algorithm({'phase': 2})
            perf = algo.performance_pattern(algo.calls * 2)  # Different pattern
            ml.update(algo, perf, {'phase': 2})
        
        # Count selections in second phase
        phase2_counts = {opt.name: 0 for opt in self.mock_opts}
        for hist in ml.history[-5:]:
            phase2_counts[hist['algorithm'].name] += 1
        
        # Verify adaptation
        self.assertNotEqual(
            max(phase1_counts.items(), key=lambda x: x[1])[0],
            max(phase2_counts.items(), key=lambda x: x[1])[0],
            "Meta-learner didn't adapt to changing conditions"
        )
    
    def test_real_optimization_tasks(self):
        """Test meta-learner on actual optimization problems"""
        ml = MetaLearner(method='bayesian')
        ml.set_algorithms(list(self.optimizers.values()))
        
        for func_name, func_class in TEST_FUNCTIONS.items():
            # Create test function
            func = func_class(dim=10, bounds=[(-5.12, 5.12)]*10)
            
            # Try different optimizers
            context = {
                'function': func_name,
                'dim': 10,
                'bounds': (-5.12, 5.12)
            }
            
            algo = ml.select_algorithm(context)
            _, score = algo.optimize(func)
            ml.update(algo, -score, context)  # Negative score as we minimize
            
            self.assertLess(score, 1000, 
                           f"Poor performance on {func_name}: {score}")
    
    def test_concurrent_learning(self):
        """Test meta-learner in concurrent optimization scenario"""
        from concurrent.futures import ThreadPoolExecutor
        
        ml = MetaLearner(method='bayesian')
        ml.set_algorithms(self.mock_opts)
        
        def run_optimization(context):
            algo = ml.select_algorithm(context)
            perf = algo.performance_pattern(algo.calls)
            ml.update(algo, perf, context)
            return algo.name, perf
        
        contexts = [{'task': i} for i in range(4)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_optimization, ctx) 
                      for ctx in contexts]
            results = [f.result() for f in futures]
        
        self.assertEqual(len(results), 4)
        self.assertEqual(len(ml.history), 4)

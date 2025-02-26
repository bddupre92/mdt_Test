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
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from benchmarking.test_functions import TEST_FUNCTIONS

class MockOptimizer:
    def __init__(self, name, performance_pattern):
        self.name = name
        self.performance_pattern = performance_pattern  # function: context -> performance
        self.calls = 0
    
    def optimize(self, func):
        self.calls += 1
        return None, self.performance_pattern(self.calls, {})

class TestMetaLearner(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.optimizers = {
            'ACO': AntColonyOptimizer(dim=30, bounds=[(-100, 100)] * 30),
            'GWO': GreyWolfOptimizer(dim=30, bounds=[(-100, 100)] * 30),
            'ES': EvolutionStrategyOptimizer(dim=30, bounds=[(-100, 100)] * 30),  
            'DE': DifferentialEvolutionOptimizer(dim=30, bounds=[(-100, 100)] * 30)
        }
        
        # Create mock optimizers with phase-dependent performance patterns
        def opt1_pattern(x, context):
            phase = context.get('phase', 1)
            if phase == 1:
                return 0.9  # Best in phase 1
            else:
                return 0.3  # Poor in phase 2
        
        def opt2_pattern(x, context):
            phase = context.get('phase', 1)
            if phase == 1:
                return 0.4  # Poor in phase 1
            else:
                return 0.8  # Best in phase 2
        
        def opt3_pattern(x, context):
            phase = context.get('phase', 1)
            return 0.5  # Mediocre in both phases
        
        self.mock_opts = [
            MockOptimizer("Opt1", opt1_pattern),
            MockOptimizer("Opt2", opt2_pattern),
            MockOptimizer("Opt3", opt3_pattern)
        ]
    
    def test_bayesian_optimization(self):
        """Test Bayesian optimization for algorithm selection"""
        ml = MetaLearner(
            method='bayesian',
            surrogate_model=GaussianProcessRegressor(
                normalize_y=True,
                alpha=0.1
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
            perf = algo.performance_pattern(algo.calls, context)
            ml.update(algo.name, perf)
        
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
        
        ml = MetaLearner(method='rl')
        ml.set_algorithms(self.mock_opts)
        
        # Test learning loop
        for _ in range(10):
            state = torch.randn(3)  # Mock state
            algo = ml.select_algorithm_rl(state)
            reward = algo.performance_pattern(algo.calls, {})
            ml.update_rl(algo.name, reward, state)
        
        self.assertGreater(len(ml.history), 0)
    
    def test_adaptive_selection(self):
        """Test meta-learner's ability to adapt to changing conditions"""
        ml = MetaLearner(method='bayesian')
        ml.set_algorithms(self.mock_opts)
        
        # Phase 1: First optimizer should perform best
        print("\nPhase 1:")
        for i in range(5):
            algo = ml.select_algorithm({'phase': 1})
            perf = algo.performance_pattern(algo.calls, {'phase': 1})
            ml.update(algo.name, perf)
            print(f"  Iteration {i+1}: Selected {algo.name}, Performance: {perf:.2f}")
        
        # Count selections in first phase
        phase1_counts = {opt.name: 0 for opt in self.mock_opts}
        for hist in ml.history[-5:]:
            phase1_counts[hist['algorithm']] += 1
        
        print("\nPhase 1 Selection Counts:", phase1_counts)
        
        # Phase 2: Second optimizer should perform best
        print("\nPhase 2:")
        for i in range(5):
            algo = ml.select_algorithm({'phase': 2})
            perf = algo.performance_pattern(algo.calls, {'phase': 2})
            ml.update(algo.name, perf)
            print(f"  Iteration {i+1}: Selected {algo.name}, Performance: {perf:.2f}")
        
        # Count selections in second phase
        phase2_counts = {opt.name: 0 for opt in self.mock_opts}
        for hist in ml.history[-5:]:
            phase2_counts[hist['algorithm']] += 1
        
        print("\nPhase 2 Selection Counts:", phase2_counts)
        
        # Verify adaptation
        phase1_best = max(phase1_counts.items(), key=lambda x: x[1])[0]
        phase2_best = max(phase2_counts.items(), key=lambda x: x[1])[0]
        
        self.assertEqual(phase1_best, "Opt1", 
                        f"Expected Opt1 to be selected most in phase 1, but got {phase1_best}")
        self.assertEqual(phase2_best, "Opt2",
                        f"Expected Opt2 to be selected most in phase 2, but got {phase2_best}")
        self.assertNotEqual(phase1_best, phase2_best,
                          "Meta-learner didn't adapt to changing conditions")
    
    def test_real_optimization_tasks(self):
        """Test meta-learner on actual optimization problems"""
        ml = MetaLearner(method='bayesian')
        
        # Define function-specific settings
        function_settings = {
            'sphere': {
                'bounds': (-5.12, 5.12),
                'threshold': 500,      # Relatively easy to optimize
                'min_improvement': 20,  # Expect at least 20% improvement
            },
            'rosenbrock': {
                'bounds': (-5.12, 5.12),
                'threshold': 5000,     # Very challenging due to narrow valley
                'min_improvement': 15,  # Harder to get big improvements
            },
            'rastrigin': {
                'bounds': (-5.12, 5.12),
                'threshold': 1000,     # Many local optima make it hard
                'min_improvement': 20,
            },
            'ackley': {
                'bounds': (-5.12, 5.12),
                'threshold': 800,      # Deceptive global landscape
                'min_improvement': 20,
            },
            'griewank': {
                'bounds': (-600, 600), # Different search space
                'threshold': 2000,     # Multimodal with different scale
                'min_improvement': 20,
            },
            'levy': {
                'bounds': (-10, 10),   # Different search space
                'threshold': 1000,     # Multimodal function
                'min_improvement': 20,
            }
        }
        
        results = {}  # Track results for each function
        
        for func_name, func_class in TEST_FUNCTIONS.items():
            print(f"\nTesting {func_name}...")
            
            # Get function-specific settings
            settings = function_settings.get(func_name.lower(), {
                'bounds': (-5.12, 5.12),
                'threshold': 1000,
                'min_improvement': 20
            })
            
            # Create test function with consistent settings
            dim = 10
            bounds = [settings['bounds']] * dim
            func = func_class(dim=dim, bounds=bounds)
            
            # Configure optimizers for this function
            optimizers = {
                'ACO': AntColonyOptimizer(dim=dim, bounds=bounds, max_evals=10000),
                'GWO': GreyWolfOptimizer(dim=dim, bounds=bounds, max_evals=10000),
                'ES': EvolutionStrategyOptimizer(dim=dim, bounds=bounds, max_evals=10000),
                'DE': DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, max_evals=10000)
            }
            ml.set_algorithms(list(optimizers.values()))
            
            # Try different optimizers
            context = {
                'function': func_name,
                'dim': dim,
                'bounds': settings['bounds']
            }
            
            try:
                algo = ml.select_algorithm(context)
                solution, score = algo.optimize(func)
                ml.update(algo.name, -score)  # Negative score as we minimize
                
                results[func_name] = {
                    'score': score,
                    'optimizer': algo.name,
                    'evaluations': algo.evaluations,
                    'time': algo.end_time - algo.start_time,
                    'initial_score': algo.convergence_curve[0],
                    'improvement': (algo.convergence_curve[0] - score) / algo.convergence_curve[0] * 100
                }
                
                # Test optimization quality with function-specific threshold
                self.assertLess(score, settings['threshold'],
                              f"Poor performance on {func_name}: score={score:.2e} > threshold={settings['threshold']:.2e}")
                
                # Test convergence
                self.assertGreater(len(algo.convergence_curve), 0,
                                 f"No convergence data for {func_name}")
                self.assertLess(algo.convergence_curve[-1], algo.convergence_curve[0],
                              f"No improvement on {func_name}")
                
                # Test minimum improvement percentage
                improvement = (algo.convergence_curve[0] - score) / algo.convergence_curve[0] * 100
                self.assertGreater(improvement, settings['min_improvement'],
                                 f"Insufficient improvement on {func_name}: {improvement:.1f}%")
                
                # Test evaluation budget
                self.assertLessEqual(algo.evaluations, algo.max_evals,
                                   f"Exceeded max evaluations on {func_name}")
                
            except Exception as e:
                self.fail(f"Failed on {func_name}: {str(e)}")
        
        # Print summary of results
        print("\nOptimization Results:")
        for func_name, result in results.items():
            settings = function_settings.get(func_name.lower())
            print(f"\n{func_name}:")
            print(f"  Initial Score: {result['initial_score']:.2e}")
            print(f"  Final Score: {result['score']:.2e} (threshold: {settings['threshold']:.2e})")
            print(f"  Improvement: {result['improvement']:.1f}%")
            print(f"  Optimizer: {result['optimizer']}")
            print(f"  Evaluations: {result['evaluations']}")
            print(f"  Time: {result['time']:.2f}s")
    
    def test_concurrent_learning(self):
        """Test meta-learner in concurrent optimization scenario"""
        from concurrent.futures import ThreadPoolExecutor
        
        ml = MetaLearner(method='bayesian')
        ml.set_algorithms(self.mock_opts)
        
        def run_optimization(context):
            algo = ml.select_algorithm(context)
            perf = algo.performance_pattern(algo.calls, {})
            ml.update(algo.name, perf)
            return algo.name, perf
        
        contexts = [{'task': i} for i in range(4)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_optimization, ctx) 
                      for ctx in contexts]
            results = [f.result() for f in futures]
        
        self.assertEqual(len(results), 4)
        self.assertEqual(len(ml.history), 4)

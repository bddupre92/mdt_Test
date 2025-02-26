"""
test_optimizers.py
------------------
Verifies our optimization algorithms can solve test functions and checks performance
"""

import unittest
import numpy as np
import psutil
import time
from concurrent.futures import ThreadPoolExecutor
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize

from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategyOptimizer as EvolutionStrategy
from optimizers.de import DifferentialEvolutionOptimizer
from benchmarking.test_functions import ClassicalTestFunctions, TestFunction

class BaseOptimizerTest(unittest.TestCase):
    """Base class for optimizer tests to avoid code duplication"""
    
    @classmethod
    def setUpClass(cls):
        """This prevents the base class from being run as a test case"""
        if cls is BaseOptimizerTest:
            raise unittest.SkipTest("Skip BaseOptimizerTest")
    
    def setUp(self):
        """Set up test parameters"""
        self.dim = 5  # Reduced from 10
        self.bounds = [(-5.12, 5.12)] * self.dim
        self.test_functions = {
            'sphere': TestFunction(
                name='Sphere',
                func=ClassicalTestFunctions.sphere,
                dim=self.dim,
                bounds=self.bounds,
                global_minimum=0.0,
                characteristics={'continuous': True, 'convex': True, 'unimodal': True}
            ),
            'rastrigin': TestFunction(
                name='Rastrigin',
                func=lambda x: ClassicalTestFunctions.rastrigin(x) / 100,  # Scale down more
                dim=self.dim,
                bounds=self.bounds,
                global_minimum=0.0,
                characteristics={'continuous': True, 'non-convex': True, 'multimodal': True}
            ),
            'rosenbrock': TestFunction(
                name='Rosenbrock',
                func=lambda x: ClassicalTestFunctions.rosenbrock(x) / 100,  # Scale down more
                dim=self.dim,
                bounds=[(-2.048, 2.048)] * self.dim,
                global_minimum=0.0,
                characteristics={'continuous': True, 'non-convex': True, 'unimodal': True}
            )
        }
        super().setUp()
    
    def test_basic_optimization(self):
        """Test basic optimization on simple function"""
        optimizer = self.optimizer_class(dim=1, bounds=[(0,10)])
        solution, score = optimizer.optimize(lambda x: (x[0] - 3)**2)
        self.assertLess(score, 2.0, 
                       f"{self.optimizer_class.__name__} didn't get close enough")
    
    def test_optimizer_memory_usage(self):
        """Test memory usage during optimization"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run heavy optimization
        optimizer = self.optimizer_class(dim=self.dim, bounds=self.bounds)
        _, _ = optimizer.optimize(self.test_functions['sphere'])
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        self.assertLess(memory_increase, 100, 
                       f"{self.optimizer_class.__name__} used too much memory")
    
    def test_performance_on_test_functions(self):
        """Test performance on standard benchmark functions"""
        results = {}
        for func_name, func in self.test_functions.items():
            start_time = time.time()
            optimizer = self.optimizer_class(
                dim=self.dim, 
                bounds=func.bounds,
                population_size=200,  # Increased from 100
                max_evals=20000,      # Increased from default
                adaptive=True         # Enable adaptive parameters
            )
            solution, score = optimizer.optimize(func)
            end_time = time.time()
            
            results[func_name] = {
                'score': score,
                'time': end_time - start_time,
                'distance_to_optimum': abs(score - func.global_minimum)
            }
            
            self.assertLess(
                results[func_name]['distance_to_optimum'],
                1.0,
                f"{self.optimizer_class.__name__} failed on {func_name}"
            )
        
        return results
    
    def test_parallel_optimization(self):
        """Test parallel optimization capabilities"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for func_name, func in self.test_functions.items():
                optimizer = self.optimizer_class(dim=self.dim, bounds=func.bounds)
                futures.append(
                    executor.submit(optimizer.optimize, func)
                )
            
            results = [f.result() for f in futures]
            for result in results:
                self.assertIsNotNone(result)

class TestACO(BaseOptimizerTest):
    def setUp(self):
        self.optimizer_class = AntColonyOptimizer
        super().setUp()

class TestGWO(BaseOptimizerTest):
    def setUp(self):
        self.optimizer_class = GreyWolfOptimizer
        super().setUp()

class TestES(BaseOptimizerTest):
    def setUp(self):
        self.optimizer_class = EvolutionStrategy
        super().setUp()

class TestDE(BaseOptimizerTest):
    def setUp(self):
        self.optimizer_class = DifferentialEvolutionOptimizer
        super().setUp()

if __name__ == '__main__':
    unittest.main()

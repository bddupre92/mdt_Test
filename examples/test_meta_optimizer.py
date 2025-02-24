"""
test_meta_optimizer.py
--------------------
Test script for the meta-learning optimizer system.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from meta.meta_optimizer import MetaOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer

def create_test_functions():
    """Create a set of test functions with different characteristics"""
    functions = {}
    
    # Unimodal function
    def sphere(x):
        """Sphere function"""
        x = np.asarray(x)
        return np.sum(x**2)
    functions['sphere'] = {
        'func': sphere,
        'bounds': [(-5.12, 5.12)],
        'dim': 2,
        'multimodal': 0,
        'discrete_vars': 0
    }
    
    # Multimodal function
    def rastrigin(x):
        """Rastrigin function"""
        x = np.asarray(x)
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    functions['rastrigin'] = {
        'func': rastrigin,
        'bounds': [(-5.12, 5.12)],
        'dim': 2,
        'multimodal': 1,
        'discrete_vars': 0
    }
    
    # Rotated multimodal function
    def rotated_rastrigin(x):
        """Rotated Rastrigin function"""
        x = np.asarray(x)
        # Apply rotation matrix
        theta = np.pi/6  # 30 degrees
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        x_rot = R @ x
        return rastrigin(x_rot)
    functions['rotated_rastrigin'] = {
        'func': rotated_rastrigin,
        'bounds': [(-5.12, 5.12)],
        'dim': 2,
        'multimodal': 1,
        'discrete_vars': 0
    }
    
    # Higher dimensional function
    def high_dim_sphere(x):
        """Higher dimensional sphere function"""
        x = np.asarray(x)
        return sphere(x)
    functions['high_dim_sphere'] = {
        'func': high_dim_sphere,
        'bounds': [(-5.12, 5.12)],
        'dim': 10,
        'multimodal': 0,
        'discrete_vars': 0
    }
    
    # Rosenbrock function
    def rosenbrock(x):
        """Rosenbrock function"""
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    functions['rosenbrock'] = {
        'func': rosenbrock,
        'bounds': [(-5.12, 5.12)],
        'dim': 10,
        'multimodal': 1,
        'discrete_vars': 0
    }
    
    # Ackley function
    def ackley(x):
        """Ackley function"""
        x = np.asarray(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
        cos_term = -np.exp(np.sum(np.cos(c * x) / d))
        return sum_sq_term + cos_term + a + np.exp(1)
    functions['ackley'] = {
        'func': ackley,
        'bounds': [(-5.12, 5.12)],
        'dim': 10,
        'multimodal': 1,
        'discrete_vars': 0
    }
    
    # Griewank function
    def griewank(x):
        """Griewank function"""
        x = np.asarray(x)
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_term - prod_term
    functions['griewank'] = {
        'func': griewank,
        'bounds': [(-5.12, 5.12)],
        'dim': 10,
        'multimodal': 1,
        'discrete_vars': 0
    }
    
    return functions

def create_optimizers(dim: int, bounds: List[Tuple[float, float]]):
    return {
        'surrogate': SurrogateOptimizer(
            dim=dim,
            bounds=bounds,
            pop_size=30,
            n_initial=10
        ),
        'de': DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        ),
        'es': EvolutionStrategyOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        ),
        'gwo': GreyWolfOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=30
        )
    }

def run_meta_optimization_test(n_trials: int = 5):
    """Run meta-optimization test across different functions"""
    
    # Create test functions
    functions = create_test_functions()
    
    # Results storage
    results = {name: {'scores': [], 'optimizer_usage': {}} for name in functions}
    
    # Run trials
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        for func_name, func_info in functions.items():
            print(f"\nOptimizing {func_name}")
            
            # Create bounds list
            bounds = func_info['bounds'] * func_info['dim']
            
            # Create optimizers for this dimension
            optimizers = create_optimizers(func_info['dim'], bounds)
            
            # Create meta-optimizer
            meta_opt = MetaOptimizer(optimizers, mode='bayesian')
            
            # Create context
            context = {
                'dim': func_info['dim'],
                'multimodal': func_info['multimodal'],
                'discrete_vars': func_info['discrete_vars']
            }
            
            try:
                # Run optimization
                best_solution = meta_opt.optimize(func_info['func'], context)
                best_score = float(func_info['func'](best_solution))  # Convert to float
                
                # Store results
                results[func_name]['scores'].append(best_score)
                
                # Update optimizer usage
                usage = meta_opt.performance_history['optimizer'].value_counts()
                for opt, count in usage.items():
                    if opt not in results[func_name]['optimizer_usage']:
                        results[func_name]['optimizer_usage'][opt] = []
                    results[func_name]['optimizer_usage'][opt].append(count)
                
                print(f"Best score: {best_score:.2e}")
                print("Optimizer usage:")
                for opt, count in usage.items():
                    print(f"  {opt}: {count}")
            except Exception as e:
                print(f"Error optimizing {func_name}: {str(e)}")
                continue
    
    # Print summary
    print("\nFinal Results:")
    for func_name in functions:
        scores = results[func_name]['scores']
        if scores:  # Only print if we have results
            print(f"\n{func_name}:")
            print(f"  Mean score: {np.mean(scores):.2e} ± {np.std(scores):.2e}")
            print(f"  Best score: {np.min(scores):.2e}")
            print(f"  Worst score: {np.max(scores):.2e}")
            print(f"  Median score: {np.median(scores):.2e}")
            
            # Print average optimizer usage
            print("  Average optimizer usage:")
            for opt in results[func_name]['optimizer_usage']:
                usage = results[func_name]['optimizer_usage'][opt]
                if usage:
                    avg_usage = np.mean(usage)
                    total_evals = sum(usage)
                    print(f"    {opt}: {avg_usage:.1f} calls/trial ({total_evals} total)")
        else:
            print(f"\n{func_name}: No successful trials")

if __name__ == '__main__':
    run_meta_optimization_test(n_trials=5)

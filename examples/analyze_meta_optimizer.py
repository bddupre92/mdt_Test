"""
analyze_meta_optimizer.py
-----------------------
Comprehensive analysis of meta-learning optimizer performance with incremental testing
and detailed analysis of meta-learner effectiveness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import json
import os
from datetime import datetime
from pathlib import Path

from meta.meta_optimizer import MetaOptimizer
from optimizers.ml_optimizers.surrogate_optimizer import SurrogateOptimizer
from optimizers.de import DifferentialEvolutionOptimizer
from optimizers.es import EvolutionStrategyOptimizer
from optimizers.gwo import GreyWolfOptimizer

def create_test_suite():
    """Create test suite with functions grouped by characteristics"""
    def sphere(x):
        x = np.asarray(x)
        return float(np.sum(x**2))
        
    def rastrigin(x):
        x = np.asarray(x)
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
        
    def rosenbrock(x):
        x = np.asarray(x)
        return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))
    
    suites = {
        'unimodal': {
            'sphere': {
                'func': sphere,
                'bounds': [(-5.12, 5.12)],
                'dim': 2,
                'multimodal': 0,
                'discrete_vars': 0,
                'optimal': 0.0
            }
        },
        'multimodal': {
            'rastrigin': {
                'func': rastrigin,
                'bounds': [(-5.12, 5.12)],
                'dim': 2,
                'multimodal': 1,
                'discrete_vars': 0,
                'optimal': 0.0
            }
        },
        'high_dimensional': {
            'rosenbrock': {
                'func': rosenbrock,
                'bounds': [(-2.048, 2.048)],
                'dim': 10,
                'multimodal': 0,
                'discrete_vars': 0,
                'optimal': 0.0
            }
        }
    }
    return suites

def create_optimizers(dim: int, bounds: List[Tuple[float, float]]):
    """Create optimizer instances"""
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

class OptimizationAnalyzer:
    def __init__(self, base_dir: str = 'results/meta_analysis'):
        """Initialize analyzer with results directory"""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create run directory
        self.run_dir = self.base_dir / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single_test(self, func_name: str, func_info: Dict, n_trials: int = 5):
        """Run test for a single function and save results"""
        print(f"\nTesting {func_name}")
        
        # Results storage
        results = {
            'scores': [],
            'optimizer_choices': [],
            'convergence': [],
            'meta_learner_accuracy': [],
            'errors': []
        }
        
        # Create bounds list
        bounds = func_info['bounds'] * func_info['dim']
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}")
            
            try:
                # Create optimizers
                optimizers = create_optimizers(func_info['dim'], bounds)
                
                # Create meta-optimizer
                meta_opt = MetaOptimizer(optimizers, mode='bayesian')
                
                # Create context
                context = {
                    'dim': func_info['dim'],
                    'multimodal': func_info['multimodal'],
                    'discrete_vars': func_info['discrete_vars']
                }
                
                # Run optimization
                best_solution = meta_opt.optimize(func_info['func'], context)
                best_score = float(func_info['func'](np.asarray(best_solution)))
                
                # Store results
                results['scores'].append(best_score)
                results['optimizer_choices'].append(
                    meta_opt.performance_history['optimizer'].tolist()
                )
                results['convergence'].append([
                    float(score) for score in meta_opt.performance_history['score']
                ])
                
                # Analyze meta-learner accuracy
                if len(meta_opt.performance_history) > 1:
                    # Compare predicted vs actual performance
                    predicted_best = meta_opt.select_optimizer(context)
                    actual_scores = meta_opt.performance_history.groupby('optimizer')['score'].mean()
                    actual_best = actual_scores.idxmin()
                    results['meta_learner_accuracy'].append(predicted_best == actual_best)
                
                print(f"    Best score: {best_score:.2e}")
                print("    Optimizer usage:")
                usage = meta_opt.performance_history['optimizer'].value_counts()
                for opt, count in usage.items():
                    print(f"      {opt}: {count}")
                    
            except Exception as e:
                print(f"    Error in trial {trial + 1}: {str(e)}")
                results['errors'].append({
                    'trial': trial + 1,
                    'error': str(e)
                })
                continue
        
        # Save results only if we have some successful trials
        if results['scores']:
            results_file = self.run_dir / f"{func_name}_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'function_info': {
                        'dim': func_info['dim'],
                        'multimodal': func_info['multimodal'],
                        'discrete_vars': func_info['discrete_vars'],
                        'optimal': func_info['optimal']
                    },
                    'results': {
                        'scores': results['scores'],
                        'optimizer_choices': results['optimizer_choices'],
                        'convergence': results['convergence'],
                        'meta_learner_accuracy': results['meta_learner_accuracy'],
                        'errors': results['errors']
                    }
                }, f, indent=2)
            
            # Analyze results if we have any successful trials
            self.analyze_results(func_name, results)
        else:
            print(f"  No successful trials for {func_name}")
        
        return results
    
    def analyze_results(self, func_name: str, results: Dict):
        """Analyze and visualize results for a single function"""
        if not results['scores']:
            print(f"\nNo results to analyze for {func_name}")
            if results['errors']:
                print("\nErrors encountered:")
                for error in results['errors']:
                    print(f"  Trial {error['trial']}: {error['error']}")
            return
            
        print(f"\nAnalyzing {func_name}")
        
        # Create function directory
        func_dir = self.run_dir / func_name
        func_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Performance Statistics
        scores = np.array(results['scores'])
        stats = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'successful_trials': len(scores),
            'failed_trials': len(results['errors'])
        }
        
        print("\nPerformance Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save statistics
        with open(func_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 2. Convergence Plot
        plt.figure(figsize=(10, 6))
        for i, conv in enumerate(results['convergence']):
            plt.plot(conv, alpha=0.5, label=f'Trial {i+1}')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(f'Convergence Plot - {func_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(func_dir / 'convergence.png')
        plt.close()
        
        # 3. Optimizer Usage Analysis
        optimizer_counts = []
        for choices in results['optimizer_choices']:
            counts = pd.Series(choices).value_counts()
            optimizer_counts.append(counts)
        
        usage_df = pd.DataFrame(optimizer_counts).fillna(0)
        
        plt.figure(figsize=(10, 6))
        usage_df.mean().plot(kind='bar')
        plt.title(f'Average Optimizer Usage - {func_name}')
        plt.xlabel('Optimizer')
        plt.ylabel('Average Number of Calls')
        plt.tight_layout()
        plt.savefig(func_dir / 'optimizer_usage.png')
        plt.close()
        
        # 4. Meta-learner Analysis
        if results['meta_learner_accuracy']:
            accuracy = np.mean(results['meta_learner_accuracy'])
            print(f"\nMeta-learner accuracy: {accuracy:.2%}")
            
            with open(func_dir / 'meta_learner_accuracy.json', 'w') as f:
                json.dump({
                    'accuracy': float(accuracy),
                    'predictions': len(results['meta_learner_accuracy'])
                }, f, indent=2)

def main():
    """Run complete analysis"""
    # Create analyzer
    analyzer = OptimizationAnalyzer()
    
    # Get test suites
    suites = create_test_suite()
    
    # Test each suite
    all_results = {}
    for suite_name, suite in suites.items():
        print(f"\nTesting suite: {suite_name}")
        suite_results = {}
        
        for func_name, func_info in suite.items():
            try:
                # Run tests
                results = analyzer.run_single_test(func_name, func_info)
                
                # Analyze results
                analyzer.analyze_results(func_name, results)
                
                suite_results[func_name] = results
                
            except Exception as e:
                print(f"Error testing {func_name}: {str(e)}")
                continue
        
        all_results[suite_name] = suite_results
    
    # Save overall summary
    summary = {
        'timestamp': analyzer.timestamp,
        'suites_tested': list(suites.keys()),
        'functions_tested': sum(len(suite) for suite in suites.values()),
        'successful_tests': sum(
            len(suite_results) for suite_results in all_results.values()
        )
    }
    
    with open(analyzer.run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()

"""
benchmark_runner.py
------------------
Comprehensive benchmarking system for optimization algorithms.
Supports both theoretical and ML-based benchmarks with parallel execution.
"""

import time
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import psutil
import ray
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize

from benchmarking.test_functions import TEST_FUNCTIONS, MLTestFunctions
from optimizers.base import BaseOptimizer
from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategy
from optimizers.de import DifferentialEvolutionOptimizer
from meta.meta_optimizer import MetaOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    optimizer_name: str
    function_name: str
    dimension: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    iterations: int
    evaluations: int
    time_taken: float
    memory_peak: float
    convergence_history: List[float]

class BenchmarkRunner:
    def __init__(
        self,
        optimizers: List[BaseOptimizer],
        test_functions: Dict[str, Any] = None,
        n_runs: int = 30,
        max_evaluations: int = 10000,
        use_ray: bool = False,
        memory_tracking: bool = True
    ):
        self.optimizers = optimizers
        self.test_functions = test_functions or TEST_FUNCTIONS
        self.n_runs = n_runs
        self.max_evaluations = max_evaluations
        self.use_ray = use_ray
        self.memory_tracking = memory_tracking
        
        if use_ray and not ray.is_initialized():
            ray.init()
    
    def _run_single_optimization(
        self,
        optimizer: BaseOptimizer,
        func: Any,
        run_id: int
    ) -> Tuple[float, List[float], int, float]:
        """Execute a single optimization run"""
        if self.memory_tracking:
            process = psutil.Process()
            initial_memory = process.memory_info().rss
        
        start_time = time.time()
        
        # Handle meta-optimizer differently
        if isinstance(optimizer, MetaOptimizer):
            # Create objective function wrapper for meta-optimizer
            def objective_func(x):
                return func(x)
            
            best_solution = optimizer.optimize(
                objective_func,
                max_evals=self.max_evaluations,
                context={'problem_type': func.name if hasattr(func, 'name') else 'unknown'}
            )
            best_fitness = func(best_solution)
            convergence_curve = optimizer.optimization_history
            evaluations = optimizer.total_evaluations
        else:
            best_solution, best_fitness = optimizer.optimize(func)
            convergence_curve = optimizer.convergence_curve
            evaluations = optimizer.evaluations
            
        time_taken = time.time() - start_time
        
        if self.memory_tracking:
            memory_peak = (process.memory_info().rss - initial_memory) / 1024 / 1024
        else:
            memory_peak = 0.0
        
        return (
            best_fitness,
            convergence_curve,
            evaluations,
            memory_peak
        )
    
    @ray.remote
    def _run_single_optimization_ray(self, *args, **kwargs):
        """Ray wrapper for parallel optimization"""
        return self._run_single_optimization(*args, **kwargs)
    
    def run_theoretical_benchmarks(
        self,
        dimensions: List[int] = [2, 10, 30],
        bounds: List[Tuple[float, float]] = [(-5.12, 5.12)]
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks on theoretical test functions"""
        results = {}
        
        for dim in dimensions:
            for func_name, func_factory in self.test_functions.items():
                logger.info(f"Running {func_name} benchmark in {dim} dimensions")
                
                func = func_factory(dim, bounds)
                func_results = []
                
                for optimizer in self.optimizers:
                    optimizer_name = (
                        optimizer.__class__.__name__ 
                        if not hasattr(optimizer, 'name') 
                        else optimizer.name
                    )
                    logger.info(f"Testing {optimizer_name}")
                    
                    # Run multiple times
                    run_results = []
                    for run in range(self.n_runs):
                        optimizer.reset()  # Reset optimizer state
                        result = self._run_single_optimization(
                            optimizer, func, run
                        )
                        run_results.append(result)
                    
                    # Aggregate results
                    best_fitnesses = [r[0] for r in run_results]
                    mean_fitness = np.mean(best_fitnesses)
                    std_fitness = np.std(best_fitnesses)
                    total_evals = sum(r[2] for r in run_results)
                    total_time = sum(r[3] for r in run_results)
                    
                    # Store result
                    benchmark_result = BenchmarkResult(
                        optimizer_name=optimizer_name,
                        function_name=func_name,
                        dimension=dim,
                        best_fitness=min(best_fitnesses),
                        mean_fitness=mean_fitness,
                        std_fitness=std_fitness,
                        iterations=self.n_runs,
                        evaluations=total_evals // self.n_runs,
                        time_taken=total_time / self.n_runs,
                        memory_peak=max(r[3] for r in run_results),
                        convergence_history=run_results[0][1]  # Use first run's history
                    )
                    func_results.append(benchmark_result)
                
                results[f"{func_name}_{dim}d"] = func_results
                
        return results
    
    def run_ml_benchmarks(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_factories: Dict[str, Any],
        param_ranges: Dict[str, Tuple[float, float, type]]
    ) -> pd.DataFrame:
        """Run benchmarks on ML hyperparameter optimization"""
        results = []
        
        for model_name, model_factory in model_factories.items():
            for opt in self.optimizers:
                logger.info(f"Testing {opt.__class__.__name__} on {model_name}")
                
                # Create objective function
                objective = MLTestFunctions.create_sklearn_cv_objective(
                    model_factory(),
                    X_train, y_train,
                    param_ranges
                )
                
                # Run optimization
                if self.use_ray:
                    futures = [
                        self._run_single_optimization_ray.remote(
                            self, opt, objective, i
                        ) for i in range(self.n_runs)
                    ]
                    run_results = ray.get(futures)
                else:
                    with ProcessPoolExecutor() as executor:
                        futures = [
                            executor.submit(
                                self._run_single_optimization,
                                opt, objective, i
                            ) for i in range(self.n_runs)
                        ]
                        run_results = [f.result() for f in futures]
                
                # Aggregate results
                result = BenchmarkResult(
                    optimizer_name=opt.__class__.__name__,
                    function_name=f"{model_name}_optimization",
                    dimension=len(param_ranges),
                    best_fitness=min(r[0] for r in run_results),
                    mean_fitness=np.mean([r[0] for r in run_results]),
                    std_fitness=np.std([r[0] for r in run_results]),
                    iterations=np.mean([len(r[1]) for r in run_results]),
                    evaluations=np.mean([r[2] for r in run_results]),
                    time_taken=np.mean([r[2] for r in run_results]),
                    memory_peak=np.mean([r[3] for r in run_results]),
                    convergence_history=np.mean([r[1] for r in run_results], axis=0).tolist()
                )
                results.append(asdict(result))
        
        return pd.DataFrame(results)
    
    def compare_with_cmaes(
        self,
        dimensions: List[int] = [2, 10, 30],
        bounds: List[Tuple[float, float]] = [(-5.12, 5.12)]
    ) -> pd.DataFrame:
        """Compare optimizers against CMA-ES using pymoo"""
        results = []
        
        for func_name, func_factory in self.test_functions.items():
            for dim in dimensions:
                logger.info(f"Testing CMA-ES on {func_name} ({dim}D)")
                
                test_func = func_factory(dim=dim, bounds=bounds*dim)
                
                # Run CMA-ES
                algorithm = CMAES(
                    x0=np.zeros(dim),
                    sigma=0.5,
                    restarts=2,
                    restart_from_best=True
                )
                
                start_time = time.time()
                res = minimize(
                    test_func,
                    algorithm,
                    ('n_gen', 100),
                    verbose=False
                )
                time_taken = time.time() - start_time
                
                result = BenchmarkResult(
                    optimizer_name='CMA-ES',
                    function_name=func_name,
                    dimension=dim,
                    best_fitness=res.F[0],
                    mean_fitness=np.mean(res.F),
                    std_fitness=np.std(res.F),
                    iterations=res.algorithm.n_gen,
                    evaluations=res.algorithm.evaluator.n_eval,
                    time_taken=time_taken,
                    memory_peak=0.0,  # Not tracked for CMA-ES
                    convergence_history=res.algorithm.callback.data['F'].tolist()
                )
                results.append(asdict(result))
        
        return pd.DataFrame(results)
    
    def save_results(self, results: pd.DataFrame, filename: str):
        """Save benchmark results to file"""
        results.to_csv(filename, index=False)
        
        # Also save convergence plots
        import matplotlib.pyplot as plt
        for func_name in results['function_name'].unique():
            func_results = results[results['function_name'] == func_name]
            plt.figure(figsize=(10, 6))
            
            for opt_name in func_results['optimizer_name'].unique():
                opt_data = func_results[func_results['optimizer_name'] == opt_name]
                history = opt_data['convergence_history'].iloc[0]
                plt.plot(history, label=opt_name)
            
            plt.title(f'Convergence on {func_name}')
            plt.xlabel('Iteration')
            plt.ylabel('Fitness')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{func_name}_convergence.png')
            plt.close()

def run_comprehensive_benchmark(
    data_size: int = 1000,
    use_ray: bool = True,
    output_dir: str = 'benchmark_results'
) -> None:
    """Run comprehensive benchmark suite"""
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from data.generate_synthetic import generate_synthetic_data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic dataset
    df = generate_synthetic_data(num_days=data_size)
    X = df.drop(['migraine_occurred'], axis=1).values
    y = df['migraine_occurred'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Initialize optimizers
    optimizers = [
        AntColonyOptimizer(dim=10, bounds=[(-5.12, 5.12)]*10),
        GreyWolfOptimizer(dim=10, bounds=[(-5.12, 5.12)]*10),
        EvolutionStrategy(dim=10, bounds=[(-5.12, 5.12)]*10),
        DifferentialEvolutionOptimizer(dim=10, bounds=[(-5.12, 5.12)]*10)
    ]
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        optimizers=optimizers,
        n_runs=30,
        max_evaluations=10000,
        use_ray=use_ray
    )
    
    # Run theoretical benchmarks
    theoretical_results = runner.run_theoretical_benchmarks()
    theoretical_results_df = pd.concat([pd.DataFrame(v) for v in theoretical_results.values()], ignore_index=True)
    theoretical_results_df.to_csv(f'{output_dir}/theoretical_results.csv')
    
    # Run ML benchmarks
    model_factories = {
        'RandomForest': lambda: RandomForestClassifier()
    }
    
    param_ranges = {
        'n_estimators': (10, 200, int),
        'max_depth': (3, 20, int),
        'min_samples_split': (2, 20, int),
        'max_features': (0.1, 1.0, float)
    }
    
    ml_results = runner.run_ml_benchmarks(
        X_train, y_train,
        X_test, y_test,
        model_factories,
        param_ranges
    )
    ml_results.to_csv(f'{output_dir}/ml_results.csv')
    
    # Compare with CMA-ES
    cmaes_results = runner.compare_with_cmaes()
    cmaes_results.to_csv(f'{output_dir}/cmaes_comparison.csv')
    
    # Generate summary report
    with open(f'{output_dir}/summary_report.txt', 'w') as f:
        f.write("Benchmark Summary Report\n")
        f.write("======================\n\n")
        
        f.write("1. Theoretical Benchmarks\n")
        f.write("-----------------------\n")
        for func in theoretical_results_df['function_name'].unique():
            f.write(f"\n{func} Function:\n")
            func_data = theoretical_results_df[
                theoretical_results_df['function_name'] == func
            ]
            for _, row in func_data.iterrows():
                f.write(f"  {row['optimizer_name']}:\n")
                f.write(f"    Best: {row['best_fitness']:.6f}\n")
                f.write(f"    Mean: {row['mean_fitness']:.6f}\n")
                f.write(f"    Std:  {row['std_fitness']:.6f}\n")
                f.write(f"    Time: {row['time_taken']:.2f}s\n")
        
        f.write("\n2. ML Benchmarks\n")
        f.write("---------------\n")
        for model in ml_results['function_name'].unique():
            f.write(f"\n{model}:\n")
            model_data = ml_results[
                ml_results['function_name'] == model
            ]
            for _, row in model_data.iterrows():
                f.write(f"  {row['optimizer_name']}:\n")
                f.write(f"    Best CV Score: {-row['best_fitness']:.4f}\n")
                f.write(f"    Mean CV Score: {-row['mean_fitness']:.4f}\n")
                f.write(f"    Time: {row['time_taken']:.2f}s\n")
        
        f.write("\n3. CMA-ES Comparison\n")
        f.write("-------------------\n")
        for func in cmaes_results['function_name'].unique():
            f.write(f"\n{func} Function:\n")
            func_data = cmaes_results[
                cmaes_results['function_name'] == func
            ]
            for _, row in func_data.iterrows():
                f.write(f"  Dimension {row['dimension']}:\n")
                f.write(f"    Best: {row['best_fitness']:.6f}\n")
                f.write(f"    Time: {row['time_taken']:.2f}s\n")

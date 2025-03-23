"""
Benchmark Service Module

This module provides services for running optimization benchmarks and comparing different optimizers.
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .benchmark_repository import BenchmarkRepository, BenchmarkFunction

class BenchmarkResult:
    """Class to store and manage benchmark results."""
    
    def __init__(self, benchmark_id: str, optimizer_id: str, params: Dict[str, Any]):
        """Initialize benchmark result."""
        self.benchmark_id = benchmark_id
        self.optimizer_id = optimizer_id
        self.params = params
        self.start_time = time.time()
        self.end_time = None
        self.runtime = None
        self.function_evaluations = 0
        self.best_fitness = float('inf')
        self.best_position = None
        self.convergence = []
        self.success = False
        self.error = None
    
    def finish(self, success: bool = True, error: Optional[str] = None):
        """Mark the benchmark as finished."""
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time
        self.success = success
        self.error = error
    
    def update(self, fitness: float, position: np.ndarray, evaluations: int):
        """Update benchmark progress."""
        self.function_evaluations = evaluations
        self.convergence.append(float(fitness))
        if fitness < self.best_fitness:
            self.best_fitness = float(fitness)
            self.best_position = position.tolist()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "benchmark_id": self.benchmark_id,
            "optimizer_id": self.optimizer_id,
            "params": self.params,
            "runtime": self.runtime,
            "function_evaluations": self.function_evaluations,
            "best_fitness": self.best_fitness,
            "best_position": self.best_position,
            "convergence": self.convergence,
            "success": self.success,
            "error": self.error,
            "timestamp": datetime.now().isoformat()
        }

class BenchmarkService:
    """Service for running optimization benchmarks."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize benchmark service.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.repository = BenchmarkRepository()
    
    def run_benchmark(self, 
                     optimizer: Any,
                     function_name: str,
                     dimension: int = 2,
                     max_evaluations: int = 1000,
                     bounds: Optional[Tuple[float, float]] = None,
                     num_runs: int = 1) -> List[BenchmarkResult]:
        """Run benchmark for a single optimizer.
        
        Args:
            optimizer: Optimizer instance
            function_name: Name of benchmark function
            dimension: Problem dimension
            max_evaluations: Maximum function evaluations
            bounds: Function bounds (min, max)
            num_runs: Number of independent runs
            
        Returns:
            List of benchmark results
        """
        results = []
        
        for run in range(num_runs):
            # Get benchmark function
            function = self.repository.get_function(function_name, dimension, bounds)
            
            # Create result object
            result = BenchmarkResult(
                benchmark_id=f"{function_name}_{dimension}d_run{run}",
                optimizer_id=optimizer.__class__.__name__,
                params={
                    "dimension": dimension,
                    "max_evaluations": max_evaluations,
                    "bounds": bounds or function.bounds,
                    "run": run
                }
            )
            
            try:
                # Run optimization
                best_position, best_fitness, evaluations = optimizer.optimize(
                    function.evaluate,
                    dimension,
                    bounds or function.bounds,
                    max_evaluations=max_evaluations
                )
                
                # Update result
                result.update(best_fitness, best_position, evaluations)
                result.finish(success=True)
                
            except Exception as e:
                result.finish(success=False, error=str(e))
            
            results.append(result)
            
            # Save result
            self._save_result(result)
        
        return results
    
    def run_comparison(self,
                      optimizers: List[Any],
                      function_names: List[str],
                      dimension: int = 2,
                      max_evaluations: int = 1000,
                      num_runs: int = 10) -> Dict[str, List[BenchmarkResult]]:
        """Run comparison between multiple optimizers.
        
        Args:
            optimizers: List of optimizer instances
            function_names: List of benchmark function names
            dimension: Problem dimension
            max_evaluations: Maximum function evaluations
            num_runs: Number of independent runs per optimizer
            
        Returns:
            Dictionary of results by optimizer
        """
        results = {}
        
        for optimizer in optimizers:
            optimizer_results = []
            for function_name in function_names:
                run_results = self.run_benchmark(
                    optimizer,
                    function_name,
                    dimension,
                    max_evaluations,
                    num_runs=num_runs
                )
                optimizer_results.extend(run_results)
            results[optimizer.__class__.__name__] = optimizer_results
        
        # Save comparison summary
        self._save_comparison_summary(results)
        
        return results
    
    def run_meta_optimizer_comparison(self,
                                   meta_optimizer: Any,
                                   base_optimizers: List[Any],
                                   function_names: List[str],
                                   dimension: int = 2,
                                   max_evaluations: int = 1000,
                                   num_runs: int = 10) -> Dict[str, Any]:
        """Run comparison between meta-optimizer and base optimizers.
        
        Args:
            meta_optimizer: Meta-optimizer instance
            base_optimizers: List of base optimizer instances
            function_names: List of benchmark function names
            dimension: Problem dimension
            max_evaluations: Maximum function evaluations
            num_runs: Number of independent runs
            
        Returns:
            Dictionary containing comparison results and analysis
        """
        # Run meta-optimizer
        meta_results = []
        for function_name in function_names:
            run_results = self.run_benchmark(
                meta_optimizer,
                function_name,
                dimension,
                max_evaluations,
                num_runs=num_runs
            )
            meta_results.extend(run_results)
        
        # Run base optimizers
        base_results = self.run_comparison(
            base_optimizers,
            function_names,
            dimension,
            max_evaluations,
            num_runs
        )
        
        # Analyze results
        analysis = self._analyze_meta_optimizer_results(meta_results, base_results)
        
        # Save results
        results = {
            "meta_optimizer": meta_results,
            "base_optimizers": base_results,
            "analysis": analysis
        }
        self._save_meta_optimizer_results(results)
        
        return results
    
    def _save_result(self, result: BenchmarkResult):
        """Save individual benchmark result."""
        filename = f"{result.benchmark_id}_{result.optimizer_id}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _save_comparison_summary(self, results: Dict[str, List[BenchmarkResult]]):
        """Save comparison summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        for optimizer_name, optimizer_results in results.items():
            summary["results"][optimizer_name] = {
                "num_runs": len(optimizer_results),
                "successful_runs": sum(1 for r in optimizer_results if r.success),
                "mean_fitness": np.mean([r.best_fitness for r in optimizer_results]),
                "std_fitness": np.std([r.best_fitness for r in optimizer_results]),
                "mean_evaluations": np.mean([r.function_evaluations for r in optimizer_results]),
                "mean_runtime": np.mean([r.runtime for r in optimizer_results])
            }
        
        filepath = self.results_dir / f"comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_meta_optimizer_results(self, results: Dict[str, Any]):
        """Save meta-optimizer comparison results."""
        filepath = self.results_dir / f"meta_optimizer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to serializable format
        serializable_results = {
            "meta_optimizer": [r.to_dict() for r in results["meta_optimizer"]],
            "base_optimizers": {
                name: [r.to_dict() for r in opt_results]
                for name, opt_results in results["base_optimizers"].items()
            },
            "analysis": results["analysis"]
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _analyze_meta_optimizer_results(self,
                                     meta_results: List[BenchmarkResult],
                                     base_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Analyze meta-optimizer performance against base optimizers."""
        analysis = {
            "overall_comparison": {},
            "function_specific": {},
            "statistical_tests": {}
        }
        
        # Calculate overall metrics
        meta_fitness = [r.best_fitness for r in meta_results]
        analysis["overall_comparison"]["meta_optimizer"] = {
            "mean_fitness": float(np.mean(meta_fitness)),
            "std_fitness": float(np.std(meta_fitness)),
            "success_rate": sum(1 for r in meta_results if r.success) / len(meta_results)
        }
        
        for opt_name, opt_results in base_results.items():
            opt_fitness = [r.best_fitness for r in opt_results]
            analysis["overall_comparison"][opt_name] = {
                "mean_fitness": float(np.mean(opt_fitness)),
                "std_fitness": float(np.std(opt_fitness)),
                "success_rate": sum(1 for r in opt_results if r.success) / len(opt_results)
            }
        
        # Group results by function
        for result in meta_results:
            func_name = result.benchmark_id.split('_')[0]
            if func_name not in analysis["function_specific"]:
                analysis["function_specific"][func_name] = {
                    "meta_optimizer": [],
                    "base_optimizers": {name: [] for name in base_results.keys()}
                }
            analysis["function_specific"][func_name]["meta_optimizer"].append(result.best_fitness)
        
        for opt_name, opt_results in base_results.items():
            for result in opt_results:
                func_name = result.benchmark_id.split('_')[0]
                analysis["function_specific"][func_name]["base_optimizers"][opt_name].append(result.best_fitness)
        
        # Calculate improvement percentages
        improvements = {}
        for opt_name, opt_results in base_results.items():
            opt_mean = np.mean([r.best_fitness for r in opt_results])
            meta_mean = np.mean([r.best_fitness for r in meta_results])
            improvement = (opt_mean - meta_mean) / opt_mean * 100
            improvements[opt_name] = float(improvement)
        
        analysis["improvements"] = improvements
        
        return analysis 
"""
Performance metrics for evaluating system performance.

This module provides metrics for measuring computational efficiency,
memory usage, scalability, and other performance characteristics of the system.
"""

from typing import Dict, List, Any, Callable, Tuple, Union, Optional
import time
import gc
import os
import platform
import psutil
import numpy as np
from functools import wraps

def computational_efficiency_metrics(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    n_runs: int = 5,
    warmup_runs: int = 1
) -> Dict[str, float]:
    """Measure computational efficiency metrics for a function.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        n_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary of computational efficiency metrics
    """
    if kwargs is None:
        kwargs = {}
    
    # Perform warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Measure execution time
    execution_times = []
    
    for _ in range(n_runs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_times.append(end_time - start_time)
    
    # Calculate metrics
    mean_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    
    # Calculate throughput (operations per second)
    throughput = 1.0 / mean_time if mean_time > 0 else float('inf')
    
    # Calculate coefficient of variation (stability indicator)
    cv = std_time / mean_time if mean_time > 0 else 0.0
    
    return {
        "mean_execution_time": mean_time,
        "std_execution_time": std_time,
        "min_execution_time": min_time,
        "max_execution_time": max_time,
        "throughput": throughput,
        "coefficient_variation": cv,
        "runs": n_runs
    }

def memory_usage_metrics(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    collect_garbage: bool = True
) -> Dict[str, float]:
    """Measure memory usage metrics for a function.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        collect_garbage: Whether to collect garbage before measurement
        
    Returns:
        Dictionary of memory usage metrics
    """
    if kwargs is None:
        kwargs = {}
    
    if collect_garbage:
        gc.collect()
    
    # Get memory usage before
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    
    # Run the function
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    execution_time = time.perf_counter() - start_time
    
    # Get memory usage after
    memory_after = process.memory_info().rss
    
    # Calculate memory metrics
    memory_used = memory_after - memory_before
    memory_efficiency = execution_time / (memory_used / 1024 / 1024) if memory_used > 0 else float('inf')
    
    return {
        "memory_before_mb": memory_before / 1024 / 1024,
        "memory_after_mb": memory_after / 1024 / 1024,
        "memory_used_mb": memory_used / 1024 / 1024,
        "memory_efficiency": memory_efficiency,
        "execution_time": execution_time
    }

def scalability_metrics(
    func: Callable,
    input_sizes: List[int],
    size_to_args: Callable[[int], Tuple],
    size_to_kwargs: Optional[Callable[[int], Dict[str, Any]]] = None,
    n_runs: int = 3
) -> Dict[str, Any]:
    """Measure scalability metrics across different input sizes.
    
    Args:
        func: Function to benchmark
        input_sizes: List of input sizes to test
        size_to_args: Function to convert size to positional arguments
        size_to_kwargs: Function to convert size to keyword arguments
        n_runs: Number of runs per input size
        
    Returns:
        Dictionary of scalability metrics
    """
    if size_to_kwargs is None:
        size_to_kwargs = lambda _: {}
    
    # Collect metrics for each input size
    execution_times = []
    memory_usages = []
    
    for size in input_sizes:
        args = size_to_args(size)
        kwargs = size_to_kwargs(size)
        
        size_execution_times = []
        size_memory_usages = []
        
        for _ in range(n_runs):
            # Measure memory usage
            if hasattr(psutil, "Process"):
                process = psutil.Process(os.getpid())
                gc.collect()
                memory_before = process.memory_info().rss
                
                # Run the function
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                memory_after = process.memory_info().rss
                memory_used = memory_after - memory_before
                size_memory_usages.append(memory_used)
            else:
                # If psutil is not available, skip memory measurements
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                size_memory_usages.append(0)
            
            size_execution_times.append(execution_time)
        
        execution_times.append(np.mean(size_execution_times))
        memory_usages.append(np.mean(size_memory_usages))
    
    # Calculate scaling factors
    time_scaling = []
    memory_scaling = []
    
    for i in range(1, len(input_sizes)):
        size_ratio = input_sizes[i] / input_sizes[i-1] if input_sizes[i-1] > 0 else 1
        time_ratio = execution_times[i] / execution_times[i-1] if execution_times[i-1] > 0 else 1
        memory_ratio = memory_usages[i] / memory_usages[i-1] if memory_usages[i-1] > 0 else 1
        
        time_scaling.append(time_ratio / size_ratio)
        memory_scaling.append(memory_ratio / size_ratio)
    
    # Calculate scalability metrics
    if len(time_scaling) > 0:
        avg_time_scaling = np.mean(time_scaling)
        avg_memory_scaling = np.mean(memory_scaling)
    else:
        avg_time_scaling = 1.0
        avg_memory_scaling = 1.0
    
    # Fit power law: time = a * size^b
    if len(input_sizes) > 1:
        try:
            log_sizes = np.log(input_sizes)
            log_times = np.log(execution_times)
            coeffs = np.polyfit(log_sizes, log_times, 1)
            time_complexity = coeffs[0]
        except Exception:
            time_complexity = 1.0
    else:
        time_complexity = 1.0
    
    return {
        "input_sizes": input_sizes,
        "execution_times": execution_times,
        "memory_usages_mb": [m / 1024 / 1024 for m in memory_usages],
        "time_scaling_factor": avg_time_scaling,
        "memory_scaling_factor": avg_memory_scaling,
        "estimated_time_complexity": time_complexity,
        "scaling_characteristics": "linear" if time_complexity <= 1.1 else (
            "n*log(n)" if time_complexity <= 1.3 else (
                "quadratic" if time_complexity <= 2.2 else "exponential"
            )
        )
    }

def latency_measurements(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    n_runs: int = 100,
    percentiles: List[float] = [50, 90, 95, 99]
) -> Dict[str, float]:
    """Measure detailed latency metrics for a function.
    
    Args:
        func: Function to benchmark
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        n_runs: Number of benchmark runs
        percentiles: Percentiles to calculate
        
    Returns:
        Dictionary of latency metrics
    """
    if kwargs is None:
        kwargs = {}
    
    # Measure execution times
    execution_times = []
    
    for _ in range(n_runs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_times.append(end_time - start_time)
    
    # Calculate basic statistics
    mean_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    
    # Calculate percentiles
    percentile_values = np.percentile(execution_times, percentiles)
    
    # Create metrics dictionary
    metrics = {
        "mean_latency": mean_time,
        "std_latency": std_time,
        "min_latency": min_time,
        "max_latency": max_time,
        "jitter": max_time - min_time,
        "runs": n_runs
    }
    
    # Add percentiles
    for p, value in zip(percentiles, percentile_values):
        metrics[f"p{p}_latency"] = value
    
    return metrics

def throughput_analysis(
    func: Callable,
    args_list: List[Tuple],
    concurrent_loads: List[int],
    n_runs: int = 3,
    time_budget: float = 10.0
) -> Dict[str, Any]:
    """Analyze throughput under different concurrent loads.
    
    Args:
        func: Function to benchmark
        args_list: List of argument tuples for concurrent tasks
        concurrent_loads: List of concurrent load levels to test
        n_runs: Number of runs per load level
        time_budget: Time budget for each run in seconds
        
    Returns:
        Dictionary of throughput metrics
    """
    throughputs = []
    latencies = []
    
    for load in concurrent_loads:
        load_throughputs = []
        load_latencies = []
        
        for _ in range(n_runs):
            start_time = time.perf_counter()
            completed = 0
            
            # Run tasks until time budget is exhausted
            while time.perf_counter() - start_time < time_budget:
                # Process up to 'load' tasks concurrently (simulated)
                batch_start_time = time.perf_counter()
                
                for i in range(min(load, len(args_list))):
                    func(*args_list[i % len(args_list)])
                    completed += 1
                
                batch_end_time = time.perf_counter()
                batch_latency = (batch_end_time - batch_start_time) / load
                load_latencies.append(batch_latency)
            
            elapsed_time = time.perf_counter() - start_time
            throughput = completed / elapsed_time
            load_throughputs.append(throughput)
        
        throughputs.append(np.mean(load_throughputs))
        latencies.append(np.mean(load_latencies))
    
    # Calculate scaling efficiency
    scaling_efficiency = []
    for i in range(1, len(concurrent_loads)):
        load_ratio = concurrent_loads[i] / concurrent_loads[i-1] if concurrent_loads[i-1] > 0 else 1
        throughput_ratio = throughputs[i] / throughputs[i-1] if throughputs[i-1] > 0 else 1
        scaling_efficiency.append(throughput_ratio / load_ratio)
    
    # Calculate peak throughput and optimal load
    peak_throughput = max(throughputs)
    optimal_load = concurrent_loads[throughputs.index(peak_throughput)]
    
    return {
        "concurrent_loads": concurrent_loads,
        "throughputs": throughputs,
        "latencies": latencies,
        "scaling_efficiency": scaling_efficiency,
        "peak_throughput": peak_throughput,
        "optimal_load": optimal_load,
        "saturation_point": optimal_load
    } 
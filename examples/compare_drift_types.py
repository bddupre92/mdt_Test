"""
Compare different drift types on the same benchmark function.

This script visualizes how different types of concept drift affect the same benchmark function
over time, showing the changing optimal values.
"""

import numpy as np
import matplotlib.pyplot as plt
from meta_optimizer.benchmark.dynamic_benchmark import create_dynamic_benchmark
from meta_optimizer.benchmark.test_functions import create_test_suite

# Create test suite
test_suite = create_test_suite()

# Select a benchmark function for comparison
function_name = 'sphere'
dim = 2
bounds = [(-5, 5)] * dim

# Create the test function instance
test_func = test_suite[function_name](dim, bounds)

# Create a wrapper function that calls test_func.evaluate
def func_wrapper(x):
    return test_func.evaluate(x)

# Define drift types to compare
drift_types = ['linear', 'oscillatory', 'sudden', 'incremental', 'random']
drift_rate = 0.05
drift_interval = 50
evaluations = 1000

# Create dynamic benchmarks for each drift type
dynamic_benchmarks = {}
for drift_type in drift_types:
    dynamic_benchmarks[drift_type] = create_dynamic_benchmark(
        base_function=func_wrapper,
        dim=dim,
        bounds=bounds,
        drift_type=drift_type,
        drift_rate=drift_rate,
        drift_interval=drift_interval
    )

# Track optimal values for each drift type
optimal_values = {drift_type: [] for drift_type in drift_types}
evaluation_points = list(range(0, evaluations, 10))  # Sample every 10 evaluations

# Evaluate each dynamic benchmark and track optimal values
for drift_type, benchmark in dynamic_benchmarks.items():
    # Reset the benchmark
    benchmark.reset()
    
    # Evaluate the function multiple times
    for i in range(evaluations):
        # Create a random point
        x = np.random.uniform(-5, 5, size=dim)
        
        # Evaluate
        benchmark.evaluate(x)
        
        # Record optimal value at regular intervals
        if i % 10 == 0:
            optimal_values[drift_type].append(benchmark.current_optimal)

# Plot the results
plt.figure(figsize=(12, 8))

for drift_type in drift_types:
    plt.plot(evaluation_points, optimal_values[drift_type], label=f'{drift_type.capitalize()} Drift')

plt.title(f'Comparison of Drift Types on {function_name.capitalize()} Function', fontsize=16)
plt.xlabel('Evaluation Number', fontsize=14)
plt.ylabel('Optimal Value', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig('results/drift_types_comparison.png', dpi=300)
print(f"Plot saved to results/drift_types_comparison.png")

# Show drift characteristics
print("\nDrift Characteristics:")
for drift_type, benchmark in dynamic_benchmarks.items():
    drift_info = benchmark.get_drift_characteristics()
    print(f"\n{drift_type.capitalize()} Drift:")
    for key, value in drift_info.items():
        if key != 'drift_history':
            print(f"  {key}: {value}") 
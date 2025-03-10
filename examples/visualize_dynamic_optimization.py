"""
Visualize optimization performance on dynamic benchmarks.

This script demonstrates how different optimizers perform on dynamic benchmark functions
with concept drift, showing their ability to track changing optima over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from meta_optimizer.benchmark.dynamic_benchmark import create_dynamic_benchmark
from meta_optimizer.benchmark.test_functions import create_test_suite
from meta_optimizer.optimizers.optimizer_factory import create_optimizers

# Create test suite
test_suite = create_test_suite()

# Select a benchmark function
function_name = 'ackley'
dim = 2
bounds = [(-5, 5)] * dim

# Create the test function instance
test_func = test_suite[function_name](dim, bounds)

# Create a wrapper function that calls test_func.evaluate
def func_wrapper(x):
    return test_func.evaluate(x)

# Create dynamic benchmark
drift_type = 'sudden'
drift_rate = 0.1  # Higher rate for more frequent sudden changes
drift_interval = 20
dynamic_benchmark = create_dynamic_benchmark(
    base_function=func_wrapper,
    dim=dim,
    bounds=bounds,
    drift_type=drift_type,
    drift_rate=drift_rate,
    drift_interval=drift_interval
)

# Create optimizers
optimizers = create_optimizers(dim=dim, bounds=bounds)
selected_optimizers = {
    'DE': optimizers['DE (Standard)'],
    'ES': optimizers['ES (Standard)'],
    'GWO': optimizers['GWO'],
    'ACO': optimizers['ACO']
}

# Parameters for the experiment
max_iterations = 500
reoptimize_interval = 50  # Re-optimize after this many function evaluations
tracking_window = 10  # Number of points to track for each optimizer

# Data structures to store results
optimal_values = []
optimizer_tracks = {name: [] for name in selected_optimizers.keys()}
evaluation_points = []
current_eval = 0

# Reset the benchmark
dynamic_benchmark.reset()

# Function to evaluate with tracking
def evaluate_with_tracking(x):
    global current_eval
    result = dynamic_benchmark.evaluate(x)
    
    # Record data at regular intervals
    if current_eval % 5 == 0:
        evaluation_points.append(current_eval)
        optimal_values.append(dynamic_benchmark.current_optimal)
    
    current_eval += 1
    return result

# Run the experiment
print(f"Running dynamic optimization experiment on {function_name} with {drift_type} drift")
print(f"Dimension: {dim}, Drift rate: {drift_rate}, Drift interval: {drift_interval}")

for iteration in range(max_iterations // reoptimize_interval):
    print(f"\nIteration {iteration+1}/{max_iterations // reoptimize_interval}")
    
    # Run each optimizer for a short period
    for name, optimizer in selected_optimizers.items():
        print(f"  Running {name}...")
        
        # Reset evaluation counter for this run
        optimizer.evaluations = 0
        
        # Run optimization for a short period
        start_time = time.time()
        best_solution, best_score = optimizer.optimize(
            evaluate_with_tracking, 
            max_evals=reoptimize_interval
        )
        end_time = time.time()
        
        # Record the best solution found
        optimizer_tracks[name].append((current_eval, best_score))
        
        print(f"    Best score: {best_score:.6f}, Current optimal: {dynamic_benchmark.current_optimal:.6f}")
        print(f"    Time: {end_time - start_time:.2f}s, Evaluations: {optimizer.evaluations}")

# Plot the results
plt.figure(figsize=(12, 8))

# Plot the changing optimal value
plt.plot(evaluation_points, optimal_values, 'k--', linewidth=2, label='Optimal Value')

# Plot each optimizer's tracking performance
colors = {'DE': 'blue', 'ES': 'green', 'GWO': 'red', 'ACO': 'purple'}
markers = {'DE': 'o', 'ES': 's', 'GWO': '^', 'ACO': 'D'}

for name, track in optimizer_tracks.items():
    x_points = [point[0] for point in track]
    y_points = [point[1] for point in track]
    plt.plot(x_points, y_points, color=colors[name], marker=markers[name], 
             linestyle='-', markersize=8, label=f'{name} Best Score')

plt.title(f'Optimizer Performance on Dynamic {function_name.capitalize()} Function ({drift_type.capitalize()} Drift)', 
          fontsize=16)
plt.xlabel('Function Evaluations', fontsize=14)
plt.ylabel('Function Value', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Save the plot
plt.savefig(f'results/dynamic_optimization_{function_name}_{drift_type}.png', dpi=300)
print(f"\nPlot saved to results/dynamic_optimization_{function_name}_{drift_type}.png")

# Show drift characteristics
drift_info = dynamic_benchmark.get_drift_characteristics()
print("\nDrift Characteristics:")
for key, value in drift_info.items():
    if key != 'drift_history':
        print(f"  {key}: {value}") 
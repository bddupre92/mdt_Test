from meta_optimizer.benchmark.dynamic_benchmark import create_dynamic_benchmark
from benchmarking.test_functions import ClassicalTestFunctions
import matplotlib.pyplot as plt
import numpy as np

# Create a dynamic version of the sphere function
sphere_fn = ClassicalTestFunctions.sphere
dynamic_fn = create_dynamic_benchmark(
    base_function=sphere_fn,
    drift_type='oscillatory',
    drift_rate=0.2,
    drift_interval=10,
    noise_level=0.05,
    dim=2,
    bounds=[(-5, 5)] * 2
)

# Evaluate the function multiple times
scores = []
x = np.array([0.5, 0.5])  # Fixed test point
for i in range(100):
    score = dynamic_fn.evaluate(x)
    scores.append(score)
    if i % 10 == 0:
        print(f'Evaluation {i+1}: score={score:.6f}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(scores)
plt.title('Dynamic Function Evaluation Over Time')
plt.xlabel('Evaluation')
plt.ylabel('Function Value')
plt.grid(True)
plt.savefig('results/dynamic_function_test.png')
print(f'Plot saved to results/dynamic_function_test.png')

# Print drift characteristics
drift_info = dynamic_fn.get_drift_characteristics()
print('\nDrift Characteristics:')
for key, value in drift_info.items():
    if key != 'magnitude_history':
        print(f'  {key}: {value}') 
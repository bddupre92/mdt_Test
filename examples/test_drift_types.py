from meta_optimizer.benchmark.dynamic_benchmark import create_dynamic_benchmark
from benchmarking.test_functions import ClassicalTestFunctions
import matplotlib.pyplot as plt
import numpy as np

# Test point
x = np.array([0.5, 0.5])

# Different drift types to test
drift_types = ['linear', 'oscillatory', 'sudden', 'incremental', 'random']
plt.figure(figsize=(15, 10))

for i, drift_type in enumerate(drift_types):
    # Create a dynamic version of the sphere function with different drift
    dynamic_fn = create_dynamic_benchmark(
        base_function=ClassicalTestFunctions.sphere,
        drift_type=drift_type,
        drift_rate=0.2,
        drift_interval=10 if drift_type in ['sudden', 'incremental'] else 1,
        noise_level=0.05,
        dim=2,
        bounds=[(-5, 5)] * 2
    )
    
    # Evaluate the function multiple times
    scores = []
    for j in range(100):
        score = dynamic_fn.evaluate(x)
        scores.append(score)
    
    # Plot the results
    plt.subplot(len(drift_types), 1, i+1)
    plt.plot(scores)
    plt.title(f'{drift_type.capitalize()} Drift')
    plt.ylabel('Function Value')
    plt.grid(True)
    
    # Print drift characteristics
    drift_info = dynamic_fn.get_drift_characteristics()
    print(f'\n{drift_type.capitalize()} Drift Characteristics:')
    for key, value in drift_info.items():
        if key != 'magnitude_history':
            print(f'  {key}: {value}')

plt.xlabel('Evaluation')
plt.tight_layout()
plt.savefig('results/drift_types_comparison.png')
print('Drift types comparison saved to results/drift_types_comparison.png') 
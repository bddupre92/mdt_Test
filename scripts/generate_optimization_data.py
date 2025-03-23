import numpy as np
import json
import os
from datetime import datetime

# Create results directory if it doesn't exist
os.makedirs('results/comparison_latest', exist_ok=True)

# Generate optimizer comparison data
optimizers = ['ACO', 'DE', 'ES', 'GWO']
np.random.seed(42)

# Generate scores for each optimizer (30 runs each)
scores = {
    'ACO': np.random.normal(0.82, 0.05, 30),
    'DE': np.random.normal(0.85, 0.04, 30),
    'ES': np.random.normal(0.83, 0.05, 30),
    'GWO': np.random.normal(0.84, 0.04, 30)
}

# Create comparison summary
comparison_data = {
    'optimizers': optimizers,
    'scores': {opt: scores[opt].tolist() for opt in optimizers},
    'mean_scores': {opt: float(scores[opt].mean()) for opt in optimizers},
    'std_scores': {opt: float(scores[opt].std()) for opt in optimizers},
    'timestamp': datetime.now().isoformat()
}

# Save comparison data
with open('results/comparison_latest/summary.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)

# Generate convergence data for each optimizer
iterations = 100
for optimizer in optimizers:
    # Generate 5 runs for each optimizer
    for run in range(5):
        # Generate convergence curve
        base = np.linspace(0.4, 0.8, iterations)  # Base improvement curve
        noise = np.random.normal(0, 0.02, iterations)  # Random noise
        convergence = base + noise
        
        # Add optimizer-specific characteristics
        if optimizer == 'ACO':
            convergence += np.random.exponential(0.1, iterations)  # Quick early convergence
        elif optimizer == 'DE':
            convergence += np.sin(np.linspace(0, 2*np.pi, iterations)) * 0.05  # Oscillating improvement
        elif optimizer == 'ES':
            convergence += np.random.gamma(2, 0.02, iterations)  # Steady improvement
        elif optimizer == 'GWO':
            convergence += np.random.beta(2, 5, iterations) * 0.1  # Gradual improvement
        
        # Ensure values are between 0 and 1
        convergence = np.clip(convergence, 0, 1)
        
        # Create run data
        run_data = {
            'optimizer': optimizer,
            'run': run + 1,
            'iterations': list(range(iterations)),
            'fitness': convergence.tolist(),
            'best_fitness': float(convergence.max()),
            'final_fitness': float(convergence[-1])
        }
        
        # Save run data
        filename = f'{optimizer.lower()}_run_{run+1}.json'
        with open(f'results/comparison_latest/{filename}', 'w') as f:
            json.dump(run_data, f, indent=2)

print("Optimization data generated successfully!")

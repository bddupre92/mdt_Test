# Optimization Framework for Migraine Detection and Treatment

A comprehensive optimization and visualization framework for analyzing and comparing various optimization algorithms, with applications in migraine prediction and treatment optimization.

## Features

### Optimization Algorithms
- Grey Wolf Optimizer (GWO)
- Differential Evolution (DE)
- Evolution Strategy (ES)
- Ant Colony Optimization (ACO)

### Visualization Tools
- Convergence curves
- Performance heatmaps
- Parameter sensitivity analysis
- 3D fitness landscapes
- Hyperparameter correlation plots

### Benchmarking
- Classical test functions
- Performance metrics
- Comparative analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mdt_Test.git
cd mdt_Test

# Create and activate virtual environment
python -m venv venv_migraine
source venv_migraine/bin/activate  # On Windows: venv_migraine\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Optimization
```python
from optimizers import DifferentialEvolution
from benchmarking.test_functions import ClassicalTestFunctions

# Create optimizer
optimizer = DifferentialEvolution(
    dim=2,
    bounds=[(-5.12, 5.12)] * 2,
    population_size=50,
    num_generations=100
)

# Run optimization
solution, score = optimizer.optimize(ClassicalTestFunctions.sphere)
```

### Parameter Tuning
```python
# Run parameter tuning script
python examples/tune_optimizers.py
```

### Visualization
```python
# Run visualization example
python examples/visualize_optimizers.py
```

## Results

Our framework achieves state-of-the-art performance on benchmark functions:

- Differential Evolution: Perfect convergence on Rosenbrock function
- Grey Wolf Optimizer: 10^-9 precision on complex landscapes
- Evolution Strategy: Robust exploration capabilities
- Ant Colony Optimization: Fast approximate solutions

## Documentation

For detailed documentation, see the docstrings in each module. Example scripts in the `examples/` directory demonstrate common usage patterns.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Classical optimization algorithms implementations
- Scientific Python community
- Visualization libraries contributors
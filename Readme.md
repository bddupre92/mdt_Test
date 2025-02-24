# Meta-Learning Optimization Framework

A sophisticated meta-learning framework for optimization that combines multiple optimization strategies with intelligent algorithm selection. The framework includes theoretical analysis, comprehensive benchmarking, and advanced performance tracking.

## Features

- **Meta-Learning Optimization**
  - Bayesian optimization with Gaussian Processes
  - Multi-armed bandit strategy
  - Context-aware algorithm selection
  - Adaptive parameter control

- **Optimization Algorithms**
  - Differential Evolution (DE)
  - Evolution Strategy (ES)
  - Grey Wolf Optimizer (GWO)
  - Surrogate-based Optimization

- **Theoretical Analysis**
  - Convergence rate analysis
  - Stability analysis
  - Parameter sensitivity analysis
  - Computational complexity analysis

- **Comprehensive Benchmarking**
  - Standard test functions suite
  - Statistical significance testing
  - Performance visualization
  - Cross-optimizer comparisons

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bddupre92/mdt_Test.git
cd mdt_Test
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Python path:
```bash
export PYTHONPATH=/path/to/mdt_Test:$PYTHONPATH
```

## Quick Start

Run a quick test to verify the framework:
```bash
python examples/quick_test.py
```

This will run a minimal optimization problem and show basic results.

## Usage Examples

### 1. Basic Usage
```python
from meta.meta_optimizer import MetaOptimizer
from optimizers import create_optimizers

# Define objective function
def my_objective(x):
    return np.sum(x**2)

# Create optimizers
optimizers = create_optimizers(dim=10, bounds=[(-5.12, 5.12)])

# Create and run meta-optimizer
meta_opt = MetaOptimizer(optimizers, mode='bayesian')
solution = meta_opt.optimize(
    my_objective,
    context={'dim': 10, 'multimodal': 0}
)
```

### 2. Running Examples

The `examples/` directory contains various scripts for different purposes:

#### Quick Testing
```bash
python examples/quick_test.py  # Fast test of core functionality
```

#### Comprehensive Analysis
```bash
python examples/analyze_meta_optimizer.py  # Detailed performance analysis
python examples/theoretical_comparison.py  # Compare with standalone optimizers
```

#### Optimizer Testing
```bash
python examples/test_meta_optimizer.py     # Test meta-learning framework
python examples/test_surrogate_optimizer.py  # Test surrogate optimization
```

#### Visualization
```bash
python examples/visualize_results.py  # Generate performance plots
```

## Framework Structure

```
mdt_Test/
├── analysis/                 # Theoretical analysis tools
├── benchmarking/            # Benchmark functions and runners
├── data/                    # Data handling utilities
├── drift_detection/         # Concept drift detection
├── examples/                # Example scripts
├── meta/                    # Meta-learning framework
├── models/                  # ML models
└── optimizers/             # Optimization algorithms
```

## Running Tests

1. **Quick Test** (for rapid verification):
```bash
python examples/quick_test.py
```

2. **Full Test Suite** (comprehensive testing):
```bash
python main.py --mode test
```

3. **Benchmark Suite** (performance evaluation):
```bash
python examples/run_benchmarks.py
```

## Example Outputs

The framework generates various outputs in the `results/` directory:

- `results/benchmarks/`: Benchmark results and statistics
- `results/theoretical/`: Theoretical analysis results
- `results/quick_test/`: Quick test results
- `results/visualization/`: Performance plots and graphs

## Advanced Usage

### 1. Custom Optimizer Integration
```python
from optimizers.base_optimizer import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def __init__(self, dim, bounds):
        super().__init__(dim, bounds)
        # Custom initialization

    def optimize(self, objective_func):
        # Implementation
        pass
```

### 2. Theoretical Analysis
```python
from analysis.theoretical_analysis import ConvergenceAnalyzer

analyzer = ConvergenceAnalyzer()
rates = analyzer.analyze_convergence_rate(
    optimizer_name='de',
    dimension=10,
    iterations=[1,2,3...],
    objective_values=[0.1,0.01,...]
)
```

### 3. Custom Benchmark Functions
```python
from benchmarking.test_functions import create_test_suite

def my_function(x):
    return np.sum(x**2) + np.prod(np.abs(x))

suite = create_test_suite()
suite['custom'] = {
    'my_func': {
        'func': my_function,
        'dim': 2,
        'bounds': [(-10, 10)],
        'optimal': 0.0
    }
}
```

## Performance Tracking

The framework automatically tracks:
- Convergence rates
- Algorithm selection patterns
- Runtime performance
- Resource usage

Access tracking data:
```python
meta_opt.performance_history  # pandas DataFrame
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new features
4. Submit a pull request

## Version History

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this framework in your research, please cite:
```bibtex
@software{mdt_test,
  title = {Meta-Learning Optimization Framework},
  author = {Dupre, Blair},
  year = {2025},
  version = {4.0.0}
}
# Meta-Learning Optimization Framework

A sophisticated meta-learning framework for optimization that combines multiple optimization strategies with intelligent algorithm selection. The framework leverages adaptive learning, parallel execution, and comprehensive performance tracking to deliver efficient optimization solutions.

## Core Components

### Meta-Learning Framework (`meta/`)
- **meta_optimizer.py**: Core component that orchestrates optimization strategy selection and execution. Features:
  - Adaptive exploration/exploitation balance
  - Parallel optimizer execution
  - Early stopping criteria
  - Problem-specific learning
  
- **optimization_history.py**: Manages optimization history and enables learning from past runs:
  - JSON-serializable history storage
  - Similar problem identification
  - Performance tracking
  
- **problem_analysis.py**: Analyzes optimization problems to extract key features:
  - Landscape characterization
  - Modality detection
  - Separability analysis
  
- **selection_tracker.py**: Tracks and analyzes optimizer selections:
  - Performance correlation analysis
  - Success rate tracking
  - Feature-based selection learning
  
- **bandit_policy.py**: Implements multi-armed bandit strategies for optimizer selection:
  - Thompson sampling
  - Upper Confidence Bound (UCB)
  - Adaptive exploration rates

### Optimizers (`optimizers/`)
- **Base Optimizers**:
  - **de.py**: Differential Evolution with adaptive parameters
  - **es.py**: Evolution Strategy with self-adaptive step sizes
  - **gwo.py**: Grey Wolf Optimizer with enhanced social learning
  - **aco.py**: Ant Colony Optimization for continuous domains
  
- **ML-Enhanced Optimizers** (`ml_optimizers/`):
  - **surrogate_optimizer.py**: Gaussian Process-based surrogate optimization
  - Support for custom ML model integration

### Analysis Tools (`analysis/`)
- **theoretical_analysis.py**: Tools for analyzing optimizer behavior:
  - Convergence rate analysis
  - Stability analysis
  - Parameter sensitivity studies

### Benchmarking (`benchmarking/`)
- **benchmark_runner.py**: Configurable benchmark execution
- **test_functions.py**: Standard optimization test suite
- **cec_functions.py**: CEC benchmark functions
- **statistical_analysis.py**: Statistical comparison tools
- **sota_comparison.py**: Comparison with state-of-the-art methods

### Drift Detection (`drift_detection/`)
- **adwin.py**: Adaptive windowing for drift detection
- **performance_monitor.py**: Real-time performance tracking
- **ensemble_adapt.py**: Ensemble-based adaptation strategies
- **statistical.py**: Statistical change detection methods

### Data Management (`data/`)
- **preprocessing.py**: Data preparation utilities
- **domain_knowledge.py**: Domain-specific knowledge integration
- **generate_synthetic.py**: Synthetic problem generation

### Models (`models/`)
- **base_model.py**: Base class for ML models
- **sklearn_model.py**: Scikit-learn model integrations
- **torch_model.py**: PyTorch model support
- **tf_model.py**: TensorFlow model support

### Examples and Testing (`examples/`)
- **quick_test.py**: Basic functionality verification
- **theoretical_comparison.py**: Comprehensive optimizer comparison
- **analyze_meta_optimizer.py**: Meta-optimizer analysis
- **visualize_optimizers.py**: Performance visualization tools
- **tune_optimizers.py**: Parameter tuning utilities

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

### Basic Usage
```python
from meta.meta_optimizer import MetaOptimizer
from optimizers import create_optimizers

# Define optimization problem
dim = 30
bounds = [(-100, 100)] * dim

# Create optimizers
optimizers = {
    'de': DifferentialEvolutionOptimizer(dim, bounds),
    'gwo': GreyWolfOptimizer(dim, bounds),
    'surrogate': SurrogateOptimizer(dim, bounds)
}

# Initialize meta-optimizer
meta_opt = MetaOptimizer(
    dim=dim,
    bounds=bounds,
    optimizers=optimizers,
    history_file='optimization_history.json',
    selection_file='selection_history.json',
    n_parallel=2  # Enable parallel execution
)

# Run optimization
solution = meta_opt.optimize(
    objective_func=my_objective,
    max_evals=1000,
    record_history=True,
    context={'problem_type': 'continuous'}
)
```

### Advanced Features

1. **Parallel Optimization**
```python
# Enable parallel execution with 3 concurrent optimizers
meta_opt = MetaOptimizer(..., n_parallel=3)
```

2. **Selection Learning**
```python
# Access optimizer selection statistics
stats = meta_opt.selection_tracker.get_selection_stats()
correlations = meta_opt.selection_tracker.get_feature_correlations()
```

3. **Problem Analysis**
```python
# Analyze problem features
features = meta_opt.analyzer.analyze_features(objective_func)
```

## Framework Workflow

1. **Problem Analysis**
   - Extract problem features
   - Identify similar historical problems
   - Determine initial optimizer selection

2. **Optimization Execution**
   - Parallel optimizer execution
   - Adaptive exploration/exploitation
   - Early stopping when criteria met

3. **Learning and Adaptation**
   - Track optimizer performance
   - Update selection strategies
   - Refine feature correlations

4. **Performance Analysis**
   - Statistical analysis of results
   - Visualization of convergence
   - Comparison with baselines

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new features
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Version

Current version: 4.0.5 - See CHANGELOG.md for version history.
# Adaptive Optimization Framework with Meta-Learning and Drift Detection

A comprehensive framework for optimization, meta-learning, drift detection, and model explainability designed for solving complex optimization problems and adapting to changing environments.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Overview

This framework provides a complete suite of tools for optimization problems with a special focus on adaptivity, explainability, and robustness. The system leverages meta-learning techniques to select the most appropriate optimization algorithm based on problem characteristics and historical performance, while also detecting and adapting to concept drift in the underlying optimization landscape.

## Key Features

### Optimization Components

- **Multiple Optimization Algorithms**: Includes implementations of Differential Evolution, Evolution Strategy, Ant Colony Optimization, and Grey Wolf Optimization
- **Meta-Optimizer**: Automatically selects the best optimizer for a given problem using machine learning techniques
- **Parameter Adaptation**: Algorithms adapt their parameters during optimization to improve performance
- **Robust Error Handling**: Comprehensive validation and error handling to gracefully manage edge cases

### Explainability and Analysis

- **Optimizer Explainability**: Visualizes and explains optimizer behavior, decision processes, and performance
- **Model Explainability**: Supports SHAP, LIME, and Feature Importance explainers for understanding model decisions
- **Drift Explainability**: Tools for explaining detected drift and its impact on model performance
- **Real-time Visualization**: Monitoring optimization progress and performance metrics in real-time
- **Drift Visualization**: Tools for visualizing drift patterns and adaptations

### Framework Infrastructure

- **Modular Design**: Components can be used independently or together
- **Extensible Architecture**: Easy to add new optimizers, explainers, or drift detectors
- **Comprehensive CLI**: Command-line interface for all framework features
- **Robust Testing**: Extensive test suite to ensure reliability

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install meta_optimizer_mdt_test
```

### Option 2: Install from Source

#### Prerequisites

- Python 3.8+
- pip package manager

#### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adaptive-optimization-framework.git
   cd adaptive-optimization-framework
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run optimization with multiple algorithms:

```bash
python main.py --optimize
```

Run meta-learning to select the best optimizer:

```bash
python main.py --meta
```

Evaluate a trained model:

```bash
python main.py --evaluate
```

### Drift Detection

Run drift detection on synthetic data:

```bash
python main.py --drift --drift-window 10 --drift-threshold 0.01 --visualize
```

Run meta-learner with drift detection:

```bash
python main.py --run-meta-learner-with-drift --drift-window 10 --drift-threshold 0.01 --visualize
```

### Explainability

Run explainability analysis on a trained model:

```bash
python main.py --explain --explainer shap --explain-plots
```

Generate explanations for detected drift:

```bash
python main.py --run-meta-learner-with-drift --explain-drift --explainer shap
```

### Real-Time Visualization

Run optimization with real-time visualization:

```bash
python main.py --optimize --live-viz
```

Save visualization results:

```bash
python main.py --optimize --live-viz --save-plots
```

## Explainability Framework

The framework provides multiple explainability methods to understand model behavior:

### SHAP Explainer

SHAP (SHapley Additive exPlanations) provides a unified approach to explaining the output of any machine learning model.

**Supported Plot Types:**
- `summary`: Summary plot of feature importance
- `bar`: Bar plot of feature importance
- `beeswarm`: Beeswarm plot of SHAP values
- `waterfall`: Waterfall plot for a single prediction
- `force`: Force plot for a single prediction
- `decision`: Decision plot for a single prediction
- `dependence`: Dependence plot for a specific feature
- `interaction`: Interaction plot between two features

### LIME Explainer

LIME (Local Interpretable Model-agnostic Explanations) explains predictions by learning an interpretable model locally.

**Supported Plot Types:**
- `local`: Plot explanation for a single instance
- `all_local`: Plot explanations for all instances
- `summary`: Summary of feature importance across all instances

### Feature Importance Explainer

A simple explainer that uses built-in feature importance methods from the model or permutation importance.

**Supported Plot Types:**
- `bar`: Vertical bar plot of feature importance
- `horizontal_bar`: Horizontal bar plot of feature importance
- `heatmap`: Heatmap of feature importance

### Programmatic Usage

```python
from explainability.explainer_factory import ExplainerFactory

# Create explainer
explainer = ExplainerFactory.create_explainer(
    explainer_type='shap',
    model=your_model,
    feature_names=feature_names
)

# Generate explanation
explanation = explainer.explain(X, y)

# Create visualization
fig = explainer.plot(plot_type='summary')
fig.savefig('shap_summary.png')

# Get feature importance
feature_importance = explainer.get_feature_importance()
```

## Real-Time Visualization System

The framework includes a real-time visualization system for monitoring optimization progress:

### Key Features

1. **Optimization Progress**: Shows the best score vs. number of evaluations for each optimizer
2. **Improvement Rate**: Displays how quickly each optimizer is improving
3. **Convergence Speed**: Tracks performance over time (rather than evaluations)
4. **Optimization Statistics**: Shows the best optimizer, best score, and other metrics

### Command-Line Arguments

The following command-line arguments control the visualization:

- `--live-viz`: Enable real-time visualization
- `--save-plots`: Save visualization results to files
- `--max-data-points`: Maximum number of data points to store per optimizer (default: 1000)
- `--no-auto-show`: Disable automatic plot display

### Memory Management

For long-running optimizations, the system includes:

1. **Automatic Downsampling**: Only keeps a fixed number of data points (controlled by `--max-data-points`)
2. **Selective Storage**: Only stores necessary data for visualization
3. **Manual Cleanup**: Call `visualization_manager.cleanup()` to free memory

## Project Structure

```
project/
├── benchmarks/         # Benchmark functions and problems
├── data/               # Data files and datasets
├── docs/               # Documentation
├── drift_detection/    # Drift detection components
├── examples/           # Example scripts
├── explainability/     # Explainability components
├── meta/               # Meta-learning components
├── meta_optimizer/     # Core package
├── models/             # Model definitions
├── optimizers/         # Optimization algorithms
├── results/            # Output and results
├── tests/              # Test suite
├── utils/              # Utility functions
├── visualization/      # Visualization components
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

## Documentation

Comprehensive documentation is available in the `docs` directory:

- [Framework Architecture](./docs/framework_architecture.md) - Overview of system architecture
- [Component Integration](./docs/component_integration.md) - How components work together
- [Command-Line Interface](./docs/command_line_interface.md) - Command-line options
- [Drift Detection Guide](./docs/drift_detection_guide.md) - Guide to drift detection features
- [Explainability Guide](./docs/explainability_guide.md) - Guide to explainability features
- [Model Explainability](./docs/model_explainability.md) - Model explanation features
- [Optimizer Explainability](./docs/optimizer_explainability.md) - Optimizer explanation features
- [Testing Guide](./docs/testing_guide.md) - Guide to testing the framework
- [Examples](./docs/examples.md) - Example usage scenarios

## Advanced Usage

For more advanced usage examples, please refer to the [Examples](./docs/examples.md) documentation or check the `examples/` directory.

### Meta-Learning with Drift Detection

```python
from meta.meta_learner import MetaLearner
from drift_detection.drift_detector import DriftDetector

# Initialize components
detector = DriftDetector(window_size=10, threshold=0.01, alpha=0.3)
meta_learner = MetaLearner(method="bayesian", strategy="bandit", exploration=0.3, drift_detector=detector)

# Initial training
X_train, y_train = get_training_data()
meta_learner.fit(X_train, y_train)

# As new data arrives
while True:
    X_new, y_new = get_new_data()
    
    # Update with drift detection
    # This will automatically detect drift and retrain if necessary
    updated = meta_learner.update(X_new, y_new)
    
    if updated:
        print("Model updated due to drift")
    
    # Make predictions with the updated model
    X_test = get_test_data()
    predictions = meta_learner.predict(X_test)
```

Alternatively, you can use the command-line interface:

```bash
# Run meta-learner with drift detection
python -m meta_optimizer.cli --run-meta-learner-with-drift --drift-window 10 --drift-threshold 0.01 --visualize
```

## Contributing

Contributions are welcome! Please read the [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The optimization algorithms are inspired by various academic papers and open-source implementations
- Explainability components build on top of established libraries like SHAP and LIME
- Thanks to all contributors who have helped shape this framework

## New Modular Structure

The codebase has been restructured in a modular way to improve maintainability and extensibility. The original monolithic `main.py` file has been broken down into the following modules:

```
.
├── cli/                     # Command-line interface
│   ├── argument_parser.py   # Argument parsing
│   ├── commands.py          # Command implementations
│   └── main.py              # CLI entry point
├── core/                    # Core functionality
│   ├── meta_learning.py     # Meta-learning functionality
│   ├── optimization.py      # Optimization functionality
│   ├── evaluation.py        # Evaluation functionality
│   └── drift_detection.py   # Drift detection functionality
├── explainability/          # Explainability functionality
│   ├── model_explainer.py   # Model explainability
│   ├── optimizer_explainer.py # Optimizer explainability
│   └── drift_explainer.py   # Drift explainability
├── migraine/                # Migraine-specific functionality
│   ├── data_import.py       # Migraine data import
│   ├── prediction.py        # Migraine prediction
│   └── explainability.py    # Migraine explainability
└── utils/                   # Utilities
    ├── environment.py       # Environment setup
    ├── json_utils.py        # JSON utilities
    ├── logging_config.py    # Logging configuration
    └── plotting.py          # Plotting utilities
```

## Usage

You can run the framework using the new modular structure:

```bash
python main_v2.py --help
```

Or continue using the original monolithic structure:

```bash
python main.py --help
```

### Example Commands

Running optimization:

```bash
python main_v2.py --optimize --optimizer DE --problem sphere --dimension 10 --max-evaluations 1000 --visualize
```

Comparing optimizers:

```bash
python main_v2.py --compare-optimizers --dimension 5 --max-evaluations 500 --benchmark-repetitions 10
```

Running meta-learning:

```bash
python main_v2.py --meta --meta-method ucb --meta-surrogate gp --visualize
```

Running enhanced meta-learning:

```bash
python main_v2.py --enhanced-meta --use-ml-selection --extract-features --dimension 2 --visualize
```

## Implementation Details

### CLI Structure

The CLI structure is implemented using the Command pattern:

1. `argument_parser.py` defines all available command-line arguments in a modular way
2. `commands.py` defines command classes that encapsulate the execution logic
3. `main.py` parses arguments, sets up the environment, and executes the appropriate command

### Core Functionality

Each core module is responsible for a specific aspect of the framework:

1. `meta_learning.py` handles meta-learning and enhanced meta-learning
2. `optimization.py` handles optimization and optimizer comparison
3. `evaluation.py` handles model evaluation
4. `drift_detection.py` handles drift detection

### Utilities

Utilities are shared across modules:

1. `environment.py` sets up the environment (directories, logging, matplotlib)
2. `json_utils.py` handles JSON serialization with numpy support
3. `logging_config.py` configures logging
4. `plotting.py` handles plotting and visualization

## Contributing

When contributing to this project, please follow the modular structure:

1. Add new functionality to the appropriate module
2. Create new modules for new major features
3. Use utilities for shared functionality
4. Update the CLI when adding new commands

## License

[MIT License](LICENSE)

# Baseline Comparison Framework for Meta Optimizer

This framework provides tools for comparing baseline algorithm selection methods against the Meta Optimizer. It includes implementations of baseline selectors, benchmark functions, and visualization tools.

## Directory Structure

```
.
├── baseline_comparison/          # Main package
│   ├── __init__.py               # Package initialization
│   ├── comparison_runner.py      # Comparison framework
│   ├── benchmark_utils.py        # Benchmark functions
│   ├── visualization.py          # Visualization tools
│   └── baseline_algorithms/      # Baseline algorithm selectors
│       ├── __init__.py
│       └── satzilla_inspired.py  # SATzilla-inspired selector
├── tests/                      # Test directory
│   ├── test_benchmark.py       # Test benchmark script
│   ├── debug_utils.py          # Debugging utilities
│   └── ... (other test files)  # Other test files
├── run_tests.sh                  # Script to run tests
├── make_scripts_executable.sh    # Make scripts executable
└── README.md                     # This file
```

## Getting Started

1. Make the scripts executable:

```bash
./make_scripts_executable.sh
```

2. Run the debugging utilities to check for issues:

```bash
./tests/debug_utils.py
```

3. Run the test benchmark:

```bash
./tests/test_benchmark.py
```

Alternatively, use the run_tests.sh script:

```bash
./run_tests.sh
```

## Components

### Baseline Comparison Framework

The `BaselineComparison` class in `comparison_runner.py` provides the framework for benchmarking baseline algorithm selectors against the Meta Optimizer. It handles running the comparison, collecting results, and generating visualizations.

### SATzilla-inspired Selector

The `SatzillaInspiredSelector` class in `baseline_algorithms/satzilla_inspired.py` implements a feature-based algorithm selector inspired by the SATzilla framework. It extracts features from optimization problems and uses machine learning to predict which algorithm will perform best.

### Benchmark Functions

The `benchmark_utils.py` module provides standard benchmark functions for testing optimization algorithms, including:

- Sphere
- Rosenbrock
- Ackley
- Rastrigin
- Griewank
- Schwefel

It also includes a `DynamicFunction` wrapper for creating dynamic optimization problems with time-varying components.

### Visualization Tools

The `visualization.py` module provides tools for generating various visualizations, including:

- Performance comparison plots
- Algorithm selection frequency plots
- Convergence curves
- Radar charts
- Boxplots
- Heatmaps

## Next Steps

After verifying that the framework works with the test benchmark, you can:

1. Run a full benchmark with more problems
2. Analyze the feature importance
3. Conduct ablation studies
4. Apply the framework to real-world problems

## Troubleshooting

If you encounter issues:

1. Run the debugging utilities to identify the problem
2. Check the import paths and make sure your workspace is in the Python path
3. Verify that the interface between your comparison framework and Meta Optimizer is compatible

## Acknowledgments

This framework is based on the SATzilla algorithm selection approach, adapted for continuous optimization problems.

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
- **Model Explainability**: Supports SHAP, LIME, and Feature Importance for model interpretability
- **Visualization Tools**: Comprehensive visualization suite for analyzing optimization results
- **Performance Analysis**: Tools for benchmarking and comparing optimizers

### Drift Detection and Adaptation

- **Concept Drift Detection**: Monitors and detects changes in the optimization landscape
- **Adaptation Strategies**: Automatically adapts to changing conditions
- **Drift Visualization**: Tools for visualizing drift patterns and adaptations

### Framework Infrastructure

- **Modular Design**: Components can be used independently or together
- **Extensible Architecture**: Easy to add new optimizers, explainers, or drift detectors
- **Comprehensive CLI**: Command-line interface for all framework features
- **Robust Testing**: Extensive test suite to ensure reliability

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

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

## Quick Start

### Basic Optimization

```bash
python main.py --optimize --summary
```

### Meta-Learning

```bash
python main.py --meta --method bayesian --summary
```

### Drift Detection

```bash
python main.py --drift --drift-window 50 --summary
```

### Explainability

```bash
python main.py --explain --explainer shap --explain-plots --summary
```

### Optimizer Explainability

```bash
python main.py --explain-optimizer --optimizer-type differential_evolution --summary
```

## Command-Line Interface

The framework provides a comprehensive command-line interface. For a complete list of options, see:

```bash
python main.py --help
```

For detailed documentation on all command-line options, see [Command-Line Interface Documentation](./docs/command_line_interface.md).

## Project Structure

```
├── benchmarking/       # Tools for benchmarking optimizers
├── command_test_results/ # Test results for CLI commands
├── docs/               # Detailed documentation
├── drift_detection/    # Drift detection algorithms
├── evaluation/         # Framework evaluation tools
├── examples/           # Example usage scripts
├── explainability/     # Explainability tools
├── meta/               # Meta-learning components
├── models/             # ML model implementations
├── optimizers/         # Optimization algorithms
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
meta_learner = MetaLearner(method='bayesian')
drift_detector = DriftDetector(window_size=50)

# Integrate components
meta_learner.add_drift_detector(drift_detector)

# Run optimization with adaptation
results = meta_learner.optimize(objective_function, max_evaluations=1000)
```

### Optimizer Explainability

```python
from optimizers.optimizer_factory import OptimizerFactory
from explainability.optimizer_explainer import OptimizerExplainer

# Create optimizer
factory = OptimizerFactory()
optimizer = factory.create_optimizer('differential_evolution')

# Run optimization
optimizer.run(objective_function)

# Create explainer
explainer = OptimizerExplainer(optimizer)

# Generate explanations
explanation = explainer.explain()
explainer.plot('convergence')
explainer.plot('parameter_adaptation')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
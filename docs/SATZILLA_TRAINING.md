# SATzilla-inspired Algorithm Selector Training

This document describes how to train the SATzilla-inspired algorithm selector for the Meta Optimizer framework.

## Overview

The SATzilla-inspired algorithm selector uses machine learning to predict which optimization algorithm will perform best for a given problem based on problem features. The training pipeline includes:

1. **Problem Generation**: Creating diverse training problems for the selector to learn from
2. **Feature Extraction**: Extracting informative features from optimization problems
3. **Algorithm Performance Collection**: Running optimization algorithms on problems to collect performance data
4. **Model Training**: Training regression models to predict algorithm performance based on problem features
5. **Feature Analysis**: Analyzing feature importance and correlation with algorithm performance

## Command-Line Interface

The training pipeline can be run through the command-line interface using the `train_satzilla` command:

```bash
python main_v2.py train_satzilla [OPTIONS]
```

### Options

- `--dimensions`, `-d`: Number of dimensions for benchmark functions (default: 2)
- `--max-evaluations`, `-e`: Maximum number of function evaluations per algorithm (default: 1000)
- `--num-problems`, `-p`: Number of training problems to generate (default: 20)
- `--functions`, `-f`: Benchmark functions to use for training (default: sphere rosenbrock rastrigin ackley griewank)
- `--all-functions`: Use all available benchmark functions
- `--output-dir`, `-o`: Output directory for training results (default: results/satzilla_training)
- `--seed`, `-s`: Random seed for reproducibility (default: 42)
- `--visualize-features`: Generate feature importance visualizations
- `--timestamp-dir`: Create a timestamped subdirectory for results (default: true)

### Helper Script

A helper script is provided to run the training pipeline with common options:

```bash
./scripts/train_satzilla.sh [OPTIONS]
```

Run `./scripts/train_satzilla.sh --help` for more information.

## Programmatic Usage

The training pipeline can also be used programmatically. Here's a basic example:

```python
from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
from baseline_comparison.benchmark_utils import get_benchmark_function
from baseline_comparison.training import train_selector, feature_analysis

# Get benchmark functions
functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank']
benchmark_functions = [
    get_benchmark_function(name, 2)  # 2D problems
    for name in functions
]

# Initialize selector
selector = SatzillaInspiredSelector()

# Generate problem variations
training_problems = train_selector.generate_problem_variations(
    benchmark_functions,
    num_problems=20,
    dimensions=2,
    random_seed=42
)

# Train the selector
trained_selector = train_selector.train_satzilla_selector(
    selector,
    training_problems,
    max_evaluations=1000
)

# Save the trained selector
train_selector.save_trained_selector(trained_selector, "models/satzilla_selector.pkl")

# Analyze feature importance
importance = feature_analysis.analyze_feature_importance(trained_selector)

# Test the trained selector
test_problem = get_benchmark_function('schwefel', 2)
selected_algorithm = trained_selector.select_algorithm(test_problem)
```

For a complete example, see `examples/train_satzilla_demo.py`.

## Training Output

The training pipeline produces the following outputs:

- **Trained Selector**: Saved as a pickle file in the `models` subdirectory
- **Problem Features**: Exported as a CSV file in the `data` subdirectory
- **Algorithm Performance**: Exported as a CSV file in the `data` subdirectory
- **Model Metrics**: JSON file with model evaluation metrics in the `data` subdirectory
- **Feature Importance Visualizations**: PNG files in the `visualizations` subdirectory
- **Training Summary**: Text file with training information in the output directory

## Feature Analysis

The training pipeline includes several tools for analyzing problem features:

1. **Feature Importance**: Visualizes which problem features are most important for algorithm selection
2. **Feature Correlation**: Analyzes correlation between problem features
3. **PCA Analysis**: Applies Principal Component Analysis to explore feature relationships
4. **Feature-Performance Correlation**: Analyzes correlation between features and algorithm performance
5. **Feature Subset Analysis**: Evaluates the impact of removing features on performance

These analyses help understand which problem characteristics are most important for effective algorithm selection. 
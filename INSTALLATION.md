# Installation Guide

This document explains how to install and use both the Meta-Optimizer framework and the Migraine Prediction package together.

## Installation Options

There are two main packages in this project:

1. **Meta-Optimizer Framework**: The core optimization framework with meta-learning, explainability, and drift detection
2. **Migraine Prediction Model**: A specialized interface for migraine prediction built on top of the Meta-Optimizer framework

You can install each package independently or together, depending on your needs.

## Installing Both Packages

For the best experience with migraine prediction, we recommend installing both packages. The migraine prediction package will automatically install the meta-optimizer as a dependency.

### Option 1: Install from Local Directory (Development)

1. First, install the meta-optimizer package in development mode:
   ```bash
   # From the root directory (mdt_Test)
   pip install -e .
   ```

2. Then, install the migraine prediction package:
   ```bash
   # From the root directory (mdt_Test)
   pip install -e ./migraine_prediction_project/
   ```

### Option 2: Install from GitHub/PyPI (Once Published)

If the packages are published to PyPI or GitHub, you can install them directly:

```bash
# Install meta-optimizer framework
pip install meta-optimizer

# Install migraine prediction package 
# (this will automatically install meta-optimizer as a dependency)
pip install migraine-prediction
```

## Verifying Installation

After installation, you should be able to access both packages:

```python
# Import from meta-optimizer
from meta_optimizer.meta.meta_optimizer import MetaOptimizer
from meta_optimizer.optimizers.optimizer_factory import DifferentialEvolutionOptimizer

# Import from migraine prediction
from migraine_model.migraine_predictor import MigrainePredictor
```

## Using the Command-Line Interfaces

### Meta-Optimizer CLI

The meta-optimizer package provides a command-line interface:

```bash
# Display help
meta-optimizer --help

# Run optimization
meta-optimizer --optimize --dim 30 --max-evals 10000

# Run with meta-learning
meta-optimizer --meta --method bayesian
```

### Migraine Prediction CLI

The migraine prediction package provides its own CLI focused on migraine prediction:

```bash
# Display help
migraine-predict --help

# Generate synthetic data
migraine-predict generate --num-samples 1000 --output-train train.csv --output-test test.csv

# Train a model
migraine-predict train --data train.csv --model-name "my_model"

# Train with optimization
migraine-predict optimize --data train.csv --optimizer meta --max-evals 500

# Make predictions
migraine-predict predict --data test.csv --detailed
```

## Troubleshooting

If you encounter any issues with imports, ensure that both packages are properly installed by checking:

```bash
pip list | grep meta-optimizer
pip list | grep migraine-prediction
```

If either package is missing, follow the installation instructions above.

If you see warnings about missing components, it may indicate that some optional dependencies are not installed. You can install them with:

```bash
# For meta-optimizer
pip install meta-optimizer[explainability,optimization]

# For migraine prediction
pip install migraine-prediction[explainability,optimization]
```

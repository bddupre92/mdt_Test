# Migraine Prediction Project

A machine learning model for predicting migraine occurrences based on various health and environmental factors, built on top of the Meta-Optimizer framework.

## Overview

The Migraine Prediction Project provides a modular, reusable machine learning framework designed to predict migraine occurrences using various health metrics and environmental factors. The project uses the Meta-Optimizer framework to leverage advanced optimization techniques for finding the best predictive model for migraine prediction.

## Relationship with Meta-Optimizer Framework

This package is built on top of the **Meta-Optimizer Framework** and depends on it for the following features:

- **Meta-Optimization**: Selecting the best optimization algorithm for training migraine prediction models
- **Explainability Tools**: Explaining model predictions and optimizer behavior
- **Advanced Optimizers**: Differential Evolution, Evolution Strategy, Grey Wolf, Ant Colony, etc.

The Meta-Optimizer Framework is automatically installed as a dependency when you install this package.

## Installation

### Installing Both Packages

For the full feature set, install both packages:

```bash
# First, install the meta-optimizer package
pip install -e /path/to/meta-optimizer

# Then install the migraine prediction package
pip install -e /path/to/migraine_prediction_project
```

The migraine prediction package will automatically try to use the Meta-Optimizer components if available, or gracefully fall back to simpler implementations if needed.

For more detailed installation instructions, see the [INSTALLATION.md](../INSTALLATION.md) in the parent project.

## Features

- Migraine prediction based on health and environmental factors
- Data ingestion pipeline for loading and processing migraine data
- Modular model architecture with meta-optimization
- Model explainability features to understand prediction factors
- Command-line interface for training, prediction, and evaluation
- Support for model serialization and reuse

## Project Structure

```
migraine_prediction_project/
├── models/                  # Saved models
├── src/
│   ├── migraine_model/      # Core migraine prediction package
│   │   ├── __init__.py
│   │   ├── migraine_predictor.py  # Main interface for predictions
│   │   └── model.py         # Model management utilities
│   ├── pipeline/            # Data pipelines
│   │   └── data_ingestion.py  # Data loading and preprocessing
│   └── main.py              # CLI entry point
├── test_data/               # Test data for development and validation
├── explanations/            # Generated explainability artifacts
└── requirements.txt         # Project dependencies
```

## Installation

You can install this package directly from PyPI, from source, or in development mode:

### From PyPI (once published)

```bash
# Install the base package
pip install migraine-prediction

# Install with explainability extras
pip install migraine-prediction[explainability]

# Install with optimization extras
pip install migraine-prediction[optimization]

# Install with all extras
pip install migraine-prediction[explainability,optimization,dev]
```

### From Source Distribution

First, create a source distribution:

```bash
# Navigate to the project directory
cd migraine_prediction_project

# Create a source distribution
python setup.py sdist

# Install from the source distribution
pip install dist/migraine_prediction-0.1.0.tar.gz
```

### Development Mode

```bash
# Clone the repository
git clone <repository-url>
cd migraine_prediction_project

# Install in development mode
pip install -e .

# Install with explainability extras (optional)
pip install -e ".[explainability]"
```

### Direct Installation from Repository

```bash
# Install directly from repository
pip install git+<repository-url>
```

## Usage

### Using as a Python Module

```python
import pandas as pd
from migraine_model import MigrainePredictor

# Create a predictor
predictor = MigrainePredictor()

# Load data
data = pd.read_csv("your_data.csv")

# Train a model
model_id = predictor.train(data, model_name="my_model", description="My migraine model")
print(f"Model trained with ID: {model_id}")

# Make predictions
predictions = predictor.predict(data)
print(f"Predictions: {predictions[:5]}")

# Evaluate the model
metrics = predictor.evaluate(data)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Using the Command-Line Interface

The package provides a command-line interface with the following commands:

#### Training a Model

```bash
python -m migraine_model.cli train --data path/to/data.csv --model-name "my_model" --description "My migraine model" --summary
```

#### Training with Meta-Optimization

```bash
python -m migraine_model.cli optimize --data path/to/data.csv --model-name "optimized_model" --optimizer meta --max-evals 500 --summary
```

#### Making Predictions

```bash
python -m migraine_model.cli predict --data path/to/data.csv --output predictions.csv
```

#### Getting Model Explanations

```bash
python -m migraine_model.cli explain --data path/to/data.csv --explainer feature_importance --explain-plots --summary
```

#### Listing Available Models

```bash
python -m migraine_model.cli list
```

#### Exporting a Model

```bash
python -m migraine_model.cli export --output model.pkl
```

#### Loading and Combining Test Data

```bash
python -m migraine_model.cli load --data-dir ./test_data --output combined_test_data.csv
```

## Integration with Main MDT Project

This package is designed to work with the main MDT optimization framework. When installed alongside the main project, it will automatically leverage the available optimizers and explainability components.

### Using MDT Optimizers

If the MDT project's MetaOptimizer is available, the migraine predictor will automatically use it for training. Otherwise, it falls back to a standard scikit-learn implementation.

### Using Explainability Features

The package can use the explainability components from the main MDT project. If these components are not available, it falls back to a simpler feature importance implementation.

## Data Format

The migraine prediction model expects data with the following features:

- `sleep_hours`: Hours of sleep
- `stress_level`: Subjective stress level (0-10)
- `weather_pressure`: Atmospheric pressure
- `heart_rate`: Heart rate (BPM)
- `hormonal_level`: Hormone level measurement

Target variable:
- `migraine_occurred`: Binary indicator of migraine occurrence (0 or 1)

Optional columns:
- `patient_id`: Identifier for the patient
- `date`: Date/time of the measurement

## Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- PyYAML
- SHAP (for explainability)
- LIME (for explainability)

## License

[MIT License](LICENSE)

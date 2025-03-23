# Migraine Prediction Example Project

This is an example project demonstrating how to use the migraine prediction package.

## Installation

First, install the migraine prediction package from the source distribution:

```bash
# Install from the source distribution
pip install ../migraine_prediction_project/dist/migraine_prediction-0.1.0.tar.gz

# Or install with explainability extras
pip install "../migraine_prediction_project/dist/migraine_prediction-0.1.0.tar.gz[explainability]"
```

## Usage

Run the sample script:

```bash
python sample_usage.py
```

This script demonstrates:
1. Creating sample data
2. Training a migraine prediction model
3. Evaluating the model performance
4. Making predictions with the model
5. Getting detailed prediction information

## Command-line Usage

You can also use the migraine prediction CLI directly:

```bash
# Train a model on the generated data
migraine-predict train --data train_data.csv --model-name "cli_model" --description "Model trained via CLI" --summary

# Make predictions
migraine-predict predict --data test_data.csv --output predictions.csv --detailed

# List available models
migraine-predict list

# Train with optimization (if MDT optimizers are available)
migraine-predict optimize --data train_data.csv --model-name "optimized_model" --optimizer meta --max-evals 100 --summary
```

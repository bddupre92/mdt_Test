# Migraine Data Import and Prediction Guide

This guide documents how to use the enhanced migraine prediction system that supports importing new data with different schemas, derived features, and handling missing values during prediction.

## Overview

The system has been enhanced to:
1. Import data from various file formats
2. Handle new columns automatically
3. Create derived features using custom formulas
4. Train models with new data
5. Make predictions even when some features are missing

## Components

The main components of the system include:

1. **DataHandler** - Manages data importing, schema evolution, and derived features.
2. **MigrainePredictorV2** - An enhanced migraine predictor that integrates with the DataHandler.
3. **Command-line Interface** - Main.py has been updated to support migraine data operations.

## Usage Examples

### Importing New Data with Different Schema

```bash
python main.py --import-migraine-data --data-path /path/to/new_data.csv --add-new-columns --data-dir data --model-dir models --summary
```

This will:
- Import data from the specified CSV file
- Add any new columns found to the schema
- Print a summary of the imported data

### Adding Derived Features

```bash
python main.py --import-migraine-data --data-path data/migraine_data.csv --derived-features "stress_sleep_ratio:df['stress_level']/df['sleep_hours']" "active_sleep:df['activity_minutes']/df['sleep_hours']" --summary
```

This will:
- Import data from the CSV file
- Create two derived features based on the formulas provided
- Print a summary including the new derived features

### Training a Model with New Data

```bash
python main.py --import-migraine-data --data-path data/migraine_data.csv --add-new-columns --train-model --model-name "enhanced_model" --model-description "Model trained with new features" --make-default --summary
```

This will:
- Import data from the CSV file
- Add new columns to the schema
- Train a model using all available features
- Set the new model as the default
- Print a summary of the training

### Making Predictions with Missing Features

```bash
python main.py --predict-migraine --prediction-data data/patient_data.csv --model-id <model_id> --save-predictions --summary
```

This will:
- Import data from the prediction CSV file
- Make predictions using the specified model
- Handle any missing features automatically
- Save the predictions to a CSV file
- Print a summary of the predictions

## Data Formats Supported

The system supports the following data formats:
- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- Parquet (.parquet)

## Schema Evolution

When importing new data, you can decide whether to add new columns to the schema or ignore them:
- With `--add-new-columns`: New columns are added to the schema and can be used for training/prediction
- Without: New columns are ignored, only existing columns are used

## Derived Features

Derived features are calculated on the fly during data import, using Python expressions that operate on the DataFrame. Example formulas:

- `df['feature_a'] / df['feature_b']` - Ratio of two features
- `df['feature_a'] * 2` - Scaling a feature
- `df['feature_a'].rolling(window=3).mean()` - Rolling average
- `np.log(df['feature_a'] + 1)` - Logarithmic transformation

## Handling Missing Values

When making predictions with incomplete data:
1. The system identifies missing features
2. Default values are used (calculated from training data)
3. Predictions are made using the available and default values
4. Feature importances help understand which features influenced the prediction

## Example Scripts

For detailed examples, refer to the following scripts:
- `examples/new_schema_data_example.py` - Demonstrates handling new data schema
- `examples/import_new_data_example.py` - Shows how to import various data formats

## Best Practices

1. Always start with core features for initial training
2. Add new features incrementally and validate their impact
3. Create derived features that have domain relevance
4. Use the `--summary` flag to get insights into the data and model
5. Regularly evaluate model performance with new features

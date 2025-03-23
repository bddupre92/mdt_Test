# Guide to Importing New Migraine Data

This guide explains how to import new migraine data into the MDT migraine prediction system, especially when the data contains new columns or different schema.

## Table of Contents
1. [Understanding Data Schema Evolution](#understanding-data-schema-evolution)
2. [Basic Data Import](#basic-data-import)
3. [Handling New Columns](#handling-new-columns)
4. [Creating Derived Features](#creating-derived-features)
5. [Handling Missing Features](#handling-missing-features)
6. [Advanced Schema Management](#advanced-schema-management)

## Understanding Data Schema Evolution

The migraine prediction system now has enhanced capabilities to handle changes in data schema over time. This is particularly useful for:

- Incorporating new source data with additional features
- Handling missing columns in new data
- Creating derived features from existing ones
- Managing different data transformations

The system maintains a schema file that tracks:
- Core features that are required for predictions
- Optional features that can enhance the model
- Derived features calculated from other features
- Transformations applied to specific columns
- Schema version history

## Basic Data Import

To import new migraine data into the system, you can use the enhanced migraine predictor:

```python
from migraine_model.migraine_predictor_extension import EnhancedMigrainePredictor

# Initialize the predictor
predictor = EnhancedMigrainePredictor(
    model_dir="models",
    data_dir="data"
)

# Import data from a CSV, Excel, or other supported format
data = predictor.import_data("path/to/your/data.csv")

# Train a model using the imported data
model_id = predictor.train(
    data=data,
    model_name="my_model",
    description="Model trained with imported data"
)
```

The `import_data` method supports various file formats:
- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- Parquet (.parquet)

## Handling New Columns

If your new data contains additional columns that weren't in the original training data, you can add them to the schema:

```python
# Import data and add new columns to the schema
data = predictor.import_data(
    data_path="path/to/new_data.csv",
    add_new_columns=True  # This flag adds new columns to the schema
)

# Train a new model with the enhanced data
model_id = predictor.train(
    data=data,
    model_name="enhanced_model",
    description="Model with new features"
)
```

Alternatively, you can use the combined method:

```python
# Import new data and train in one step
model_id = predictor.train_with_new_data(
    data_path="path/to/new_data.csv",
    model_name="enhanced_model",
    add_new_columns=True
)
```

## Creating Derived Features

You can create new features derived from existing ones:

```python
# Add a derived feature
predictor.add_derived_feature(
    name="stress_sleep_ratio",
    formula="df['stress_level'] / df['sleep_hours']"
)

# Train a model that will use this derived feature
model_id = predictor.train(data, model_name="model_with_derived_features")
```

The formula is a Python expression that uses:
- `df` to refer to the DataFrame
- NumPy functions via `np`
- Standard Python operations

Examples of useful derived features:
- Ratios: `"df['feature1'] / df['feature2']"`
- Moving averages: `"df['feature'].rolling(window=3).mean()"`
- Polynomial features: `"df['feature'] ** 2"`
- Logarithmic transformations: `"np.log1p(df['feature'])"`

## Handling Missing Features

The system can handle missing features during prediction:

```python
# Create data with some missing features
test_data = pd.DataFrame({
    'sleep_hours': [6.5],
    'stress_level': [8],
    # 'weather_pressure' is missing
    'heart_rate': [72],
    'hormonal_level': [4.5]
})

# Make predictions despite missing features
predictions = predictor.predict_with_missing_features(test_data)
```

The system will use default values (usually means from the training data) for missing features.

## Advanced Schema Management

You can view and manage the current schema:

```python
# Get schema information
schema_info = predictor.get_schema_info()
print(schema_info)

# Add a transformation for a column
predictor.add_transformation(
    column="stress_level",
    transform_type="log"  # Options: "log", "sqrt", "standard"
)

# Export schema to a file
predictor.data_handler.export_schema("my_schema.json")

# Import schema from a file
predictor.data_handler.import_schema("my_schema.json")
```

### Schema Versioning

The system automatically tracks schema changes:
- Each change increments the schema version
- Changes are recorded in the schema history
- Previous schema versions are backed up

This ensures reproducibility and traceability of model training.

## Command-Line Usage

For command-line usage, you can extend the main script with additional arguments:

```bash
# Import new data with new columns
python main.py --import-data path/to/data.csv --add-new-columns

# Train a model using the imported data
python main.py --train-with-new-data path/to/data.csv --model-name "new_model" --add-new-columns
```

## Conclusion

With these enhanced capabilities, you can now incorporate new migraine data with different schemas into your prediction system. The system handles schema evolution gracefully, allowing you to track changes and maintain model reproducibility.

For more advanced usage, refer to the API documentation in the code.

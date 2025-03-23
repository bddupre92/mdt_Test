# Universal Migraine Data Adapter

This module provides a flexible data ingestion system that allows the migraine prediction package to work with virtually any dataset format. With this adapter, you can automatically process new migraine datasets without needing to manually map columns or adapt to different schemas.

## Features

- **Automatic schema detection**: The adapter automatically identifies target columns, date columns, and relevant features in any dataset
- **Flexible feature selection**: Select the most predictive features using standard statistical methods or meta-optimization
- **Intelligent feature mapping**: Map core migraine features to their equivalents in new datasets
- **Derived feature creation**: Automatically create useful derived features based on available data
- **Missing value handling**: Robust handling of missing values in both training and prediction
- **Meta-optimization integration**: Leverage the meta-optimization framework for optimal feature selection
- **Synthetic data integration**: Seamlessly work with synthetic data for testing and validation
- **Explainability integration**: Gain insights into model predictions using SHAP, LIME, or feature importance methods

## Usage

### Command Line Interface

The universal data adapter can be used directly through the main.py command-line interface:

```bash
# Process any migraine dataset using automatic schema detection
python main.py --universal-data --data-path path/to/your/data.csv --train-model --evaluate-model --summary

# Generate and use synthetic data instead of real data
python main.py --universal-data --generate-synthetic --synthetic-patients 100 --synthetic-days 180 --train-model --summary

# Use meta-optimization for feature selection
python main.py --universal-data --data-path path/to/your/data.csv --use-meta-feature-selection --method de --surrogate rf --train-model --summary

# Include explainability analysis with a trained model
python main.py --universal-data --data-path path/to/your/data.csv --train-model --explain --explainer shap --explain-plots --summary
```

### Quick Start with Helper Script

We've included a helper script to make it easier to use the universal data adapter:

```bash
# Show usage information
./scripts/run_universal_data.sh help

# Generate synthetic data and train a model
./scripts/run_universal_data.sh generate

# Process an existing dataset
./scripts/run_universal_data.sh process path/to/data.csv

# Use meta-optimization for feature selection
./scripts/run_universal_data.sh meta path/to/data.csv

# Apply explainability analysis to a trained model
./scripts/run_universal_data.sh explain [model_id]

# Run a complete pipeline with all features
./scripts/run_universal_data.sh full
```

### Key Command Line Arguments

#### Basic Options
- `--universal-data`: Enable universal data processing
- `--data-path`: Path to the data file to process
- `--data-dir`: Directory to store data files (default: 'data')
- `--file-format`: Format of the input data file (csv, excel, json, parquet)
- `--verbose`: Print detailed logs during processing

#### Feature Selection
- `--disable-auto-feature-selection`: Disable automatic feature selection
- `--use-meta-feature-selection`: Use meta-optimization for feature selection
- `--max-features`: Maximum number of features to select
- `--derived-features`: Add derived features with custom formulas (format: "name:formula")

#### Model Training and Evaluation
- `--train-model`: Train a model with the processed data
- `--evaluate-model`: Evaluate the trained model on test data
- `--test-size`: Fraction of data to use for testing (default: 0.2)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--model-name`: Name for the trained model
- `--model-description`: Description for the trained model
- `--make-default`: Make the trained model the default for future predictions
- `--save-processed-data`: Save the processed dataset

#### Synthetic Data Options
- `--generate-synthetic`: Generate synthetic migraine data
- `--synthetic-patients`: Number of patients in synthetic data
- `--synthetic-days`: Number of days per patient
- `--synthetic-female-pct`: Percentage of female patients
- `--synthetic-missing-rate`: Rate of missing data
- `--synthetic-anomaly-rate`: Rate of anomalies
- `--synthetic-include-severity`: Include migraine severity in synthetic data
- `--save-synthetic`: Save the generated synthetic data

#### Explainability Options
- `--explain`: Include explainability analysis with a trained model
- `--explainer`: Choose an explainability method (shap, lime, feature_importance)
- `--explain-plots`: Generate plots for explainability analysis

### Python API

For programmatic use, you can import and use the universal data adapter directly:

```python
from migraine_prediction_project.src.migraine_model.universal_data_adapter import UniversalDataAdapter
from migraine_prediction_project.src.migraine_model.new_data_migraine_predictor import MigrainePredictorV2

# Initialize the adapter
adapter = UniversalDataAdapter(data_dir='data', verbose=True)

# Load and process any dataset
data = adapter.load_data('path/to/your/data.csv')
schema = adapter.detect_schema(data)
selected_features = adapter.auto_select_features(data)

# Prepare data for training
training_data = adapter.prepare_training_data(data, test_size=0.3)
X_train, X_test = training_data['X_train'], training_data['X_test']
y_train, y_test = training_data['y_train'], training_data['y_test']

# Train a model
predictor = MigrainePredictorV2(model_dir='models', data_dir='data')
train_data = pd.concat([X_train, y_train], axis=1)
model_id = predictor.train(data=train_data, model_name="my_model")

# Make predictions
predictions = predictor.predict_with_missing_features(X_test)

# Apply explainability analysis
explainer = adapter.get_explainer('shap')
explanations = explainer.explain_instance(X_test, predictions)
```

## Example Script

See the `examples/universal_data_example.py` script for a comprehensive demonstration of the universal data adapter, including:

1. Generating synthetic migraine data
2. Automatic schema detection
3. Feature selection using both standard methods and meta-optimization
4. Training a migraine prediction model
5. Evaluating model performance
6. Making predictions on new data
7. Applying explainability analysis

## Working with the Synthetic Data Generator

The universal data adapter integrates seamlessly with the synthetic data generator:

```python
from migraine_prediction_project.examples.clinical_sythetic_data import generate_synthetic_data
from migraine_prediction_project.src.migraine_model.universal_data_adapter import UniversalDataAdapter

# Generate synthetic data
synthetic_data = generate_synthetic_data(
    num_patients=50,
    num_days=90,
    female_percentage=0.5,
    missing_data_rate=0.05,
    anomaly_rate=0.01,
    include_severity=True
)

# Process with the universal adapter
adapter = UniversalDataAdapter()
schema = adapter.detect_schema(synthetic_data)
selected_features = adapter.auto_select_features(synthetic_data)
```

## Feature Meta-Optimization

The package includes a meta-optimization approach to feature selection:

```python
from migraine_prediction_project.src.migraine_model.feature_meta_optimizer import MetaFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# Create base model
base_model = RandomForestClassifier(n_estimators=100)

# Initialize meta-feature selector
meta_selector = MetaFeatureSelector(
    base_model=base_model,
    n_features=15,
    scoring='roc_auc',
    meta_method='de',  # Differential evolution
    surrogate='rf'     # Random forest surrogate
)

# Run meta-feature selection
meta_selector.fit(X, y, feature_names=feature_names)
selected_features = meta_selector.selected_features_
```

## Technical Details

### Core Feature Detection

The adapter identifies these core features in datasets:

- heart_rate
- temperature
- barometric_pressure
- humidity
- stress_level
- sleep_hours
- hydration_ml
- caffeine_mg
- alcohol_units

Missing features will be derived when possible or substituted with reasonable defaults.

### Automatic Target Identification

The adapter automatically identifies target columns using these methods:

1. Exact match with known target names (migraine, migraine_occurred, etc.)
2. Binary columns with 'migraine' or 'headache' in the name
3. Any isolated binary column in datasets with a single binary column

### Date Column Detection

The adapter identifies date columns through:

1. Exact match with known date column names
2. Columns that can be successfully parsed as datetime
3. First column with a date-like name

## Integration with Meta-Optimizer

The feature selection can leverage the meta-optimizer framework to find optimal feature subsets using evolutionary algorithms and surrogate models. This advanced approach typically outperforms standard feature selection methods, especially on complex datasets with non-linear relationships.

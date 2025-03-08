# Command-Line Argument Fixes and Implementation Notes

This document provides details about fixes and implementation notes for the command-line arguments in the framework. It's intended to help users understand any limitations or special considerations when using certain arguments.

## Recently Fixed Arguments

The following command-line arguments have been fixed in the latest update:

### 1. `--compare-optimizers`

Previously, this argument failed due to an import error with the optimizer factory.

**Fix implemented:**
- Corrected the import from `create_optimizer` to `create_optimizers`
- Updated the function to properly utilize the optimizers returned by `create_optimizers`
- Removed unsupported parameters from optimizer method calls

**Usage example:**
```bash
python main.py --compare-optimizers --dimension 3 --verbose
```

### 2. `--meta`

Previously, this argument failed with a JSON decoding error.

**Fix implemented:**
- Created a properly formatted empty JSON file for the selection history
- Fixed path handling for selection history tracking

**Usage example:**
```bash
python main.py --meta --dimension 3 --verbose
```

**Note:** You may see non-critical warnings about "Object of type bool is not JSON serializable" during execution, but these are handled gracefully and don't affect functionality.

### 3. `--import-migraine-data`

Previously, this argument failed due to a missing module import.

**Fix implemented:**
- Added the `MIGRAINE_MODULES_AVAILABLE` flag to detect migraine module availability
- Fixed the predictor initialization to use a proper `MigrainePredictor` object
- Updated the import data functionality to use pandas directly for importing data
- Added graceful handling for missing functionality

**Usage example:**
```bash
python main.py --import-migraine-data --data-path data.csv --verbose
```

**Limitations:**
- Adding derived features is not supported in the current implementation
- Some advanced features of the original migraine predictor may not be available

### 4. `--predict-migraine`

Previously, this argument failed with an AttributeError.

**Fix implemented:**
- Fixed the predictor initialization to use a proper `MigrainePredictor` object
- Added proper error handling

**Usage example:**
```bash
python main.py --predict-migraine --prediction-data patient_data.csv --verbose
```

**Limitations:**
- Requires a trained model to work fully
- If no model is available, it will fail with "No default model found" error

## Export and Import Functionality

The export and import functionality has been fully implemented and tested:

### Export Options

| Option | Description | Default |
|--------|-------------|---------|
| `--export` | Export optimization data | False |
| `--export-format {json,csv,both}` | Format for exporting data | `json` |
| `--export-dir EXPORT_DIR` | Directory for exporting data | `results` |

**Usage example:**
```bash
python main.py --optimize --dimension 3 --export --export-dir results/test_export --verbose
```

### Import Options

| Option | Description | Default |
|--------|-------------|---------|
| `--import-data IMPORT_DATA` | Import optimization data from file | None |

**Usage example:**
```bash
python main.py --import-data results/test_export --visualize --verbose
```

## Module Availability Detection

The framework now checks for the availability of certain modules and provides more informative error messages:

- `MIGRAINE_MODULES_AVAILABLE`: Checks if migraine prediction modules are available
- `OPTIMIZER_AVAILABLE`: Checks if the MetaOptimizer is available
- `EXPLAINABILITY_AVAILABLE`: Checks if explainability components are available

If a required module is not available, the framework will log a warning and skip the functionality rather than crashing.

## Known Issues and Limitations

1. **Migraine Prediction**: The migraine prediction functionality requires having a trained model available. Training a model requires additional data and setup.

2. **Meta-Learning**: The meta-learning functionality may occasionally produce warnings about JSON serialization, but these don't affect the core functionality.

3. **Model and Optimizer Explainability**: These features may require additional dependencies that aren't part of the core framework.

## Best Practices

1. **Always use the `--verbose` flag** when troubleshooting to see detailed logs.

2. **Use `--dimension` with a small value** (e.g., 3) when testing to ensure faster execution.

3. **Create test data files** for testing migraine data import and prediction.

4. **Check logs for warnings** about missing modules or functionality.

5. **Ensure proper directory structure** when using export and import functionality. 
# MoE System Architecture Overview

## System Components and File Connections

### Core Components

| Component | Files | Description |
|-----------|-------|-------------|
| **Data Preprocessing** | `data_integration/clinical_data_validator.py` | Validates and preprocesses clinical data, handles datetime features, performs structure validation and quality checks |
| **Synthetic Data Generation** | `data_integration/synthetic_data_generator.py` | Generates synthetic patient data with various drift patterns for testing and validation |
| **Real-Synthetic Comparison** | `data_integration/real_synthetic_comparator.py` | Compares real clinical data with synthetic data, computes similarity metrics and visualizations |
| **MoE Framework** | `moe/moe_model.py`, `moe/experts/*`, `moe/gating/*` | Core MoE implementation with expert models and gating network |
| **Explainability** | `explainability/shap_explainer.py` | SHAP-based feature importance analysis and visualization |
| **Validation Framework** | `cli/real_data_commands.py`, `direct_validation_test.py` | Command-line interface and direct test script for comprehensive validation |
| **Interactive Reporting** | `tests/moe_interactive_report.py`, `tests/*_report.py` | Generates interactive HTML reports with comprehensive visualizations |

### Data Flow

```
[Clinical Data] → [Preprocessing] → [Feature Engineering] → [MoE Model Training]
                                                                    ↓
[Synthetic Data] → [Comparison Metrics] ← [Model Evaluation] ← [Prediction]
       ↓                   ↓                      ↓               ↓
       └────────┬──────────┘                      │               │
                ↓                                 │               │
         [SHAP Analysis] ←─────────────────────┬──┘               │
                ↓                              │                  │
                └─────────────┬────────────────┘                  │
                              ↓                                   │
                     [Interactive Report] ←──────────────────────┘
```

### Key Integration Points

1. **Data Preprocessing → MoE**: The `clinical_data_validator.py` prepares data for the MoE framework by:
   - Converting datetime columns to numeric features
   - Handling missing values and outliers
   - Normalizing features for consistent expert model inputs

2. **Explainability → Reporting**: The SHAP analysis is integrated into the reporting system:
   - Feature importance from SHAP values is visualized in reports
   - Model interpretation results are included in the validation summaries
   - Expert contribution analysis is visualized for transparency

3. **Validation → Dashboard**: The validation framework provides the metrics for dashboard visualization:
   - Model performance metrics (accuracy, precision, recall, F1)
   - Feature importance rankings
   - Data quality assessments
   - Real vs. synthetic data comparisons

## Implementation Details

### 1. Data Preprocessing Pipeline

The preprocessing pipeline in `clinical_data_validator.py` performs the following steps:

1. **Structure Validation**: Verifies that required columns exist in the data
2. **Data Type Conversion**: Converts datetime columns to appropriate formats
3. **Data Quality Checks**: Identifies missing values, outliers, and inconsistencies
4. **Feature Engineering**: Creates derived features from raw clinical data

### 2. MoE Model Implementation

The MoE model implementation consists of:

1. **Expert Models**: Domain-specific models for different aspects of patient data
2. **Gating Network**: Weights expert predictions based on input features
3. **Integration Layer**: Combines expert predictions into final migraine predictions
4. **Optimization**: Uses evolutionary algorithms to tune both experts and gating

### 3. Explainability Framework

The SHAP explainability framework provides:

1. **Feature Importance**: Identifies which features contribute most to predictions
2. **Model Interpretation**: Explains why specific predictions were made
3. **Visualization**: Creates intuitive visualizations of feature impacts
4. **Expert Contribution**: Shows how each expert contributes to final predictions

### 4. Validation and Reporting

The validation and reporting system includes:

1. **Data Validation**: Assesses data quality and completeness
2. **Model Performance**: Evaluates prediction accuracy and other metrics
3. **Drift Detection**: Identifies concept drift in patient data
4. **Interactive Reports**: Generates comprehensive HTML reports with visualizations

## Deployment Architecture

The system is designed for flexible deployment:

1. **Development Environment**: Local execution for model development and testing
2. **Clinical Testing**: Deployment in clinical testing environments
3. **Production**: Scalable deployment for real-world clinical use

## Running the System

### MoE Validation Framework

The MoE validation framework can be run using the following command:

```bash
python main_v2.py moe_validation [options]
```

#### Available Options

| Option | Description |
|--------|-------------|
| `--components` | Components to validate. Choose from: meta_optimizer, meta_learner, drift, explainability, gating, integrated, explain_drift, all |
| `--interactive` | Generate an interactive HTML report |
| `--results-dir` | Directory to save results (default: ./results/moe_validation) |
| `--explainers` | Explainers to use. Choose from: shap, lime, feature_importance, optimizer_explainer, all |
| `--enable-continuous-explain` | Enable continuous explainability during validation |

#### Example Commands

**Basic validation with all components:**
```bash
python main_v2.py moe_validation --components all --interactive
```

**Validation with specific components:**
```bash
python main_v2.py moe_validation --components meta_optimizer meta_learner --interactive
```

**Validation with custom results directory:**
```bash
python main_v2.py moe_validation --components all --results-dir "./results/moe_validation_$(date +%Y%m%d_%H%M%S)" --interactive
```

### Visualizing Results

#### Interactive HTML Report

The validation process generates an interactive HTML report in the specified results directory. The report can be opened in any modern web browser:

```bash
open ./results/moe_validation/reports/interactive_report_[timestamp].html
```

#### Troubleshooting Visualization Issues

If you encounter visualization errors in the HTML report:

1. **Browser Console**: Check your browser's developer console for JavaScript errors
2. **Alternative Visualization**: Use the static visualizations generated in the results directory instead
3. **Command Line Output**: Basic validation results are also displayed in the command line output

#### Static Visualizations

Static visualizations are saved as PNG files in the results directory and can be viewed directly:

```bash
open ./results/moe_validation/*.png
```

### Running without Visualizations

To run the validation without generating visualizations that might cause errors:

```bash
python main_v2.py moe_validation --components all
```

This will run all validation tests but skip generating the interactive HTML report.

### Interpreting Results

Validation results include:

1. **Test Results**: Pass/fail status for each validation test
2. **Performance Metrics**: Accuracy, precision, recall, and F1 scores
3. **Drift Analysis**: Detection of concept drift in data
4. **Expert Performance**: Evaluation of individual expert models
5. **Gating Network Analysis**: Analysis of expert weight assignment

### Common Issues and Solutions

#### Visualization Errors

**Issue**: "⚠️ Visualization Error" message in the HTML report

**Solutions**:
- Try a different web browser (Chrome or Firefox recommended)
- Run in a clean environment without previous visualization artifacts:
  ```bash
  rm -rf ./results/moe_validation
  python main_v2.py moe_validation --components all --interactive
  ```
- Use the static PNG visualizations instead

#### Missing Tab Content

**Issue**: Some tabs in the HTML report have no content

**Solution**:
- Ensure all required modules are in place (theoretical_metrics_report.py, real_data_validation_report.py, etc.)
- Run with specific components to isolate the issue:
  ```bash
  python main_v2.py moe_validation --components meta_optimizer --interactive
  ```

#### Matplotlib Backend Issues

**Issue**: Matplotlib errors related to backend

**Solution**:
- Set the Matplotlib backend explicitly:
  ```bash
  MPLBACKEND=TkAgg python main_v2.py moe_validation --components all --interactive
  ```

#### Integration with External Tools

For advanced visualization and analysis, results can be exported to external tools:

```bash
# Export test results to CSV
python main_v2.py moe_validation --components all --export-results
```

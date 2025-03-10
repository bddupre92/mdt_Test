# Command-Line Interface Documentation

This document provides comprehensive documentation for all command-line arguments supported by the framework.

## Basic Usage

```bash
python main.py [OPTIONS]
```

## General Options

| Option | Description |
|--------|-------------|
| `--config CONFIG` | Path to configuration file |
| `--summary` | Print summary of results |
| `--visualize` | Visualize results |
| `--verbose` | Print detailed logs |
| `--dimension DIMENSION` | Dimension for optimization problems (default: 10) |

## Operation Modes

The framework supports several operation modes, each activated by a specific flag:

| Flag | Description |
|------|-------------|
| `--optimize` | Run optimization with multiple optimizers |
| `--evaluate` | Evaluate a trained model |
| `--meta` | Run meta-learning to select best optimizer |
| `--drift` | Run drift detection |
| `--run-meta-learner-with-drift` | Run meta-learner with drift detection |
| `--explain` | Run explainability analysis |
| `--explain-drift` | Explain drift when detected |
| `--test-algorithm-selection` | Run a demo of algorithm selection visualization |
| `--compare-optimizers` | Run comparison of all available optimizers on benchmark functions |
| `--dynamic-optimization` | Run dynamic optimization visualization |
| `--enhanced-meta` | Run enhanced meta-learning |

## Modular Command Framework

The framework also supports a more modular command structure using subcommands:

```bash
python main_v2.py [SUBCOMMAND] [OPTIONS]
```

### Available Subcommands

- `baseline_comparison`: Run baseline comparison between SATzilla and Meta Optimizer
- `train_satzilla`: Train the SATzilla-inspired algorithm selector
- `meta`: Run meta-learning
- `dynamic_optimization`: Run dynamic optimization visualization

## Optimization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method METHOD` | Method for meta-learner | `bayesian` |
| `--surrogate SURROGATE` | Surrogate model for meta-learner | None |
| `--selection SELECTION` | Selection strategy for meta-learner | None |
| `--exploration EXPLORATION` | Exploration factor for meta-learner | `0.2` |
| `--history HISTORY` | History weight for meta-learner | `0.7` |

## Data Export and Import Options

| Option | Description | Default |
|--------|-------------|---------|
| `--export` | Export optimization data | False |
| `--export-format {json,csv,both}` | Format for exporting data | `json` |
| `--export-dir EXPORT_DIR` | Directory for exporting data | `results` |
| `--import-data IMPORT_DATA` | Import optimization data from file | None |

## Drift Detection Options

| Option | Description | Default |
|--------|-------------|---------|
| `--drift-window DRIFT_WINDOW` | Window size for drift detection | `50` |
| `--drift-threshold DRIFT_THRESHOLD` | Threshold for drift detection | `0.5` |
| `--drift-significance DRIFT_SIGNIFICANCE` | Significance level for drift detection | `0.05` |

## Dynamic Optimization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--function` | Test function to use ('ackley', 'rastrigin', etc.) | Required |
| `--drift-type` | Type of drift ('sudden', 'oscillatory', 'linear', etc.) | Required |
| `--dim` | Problem dimensionality | `10` |
| `--drift-rate` | Rate of drift (0.0 to 1.0) | `0.1` |
| `--drift-interval` | Interval between drift events | `20` |
| `--severity` | Severity of drift (0.0 to 1.0) | `1.0` |
| `--max-iterations` | Maximum number of iterations | `500` |
| `--reoptimize-interval` | Re-optimize after this many evaluations | `50` |
| `--show-plot` | Show plot in addition to saving it | `False` |

## Algorithm Selection Visualization

| Option | Description | Default |
|--------|-------------|---------|
| `--visualize-algorithm-selection` | Visualize algorithm selection process | False |
| `--algo-viz-dir ALGO_VIZ_DIR` | Directory to save algorithm selection visualizations | `results/algorithm_selection_demo` |
| `--algo-viz-plots` | Algorithm selection plot types to generate | None |
| `--interactive` | Generate interactive HTML dashboards | False |

## Performance Visualization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--boxplot` | Generate performance boxplots | False |
| `--show-significance` | Show standard deviation bands on convergence plots | False |
| `--dynamic` | Compare optimizers on dynamic problems with radar charts | False |
| `--visualize-drift` | Generate drift visualization plots | False |

## Explainability Options

### Model Explainability

| Option | Description | Default |
|--------|-------------|---------|
| `--explainer {shap,lime,feature_importance,optimizer}` | Explainer type to use | `shap` |
| `--explain-plots` | Generate and save explainability plots | False |
| `--explain-plot-types EXPLAIN_PLOT_TYPES [EXPLAIN_PLOT_TYPES ...]` | Specific plot types to generate | None |
| `--explain-samples EXPLAIN_SAMPLES` | Number of samples to use for explainability | `5` |
| `--auto-explain` | Automatically run explainability after other operations | False |

### Optimizer Explainability

| Option | Description | Default |
|--------|-------------|---------|
| `--explain-optimizer` | Run explainability analysis on optimizer | False |
| `--optimizer-type {differential_evolution,evolution_strategy,ant_colony,grey_wolf}` | Type of optimizer to explain | `differential_evolution` |
| `--optimizer-dim OPTIMIZER_DIM` | Dimension for optimizer | `10` |
| `--optimizer-bounds OPTIMIZER_BOUNDS OPTIMIZER_BOUNDS` | Bounds for optimizer (min max) | `[-5, 5]` |
| `--optimizer-plot-types OPTIMIZER_PLOT_TYPES [OPTIMIZER_PLOT_TYPES ...]` | Plot types to generate for optimizer explainability | See below |
| `--test-functions TEST_FUNCTIONS [TEST_FUNCTIONS ...]` | Test functions to run optimizer on | `['sphere', 'rosenbrock']` |
| `--max-evals MAX_EVALS` | Maximum number of function evaluations | `500` |

Default optimizer plot types:
- `convergence`
- `parameter_adaptation`
- `diversity`
- `landscape_analysis`
- `decision_process`
- `exploration_exploitation`
- `gradient_estimation`
- `performance_comparison`

## Baseline Comparison Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dimensions -d` | Number of dimensions for benchmark functions | `2` |
| `--max-evaluations -e` | Maximum number of function evaluations per algorithm | `1000` |
| `--num-trials -t` | Number of trials to run for statistical significance | `3` |
| `--functions -f` | Benchmark functions to use | `['sphere', 'rosenbrock']` |
| `--all-functions` | Use all available benchmark functions | False |
| `--output-dir -o` | Output directory for results | `results/baseline_comparison` |
| `--timestamp-dir` | Create a timestamped subdirectory for results | True |
| `--no-visualizations` | Disable visualization generation | False |

## SATzilla Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dimensions -d` | Number of dimensions for benchmark functions | `2` |
| `--max-evaluations -e` | Maximum number of function evaluations per algorithm | `1000` |
| `--num-problems -p` | Number of training problems to generate | `20` |
| `--functions -f` | Benchmark functions to use for training | `['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank']` |
| `--all-functions` | Use all available benchmark functions | False |
| `--output-dir -o` | Output directory for training results | `results/satzilla_training` |
| `--timestamp-dir` | Create a timestamped subdirectory for results | True |
| `--seed -s` | Random seed for reproducibility | `42` |
| `--visualize-features` | Generate feature importance visualizations | False |

## Migraine Data Processing

| Option | Description | Default |
|--------|-------------|---------|
| `--import-migraine-data` | Import new migraine data | False |
| `--data-path` | Path to migraine data file | None |
| `--data-dir` | Directory to store data files | `data` |
| `--model-dir` | Directory to store model files | `models` |
| `--file-format {csv,excel,json,parquet}` | Format of the input data file | `csv` |
| `--add-new-columns` | Add new columns found in the data to the schema | False |
| `--derived-features` | Derived features to create (format: "name:formula") | None |
| `--train-model` | Train a model with the imported data | False |
| `--model-name` | Name for the trained model | None |
| `--model-description` | Description for the trained model | None |
| `--make-default` | Make the trained model the default | False |
| `--save-processed-data` | Save the processed data | False |

## Migraine Prediction

| Option | Description | Default |
|--------|-------------|---------|
| `--predict-migraine` | Run migraine prediction | False |
| `--prediction-data` | Path to data for prediction | None |
| `--model-id` | ID of the model to use for prediction | None |
| `--save-predictions` | Save prediction results | False |

## Universal Data Adapter

| Option | Description | Default |
|--------|-------------|---------|
| `--universal-data` | Process any migraine dataset using the universal adapter | False |
| `--disable-auto-feature-selection` | Disable automatic feature selection | False |
| `--use-meta-feature-selection` | Use meta-optimization for feature selection | False |
| `--max-features` | Maximum number of features to select | None |
| `--test-size` | Fraction of data to use for testing | `0.2` |
| `--random-seed` | Random seed for reproducibility | `42` |
| `--evaluate-model` | Evaluate the trained model on test data | False |

## Synthetic Data Generation

| Option | Description | Default |
|--------|-------------|---------|
| `--generate-synthetic` | Generate synthetic migraine data | False |
| `--synthetic-patients` | Number of patients for synthetic data | None |
| `--synthetic-days` | Number of days per patient for synthetic data | None |
| `--synthetic-female-pct` | Percentage of female patients in synthetic data | None |
| `--synthetic-missing-rate` | Rate of missing data in synthetic data | None |
| `--synthetic-anomaly-rate` | Rate of anomalies in synthetic data | None |
| `--synthetic-include-severity` | Include migraine severity in synthetic data | False |
| `--save-synthetic` | Save the generated synthetic data | False |

## Enhanced Meta-Learning Options

| Option | Description | Default |
|--------|-------------|---------|
| `--enhanced-meta` | Run enhanced meta-learning | False |
| `--export-dir` | Directory for saving results and visualizations | `results/enhanced_meta_main` |
| `--visualize` | Enable visualization | False |

## Examples

### Running Optimization

```bash
python main.py --optimize --dimension 3 --verbose --summary
```

### Optimizing with Export

```bash
python main.py --optimize --dimension 3 --export --export-dir results/test_export --verbose
```

### Importing and Visualizing Optimization Data

```bash
python main.py --import-data results/test_export --visualize --verbose
```

### Running Algorithm Selection Demo

```bash
python main.py --test-algorithm-selection --verbose
```

### Comparing Optimizers

```bash
python main.py --compare-optimizers --dimension 3 --verbose
```

### Running Meta-Learning

```bash
python main.py --meta --dimension 3 --method bayesian --exploration 0.3 --history 0.6 --verbose
```

### Running Model Explainability with SHAP

```bash
python main.py --explain --explainer shap --explain-plots --explain-plot-types summary waterfall --verbose
```

### Running Drift Detection

```bash
python main.py --drift --drift-window 20 --drift-threshold 0.02 --drift-significance 0.95 --verbose
```

### Running Dynamic Optimization

```bash
python main_v2.py --dynamic-optimization --function=ackley --drift-type=sudden --dim=5 --verbose
```

### Running Baseline Comparison

```bash
python main_v2.py baseline_comparison --dimensions 2 --max-evaluations 1000 --num-trials 5 --functions sphere rosenbrock ackley
```

### Training SATzilla Selector

```bash
python main_v2.py train_satzilla --dimensions 2 --num-problems 30 --functions sphere rosenbrock ackley rastrigin --visualize-features
```

### Running Extended Comparison Analysis

```bash
./scripts/run_extended_comparison.sh
```

### Analyzing Existing Benchmark Results

```bash
./scripts/run_extended_comparison.sh --skip-benchmarks results/baseline_comparison/full_benchmark_YYYYMMDD
```

### Importing Migraine Data

```bash
python main.py --import-migraine-data --data-path data.csv --verbose
```

### Predicting Migraines

```bash
python main.py --predict-migraine --prediction-data patient_data.csv --verbose
```

### Generating Synthetic Data

```bash
python main.py --generate-synthetic --synthetic-patients 5 --verbose
```

### Running Algorithm Selection Demo with Interactive Visualizations

```bash
python main.py --test-algorithm-selection --interactive --algo-viz-dir results/algorithm_selection_demo --verbose
```

### Comparing Optimizers with Performance Boxplots and Significance Bands

```bash
python main.py --compare-optimizers --boxplot --show-significance --dimension 3 --verbose
```

### Evaluating a Model with Visualization

```bash
python main.py --evaluate --visualize --export-dir results --verbose
```

### Visualizing Drift for Specific Functions and Drift Types

```bash
python main.py --drift --visualize-drift --function ackley --drift-type linear --verbose
```

### Running Enhanced Meta-Learning with Visualizations

```bash
python main.py --enhanced-meta --visualize --export-dir results/enhanced_meta_main --verbose
```

### Comparing Optimizers on Dynamic Problems

```bash
python main.py --compare-optimizers --dynamic --function ackley --drift-type linear --verbose
```

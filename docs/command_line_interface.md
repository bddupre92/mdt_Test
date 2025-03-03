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

## Optimization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method METHOD` | Method for meta-learner | `bayesian` |
| `--surrogate SURROGATE` | Surrogate model for meta-learner | None |
| `--selection SELECTION` | Selection strategy for meta-learner | None |
| `--exploration EXPLORATION` | Exploration factor for meta-learner | `0.2` |
| `--history HISTORY` | History weight for meta-learner | `0.7` |

## Drift Detection Options

| Option | Description | Default |
|--------|-------------|---------|
| `--drift-window DRIFT_WINDOW` | Window size for drift detection | `10` |
| `--drift-threshold DRIFT_THRESHOLD` | Threshold for drift detection | `0.01` |
| `--drift-significance DRIFT_SIGNIFICANCE` | Significance level for drift detection | `0.9` |

## Explainability Options

### Model Explainability

| Option | Description | Default |
|--------|-------------|---------|
| `--explainer {shap,lime,feature_importance,optimizer}` | Explainer type to use | `shap` |
| `--explain-plots` | Generate and save explainability plots | False |
| `--explain-plot-types EXPLAIN_PLOT_TYPES [EXPLAIN_PLOT_TYPES ...]` | Specific plot types to generate | None |
| `--explain-samples EXPLAIN_SAMPLES` | Number of samples to use for explainability | `50` |

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

## Examples

### Running Optimization

```bash
python main.py --optimize --summary
```

### Running Model Explainability with SHAP

```bash
python main.py --explain --explainer shap --explain-plots --explain-plot-types summary waterfall --summary
```

### Running Model Explainability with LIME

```bash
python main.py --explain --explainer lime --explain-plots --explain-plot-types local summary --explain-samples 30 --summary
```

### Running Optimizer Explainability

```bash
python main.py --explain --explain-optimizer --optimizer-type differential_evolution --optimizer-dim 3 --optimizer-bounds -5 5 --test-functions sphere rosenbrock --explain-plots --summary
```

### Running Optimizer Explainability with Specific Plot Types

```bash
python main.py --explain --explain-optimizer --optimizer-type evolution_strategy --optimizer-dim 2 --optimizer-bounds -10 10 --optimizer-plot-types convergence diversity --explain-plots --summary
```

### Running Meta-Learning

```bash
python main.py --meta --method bayesian --exploration 0.3 --history 0.6 --summary
```

### Running Drift Detection

```bash
python main.py --drift --drift-window 20 --drift-threshold 0.02 --drift-significance 0.95 --summary
```

### Running Meta-Learner with Drift Detection

```bash
python main.py --run-meta-learner-with-drift --drift-window 15 --drift-threshold 0.015 --summary
```

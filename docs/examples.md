# Examples

This document provides practical examples of using the optimization framework for various tasks.

## Table of Contents

1. [Basic Optimization](#basic-optimization)
2. [Comparing Multiple Optimizers](#comparing-multiple-optimizers)
3. [Meta-Learning](#meta-learning)
4. [Drift Detection](#drift-detection)
5. [Model Explainability](#model-explainability)
6. [Optimizer Explainability](#optimizer-explainability)
7. [Combined Workflows](#combined-workflows)

## Basic Optimization

### Running a Single Optimizer on a Test Function

```python
from optimizers.optimizer_factory import OptimizerFactory
from benchmarking.test_functions import ClassicalTestFunctions

# Create optimizer
factory = OptimizerFactory()
optimizer = factory.create_optimizer(
    'differential_evolution',
    dim=10,
    bounds=[(-5, 5)] * 10
)

# Run optimization
result = optimizer.run(ClassicalTestFunctions.sphere, max_evals=1000)

# Print results
print(f"Best solution: {result['best_solution']}")
print(f"Best score: {result['best_score']}")
print(f"Number of evaluations: {result['evaluations']}")
```

**Command Line Equivalent:**
```bash
python main.py --optimize --summary
```

### Customizing Optimizer Parameters

```python
from optimizers.optimizer_factory import OptimizerFactory
from benchmarking.test_functions import ClassicalTestFunctions

# Create optimizer with custom parameters
factory = OptimizerFactory()
optimizer = factory.create_optimizer(
    'differential_evolution',
    dim=10,
    bounds=[(-5, 5)] * 10,
    population_size=50,
    mutation_factor=0.8,
    crossover_rate=0.7,
    adaptation_strategy='adaptive'
)

# Run optimization
result = optimizer.run(ClassicalTestFunctions.rosenbrock, max_evals=2000)

# Print results
print(f"Best solution: {result['best_solution']}")
print(f"Best score: {result['best_score']}")
print(f"Number of evaluations: {result['evaluations']}")
```

## Comparing Multiple Optimizers

### Benchmarking Multiple Optimizers on Test Functions

```python
from optimizers.optimizer_factory import OptimizerFactory
from benchmarking.test_functions import ClassicalTestFunctions
import pandas as pd
import matplotlib.pyplot as plt

# Create optimizers
factory = OptimizerFactory()
optimizers = {
    'DE': factory.create_optimizer('differential_evolution', dim=10, bounds=[(-5, 5)] * 10),
    'ES': factory.create_optimizer('evolution_strategy', dim=10, bounds=[(-5, 5)] * 10),
    'AC': factory.create_optimizer('ant_colony', dim=10, bounds=[(-5, 5)] * 10)
}

# Test functions
test_functions = {
    'Sphere': ClassicalTestFunctions.sphere,
    'Rosenbrock': ClassicalTestFunctions.rosenbrock,
    'Rastrigin': ClassicalTestFunctions.rastrigin,
    'Ackley': ClassicalTestFunctions.ackley
}

# Run benchmarks
results = []
for opt_name, optimizer in optimizers.items():
    for func_name, func in test_functions.items():
        optimizer.reset()
        result = optimizer.run(func, max_evals=1000)
        results.append({
            'Optimizer': opt_name,
            'Function': func_name,
            'Best Score': result['best_score'],
            'Evaluations': result['evaluations']
        })

# Create DataFrame
df = pd.DataFrame(results)

# Visualize results
plt.figure(figsize=(12, 8))
for func_name in test_functions.keys():
    plt.subplot(2, 2, list(test_functions.keys()).index(func_name) + 1)
    func_df = df[df['Function'] == func_name]
    plt.bar(func_df['Optimizer'], func_df['Best Score'])
    plt.title(func_name)
    plt.ylabel('Best Score')
plt.tight_layout()
plt.savefig('benchmark_results.png')
```

**Command Line Equivalent:**
```bash
python main.py --optimize --summary --visualize
```

## Meta-Learning

### Using Meta-Learning to Select the Best Optimizer

```python
from meta_learning.meta_learner import MetaLearner
from benchmarking.test_functions import ClassicalTestFunctions

# Create meta-learner
meta_learner = MetaLearner(
    method='bayesian',
    exploration=0.2,
    history_weight=0.7
)

# Define problem
problem = {
    'function': ClassicalTestFunctions.rastrigin,
    'dim': 10,
    'bounds': [(-5, 5)] * 10,
    'max_evals': 1000
}

# Run meta-learning
result = meta_learner.optimize(problem)

# Print results
print(f"Selected optimizer: {result['selected_optimizer']}")
print(f"Best solution: {result['best_solution']}")
print(f"Best score: {result['best_score']}")
print(f"Number of evaluations: {result['evaluations']}")
```

**Command Line Equivalent:**
```bash
python main.py --meta --method bayesian --exploration 0.2 --history 0.7 --summary
```

## Drift Detection

### Detecting and Adapting to Concept Drift

```python
from drift_detection.drift_detector import DriftDetector
from meta_learning.meta_learner import MetaLearner
import numpy as np

# Generate synthetic data with drift
n_samples = 1000
n_features = 10
X = np.random.randn(n_samples, n_features)
y = np.zeros(n_samples)

# Introduce drift at specific points
drift_points = [250, 500, 750]
for i in range(n_samples):
    if i < drift_points[0]:
        y[i] = np.sum(X[i, :3]) + np.random.normal(0, 0.1)
    elif i < drift_points[1]:
        y[i] = np.sum(X[i, 3:6]) + np.random.normal(0, 0.1)
    elif i < drift_points[2]:
        y[i] = np.sum(X[i, 6:9]) + np.random.normal(0, 0.1)
    else:
        y[i] = X[i, 9] + np.random.normal(0, 0.1)

# Create drift detector
drift_detector = DriftDetector(
    window_size=20,
    threshold=0.01,
    significance_level=0.95
)

# Create meta-learner
meta_learner = MetaLearner()

# Process data stream
results = []
for i in range(0, n_samples, 10):
    # Get current window
    X_window = X[i:i+10]
    y_window = y[i:i+10]
    
    # Check for drift
    drift_detected, drift_stats = drift_detector.detect_drift(X_window, y_window)
    
    if drift_detected:
        print(f"Drift detected at sample {i}")
        # Adapt meta-learner
        meta_learner.adapt_to_drift(drift_stats)
    
    # Update results
    results.append({
        'sample': i,
        'drift_detected': drift_detected,
        'meta_model_version': meta_learner.version
    })
```

**Command Line Equivalent:**
```bash
python main.py --run-meta-learner-with-drift --drift-window 20 --drift-threshold 0.01 --drift-significance 0.95 --summary
```

## Model Explainability

### Explaining Model Predictions with SHAP

```python
from explainability.explainer_factory import ExplainerFactory
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# Create and train a model
X = np.random.rand(100, 10)
y = np.sum(X[:, [0, 2, 5]] * [3, 1, 2], axis=1) + np.random.normal(0, 0.1, 100)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create explainer
factory = ExplainerFactory()
explainer = factory.create_explainer('shap', model)

# Generate explanation
explanation = explainer.explain(X, y)

# Generate plots
plot_types = ['summary', 'waterfall', 'dependence']
for plot_type in plot_types:
    fig = explainer.plot(plot_type)
    plt.figure(fig.number)
    plt.title(f"SHAP {plot_type} plot")
    plt.savefig(f"shap_{plot_type}.png")

# Get feature importance
feature_importance = explainer.get_feature_importance()
print("Feature Importance:")
for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {importance:.6f}")
```

**Command Line Equivalent:**
```bash
python main.py --explain --explainer shap --explain-plots --explain-plot-types summary waterfall dependence --summary
```

## Optimizer Explainability

### Explaining Optimizer Behavior

```python
from optimizers.optimizer_factory import OptimizerFactory
from explainability.explainer_factory import ExplainerFactory
from benchmarking.test_functions import ClassicalTestFunctions
import matplotlib.pyplot as plt

# Create optimizer
factory = OptimizerFactory()
optimizer = factory.create_optimizer(
    'differential_evolution',
    dim=2,
    bounds=[(-5, 5)] * 2
)

# Run optimization
optimizer.run(ClassicalTestFunctions.rastrigin, max_evals=500)

# Create explainer
explainer_factory = ExplainerFactory()
explainer = explainer_factory.create_explainer('optimizer', optimizer)

# Generate explanation
explanation = explainer.explain()

# Generate plots
plot_types = ['convergence', 'diversity', 'landscape_analysis', 'exploration_exploitation']
for plot_type in plot_types:
    fig = explainer.plot(plot_type)
    plt.figure(fig.number)
    plt.title(f"Optimizer {plot_type} plot")
    plt.savefig(f"optimizer_{plot_type}.png")

# Print explanation summary
print("Optimizer Explanation:")
print(f"  Convergence rate: {explanation['convergence_rate']:.6f}")
print(f"  Final diversity: {explanation['final_diversity']:.6f}")
print(f"  Exploration/exploitation balance: {explanation['exploration_exploitation_balance']:.6f}")
print(f"  Problem landscape ruggedness: {explanation['landscape_ruggedness']:.6f}")
```

**Command Line Equivalent:**
```bash
python main.py --explain --explain-optimizer --optimizer-type differential_evolution --optimizer-dim 2 --optimizer-bounds -5 5 --test-functions rastrigin --explain-plots --optimizer-plot-types convergence diversity landscape_analysis exploration_exploitation --summary
```

## Combined Workflows

### Meta-Learning with Explainability

```python
from meta_learning.meta_learner import MetaLearner
from explainability.explainer_factory import ExplainerFactory
from benchmarking.test_functions import ClassicalTestFunctions

# Create meta-learner
meta_learner = MetaLearner()

# Define problem
problem = {
    'function': ClassicalTestFunctions.rastrigin,
    'dim': 10,
    'bounds': [(-5, 5)] * 10,
    'max_evals': 1000
}

# Run meta-learning
result = meta_learner.optimize(problem)

# Get selected optimizer
selected_optimizer = result['optimizer']

# Create explainer
factory = ExplainerFactory()
explainer = factory.create_explainer('optimizer', selected_optimizer)

# Generate explanation
explanation = explainer.explain()

# Generate plots
plot_types = ['convergence', 'diversity']
for plot_type in plot_types:
    fig = explainer.plot(plot_type)
    plt.figure(fig.number)
    plt.title(f"Selected Optimizer ({selected_optimizer.__class__.__name__}) {plot_type}")
    plt.savefig(f"selected_optimizer_{plot_type}.png")

# Print results
print(f"Selected optimizer: {selected_optimizer.__class__.__name__}")
print(f"Best score: {result['best_score']}")
print(f"Explanation: {explanation}")
```

**Command Line Equivalent:**
```bash
python main.py --meta --method bayesian --summary && python main.py --explain --explain-optimizer --optimizer-type <selected_optimizer> --explain-plots --summary
```

### Drift Detection with Explainability

```python
from drift_detection.drift_detector import DriftDetector
from explainability.explainer_factory import ExplainerFactory
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Generate synthetic data with drift
n_samples = 1000
n_features = 10
X = np.random.randn(n_samples, n_features)
y = np.zeros(n_samples)

# Introduce drift at specific points
drift_points = [500]
for i in range(n_samples):
    if i < drift_points[0]:
        y[i] = np.sum(X[i, :5]) + np.random.normal(0, 0.1)
    else:
        y[i] = np.sum(X[i, 5:]) + np.random.normal(0, 0.1)

# Split data
X_before = X[:drift_points[0]]
y_before = y[:drift_points[0]]
X_after = X[drift_points[0]:]
y_after = y[drift_points[0]:]

# Train models
model_before = RandomForestRegressor(n_estimators=100, random_state=42)
model_before.fit(X_before, y_before)

model_after = RandomForestRegressor(n_estimators=100, random_state=42)
model_after.fit(X_after, y_after)

# Create explainers
factory = ExplainerFactory()
explainer_before = factory.create_explainer('feature_importance', model_before)
explainer_after = factory.create_explainer('feature_importance', model_after)

# Generate explanations
explanation_before = explainer_before.explain(X_before, y_before)
explanation_after = explainer_after.explain(X_after, y_after)

# Get feature importance
importance_before = explainer_before.get_feature_importance()
importance_after = explainer_after.get_feature_importance()

# Compare feature importance
print("Feature Importance Before Drift:")
for feature, importance in sorted(importance_before.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature}: {importance:.6f}")

print("\nFeature Importance After Drift:")
for feature, importance in sorted(importance_after.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature}: {importance:.6f}")

# Visualize change in feature importance
features = list(importance_before.keys())
importance_diff = {f: importance_after[f] - importance_before[f] for f in features}

plt.figure(figsize=(10, 6))
plt.bar(features, [importance_diff[f] for f in features])
plt.title('Change in Feature Importance After Drift')
plt.xlabel('Feature')
plt.ylabel('Importance Difference')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('drift_feature_importance_change.png')
```

**Command Line Equivalent:**
```bash
python main.py --drift --explain-drift --drift-window 20 --drift-threshold 0.01 --summary
```

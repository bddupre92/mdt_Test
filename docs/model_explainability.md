# Model Explainability

This document describes the model explainability features in the framework, which allow you to understand and interpret model predictions and behavior.

## Overview

The model explainability module provides tools to:

1. Explain individual predictions of machine learning models
2. Identify important features that influence model predictions
3. Visualize feature importance and interactions
4. Compare different models based on their explainability

## Supported Explainers

The framework supports the following explainer types:

### 1. SHAP (SHapley Additive exPlanations)

SHAP values provide a unified measure of feature importance based on cooperative game theory. They explain how much each feature contributes to the prediction for a specific instance.

**Supported Plot Types:**
- `summary`: Overall feature importance across the dataset
- `waterfall`: Detailed breakdown of a single prediction
- `force`: Force plot showing feature contributions
- `dependence`: Dependence plots showing how a feature affects predictions
- `interaction`: Feature interaction plots

### 2. LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by approximating the model locally with an interpretable model.

**Supported Plot Types:**
- `local`: Explanation for a single instance
- `all_local`: Explanations for multiple instances
- `summary`: Summary of explanations across the dataset

### 3. Feature Importance

Direct feature importance from the model (if supported) or permutation importance.

**Supported Plot Types:**
- `bar`: Bar chart of feature importance
- `horizontal_bar`: Horizontal bar chart of feature importance
- `heatmap`: Heatmap of feature importance

## Usage

### Command Line Interface

You can use the model explainability features through the command line interface:

```bash
python main.py --explain --explainer shap --explain-plots
```

Optional arguments:
- `--explain-plot-types`: Specify which plot types to generate (e.g., `--explain-plot-types summary waterfall`)
- `--explain-samples`: Number of samples to use for explanation (default: 50)
- `--summary`: Print a summary of the explainability results

### Programmatic Usage

You can also use the model explainability features programmatically:

```python
from explainability.explainer_factory import ExplainerFactory
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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
for plot_type in ['summary', 'waterfall', 'dependence']:
    fig = explainer.plot(plot_type)
    fig.savefig(f"shap_{plot_type}.png")

# Get feature importance
feature_importance = explainer.get_feature_importance()
print(feature_importance)
```

## Examples

### SHAP Explainer

```bash
python main.py --explain --explainer shap --explain-plots --explain-plot-types summary waterfall dependence --summary
```

This will:
1. Create a synthetic dataset and train a RandomForestRegressor
2. Create a SHAP explainer for the model
3. Generate SHAP values for the test data
4. Create summary, waterfall, and dependence plots
5. Print a summary of feature importance

### LIME Explainer

```bash
python main.py --explain --explainer lime --explain-plots --explain-plot-types local summary --explain-samples 30 --summary
```

This will:
1. Create a synthetic dataset and train a RandomForestRegressor
2. Create a LIME explainer for the model
3. Generate LIME explanations for 30 test samples
4. Create local and summary plots
5. Print a summary of feature importance

### Feature Importance Explainer

```bash
python main.py --explain --explainer feature_importance --explain-plots --explain-plot-types horizontal_bar --summary
```

This will:
1. Create a synthetic dataset and train a RandomForestRegressor
2. Extract feature importance from the model
3. Create a horizontal bar plot of feature importance
4. Print a summary of feature importance

## Interpreting Results

### SHAP Values

- **Positive SHAP value**: The feature increases the prediction
- **Negative SHAP value**: The feature decreases the prediction
- **Magnitude**: The importance of the feature for the prediction

### LIME Explanations

- **Positive weight**: The feature increases the prediction
- **Negative weight**: The feature decreases the prediction
- **Magnitude**: The importance of the feature for the prediction

### Feature Importance

- **Higher value**: More important feature
- **Lower value**: Less important feature

## Integration with Other Framework Components

The model explainability features can be integrated with other components of the framework:

- **Meta-learning**: Use explainability to understand why certain models perform better on specific problems
- **Drift detection**: Explain changes in feature importance when drift is detected
- **Optimizer selection**: Use explainability to understand which features are most important for optimizer selection

## Future Enhancements

Potential future enhancements to the model explainability module:

1. Support for more explainers (e.g., Integrated Gradients, DeepLIFT)
2. Interactive visualizations using Plotly
3. Explainability for deep learning models
4. Counterfactual explanations
5. Explainability for time series models

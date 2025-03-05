# [DEPRECATED] Explainability Framework for Optimization

**Note: This README has been consolidated into the main README.md file. Please refer to the main README.md for up-to-date information on the explainability framework.**

This document describes the explainability framework for the optimization system, which provides insights into model behavior and feature importance.

## Overview

The explainability framework is designed to be modular and extensible, supporting multiple explainability methods including:

1. **SHAP (SHapley Additive exPlanations)** - Provides global and local explanations with strong theoretical foundations
2. **LIME (Local Interpretable Model-agnostic Explanations)** - Focuses on local explanations for individual predictions
3. **Feature Importance** - Simple feature importance based on model attributes or permutation

## Usage

### Command Line Interface

The explainability framework can be used directly from the command line:

```bash
python main.py --explain --explainer shap --save-plots
```

Available command line arguments:

- `--explain`: Enable explainability analysis
- `--explainer`: Choose explainer type (`shap`, `lime`, or `feature_importance`)
- `--explain-plots`: Specify plot types to generate (optional)
- `--explain-samples`: Number of samples to use for explanation (default: 100)
- `--save-plots`: Save visualization plots
- `--explain-drift`: Generate explanations when drift is detected (requires `--run-meta-learner-with-drift`)

### Drift Detection Explainability

The framework can automatically generate explanations when drift is detected:

```bash
python main.py --run-meta-learner-with-drift --explain-drift --explainer shap
```

This will:
1. Run the meta-learner with drift detection
2. Generate explanations whenever drift is detected
3. Save visualizations to `results/explainability/drift_[sample_number]/`

These explanations help understand which features contributed most to the detected drift and how the model's behavior changed after the drift.

### Programmatic Usage

You can also use the explainability framework programmatically:

```python
from explainability.explainer_factory import ExplainerFactory

# Create explainer
explainer = ExplainerFactory.create_explainer(
    explainer_type='shap',
    model=your_model,
    feature_names=feature_names
)

# Generate explanation
explanation = explainer.explain(X, y)

# Create visualization
fig = explainer.plot(plot_type='summary')
fig.savefig('shap_summary.png')

# Get feature importance
feature_importance = explainer.get_feature_importance()
```

## Explainer Types

### SHAP Explainer

SHAP (SHapley Additive exPlanations) provides a unified approach to explaining the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory.

**Supported Plot Types:**
- `summary`: Summary plot of feature importance
- `bar`: Bar plot of feature importance
- `beeswarm`: Beeswarm plot of SHAP values
- `waterfall`: Waterfall plot for a single prediction
- `force`: Force plot for a single prediction
- `decision`: Decision plot for a single prediction
- `dependence`: Dependence plot for a specific feature
- `interaction`: Interaction plot between two features

**Dependencies:**
- shap

### LIME Explainer

LIME (Local Interpretable Model-agnostic Explanations) explains the predictions of any classifier by learning an interpretable model locally around the prediction.

**Supported Plot Types:**
- `local`: Plot explanation for a single instance
- `all_local`: Plot explanations for all instances
- `summary`: Summary of feature importance across all instances

**Dependencies:**
- lime

### Feature Importance Explainer

A simple explainer that uses built-in feature importance methods from the model or permutation importance.

**Supported Plot Types:**
- `bar`: Vertical bar plot of feature importance
- `horizontal_bar`: Horizontal bar plot of feature importance
- `heatmap`: Heatmap of feature importance

**Dependencies:**
- scikit-learn

## Extending the Framework

You can extend the framework by creating new explainer classes that inherit from `BaseExplainer`:

```python
from explainability.base_explainer import BaseExplainer

class MyCustomExplainer(BaseExplainer):
    def __init__(self, model=None, feature_names=None, **kwargs):
        super().__init__('MyCustom', model, feature_names)
        # Initialize your explainer
        
    def explain(self, X, y=None, **kwargs):
        # Implement explanation logic
        return explanation_data
        
    def plot(self, plot_type='summary', **kwargs):
        # Implement visualization logic
        return fig
```

Then register your explainer in the `ExplainerFactory` class.

## Example Outputs

The explainability framework generates various visualizations to help understand model behavior:

1. **SHAP Summary Plot**: Shows the impact of each feature on the model output
2. **Feature Importance Bar Plot**: Displays the relative importance of each feature
3. **LIME Local Explanation**: Explains individual predictions
4. **Dependence Plots**: Shows how the model depends on a specific feature

All visualizations are saved to the `results/explainability/` directory when using the `--save-plots` flag.

## Drift Detection Explainability

The explainability framework integrates with the drift detection system to provide insights into data drift:

### Understanding Drift with SHAP

When drift is detected, SHAP explanations can help understand:
1. **Which features contributed most to the drift**
2. **How feature importance changed before and after drift**
3. **Whether the drift was caused by specific feature interactions**

### Interpreting Drift Explanations

Drift explanations are stored in the `results/explainability/drift_[sample_number]/` directory and include:

1. **SHAP Summary Plot**: Shows the overall feature importance at the time of drift
2. **SHAP Bar Plot**: Displays the ranked feature importance
3. **Explanation JSON**: Contains the raw explanation data for further analysis

### Programmatic Access to Drift Explanations

You can access drift explanations programmatically:

```python
# Run meta-learner with drift detection and explainability
results = run_meta_learner_with_drift(explain_drift=True, explainer_type='shap')

# Access drift explanations
drift_explanations = results['drift_explanations']

# Print feature importance for each drift point
for explanation in drift_explanations:
    print(f"Drift at sample {explanation['drift_point']}:")
    for feature, importance in explanation['feature_importance'].items():
        print(f"  {feature}: {importance:.4f}")
```

This integration provides valuable insights into when and why drift occurs, helping to build more robust and adaptive models.

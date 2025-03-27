# Adaptive Weighting System Guide

This document provides a comprehensive guide to the quality-aware adaptive weighting system that integrates with the Meta_Learner and existing drift detection components.

## Overview

The adaptive weighting system dynamically adjusts expert weights based on:

1. **Data Domain Compatibility**: Matching experts to their specialty domains (physiological, behavioral, environmental)
2. **Data Quality Metrics**: Adjusting weights based on completeness, consistency, and overall quality
3. **Drift Detection**: Reducing reliance on experts when their domain data exhibits concept drift

This implementation leverages existing drift detection capabilities while adding quality-aware weighting for better personalization.

## Components

### 1. AdaptiveWeightIntegration Class

This is the main coordinator that brings together:

- The existing `DriftDetector` from `meta_optimizer/drift_detection/drift_detector.py`
- The `MetaLearner` with its expert management capabilities
- New quality assessment algorithms

```python
from meta.adaptive_weight_integration import AdaptiveWeightIntegration
from meta.meta_learner import MetaLearner

# Create integration with specific parameters
integration = AdaptiveWeightIntegration(
    window_size=50,          # Window size for drift detection
    drift_threshold=0.05,    # Threshold for drift detection
    significance_level=0.05, # Statistical significance level for tests
    drift_method='ks',       # Detection method (ks, ad, error_rate, etc.)
    quality_impact=0.4,      # How much quality affects weights (0-1)
    drift_impact=0.3,        # How much drift affects weights (0-1)
    history_weight=0.7       # Weight given to historical metrics
)

# Create Meta_Learner
meta_learner = MetaLearner(method='bayesian')

# Register Meta_Learner with integration
integration.register_meta_learner(meta_learner)
```

### 2. Enhanced MetaLearner

The `MetaLearner` has been enhanced with:

- Quality-aware weighting capabilities in `predict_weights()`
- Support for domain-specific data quality tracking
- Integration with the drift detector

## Usage Flow

### 1. Set Up the Integration

```python
from meta.adaptive_weight_integration import AdaptiveWeightIntegration
from meta.meta_learner import MetaLearner
from meta_optimizer.drift_detection.drift_detector import DriftDetector

# Create and set up components
meta_learner = MetaLearner(method='bayesian')
integration = AdaptiveWeightIntegration()
integration.register_meta_learner(meta_learner)

# Register experts with Meta_Learner
meta_learner.register_expert(1, physiological_expert)
meta_learner.register_expert(2, behavioral_expert)
meta_learner.register_expert(3, environmental_expert)
```

### 2. Update Domain-Specific Quality Metrics

When new data arrives for a particular domain:

```python
# When physiological data arrives
integration.update_for_domain(
    physiological_data,   # Features array
    physiological_targets, # Target values
    'physiological'       # Domain name
)

# When behavioral data arrives
integration.update_for_domain(
    behavioral_data,
    behavioral_targets,
    'behavioral'
)
```

### 3. Prepare Context with Quality Metrics

```python
# Create context with data type flags
base_context = {
    'has_physiological': True,
    'has_behavioral': True,
    'has_environmental': False
}

# Enhance with quality metrics and drift detection data
enhanced_context = integration.prepare_context_with_quality(base_context)
```

### 4. Get Quality-Aware Expert Weights

```python
# Get weights that account for specialties, quality, and drift
weights = meta_learner.predict_weights(enhanced_context)

# Result: {1: 0.45, 2: 0.35, 3: 0.20} - dynamically adjusted weights
```

## Quality Metrics

The system tracks multiple quality dimensions for each data domain:

1. **Completeness**: Percentage of non-missing values
2. **Consistency**: Uniformity of feature scales and distributions
3. **Recency**: How recently the data was updated
4. **Drift Stability**: Measure of how stable (non-drifting) the data is over time
5. **Overall**: Weighted combination of the above metrics

## Drift Detection Integration

The system uses your existing drift detector to identify concept drift:

1. It maintains reference windows for each domain
2. When drift is detected, it reduces the weight of affected experts
3. The severity of weight reduction depends on the drift score

### Weight Adjustment Implementation

When drift is detected, the `_adjust_weights_for_drift` method applies these adjustments:

```python
# Inside _adjust_weights_for_drift
if is_drift:
    # Reduce weight based on drift severity
    drift_factor = 1.0 - min(0.8, drift_score)  # Cap reduction at 80%
    adjusted_weights[expert_id] *= (1.0 - drift_impact + (drift_impact * drift_factor))
```

Key points about this implementation:

1. **Selective Reduction**: Only experts with detected drift have their weights reduced
2. **No Automatic Redistribution**: The method only reduces weights - it doesn't redistribute to other experts
3. **Normalization After All Adjustments**: The `predict_weights` method normalizes all weights as its final step

This design makes the adjustment process more modular and easier to understand, with each component focused on its specific task.

## Example Use Case

```python
# Initial expert weights before quality/drift assessment
initial_weights = meta_learner.predict_weights({
    'has_physiological': True,
    'has_behavioral': True,
    'has_environmental': True
})
# Result: {1: 0.33, 2: 0.33, 3: 0.33} - equal weights

# After processing data with varying quality
physiological_data = high_quality_data  # Complete, consistent
behavioral_data = medium_quality_data   # Some missing values
environmental_data = poor_quality_data  # Inconsistent features

integration.update_for_domain(physiological_data, targets, 'physiological')
integration.update_for_domain(behavioral_data, targets, 'behavioral')
integration.update_for_domain(environmental_data, targets, 'environmental')

# Get quality-adjusted weights
context = integration.prepare_context_with_quality({
    'has_physiological': True,
    'has_behavioral': True,
    'has_environmental': True
})

quality_weights = meta_learner.predict_weights(context)
# Result: {1: 0.50, 2: 0.35, 3: 0.15} - weights adjusted by quality

# After detecting drift in behavioral data
# Behavioral data shifts significantly
integration.update_for_domain(shifted_behavioral_data, targets, 'behavioral')

drift_context = integration.prepare_context_with_quality({
    'has_physiological': True,
    'has_behavioral': True,
    'has_environmental': True
})

drift_weights = meta_learner.predict_weights(drift_context)
# Result: {1: 0.60, 2: 0.15, 3: 0.25} - behavioral expert downweighted due to drift
```

## Implementation Notes

- The system uses existing drift detection logic rather than duplicating it
- Quality metrics are assessed independently for each data domain
- The interface is backward-compatible with existing Meta_Learner usage
- Weights are always normalized to sum to 1.0 as the final step in `predict_weights`

### Weight Calculation Process

The complete weight calculation process follows these steps:

1. **Base Weight Calculation**: Weights start based on domain availability and expert specialties
2. **Quality Adjustment**: Each expert's weight is adjusted based on its domain's quality metrics
3. **Drift Adjustment**: Experts with detected drift have their weights reduced
4. **Final Normalization**: All weights are normalized to ensure they sum to 1.0

```python
# Inside MetaLearner.predict_weights

# 1. Calculate base weights from specialty matching
weights = self._calculate_specialty_weights(context)

# 2. Adjust for quality if metrics available
if 'domain_quality' in context:
    weights = self._adjust_weights_by_quality(weights, context['domain_quality'])

# 3. Adjust for drift if recent data available
if 'recent_data' in context:
    weights = self._adjust_weights_for_drift(weights, context['recent_data'])

# 4. Normalize to ensure weights sum to 1.0
weight_sum = sum(weights.values())
if weight_sum > 0:
    weights = {k: v / weight_sum for k, v in weights.items()}
```

See the test modules `tests/test_adaptive_weighting.py` and `tests/test_adaptive_weighting_integration.py` for detailed examples.

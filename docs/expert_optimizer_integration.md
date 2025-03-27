# Expert-Optimizer Integration Guide

This document provides a comprehensive guide to the expert-optimizer integration in the MoE framework, detailing how domain-specific optimizers are selected, configured, and used with each expert type.

## Overview

The expert-optimizer integration connects domain-specific expert models with evolutionary computation algorithms tailored for their specific characteristics. This integration has the following key components:

1. **ExpertOptimizerIntegration**: Main class that manages the connection between experts and optimizers
2. **OptimizerFactory**: Factory that creates pre-configured optimizers for specific expert domains
3. **Expert Evaluation Functions**: Domain-specific fitness functions for each expert type

## Expert Types and Optimizer Mappings

Each expert type is paired with an optimizer and evaluation function tailored to its domain characteristics:

| Expert Type | Recommended Optimizer | Evaluation Function | Domain Characteristics |
|-------------|----------------------|---------------------|------------------------|
| Physiological | Differential Evolution | rmse_with_smoothness_penalty | Continuous temporal patterns, smoothness constraints |
| Environmental | Evolution Strategy | mae_with_lag_penalty | Seasonal patterns, lag effects, multimodality |
| Behavioral | Ant Colony Optimization | weighted_rmse_mae | Sparse feature interactions, categorical variables |
| Medication History | Hybrid Evolutionary | treatment_response_score | Mixed variable types, temporal dependencies |

## Using Expert-Specific Optimizers

The `OptimizerFactory` now provides specialized methods for creating optimizers for each expert domain:

```python
from meta_optimizer.optimizers.optimizer_factory import OptimizerFactory

# Create the factory
factory = OptimizerFactory()

# Create domain-specific optimizers
physiological_optimizer = factory.create_physiological_optimizer(
    dim=10,
    bounds=[(-100, 100)] * 10,
    population_size=40
)

environmental_optimizer = factory.create_environmental_optimizer(
    dim=8,
    bounds=[(-10, 10)] * 8,
    adaptation_rate=0.3
)

behavioral_optimizer = factory.create_behavioral_optimizer(
    dim=12,
    bounds=[(0, 1)] * 12,
    alpha=1.5
)

medication_history_optimizer = factory.create_medication_history_optimizer(
    bounds=[(-5, 5)] * 6,
    population_size=25
)
```

## Domain-Specific Parameter Encodings

Each expert domain has specialized parameter encodings tailored to its characteristics:

### Physiological Domain Parameters

```python
{
    'adaptive': True,        # Enable adaptive mutation for handling smooth transitions
    'strategy': 'best1bin',  # Strategy that works well for smooth continuous spaces
    'mutation': 0.7,         # Higher mutation factor for escaping local optima
    'recombination': 0.9,    # Higher crossover rate for effective space exploration
    'init': 'latin'          # Latin hypercube sampling for better initial coverage
}
```

### Environmental Domain Parameters

```python
{
    'population_size': 50,             # Larger population helps with multimodal spaces
    'adaptation_rate': 0.2,            # Moderate adaptation rate for seasonal pattern detection
    'offspring_ratio': 2.0,            # Generate more offspring to increase exploration
    'selective_pressure': 0.3,         # Higher selective pressure for multimodal spaces
    'recombination': 'intermediate'    # Intermediate recombination good for smooth landscapes
}
```

### Behavioral Domain Parameters

```python
{
    'population_size': 30,      # Good balance for behavioral feature selection
    'alpha': 2.0,               # Higher pheromone importance for variable interactions
    'beta': 4.0,                # Higher heuristic importance for behavioral patterns
    'evaporation_rate': 0.15,   # Moderate evaporation for temporal patterns
    'q': 2.0,                   # Higher deposit factor for strong features
    'elitist': True             # Enable elitism to preserve good feature combinations
}
```

### Medication History Domain Parameters

```python
{
    'population_size': 30,            # Moderate population for medication patterns
    'crossover_rate': 0.8,            # Higher crossover for effective treatment combinations
    'mutation_rate': 0.2,             # Moderate mutation for exploring treatment alternatives
    'local_search': 'pattern',        # Pattern search for local optimization of treatment patterns
    'local_search_iterations': 5,     # Moderate local search iterations
    'mixed_integer': True             # Support for mixed integer parameters common in medication
}
```

## Using the Integration in Practice

The `ExpertOptimizerIntegration` class connects experts with their optimizers:

```python
from moe_framework.experts.expert_optimizer_integration import ExpertOptimizerIntegration
from moe_framework.experts.physiological_expert import PhysiologicalExpert

# Create an expert
expert = PhysiologicalExpert(...)

# Create the integration
integration = ExpertOptimizerIntegration(
    expert=expert,
    hyperparameter_space={
        'learning_rate': (0.001, 0.1),
        'batch_size': (16, 128),
        'hidden_units': (32, 256)
    },
    max_iterations=50
)

# Optimize the expert's hyperparameters
best_params = integration.optimize()
```

## Error Handling and Fallbacks

The integration includes robust error handling and fallback mechanisms:

1. If a specialized optimizer is not available, it falls back to a general-purpose optimizer
2. If an evaluation function is not found, it uses a default evaluation function
3. Parameter mapping ensures compatibility between expert configurations and optimizer requirements

## Best Practices

1. Always specify bounds for the hyperparameter space
2. Use domain-specific evaluation functions when available
3. Consider problem characteristics when selecting optimizers
4. Provide early stopping criteria to improve efficiency

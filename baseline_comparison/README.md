# Baseline Comparison Framework

This module provides tools for comparing the Meta Optimizer against established algorithm selection methods, with a focus on SATzilla-inspired feature-based selection.

## Overview

The baseline comparison framework includes:

1. **Feature-based algorithm selection** inspired by SATzilla
2. **Comparison runner** for benchmarking against Meta Optimizer
3. **Visualization tools** for comparative analysis

## Components

### SATzilla-Inspired Selector

The `SATzillaInspiredSelector` extracts features from optimization problems and uses machine learning to predict which algorithm will perform best. Key features include:

- **Problem feature extraction**: Analyzes optimization landscapes to extract meaningful features
- **Performance prediction**: Uses random forest regression to predict algorithm performance
- **Algorithm selection**: Selects the algorithm with the best predicted performance
- **Confidence estimation**: Provides confidence scores for algorithm selection

### Comparison Runner

The `BaselineComparison` class provides a framework for benchmarking the Meta Optimizer against baseline methods:

- **Training phase**: Collects data for training the baseline selectors
- **Comparison phase**: Evaluates selection methods on test problems
- **Performance analysis**: Calculates statistics and improvement percentages
- **Results collection**: Stores performance data for visualization

### Visualization Tools

The `ComparisonVisualizer` generates publication-quality visualizations for comparative analysis:

- **Head-to-head comparison**: Boxplots comparing performance distributions
- **Performance profiles**: Curves showing proportion of problems solved within a factor of the best
- **Ranking tables**: Statistics on algorithm rankings across problems
- **Critical difference diagrams**: Statistical significance of performance differences
- **Improvement heatmaps**: Pairwise performance improvements between methods

## Usage

```python
from baseline_comparison import BaselineComparison, ComparisonVisualizer

# Initialize with your Meta Optimizer and problem generator
comparison = BaselineComparison(meta_optimizer, problem_generator, algorithms)

# Train the baseline selectors
comparison.run_training_phase(n_problems=50)

# Run comparison
results = comparison.run_comparison(n_problems=20)

# Visualize results
visualizer = ComparisonVisualizer(results, export_dir='results/comparison')
visualizer.create_all_visualizations()
```

See `examples/baseline_comparison_demo.py` for a complete demonstration.

## Output

The framework generates various visualizations saved to the specified export directory:

- `head_to_head_comparison.png`: Boxplot of performance distributions
- `performance_profile.png`: Performance profile curves
- `rank_table.csv`: Table of algorithm rankings
- `critical_difference.png`: Critical difference diagram
- `improvement_heatmap.png`: Heatmap of pairwise improvements 
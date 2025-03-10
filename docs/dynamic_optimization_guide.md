# Dynamic Optimization Guide

This document provides a comprehensive guide to running dynamic optimization visualizations using the framework.

## Overview

Dynamic optimization visualizes how different optimization algorithms perform on problems that change over time, simulating real-world scenarios where the optimization landscape can drift due to external factors. This functionality helps identify which algorithms are most robust to changes in the problem space.

## Command Line Interface

Dynamic optimization visualizations can be generated using the `main_v2.py` script with the `--dynamic-optimization` argument.

### Basic Usage

```bash
python main_v2.py --dynamic-optimization --function=ackley --drift-type=sudden
```

### Available Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--function` | Test function to use | N/A | Yes |
| `--drift-type` | Type of drift | N/A | Yes |
| `--dim` | Problem dimensionality | `10` | No |
| `--drift-rate` | Rate of drift (0.0 to 1.0) | `0.1` | No |
| `--drift-interval` | Interval between drift events | `20` | No |
| `--severity` | Severity of drift (0.0 to 1.0) | `1.0` | No |
| `--max-iterations` | Maximum number of iterations | `500` | No |
| `--reoptimize-interval` | Re-optimize after this many evaluations | `50` | No |
| `--export-dir` | Directory for saving visualizations | `results` | No |
| `--show-plot` | Show plot in addition to saving it | `False` | No |

### Available Test Functions

The following test functions are available for dynamic optimization:

- `ackley`: Ackley function, a widely used test function with many local minima
- `rastrigin`: Rastrigin function, featuring many regular local minima
- `rosenbrock`: Rosenbrock function (banana function), a challenging valley-shaped function
- `sphere`: Sphere function, a simple convex function for testing basic optimization
- `griewank`: Griewank function, with many widespread local minima
- `levy`: Lévy function, with difficult local optima
- `schwefel`: Schwefel function, with a distant global optimum

### Available Drift Types

The following drift types are available:

- `sudden`: Abrupt change in the optimum location
- `oscillatory`: The optimum oscillates between positions
- `linear`: Gradual linear movement of the optimum
- `incremental`: Step-wise movement of the optimum
- `gradual`: Slow transition from one optimum to another
- `random`: Random movements of the optimum
- `noise`: Increased noise in the objective function over time

## Output Visualizations

The dynamic optimization functionality generates a visualization named according to the pattern:

```
dynamic_optimization_[function]_[drift_type].png
```

For example:
- `dynamic_optimization_ackley_sudden.png`
- `dynamic_optimization_rastrigin_oscillatory.png`
- `dynamic_optimization_sphere_linear.png`

These visualizations are saved in the specified `export-dir` (default: `results/`).

### Understanding the Visualization

The dynamic optimization visualization includes:

1. **Optimal Value Line**: A dashed line showing how the optimal value changes due to drift
2. **Algorithm Performance**: Lines showing the best score found by each optimizer over time
3. **Drift Events**: Visualized changes in the optimization landscape

The visualization helps you understand:
- Which algorithms adapt quickly to changes
- Which algorithms maintain good performance despite drift
- How different drift types affect different algorithms
- The relative stability of algorithms in dynamic environments

## Example Usage

### Basic Examples

```bash
# Ackley function with sudden drift
python main_v2.py --dynamic-optimization --function=ackley --drift-type=sudden

# Rastrigin function with oscillatory drift
python main_v2.py --dynamic-optimization --function=rastrigin --drift-type=oscillatory

# Sphere function with linear drift
python main_v2.py --dynamic-optimization --function=sphere --drift-type=linear
```

### Advanced Examples

```bash
# Lower dimensionality (3D) with custom drift rate
python main_v2.py --dynamic-optimization --function=rosenbrock --drift-type=gradual --dim=3 --drift-rate=0.15

# Higher severity drift with more frequent reoptimization
python main_v2.py --dynamic-optimization --function=levy --drift-type=random --severity=2.0 --reoptimize-interval=30

# Longer run with more iterations
python main_v2.py --dynamic-optimization --function=griewank --drift-type=incremental --max-iterations=1000

# Custom export directory
python main_v2.py --dynamic-optimization --function=schwefel --drift-type=noise --export-dir=results/custom_folder
```

## Comparing Multiple Visualizations

To compare how different optimization algorithms perform across various drift types and test functions, you can run a series of commands and then compare the generated visualizations:

```bash
# Compare drift types on the same function
python main_v2.py --dynamic-optimization --function=ackley --drift-type=sudden
python main_v2.py --dynamic-optimization --function=ackley --drift-type=oscillatory
python main_v2.py --dynamic-optimization --function=ackley --drift-type=linear

# Compare functions with the same drift type
python main_v2.py --dynamic-optimization --function=ackley --drift-type=sudden
python main_v2.py --dynamic-optimization --function=rastrigin --drift-type=sudden
python main_v2.py --dynamic-optimization --function=sphere --drift-type=sudden
```

## Interpreting Results

When analyzing dynamic optimization visualizations, look for:

1. **Adaptation Speed**: How quickly does each algorithm adapt to changes?
2. **Stability**: Which algorithms maintain stable performance despite drift?
3. **Recovery**: Can the algorithm recover after a significant drift event?
4. **Final Performance**: Which algorithm finds the best solution by the end of the run?
5. **Drift Type Sensitivity**: Are some algorithms more affected by certain drift types?

This information can help you select the most appropriate optimization algorithm for problems that may change over time.

## Technical Details

The dynamic optimization functionality operates by:

1. Creating a benchmark function based on the specified test function
2. Applying drift to this function according to the specified drift type and parameters
3. Running multiple optimizers on the changing function
4. Tracking performance and the optimal value over time
5. Generating a visualization of the results

The implementation is located in the following files:
- `visualization/dynamic_optimization_viz.py`: Contains the visualization code
- `cli/commands/dynamic_optimization.py`: Command implementation
- `meta_optimizer/benchmark/dynamic_benchmark.py`: Dynamic benchmark implementation

## Troubleshooting

If you encounter issues with dynamic optimization visualizations:

1. **No visualization generated**: Ensure the `--export-dir` directory exists
2. **Dimension mismatch errors**: Try reducing the dimension (`--dim`) to a smaller value
3. **Poor algorithm performance**: Try increasing `--max-iterations` or decreasing `--reoptimize-interval`
4. **No visible drift effect**: Try increasing the `--drift-rate` or `--severity` parameters 
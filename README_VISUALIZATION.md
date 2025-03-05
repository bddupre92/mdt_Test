# [DEPRECATED] Real-Time Optimization Visualization

**Note: This README has been consolidated into the main README.md file. Please refer to the main README.md for up-to-date information on the real-time visualization features.**

## Overview

The optimization framework now includes a real-time visualization system that allows you to monitor the progress of different optimization algorithms as they run. The visualization provides:

1. **Optimization Progress**: Shows the best score vs. number of evaluations for each optimizer
2. **Improvement Rate**: Displays how quickly each optimizer is improving
3. **Convergence Speed**: Tracks performance over time (rather than evaluations)
4. **Optimization Statistics**: Shows the best optimizer, best score, and other metrics

## Command-Line Arguments

The following command-line arguments control the visualization:

- `--live-viz`: Enable real-time visualization
- `--save-plots`: Save visualization results to files
- `--max-data-points`: Maximum number of data points to store per optimizer (default: 1000)
- `--no-auto-show`: Disable automatic plot display

## Examples

### Basic Visualization

To run optimization with real-time visualization:

```bash
python main.py --live-viz
```

### Save Visualization Results

To save the visualization plots and data:

```bash
python main.py --live-viz --save-plots
```

### Limit Data Storage

To limit the amount of data stored (useful for long-running optimizations):

```bash
python main.py --live-viz --max-data-points 500
```

## How It Works

The visualization system works by:

1. Each optimizer reports its progress after each iteration
2. The data is collected by the LiveOptimizationMonitor
3. The monitor updates the plots in real-time
4. Data is automatically downsampled to prevent excessive memory usage

## Memory Management

To prevent excessive memory usage during long-running optimizations:

1. The system limits the number of data points stored per optimizer
2. When the limit is reached, it keeps the first few points and the most recent points, downsampling the middle
3. This ensures that you can see both the initial behavior and the most recent progress

## Saving Results

When saving is enabled (with `--save-plots`), the system will save:

1. A PNG image of the final visualization state
2. A CSV file with the raw optimization data

These files are saved to the `results` directory.

## Customization

You can customize the visualization by modifying the `LiveOptimizationMonitor` class in `visualization/live_visualization.py`. Some possible customizations include:

- Changing the plot layout
- Adding new metrics to track
- Modifying the update frequency
- Changing the appearance of the plots

## Troubleshooting

If the visualization window doesn't appear:
- Make sure you're using a backend that supports interactive plotting
- Try running with `--no-auto-show` and then call `plt.show()` manually
- Check if your environment supports GUI applications

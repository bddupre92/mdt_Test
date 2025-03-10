#!/bin/bash
# Script to run a modular baseline comparison using main_v2.py

# Parse command line arguments
dimensions=2
num_trials=3
functions="sphere rosenbrock"
output_dir=""
all_functions=false
additional_args=""

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dimensions)
      dimensions="$2"
      shift 2
      ;;
    --num-trials)
      num_trials="$2"
      shift 2
      ;;
    --functions)
      functions="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --all-functions)
      all_functions=true
      shift
      ;;
    *)
      additional_args="$additional_args $1"
      shift
      ;;
  esac
done

# If all_functions is true, override the functions list
if [ "$all_functions" = true ]; then
    functions="sphere rosenbrock ackley rastrigin griewank schwefel"
fi

# Create timestamp for results directory if not specified
if [ -z "$output_dir" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_dir="results/baseline_comparison/${dimensions}D_${functions//[^a-zA-Z0-9]/_}/$timestamp"
fi

# Create output directory
mkdir -p "$output_dir/data"
mkdir -p "$output_dir/visualizations"

# Log configuration
echo "Running baseline comparison with:"
echo "  Dimensions: $dimensions"
echo "  Number of trials: $num_trials"
echo "  Functions: $functions"
echo "  Output directory: $output_dir"

# Run the comparison for each function
for func in $functions; do
    echo "Running comparison for function: $func"
    
    # Execute the baseline comparison command
    python main_v2.py baseline_comparison \
        --function "$func" \
        --dimensions "$dimensions" \
        --num-trials "$num_trials" \
        --output-dir "$output_dir" \
        $additional_args
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Baseline comparison failed for function $func"
        exit 1
    fi
done

# Generate a summary of the results
echo "Generating summary..."

echo "# Baseline Comparison Results" > "$output_dir/summary.md"
echo "" >> "$output_dir/summary.md"
echo "## Configuration" >> "$output_dir/summary.md"
echo "" >> "$output_dir/summary.md"
echo "- **Dimensions**: $dimensions" >> "$output_dir/summary.md"
echo "- **Number of trials**: $num_trials" >> "$output_dir/summary.md"
echo "- **Functions**: $functions" >> "$output_dir/summary.md"
echo "- **Run date**: $(date)" >> "$output_dir/summary.md"
echo "" >> "$output_dir/summary.md"
echo "## Results" >> "$output_dir/summary.md"
echo "" >> "$output_dir/summary.md"

# Extract results from the output files and append to summary
if [ -f "$output_dir/data/benchmark_results.json" ]; then
    echo "Results are available in: $output_dir/data/benchmark_results.json" >> "$output_dir/summary.md"
else
    echo "No results file found." >> "$output_dir/summary.md"
fi

echo "" >> "$output_dir/summary.md"
echo "## Visualizations" >> "$output_dir/summary.md"
echo "" >> "$output_dir/summary.md"

# List any visualization files
if [ "$(ls -A "$output_dir/visualizations" 2>/dev/null)" ]; then
    echo "Visualizations are available in: $output_dir/visualizations/" >> "$output_dir/summary.md"
    for viz in "$output_dir"/visualizations/*; do
        if [ -f "$viz" ]; then
            echo "- $(basename "$viz")" >> "$output_dir/summary.md"
        fi
    done
else
    echo "No visualizations generated." >> "$output_dir/summary.md"
fi

echo "Baseline comparison completed successfully!"
echo "Results are available in: $output_dir" 
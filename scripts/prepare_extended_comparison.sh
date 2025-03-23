#!/bin/bash
# Prepare the extended comparison scripts

# Make the scripts executable
chmod +x scripts/run_full_benchmark_suite.sh
chmod +x scripts/analyze_benchmark_results.py

echo "Extended comparison scripts prepared!"
echo
echo "To run the comprehensive benchmark suite, use:"
echo "  ./scripts/run_full_benchmark_suite.sh"
echo
echo "After running the benchmarks, analyze the results with:"
echo "  python scripts/analyze_benchmark_results.py --results-dir results/baseline_comparison/full_benchmark_YYYYMMDD"
echo "  (replace YYYYMMDD with the actual date)"
echo
echo "The analysis will generate visualizations, statistical tests, and summary reports." 
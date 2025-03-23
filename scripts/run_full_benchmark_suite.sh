#!/bin/bash
# Comprehensive benchmark suite for baseline comparison

# Set up fancy output formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    COMPREHENSIVE BASELINE COMPARISON SUITE       ${NC}"
echo -e "${BLUE}==================================================${NC}"

# Make sure the scripts are executable
chmod +x scripts/run_modular_baseline_comparison.sh
chmod +x main_v2.py

# Create a directory for the full benchmark results
RESULTS_DIR="results/baseline_comparison/full_benchmark_$(date +"%Y%m%d")"
mkdir -p "$RESULTS_DIR"

# Create a log file
LOG_FILE="$RESULTS_DIR/benchmark_log.txt"
touch "$LOG_FILE"

# Log start time
echo "Starting comprehensive benchmark suite at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to $RESULTS_DIR" | tee -a "$LOG_FILE"

# Function to run a benchmark and log results
run_benchmark() {
    local dims=$1
    local trials=$2
    local funcs=$3
    local additional_args=$4
    
    echo -e "\n${YELLOW}Running benchmark with:${NC}" | tee -a "$LOG_FILE"
    echo "  Dimensions: $dims" | tee -a "$LOG_FILE"
    echo "  Trials: $trials" | tee -a "$LOG_FILE"
    echo "  Functions: $funcs" | tee -a "$LOG_FILE"
    echo "  Additional args: $additional_args" | tee -a "$LOG_FILE"
    
    # Build the command
    local cmd="scripts/run_modular_baseline_comparison.sh --dimensions $dims --num-trials $trials --functions $funcs $additional_args --output-dir $RESULTS_DIR/${dims}D_${funcs//[^a-zA-Z0-9]/_}"
    
    echo "Executing: $cmd" | tee -a "$LOG_FILE"
    eval "$cmd"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Benchmark completed successfully!${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}Benchmark failed with exit code $exit_code${NC}" | tee -a "$LOG_FILE"
    fi
    
    echo "Completed at $(date)" | tee -a "$LOG_FILE"
    echo -e "${BLUE}--------------------------------------------------${NC}" | tee -a "$LOG_FILE"
}

# 1. Basic benchmark functions across dimensions
echo -e "\n${BLUE}Running basic benchmark functions across dimensions...${NC}" | tee -a "$LOG_FILE"

# 2D benchmarks (quick)
run_benchmark 2 5 "sphere rosenbrock ackley rastrigin"

# 5D benchmarks
run_benchmark 5 5 "sphere rosenbrock ackley rastrigin"

# 10D benchmarks
run_benchmark 10 5 "sphere rosenbrock ackley rastrigin"

# 2. Extended benchmarks with more trials
echo -e "\n${BLUE}Running extended benchmarks with more trials...${NC}" | tee -a "$LOG_FILE"

# 2D with more trials
run_benchmark 2 10 "sphere rosenbrock"

# 5D with more trials
run_benchmark 5 10 "sphere rosenbrock"

# 3. Dynamic function benchmarks
echo -e "\n${BLUE}Running dynamic function benchmarks...${NC}" | tee -a "$LOG_FILE"

# Linear drift
run_benchmark 2 5 "dynamic_sphere_linear dynamic_rosenbrock_linear"

# Oscillatory drift
run_benchmark 2 5 "dynamic_sphere_oscillatory dynamic_rosenbrock_oscillatory"

# Random drift
run_benchmark 2 5 "dynamic_sphere_random dynamic_rosenbrock_random"

# 4. All available functions (comprehensive)
echo -e "\n${BLUE}Running comprehensive benchmark with all functions...${NC}" | tee -a "$LOG_FILE"
run_benchmark 2 3 "all" "--all-functions"

# Create a summary file
echo -e "\n${BLUE}Creating benchmark summary...${NC}" | tee -a "$LOG_FILE"
cat > "$RESULTS_DIR/summary.md" << EOL
# Comprehensive Benchmark Results

## Overview

This directory contains results from a comprehensive benchmark suite comparing the Meta Optimizer against the SATzilla-inspired baseline selector.

## Benchmark Configurations

1. **Dimensions Tested**: 
   - 2D (basic)
   - 5D (medium)
   - 10D (challenging)

2. **Benchmark Functions**:
   - Standard functions: sphere, rosenbrock, ackley, rastrigin
   - Dynamic functions with different drift types:
     - Linear drift
     - Oscillatory drift
     - Random drift

3. **Trial Counts**:
   - 5 trials (standard)
   - 10 trials (extended for statistical significance)

## Directory Structure

Each subdirectory is named according to its configuration:
\`\`\`
{dimensions}D_{function_names}
\`\`\`

For example, \`5D_sphere_rosenbrock\` contains results for 5-dimensional sphere and rosenbrock functions.

## Analysis

To analyze these results, run:
\`\`\`
python scripts/analyze_benchmark_results.py --results-dir "$RESULTS_DIR"
\`\`\`

This will generate comparative visualizations and statistical analysis of the benchmark results.

## Conducted on

Date: $(date)
EOL

echo -e "${GREEN}Comprehensive benchmark suite completed!${NC}" | tee -a "$LOG_FILE"
echo -e "Results and logs saved to: ${YELLOW}$RESULTS_DIR${NC}" | tee -a "$LOG_FILE"
echo -e "${BLUE}==================================================${NC}" 
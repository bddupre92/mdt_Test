#!/bin/bash
# Comprehensive script to run the baseline comparison and organize results

# Set up fancy output formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}    BASELINE COMPARISON BENCHMARK RUNNER      ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p results/baseline_comparison/logs
mkdir -p results/baseline_comparison/data
mkdir -p results/baseline_comparison/visualizations

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/baseline_comparison/logs"
DATA_DIR="results/baseline_comparison/data"
VIZ_DIR="results/baseline_comparison/visualizations"

# Make scripts executable
echo -e "\n${YELLOW}Setting up environment...${NC}"
chmod +x run_tests.sh
chmod +x tests/debug_utils.py
chmod +x tests/test_benchmark.py

# Check if the current directory is in the Python path
# If not, set the PYTHONPATH environment variable
DIR="$(pwd)"
if [[ ":$PYTHONPATH:" != *":$DIR:"* ]]; then
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$DIR"
    echo -e "${GREEN}Added $DIR to PYTHONPATH${NC}"
fi

# Run debug utilities to check setup
echo -e "\n${YELLOW}Running debug utilities...${NC}"
python tests/debug_utils.py > "$LOG_DIR/debug_${TIMESTAMP}.log" 2>&1

# Check if debug utilities ran successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Debug checks passed!${NC}"
    
    # Run test benchmark
    echo -e "\n${YELLOW}Running baseline comparison benchmark...${NC}"
    echo -e "${YELLOW}This may take a few minutes depending on the number of benchmark functions and trials...${NC}"
    
    python tests/test_benchmark.py > "$LOG_DIR/test_benchmark_${TIMESTAMP}.log" 2>&1
    
    # Check if test benchmark ran successfully
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Benchmark completed successfully!${NC}"
        
        # Organize results
        echo -e "\n${YELLOW}Organizing results...${NC}"
        
        # Move JSON and TXT files to data directory
        find results/baseline_comparison -maxdepth 1 -name "*.json" -exec mv {} "$DATA_DIR/" \;
        find results/baseline_comparison -maxdepth 1 -name "*.txt" -exec mv {} "$DATA_DIR/" \;
        
        # Move image files to visualizations directory
        find results/baseline_comparison -maxdepth 1 -name "*.png" -exec mv {} "$VIZ_DIR/" \;
        
        # Create index file
        echo -e "\n${YELLOW}Creating result index...${NC}"
        cat > "results/baseline_comparison/index.md" << EOL
# Baseline Comparison Results

## Overview

This directory contains the results of comparing the Meta Optimizer against the SATzilla-inspired baseline algorithm selector.

## Directory Structure

- \`data/\`: Contains raw data in JSON and TXT formats
- \`visualizations/\`: Contains performance plots and other visualizations
- \`logs/\`: Contains execution logs with timestamps

## Latest Run

- **Date**: $(date "+%Y-%m-%d %H:%M:%S")
- **Logs**: \`logs/debug_${TIMESTAMP}.log\` and \`logs/test_benchmark_${TIMESTAMP}.log\`

## Key Results

### Performance Summary

$(cat "$DATA_DIR/benchmark_summary.txt" 2>/dev/null || echo "Summary not available yet.")

### Available Visualizations

$(find "$VIZ_DIR" -type f -name "*.png" | while read -r file; do
    echo "- [$(basename "$file")](./visualizations/$(basename "$file"))"
done)

EOL
        
        echo -e "${GREEN}Results organized successfully!${NC}"
        echo -e "\n${BLUE}===============================================${NC}"
        echo -e "${GREEN}BENCHMARK COMPLETED SUCCESSFULLY${NC}"
        echo -e "${BLUE}===============================================${NC}"
        echo -e "\nResults available in: ${YELLOW}results/baseline_comparison/${NC}"
        echo -e "Summary index: ${YELLOW}results/baseline_comparison/index.md${NC}"
    else
        echo -e "${RED}Benchmark failed. Check logs for details:${NC}"
        echo -e "${YELLOW}$LOG_DIR/test_benchmark_${TIMESTAMP}.log${NC}"
        echo -e "\nLast 10 lines of log:"
        tail -n 10 "$LOG_DIR/test_benchmark_${TIMESTAMP}.log"
        exit 1
    fi
else
    echo -e "${RED}Debug checks failed. Check logs for details:${NC}"
    echo -e "${YELLOW}$LOG_DIR/debug_${TIMESTAMP}.log${NC}"
    echo -e "\nLast 10 lines of log:"
    tail -n 10 "$LOG_DIR/debug_${TIMESTAMP}.log"
    exit 1
fi 
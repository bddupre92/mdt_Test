#!/bin/bash
# Run MoE validation with enhanced synthetic data
# This script demonstrates how to use the enhanced synthetic data features

# Set current directory to script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Ensure enhanced synthetic data is prepared
echo "Preparing enhanced synthetic data..."
# First remove existing visualization files to avoid symlink errors
rm -f results/moe_validation/visualizations/*
python prepare_enhanced_validation.py

# Define configuration paths
CONFIG_PATH="results/moe_validation/enhanced_validation_config.json"
RESULTS_DIR="results/moe_validation/enhanced_run_$(date +%Y%m%d_%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run MoE validation with enhanced data
echo "Running MoE validation with enhanced synthetic data..."

# Set environment variable for enhanced data configuration
export ENHANCED_DATA_CONFIG="$CONFIG_PATH"
echo "Set ENHANCED_DATA_CONFIG=$CONFIG_PATH"

# Run MoE validation with enhanced data
python main_v2.py moe_validation \
  --interactive \
  --explainers shap feature_importance \
  --notify \
  --results-dir "$RESULTS_DIR" \
  --components all \
  --enable-continuous-explain \
  --notify \
  --notify-with-visuals

echo "MoE validation completed. Results available in $RESULTS_DIR"
echo "Interactive report is available in $RESULTS_DIR/reports/"

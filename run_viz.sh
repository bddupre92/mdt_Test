#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d ".venv_minimal" ]; then
    python -m venv .venv_minimal
fi

# Activate virtual environment
source .venv_minimal/bin/activate

# Create output directory
mkdir -p results/visualizations/benchmarks

# Run visualization script with interactive mode enabled
python visualization/visualization_generator.py \
    --dims 2 5 \
    --noise-levels 0.0 \
    --runs 3 \
    --max-evals 200 \
    --output-dir 'results/visualizations/benchmarks' \
    --interactive \
    --verbose 
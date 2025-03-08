#!/bin/bash
# Make sure the script is executable
chmod +x run.sh

# Create results directory
mkdir -p results/visualizations

# Run the visualization generator
python3 visualization/visualization_generator.py --dims 2 5 --noise-levels 0.0 --runs 2 --max-evals 200 --output-dir 'results/visualizations' --verbose 

# Run the script
./run.sh 
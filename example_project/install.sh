#!/bin/bash
# Simple installation script for the example project

echo "Creating a virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing migraine prediction package from source distribution..."
pip install ../migraine_prediction_project/dist/migraine_prediction-0.1.0.tar.gz

echo "Installation complete! You can now run the sample_usage.py script:"
echo "source venv/bin/activate"
echo "python sample_usage.py"

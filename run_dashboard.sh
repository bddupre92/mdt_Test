#!/bin/bash

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Install MoE framework in development mode
echo "Installing MoE framework..."
pip install -e .

# Check dependencies
echo "Checking dependencies..."
python check_dependencies.py
if [ $? -ne 0 ]; then
    echo "Dependency check failed. Please fix the issues and try again."
    exit 1
fi

# Start Streamlit dashboard
echo "Starting dashboard..."
streamlit run moe_dashboard.py 
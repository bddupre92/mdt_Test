#!/bin/bash

# Run dashboard script with error filtering
# This script runs the Streamlit dashboard while filtering out specific PyTorch errors

# Change to the project directory
cd "$(dirname "$0")"

# Run Streamlit with error filtering
streamlit run app/ui/benchmark_dashboard.py 2> >(grep -v "Tried to instantiate class '__path__._path'" | grep -v "no running event loop")

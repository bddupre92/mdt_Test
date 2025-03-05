#!/bin/bash
# This script uploads packages to the main PyPI repository

# Clean build directories
rm -rf dist build meta_optimizer_mdt_test.egg-info

# Build the package
echo "Building package..."
python -m build

# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload dist/*

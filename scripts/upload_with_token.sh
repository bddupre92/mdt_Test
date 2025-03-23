#!/bin/bash
# This script uploads packages to TestPyPI without requiring interactive token input

# Check if token is provided
if [ -z "$1" ]; then
  echo "Usage: ./upload_with_token.sh YOUR_TESTPYPI_TOKEN"
  exit 1
fi

# Clean build directories
rm -rf dist build meta_optimizer_mdt.egg-info

# Build the package
python -m build

# Upload to TestPyPI using the provided token
TWINE_USERNAME=__token__ TWINE_PASSWORD="$1" python -m twine upload --repository testpypi dist/*

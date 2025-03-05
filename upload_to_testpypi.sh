#!/bin/bash
# Build the package
python -m build

# Upload to TestPyPI
# You'll be prompted for your TestPyPI token
python -m twine upload --repository testpypi dist/*

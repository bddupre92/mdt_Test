#!/bin/bash

# Run tests with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing tests/

# Run specific test suites if provided
if [ $# -gt 0 ]; then
    pytest --cov=app --cov-report=html --cov-report=term-missing tests/test_$1.py
fi

# Check test coverage threshold
coverage_threshold=80
coverage_score=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')

if (( $(echo "$coverage_score < $coverage_threshold" | bc -l) )); then
    echo "Warning: Test coverage ($coverage_score%) is below threshold ($coverage_threshold%)"
    exit 1
fi

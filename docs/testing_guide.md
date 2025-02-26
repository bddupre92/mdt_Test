# Migraine Prediction Testing Guide

This guide covers the testing infrastructure and procedures for the Migraine Prediction application.

## Table of Contents
1. [Setup](#setup)
2. [Data Generation](#data-generation)
3. [Running Tests](#running-tests)
4. [Test Scenarios](#test-scenarios)
5. [Analyzing Results](#analyzing-results)

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Initialize development environment:
   ```bash
   ./scripts/setup_dev.sh
   ```

## Data Generation

### Test Data
Generate data for testing purposes:
```bash
# Generate 30 days of test data
python scripts/test_environment.py generate-test --days 30

# Generate data with more days
python scripts/test_environment.py generate-test --days 90
```

### Production-like Data
Generate more realistic data with missing values and drift:
```bash
# Generate data for 5 patients over 90 days
python scripts/test_environment.py generate-production --patients 5 --days 90
```

### Validation Data
Generate special datasets for drift detection validation:
```bash
python scripts/test_environment.py analyze-drift
```

## Running Tests

### All Tests
Run the complete test suite:
```bash
./scripts/run_tests.sh
```

### Specific Test Suites
Run individual test suites:
```bash
# Data generation tests
pytest tests/test_data_generation.py -v

# Prediction service tests
pytest tests/test_prediction_service.py -v

# API route tests
pytest tests/test_api_routes.py -v

# Drift detection tests
pytest tests/test_drift_detection.py -v
```

### Test Coverage
Generate and view coverage report:
```bash
pytest --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

## Test Scenarios

### 1. Basic Prediction Testing
```bash
# Run simulation without drift
python scripts/test_environment.py simulate --days 30 --no-drift --scenario "basic_test"
```

### 2. Drift Detection Testing
```bash
# Run simulation with drift
python scripts/test_environment.py simulate --days 90 --drift --scenario "drift_test"
```

### 3. Long-term Performance Testing
```bash
# Run extended simulation
python scripts/test_environment.py simulate --days 180 --drift --scenario "long_term"
```

## Analyzing Results

### 1. View Test Results
Test results are saved in `test_data/results/` with the following naming convention:
- `{scenario_name}_{timestamp}.json`

### 2. Drift Analysis
```bash
# Run drift analysis
python scripts/test_environment.py analyze-drift
```

### 3. Performance Metrics
Each test scenario provides:
- Accuracy
- Precision
- Recall
- Number of days drift was detected

## Common Testing Workflows

### 1. Quick Test
```bash
# Generate test data
python scripts/test_environment.py generate-test --days 30

# Run basic simulation
python scripts/test_environment.py simulate --days 30 --no-drift --scenario "quick_test"
```

### 2. Comprehensive Test
```bash
# Generate production data
python scripts/test_environment.py generate-production --patients 5 --days 90

# Run simulation with drift
python scripts/test_environment.py simulate --days 90 --drift --scenario "full_test"

# Analyze drift
python scripts/test_environment.py analyze-drift
```

### 3. Development Testing
```bash
# Run specific test file
pytest tests/test_prediction_service.py -v

# Run tests with print statements
pytest -s tests/test_prediction_service.py
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Database Issues**
   ```bash
   python -m app.core.db.init_db
   ```

3. **Data Generation Errors**
   - Check disk space
   - Verify write permissions in test_data directory
   - Ensure all required packages are installed

### Getting Help

1. Check the logs in `logs/` directory
2. Run tests with increased verbosity:
   ```bash
   pytest -vv tests/
   ```
3. Enable debug logging:
   ```bash
   export DEBUG=1
   pytest tests/
   ```

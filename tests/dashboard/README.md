# Performance Analysis Dashboard Tests

This directory contains automated tests for the Performance Analysis Dashboard, focusing on verifying its flexibility in handling various data formats and structures.

## Test Structure

The tests use Playwright for browser automation and pytest as the test framework. They verify that the dashboard can:

1. Load different checkpoint formats without errors
2. Display data correctly regardless of the underlying structure
3. Dynamically adapt visualizations to different data formats
4. Allow users to customize how they want to interpret the data

## Test Data

Test data is located in `/test_data/performance_formats/` and includes:

- `standard_format.json` - Data in the standard expected format
- `nested_format.json` - Deeply nested data structure with performance metrics inside other structures
- `flat_format.json` - A flat key-value structure without hierarchy
- `unusual_format.json` - A complex, atypical data structure to test flexibility

## Running Tests

To run all dashboard tests:

```bash
python run_dashboard_tests.py
```

Or to run specific tests:

```bash
python -m pytest tests/dashboard/test_dashboard_data_format_flexibility.py -v
```

## Test Coverage

The tests verify:

1. **Data Loading Flexibility**: Tests that the dashboard correctly loads and processes data in various formats
2. **UI Adaptation**: Verifies that UI components adapt based on the data structure
3. **Data Inspector**: Tests the data structure inspector in the dashboard sidebar
4. **Error Handling**: Ensures the dashboard gracefully handles unexpected data formats
5. **Path Selection**: Tests the custom data path functionality that allows users to specify where metrics are located

## Test Configuration

Tests are configured in `conftest.py`, which:

1. Creates test checkpoint files from the different data formats
2. Starts a test instance of the dashboard server
3. Sets up browser automation with Playwright
4. Cleans up after tests complete

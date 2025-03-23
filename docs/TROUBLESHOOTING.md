# Troubleshooting Guide

This document provides solutions for common issues when running the baseline comparison framework tests.

## Import Errors

### Module Not Found Errors

If you see errors like "No module named 'baseline_comparison'", try the following:

1. Make sure the current directory is in your Python path:
   ```bash
   export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
   ```

2. Verify the directory structure is correct:
   ```
   .
   ├── baseline_comparison/
   │   ├── __init__.py
   │   ├── ...
   └── tests/
       ├── __init__.py
       ├── ...
   ```

3. Check that all `__init__.py` files exist and have the correct imports.

### AttributeError or ImportError for ComparisonVisualizer

If you see errors related to `ComparisonVisualizer`, ensure that:

1. The `visualization.py` file includes the `ComparisonVisualizer` class.
2. The `__init__.py` file imports it correctly.
3. There are no circular imports.

## Execution Errors

### Permission Denied

If you see "Permission denied" errors when running scripts, make them executable:

```bash
chmod +x script_name.sh
```

### Script Not Found

If you get "No such file or directory" errors, check:

1. The file exists in the expected location.
2. You're running the command from the correct directory.
3. On Windows, ensure you're using the correct path separators.

## Test Failures

### Function Evaluation Errors

If test functions fail to evaluate:

1. Check that benchmark function implementations are correct.
2. Verify input dimensions match expected dimensions.
3. Ensure bounds are respected in the optimization process.

### Visualization Errors

If visualization fails:

1. Make sure matplotlib is installed: `pip install matplotlib`.
2. Check that the results dictionary has the expected structure.
3. Verify that all required keys exist in the results.

## MetaOptimizer Integration Issues

If MetaOptimizer integration fails:

1. Check if the MetaOptimizer class exists and is importable.
2. Verify the interface (method signatures) matches what's expected.
3. If using a mock, ensure it implements all required methods.

## Performance Issues

If the tests run but are very slow:

1. Reduce `max_evaluations` and `num_trials` for faster testing.
2. Use simpler benchmark functions with lower dimensions.
3. Check for any infinite loops in the optimization algorithms.

## Getting More Information

To get more detailed error information, run:

```bash
./run_and_log_tests.sh
```

This will create log files in the `logs` directory with detailed output. 
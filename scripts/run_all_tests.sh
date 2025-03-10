#!/bin/bash
# Comprehensive script to run all tests and check for errors

# Exit on error
set -e

echo "===== Making all scripts executable ====="
chmod +x run_tests.sh
chmod +x run_tests_now.sh
chmod +x make_scripts_executable.sh
chmod +x make_executable.sh
chmod +x tests/debug_utils.py
chmod +x tests/test_benchmark.py

echo "===== Setting up Python path ====="
# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "===== Running debug utilities to check imports and setup ====="
python tests/debug_utils.py

# Store the exit code of the debug utils
DEBUG_EXIT_CODE=$?

if [ $DEBUG_EXIT_CODE -eq 0 ]; then
    echo "===== Debug checks passed, running test benchmark ====="
    python tests/test_benchmark.py
    
    # Store the exit code of the test benchmark
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "===== SUCCESS: All tests passed! ====="
    else
        echo "===== ERROR: Test benchmark failed with exit code $TEST_EXIT_CODE ====="
        exit $TEST_EXIT_CODE
    fi
else
    echo "===== ERROR: Debug checks failed with exit code $DEBUG_EXIT_CODE ====="
    exit $DEBUG_EXIT_CODE
fi

echo "===== Test execution completed successfully =====" 
#!/bin/bash
# Run tests and log the output for error analysis

# Create logs directory if it doesn't exist
mkdir -p results/baseline_comparison/logs

# Timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Make scripts executable
echo "===== Making scripts executable ====="
chmod +x run_tests.sh
chmod +x tests/debug_utils.py
chmod +x tests/test_benchmark.py
chmod +x run_all_tests.sh

# Run debug utilities and log output
echo "===== Running debug utilities ====="
python tests/debug_utils.py > results/baseline_comparison/logs/debug_${TIMESTAMP}.log 2>&1

# Check if debug utilities ran successfully
if [ $? -eq 0 ]; then
    echo "Debug utilities completed successfully"
    
    # Run test benchmark and log output
    echo "===== Running test benchmark ====="
    python tests/test_benchmark.py > results/baseline_comparison/logs/test_benchmark_${TIMESTAMP}.log 2>&1
    
    # Check if test benchmark ran successfully
    if [ $? -eq 0 ]; then
        echo "Test benchmark completed successfully"
        echo "All tests passed!"
    else
        echo "Test benchmark failed. Check results/baseline_comparison/logs/test_benchmark_${TIMESTAMP}.log for details"
        echo "Last 10 lines of test benchmark log:"
        tail -n 10 results/baseline_comparison/logs/test_benchmark_${TIMESTAMP}.log
    fi
else
    echo "Debug utilities failed. Check results/baseline_comparison/logs/debug_${TIMESTAMP}.log for details"
    echo "Last 10 lines of debug log:"
    tail -n 10 results/baseline_comparison/logs/debug_${TIMESTAMP}.log
fi

echo "===== Test run completed ====="
echo "Log files:"
echo "  - results/baseline_comparison/logs/debug_${TIMESTAMP}.log"
echo "  - results/baseline_comparison/logs/test_benchmark_${TIMESTAMP}.log" 
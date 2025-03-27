#!/bin/bash
# Script to run the dashboard test with the correct environment setup

# Make sure we're using Python from the virtual environment
source .venv/bin/activate

# Set environment variables for the test
export MOE_CHECKPOINT_DIR="checkpoints/dev"

# Print debug info
echo "Using checkpoint directory: $MOE_CHECKPOINT_DIR"
echo "Checking if directory exists:"
ls -la $MOE_CHECKPOINT_DIR

# First make sure we have the right dependencies
npm install

# Start the Streamlit server
echo "Starting Streamlit server..."
streamlit run app/ui/performance_analysis_dashboard.py --server.port=8506 --server.headless=true > streamlit_server.log 2>&1 &
STREAMLIT_PID=$!

# Wait for the server to start
echo "Waiting for Streamlit server to start..."
sleep 10  # Give it some time to initialize

# Check if the server is running
if ! curl -s http://localhost:8506 > /dev/null; then
  echo "WARNING: Streamlit server may not be running correctly. Check streamlit_server.log for details."
  cat streamlit_server.log
else
  echo "Streamlit server is running"
fi

# Run the test using Playwright
echo "Running tests..."
npx playwright test tests/dashboard.spec.ts --headed
TEST_RESULT=$?

# Shutdown the Streamlit server
echo "Shutting down Streamlit server (PID: $STREAMLIT_PID)..."
kill $STREAMLIT_PID

# Note: The --headed flag shows the browser UI, remove for CI environments

# Return the test result
exit $TEST_RESULT
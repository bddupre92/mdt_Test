#!/bin/bash
# Setup and run the application with test data

# Set up environment
echo "Setting up environment..."
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create test user if not exists
echo "Creating test user..."
python scripts/create_test_user.py --username testuser --password password123 --email testuser@example.com

# Generate test data for drift detection
echo "Generating test data for drift detection..."
python scripts/populate_drift_data.py --username testuser --days 90 --drift-points 5

# Run the application
echo "Starting the application..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

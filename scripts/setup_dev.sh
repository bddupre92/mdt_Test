#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file. Please update with your configuration."
fi

# Generate synthetic test data
echo "Generating synthetic test data..."
python scripts/generate_test_data.py

# Initialize database
python -m app.core.db.init_db

# Run tests to verify setup
./scripts/run_tests.sh

echo "Development environment setup complete!"
echo "Generated test data is available in the data/ directory:"
ls -l data/
echo ""
echo "Next steps:"
echo "1. Update .env with your configuration"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Start the development server: uvicorn app.main:app --reload"

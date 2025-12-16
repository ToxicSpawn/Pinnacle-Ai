#!/bin/bash
# Run tests for Pinnacle AI

set -e

echo "Running Pinnacle AI tests..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

# Run integration tests
if [ -d "tests/integration" ] && [ "$(ls -A tests/integration)" ]; then
    echo "Running integration tests..."
    python -m pytest tests/integration/ -v
fi

# Run end-to-end tests
if [ -d "tests/e2e" ] && [ "$(ls -A tests/e2e)" ]; then
    echo "Running end-to-end tests..."
    python -m pytest tests/e2e/ -v
fi

echo "All tests completed!"


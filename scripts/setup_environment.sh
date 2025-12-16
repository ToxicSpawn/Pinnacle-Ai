#!/bin/bash
# Setup environment for Pinnacle AI

set -e

echo "Setting up Pinnacle AI environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p config/prompts

# Copy example config if it doesn't exist
if [ ! -f "config/settings.yaml" ]; then
    echo "Creating config file from example..."
    cp config/settings.yaml.example config/settings.yaml
    echo "Please edit config/settings.yaml with your API keys"
fi

echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"


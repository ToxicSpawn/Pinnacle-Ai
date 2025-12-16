#!/bin/bash
# Comprehensive setup script for Pinnacle AI

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$(id -u)" -eq 0 ]; then
    echo -e "${RED}Error: This script should not be run as root${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install package
install_package() {
    if ! command_exists "$1"; then
        echo -e "${YELLOW}Installing $1...${NC}"
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y "$1"
        elif command_exists yum; then
            sudo yum install -y "$1"
        elif command_exists brew; then
            brew install "$1"
        else
            echo -e "${RED}Error: Package manager not found. Please install $1 manually.${NC}"
            exit 1
        fi
    fi
}

# Check for Python
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3.9 or later.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$(printf '%s\n' "3.9" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.9" ]; then
    echo -e "${RED}Error: Python 3.9 or later is required. Found Python $PYTHON_VERSION.${NC}"
    exit 1
fi

# Install required system packages
echo -e "${GREEN}Installing system dependencies...${NC}"
install_package git
install_package python3-pip
install_package python3-venv
install_package build-essential
install_package libgl1  # For OpenCV
install_package libglib2.0-0  # For OpenCV

# Create virtual environment
echo -e "${GREEN}Creating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies if in development mode
if [ "$1" = "--dev" ]; then
    echo -e "${GREEN}Installing development dependencies...${NC}"
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
fi

# Set up configuration
echo -e "${GREEN}Setting up configuration...${NC}"
if [ ! -f "config/settings.yaml" ]; then
    if [ -f "config/settings.yaml.example" ]; then
        cp config/settings.yaml.example config/settings.yaml
        echo -e "${YELLOW}Please edit config/settings.yaml with your API keys and settings.${NC}"
    else
        echo -e "${YELLOW}Configuration file not found. Please create config/settings.yaml.${NC}"
    fi
else
    echo -e "${YELLOW}Configuration file already exists. Skipping.${NC}"
fi

# Create data directories
echo -e "${GREEN}Creating data directories...${NC}"
mkdir -p data/logs
mkdir -p data/models
mkdir -p data/cache

# Set up aliases
echo -e "${GREEN}Setting up command aliases...${NC}"
{
    echo ""
    echo "# Pinnacle AI aliases"
    echo "alias pinnacle='python $(pwd)/src/main.py'"
    echo "alias pinnacle-interactive='python $(pwd)/src/main.py --interactive'"
    echo "alias pinnacle-benchmark='python $(pwd)/src/main.py --benchmark'"
    echo "alias pinnacle-web='python $(pwd)/src/main.py --web'"
    echo "alias pinnacle-api='python $(pwd)/src/main.py --api'"
    echo "alias pinnacle-update='git pull && pip install -r requirements.txt'"
} >> ~/.bashrc

# Source the bashrc to make aliases available
source ~/.bashrc

# Check for Docker
if command_exists docker; then
    echo -e "${GREEN}Docker is installed.${NC}"
    echo -e "${YELLOW}You can run Pinnacle AI with Docker using: docker-compose up --build${NC}"
else
    echo -e "${YELLOW}Docker not found. Install Docker for containerized deployment.${NC}"
fi

# Check for Kubernetes
if command_exists kubectl; then
    echo -e "${GREEN}Kubernetes is installed.${NC}"
    echo -e "${YELLOW}You can deploy Pinnacle AI to Kubernetes using the provided manifests.${NC}"
else
    echo -e "${YELLOW}kubectl not found. Install Kubernetes for cluster deployment.${NC}"
fi

# Run tests
echo -e "${GREEN}Running tests...${NC}"
if [ -d "tests" ]; then
    python -m pytest tests/unit/ -v --tb=short || echo -e "${YELLOW}Some tests failed, but continuing...${NC}"
fi

# Completion message
echo -e "${GREEN}"
echo "============================================="
echo " Pinnacle AI setup completed successfully! "
echo "============================================="
echo ""
echo "To get started:"
echo "1. Edit config/settings.yaml with your API keys"
echo "2. Try these commands:"
echo "   - pinnacle-interactive"
echo "   - pinnacle 'Your task here'"
echo "   - pinnacle-benchmark"
echo "   - pinnacle-web (for web interface)"
echo "   - pinnacle-api (for API server)"
echo ""
echo "For Docker deployment:"
echo "   docker-compose up --build"
echo ""
echo "For production deployment, see the deployment documentation."
echo -e "${NC}"


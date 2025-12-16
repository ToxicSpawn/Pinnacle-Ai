# Setup environment for Pinnacle AI (PowerShell)

Write-Host "Setting up Pinnacle AI environment..." -ForegroundColor Green

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install development dependencies
if (Test-Path "requirements-dev.txt") {
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    pip install -r requirements-dev.txt
}

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path data | Out-Null
New-Item -ItemType Directory -Force -Path config/prompts | Out-Null

# Copy example config if it doesn't exist
if (-not (Test-Path "config/settings.yaml")) {
    Write-Host "Creating config file from example..." -ForegroundColor Yellow
    Copy-Item config/settings.yaml.example config/settings.yaml
    Write-Host "Please edit config/settings.yaml with your API keys" -ForegroundColor Cyan
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan


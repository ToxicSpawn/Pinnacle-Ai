#!/usr/bin/env python3
"""
Deployment script for Pinnacle AI
"""

import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if all dependencies are installed."""
    required = ['docker', 'docker-compose']
    missing = []
    
    for dep in required:
        if not shutil.which(dep):
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        return False
    return True

def build_docker():
    """Build Docker image."""
    print("Building Docker image...")
    subprocess.run(['docker', 'build', '-t', 'pinnacle-ai', '.'], check=True)
    print("Docker image built successfully!")

def deploy():
    """Deploy Pinnacle AI."""
    print("Deploying Pinnacle AI...")
    
    if not check_dependencies():
        print("Please install missing dependencies")
        return
    
    # Build Docker image
    build_docker()
    
    # Start with docker-compose
    print("Starting services with docker-compose...")
    subprocess.run(['docker-compose', 'up', '-d'], check=True)
    
    print("Deployment complete!")
    print("Pinnacle AI is running. Check logs with: docker-compose logs -f")

if __name__ == "__main__":
    deploy()


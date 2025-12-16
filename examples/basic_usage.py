#!/usr/bin/env python3
"""
Basic Usage Example - Pinnacle AI

This example demonstrates basic usage of Pinnacle AI.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import PinnacleAI

def main():
    """Demonstrate basic Pinnacle AI usage."""
    print("=" * 60)
    print("Pinnacle AI - Basic Usage Example")
    print("=" * 60)
    
    # Initialize Pinnacle AI
    try:
        pinnacle = PinnacleAI()
        print("✓ Pinnacle AI initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        print("\nMake sure config/settings.yaml exists and is properly configured")
        return
    
    # Example 1: Simple task
    print("Example 1: Simple Task")
    print("-" * 60)
    result = pinnacle.execute_task("Write a Python function to calculate factorial")
    print(f"Success: {result['evaluation']['success']}")
    print(f"Quality: {result['evaluation']['quality']:.1%}\n")
    
    # Example 2: Research task
    print("Example 2: Research Task")
    print("-" * 60)
    result = pinnacle.execute_task("Research the latest developments in AI")
    print(f"Success: {result['evaluation']['success']}")
    print(f"Agents used: {[r.get('agent') for r in result['execution']['execution']]}\n")
    
    # Example 3: Creative task
    print("Example 3: Creative Task")
    print("-" * 60)
    result = pinnacle.execute_task("Write a short poem about artificial intelligence")
    print(f"Success: {result['evaluation']['success']}\n")
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()


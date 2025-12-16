#!/usr/bin/env python3
"""
Benchmark script for Pinnacle AI
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import PinnacleAI

def main():
    """Run benchmark tests."""
    print("=" * 60)
    print("Pinnacle AI Benchmark Suite")
    print("=" * 60)
    
    # Initialize Pinnacle AI
    try:
        pinnacle = PinnacleAI()
    except Exception as e:
        print(f"Failed to initialize Pinnacle AI: {e}")
        print("Make sure config/settings.yaml exists and is properly configured")
        return
    
    # Benchmark tasks
    benchmark_tasks = [
        {
            "description": "Write a Python script that sorts a list of numbers",
            "category": "coding"
        },
        {
            "description": "Research the latest advancements in quantum computing",
            "category": "research"
        },
        {
            "description": "Create a short story about an AI that gains consciousness",
            "category": "creative"
        },
        {
            "description": "Plan a complex project with multiple dependencies",
            "category": "planning"
        },
        {
            "description": "Analyze the philosophical implications of artificial general intelligence",
            "category": "philosophy"
        }
    ]
    
    print(f"\nRunning {len(benchmark_tasks)} benchmark tasks...\n")
    
    # Run benchmark
    results = pinnacle.benchmark(benchmark_tasks)
    
    # Display results
    pinnacle._display_benchmark_results(results)
    
    # Save results
    import json
    results_file = Path("data/benchmark_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()


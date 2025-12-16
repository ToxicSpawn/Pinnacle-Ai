#!/usr/bin/env python3
"""
Agent Demo - Pinnacle AI

Demonstrates how to use individual agents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import OmniAIOrchestrator
from src.tools.config_loader import get_default_config

def main():
    """Demonstrate agent usage."""
    print("=" * 60)
    print("Pinnacle AI - Agent Demo")
    print("=" * 60)
    
    # Load configuration
    config = get_default_config()
    
    try:
        # Initialize orchestrator
        orchestrator = OmniAIOrchestrator(config)
        print("✓ Orchestrator initialized\n")
        
        # Access individual agents
        if "planner" in orchestrator.agents:
            print("Testing Planner Agent...")
            result = orchestrator.agents["planner"].execute(
                "Plan a software development project"
            )
            print(f"  Agent: {result.get('agent')}")
            print(f"  Task: {result.get('task')}\n")
        
        if "researcher" in orchestrator.agents:
            print("Testing Researcher Agent...")
            result = orchestrator.agents["researcher"].execute(
                "Research quantum computing"
            )
            print(f"  Agent: {result.get('agent')}")
            print(f"  Sources found: {result.get('sources', 0)}\n")
        
        if "coder" in orchestrator.agents:
            print("Testing Coder Agent...")
            result = orchestrator.agents["coder"].execute(
                "Write a Python function"
            )
            print(f"  Agent: {result.get('agent')}")
            print(f"  Code generated: {'code' in result}\n")
        
        print("=" * 60)
        print("Agent demo completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure all dependencies are installed and configured")

if __name__ == "__main__":
    main()


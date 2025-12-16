#!/usr/bin/env python3
"""
Interactive Demo - Pinnacle AI

Demonstrates interactive mode usage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import PinnacleAI

def main():
    """Run interactive demo."""
    print("=" * 60)
    print("Pinnacle AI - Interactive Demo")
    print("=" * 60)
    print("\nThis will start interactive mode.")
    print("Try commands like:")
    print("  - 'Write a Python script'")
    print("  - 'Research AI developments'")
    print("  - 'Create a story'")
    print("  - 'improve' (trigger self-improvement)")
    print("  - 'exit' (quit)\n")
    
    try:
        pinnacle = PinnacleAI()
        pinnacle.interactive_mode()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()


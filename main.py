#!/usr/bin/env python3
"""
Pinnacle-AI: The Ultimate AGI System
====================================

Run this script to start Pinnacle-AI.
"""

import asyncio
from loguru import logger

def main():
    logger.info("=" * 60)
    logger.info("  PINNACLE-AI: THE ULTIMATE AGI SYSTEM")
    logger.info("  Version: 1.0.0")
    logger.info("=" * 60)
    
    # Initialize AI
    from pinnacle_ai.core.model import PinnacleAI
    from pinnacle_ai.core.config import PinnacleConfig
    
    config = PinnacleConfig(
        use_4bit=True,
        memory_enabled=True,
        consciousness_enabled=True,
        emotional_enabled=True,
        causal_reasoning_enabled=True,
        simulation_enabled=True,
        evolution_enabled=True,
        swarm_enabled=True,
        knowledge_enabled=True,
        autonomous_lab_enabled=True
    )
    
    ai = PinnacleAI(config)
    
    # Interactive mode
    print("\n" + "="*60)
    print("Pinnacle-AI is ready. Type 'quit' to exit.")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = ai.generate(user_input)
            print(f"\nPinnacle-AI: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

if __name__ == "__main__":
    main()

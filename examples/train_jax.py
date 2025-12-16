#!/usr/bin/env python3
"""
JAX Training Example - Pinnacle AI

Demonstrates how to train a model using JAX backend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import ModelTrainer

def main():
    """Demonstrate JAX training."""
    print("=" * 60)
    print("Pinnacle AI - JAX Training Example")
    print("=" * 60)
    
    try:
        # Initialize trainer with JAX backend
        print("\n1. Initializing JAX trainer...")
        trainer = ModelTrainer(
            backend="jax",
            distributed=False,
            mixed_precision=False,
            input_size=784,
            output_size=10
        )
        print("✓ JAX trainer initialized")
        
        # Train model
        print("\n2. Training model...")
        # Note: In production, you would provide actual data path
        # For this example, we'll use a placeholder
        metrics = trainer.train("data/example_dataset.csv", epochs=5)
        
        print("\n3. Training Results:")
        print(f"   Loss: {metrics.get('loss', 'N/A')}")
        print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
        print(f"   Epochs: {metrics.get('epochs', 'N/A')}")
        
        # Save model
        print("\n4. Saving model...")
        trainer.save("models/jax_model.pkl")
        print("✓ Model saved")
        
        print("\n" + "=" * 60)
        print("JAX training example completed!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo use JAX, install dependencies:")
        print("  pip install jax flax optax")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()


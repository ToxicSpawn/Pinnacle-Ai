#!/usr/bin/env python3
"""
Mixed Precision Training Example - Pinnacle AI

Demonstrates mixed precision (FP16) training for faster training and lower memory usage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import ModelTrainer

def main():
    """Demonstrate mixed precision training."""
    print("=" * 60)
    print("Pinnacle AI - Mixed Precision Training Example")
    print("=" * 60)
    
    backends = ["pytorch", "tensorflow"]
    
    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Testing {backend.upper()} Mixed Precision Training")
        print(f"{'='*60}")
        
        try:
            # Initialize trainer with mixed precision
            print(f"\n1. Initializing {backend} trainer with mixed precision...")
            trainer = ModelTrainer(
                backend=backend,
                distributed=False,
                mixed_precision=True,
                input_size=784,
                output_size=10
            )
            print(f"✓ {backend} mixed precision trainer initialized")
            
            # Train model
            print(f"\n2. Training {backend} model (mixed precision FP16)...")
            metrics = trainer.train("data/example_dataset.csv", epochs=5, batch_size=128)
            
            print(f"\n3. Training Results ({backend}):")
            print(f"   Loss: {metrics.get('loss', 'N/A')}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
            print(f"   Note: Mixed precision can provide 2x speedup on compatible GPUs")
            
        except ImportError as e:
            print(f"\n✗ {backend} not available: {e}")
        except Exception as e:
            print(f"\n✗ Error with {backend}: {e}")
    
    print("\n" + "=" * 60)
    print("Mixed precision training examples completed!")
    print("=" * 60)
    print("\nBenefits of Mixed Precision:")
    print("  - 2x faster training on compatible GPUs")
    print("  - Lower memory usage")
    print("  - Same or better accuracy")

if __name__ == "__main__":
    main()


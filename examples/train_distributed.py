#!/usr/bin/env python3
"""
Distributed Training Example - Pinnacle AI

Demonstrates distributed training with PyTorch DDP and TensorFlow MirroredStrategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import ModelTrainer

def main():
    """Demonstrate distributed training."""
    print("=" * 60)
    print("Pinnacle AI - Distributed Training Example")
    print("=" * 60)
    
    backends = ["pytorch", "tensorflow"]
    
    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Testing {backend.upper()} Distributed Training")
        print(f"{'='*60}")
        
        try:
            # Initialize trainer with distributed training
            print(f"\n1. Initializing {backend} distributed trainer...")
            trainer = ModelTrainer(
                backend=backend,
                distributed=True,
                mixed_precision=False,
                input_size=784,
                output_size=10
            )
            print(f"✓ {backend} distributed trainer initialized")
            
            # Train model
            print(f"\n2. Training {backend} model (distributed)...")
            metrics = trainer.train("data/example_dataset.csv", epochs=3, batch_size=64)
            
            print(f"\n3. Training Results ({backend}):")
            print(f"   Loss: {metrics.get('loss', 'N/A')}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
            
        except ImportError as e:
            print(f"\n✗ {backend} not available: {e}")
        except Exception as e:
            print(f"\n✗ Error with {backend}: {e}")
    
    print("\n" + "=" * 60)
    print("Distributed training examples completed!")
    print("=" * 60)
    print("\nNote: For actual distributed training, run with:")
    print("  PyTorch: torchrun --nproc_per_node=4 train_distributed.py")
    print("  TensorFlow: Configure TF_CONFIG environment variable")

if __name__ == "__main__":
    main()


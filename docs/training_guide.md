# Training Guide - Pinnacle AI

This guide covers model training capabilities in Pinnacle AI, including multi-backend support, distributed training, and performance optimizations.

## Multi-Backend Support

Pinnacle AI supports three major ML frameworks:

### PyTorch

```python
from src.training.trainer import ModelTrainer

trainer = ModelTrainer(backend="pytorch")
metrics = trainer.train("data.csv", epochs=10)
```

### TensorFlow

```python
trainer = ModelTrainer(backend="tensorflow")
metrics = trainer.train("data.csv", epochs=10)
```

### JAX

```python
trainer = ModelTrainer(backend="jax")
metrics = trainer.train("data.csv", epochs=10)
```

**Installation:**
```bash
pip install jax flax optax
```

## Distributed Training

### PyTorch Distributed Data Parallel (DDP)

```python
trainer = ModelTrainer(
    backend="pytorch",
    distributed=True
)
metrics = trainer.train("data.csv", epochs=10)
```

**Run with multiple GPUs:**
```bash
torchrun --nproc_per_node=4 train_distributed.py
```

### TensorFlow MirroredStrategy

```python
trainer = ModelTrainer(
    backend="tensorflow",
    distributed=True
)
metrics = trainer.train("data.csv", epochs=10)
```

**Configure TF_CONFIG:**
```bash
export TF_CONFIG='{"cluster":{"worker":["localhost:12345"]},"task":{"type":"worker","index":0}}'
```

## Mixed Precision Training

Mixed precision (FP16) training provides:
- **2x faster training** on compatible GPUs
- **Lower memory usage**
- **Same or better accuracy**

### PyTorch

```python
trainer = ModelTrainer(
    backend="pytorch",
    mixed_precision=True
)
metrics = trainer.train("data.csv", epochs=10)
```

### TensorFlow

```python
trainer = ModelTrainer(
    backend="tensorflow",
    mixed_precision=True
)
metrics = trainer.train("data.csv", epochs=10)
```

## Optimized Data Loading

```python
from src.training.data_loader import get_data_loader

# Create optimized DataLoader
loader = get_data_loader(
    data_path="data.csv",
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    shuffle=True
)

# Use in training loop
for batch in loader:
    features, labels = batch
    # Training logic...
```

**Benefits:**
- Multi-process data loading
- Pinned memory for faster GPU transfer
- Automatic batching and shuffling

## Complete Training Example

```python
from src.training.trainer import ModelTrainer

# Initialize trainer with all optimizations
trainer = ModelTrainer(
    backend="pytorch",
    distributed=True,      # Use multiple GPUs
    mixed_precision=True,  # Use FP16
    input_size=784,
    output_size=10
)

# Train model
metrics = trainer.train(
    data_path="data/train.csv",
    epochs=50,
    batch_size=128
)

# Save model
trainer.save("models/trained_model.pth")

# Deploy model
trainer.deploy("models/trained_model.pth", backend="onnx")
```

## Performance Tips

1. **Use Mixed Precision**: Enable for 2x speedup on compatible GPUs
2. **Optimize Data Loading**: Use multiple workers and pin memory
3. **Distributed Training**: Scale to multiple GPUs/nodes
4. **Batch Size**: Larger batches = better GPU utilization
5. **JAX Backend**: Fastest for research and experimentation

## Backend Comparison

| Feature | PyTorch | TensorFlow | JAX |
|---------|---------|------------|-----|
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Performance | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Production Ready | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Research | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Distributed | ✅ DDP | ✅ MirroredStrategy | ✅ pmap |
| Mixed Precision | ✅ | ✅ | ✅ |

## Examples

See example scripts:
- `examples/train_jax.py` - JAX training
- `examples/train_distributed.py` - Distributed training
- `examples/train_mixed_precision.py` - Mixed precision training

## Troubleshooting

### JAX Installation Issues

```bash
# CPU only
pip install jax flax optax

# GPU support (CUDA)
pip install jax[cuda12] flax optax -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Distributed Training Issues

- Ensure all nodes can communicate
- Check firewall settings
- Verify GPU availability
- Check CUDA/ROCm installation

### Mixed Precision Issues

- Requires compatible GPU (Volta or newer)
- May need to adjust loss scaling
- Some operations may not support FP16

## Next Steps

- [API Reference](api_reference.md)
- [Deployment Guide](deployment_guide.md)
- [Examples](../examples/)


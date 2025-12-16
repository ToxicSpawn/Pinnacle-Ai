"""
Advanced Quantization: QLoRA, Sparse Attention, Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import bitsandbytes for 4-bit quantization
try:
    from bitsandbytes.nn import Linear4bit
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes not available. Install with: pip install bitsandbytes")


class QuantizedMistral(nn.Module):
    """Mistral model with 4-bit quantization (QLoRA)."""
    
    def __init__(self, base_model: nn.Module):
        """
        Initialize quantized model.
        
        Args:
            base_model: Base Mistral model
        """
        super().__init__()
        self.base_model = base_model
        if BITSANDBYTES_AVAILABLE:
            self._quantize()
        else:
            logger.warning("Using standard linear layers (bitsandbytes not available)")
    
    def _quantize(self):
        """Quantize linear layers to 4-bit."""
        if not BITSANDBYTES_AVAILABLE:
            return
        
        # Replace linear layers with 4-bit quantized versions
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Create 4-bit linear layer
                quantized = Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                )
                # Copy weights (will be quantized automatically)
                with torch.no_grad():
                    quantized.weight.data = module.weight.data
                    if module.bias is not None:
                        quantized.bias.data = module.bias.data
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = self.base_model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, quantized)
                else:
                    setattr(self.base_model, child_name, quantized)


def sparse_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparsity: float = 0.5,
    pattern: str = "random",
) -> Tensor:
    """
    Sparse attention with configurable sparsity patterns.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        sparsity: Sparsity ratio (0.0 = dense, 1.0 = fully sparse)
        pattern: Sparsity pattern ("random", "local", "strided")
        
    Returns:
        Attention output
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # Compute attention weights
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Apply sparsity mask
    if pattern == "random":
        mask = torch.rand_like(attn_weights) > sparsity
    elif pattern == "local":
        # Local attention window
        window_size = int(seq_len * (1 - sparsity))
        mask = torch.ones_like(attn_weights, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2)
            mask[:, :, i, :start] = False
            mask[:, :, i, end:] = False
    elif pattern == "strided":
        # Strided pattern
        stride = int(1 / (1 - sparsity))
        mask = torch.zeros_like(attn_weights, dtype=torch.bool)
        for i in range(0, seq_len, stride):
            mask[:, :, i, :] = True
    else:
        mask = torch.ones_like(attn_weights, dtype=torch.bool)
    
    # Apply mask
    attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # Apply to values
    output = torch.matmul(attn_weights, value)
    return output


class DistillationTrainer:
    """Knowledge distillation trainer."""
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.5,
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher: Teacher model (larger, pre-trained)
            student: Student model (smaller, to train)
            temperature: Distillation temperature
            alpha: Weight for distillation loss vs. task loss
        """
        self.teacher = teacher.eval()
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.distill_loss = nn.KLDivLoss(reduction="batchmean")
        self.task_loss = nn.CrossEntropyLoss()
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Perform distillation training step.
        
        Args:
            batch: Training batch with input_ids and labels
            
        Returns:
            Dictionary with loss metrics
        """
        self.student.train()
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**batch)
            if isinstance(teacher_outputs, tuple):
                teacher_logits = teacher_outputs[0]
            else:
                teacher_logits = teacher_outputs
        
        # Student forward
        student_outputs = self.student(**batch)
        if isinstance(student_outputs, tuple):
            student_logits = student_outputs[0]
        else:
            student_logits = student_outputs
        
        # Task loss (hard labels)
        task_loss = self.task_loss(
            student_logits.view(-1, student_logits.size(-1)),
            batch["labels"].view(-1)
        )
        
        # Distillation loss (soft labels)
        distill_loss = self.distill_loss(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        
        return {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "distill_loss": distill_loss.item(),
        }


"""
Training Optimizations: Memory-efficient attention, gradient checkpointing, AMP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_qkvpacked_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.warning("Flash Attention not available. Install with: pip install flash-attn")


def memory_efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Memory-efficient attention using Flash Attention if available.
    
    Args:
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, heads, seq_len, head_dim]
        value: Value tensor [batch, heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        scale: Attention scale (1/sqrt(head_dim))
        
    Returns:
        Attention output
    """
    if FLASH_ATTN_AVAILABLE and query.is_cuda:
        # Use Flash Attention
        qkv = torch.stack([query, key, value], dim=2)  # [batch, heads, 3, seq_len, head_dim]
        qkv = qkv.transpose(1, 2)  # [batch, 3, heads, seq_len, head_dim]
        
        return flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=False
        )
    else:
        # Fallback to standard attention
        if scale is None:
            scale = 1.0 / (query.size(-1) ** 0.5)
        
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=query.requires_grad)
        
        output = torch.matmul(attn_weights, value)
        return output


class CheckpointedMistral(nn.Module):
    """Mistral model with gradient checkpointing."""
    
    def __init__(self, base_model: nn.Module, gradient_checkpointing: bool = True):
        """
        Initialize checkpointed model.
        
        Args:
            base_model: Base Mistral model
            gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        self.base_model = base_model
        self.gradient_checkpointing = gradient_checkpointing
    
    def forward(self, *args, **kwargs):
        """Forward pass with optional gradient checkpointing."""
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.base_model,
                *args,
                **kwargs,
                use_reentrant=False
            )
        return self.base_model(*args, **kwargs)


class AMPTrainer:
    """Automatic Mixed Precision trainer with gradient clipping."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize AMP trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Perform training step with AMP and gradient clipping.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary with metrics
        """
        self.model.train()
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast():
            outputs = self.model(**batch)
            if isinstance(outputs, tuple):
                loss = outputs[1] if len(outputs) > 1 else outputs[0]
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return {"loss": loss.item()}


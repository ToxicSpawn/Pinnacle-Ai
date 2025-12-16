"""
Advanced Learning Rate Schedulers
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class WarmupStableDecayScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup, cosine decay, and stable decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr_ratio: Minimum LR as ratio of initial LR
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            lr_scale = min(1.0, self.last_epoch / self.warmup_steps)
        elif self.last_epoch < self.total_steps:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            # Stable decay (constant at min_lr_ratio)
            lr_scale = self.min_lr_ratio
        
        return [base_lr * lr_scale for base_lr in self.base_lrs]


"""
Learning rate scheduler builder
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional


class SchedulerBuilder:
    """Builder for learning rate schedulers."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        min_lr_ratio: float = 0.1,
    ):
        """
        Initialize scheduler builder.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            min_lr_ratio: Minimum LR as ratio of initial LR
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

    def build(self) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Build scheduler with warmup and cosine decay.
        
        Returns:
            SequentialLR scheduler with linear warmup and cosine decay
        """
        initial_lr = self.optimizer.param_groups[0]["lr"]
        
        # Linear warmup
        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        
        # Cosine annealing
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=initial_lr * self.min_lr_ratio,
        )
        
        # Sequential: warmup then cosine
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_steps],
        )


"""
Optimizer builder with weight decay handling
"""

import torch
from typing import Tuple


class OptimizerBuilder:
    """Builder for optimizers with proper weight decay handling."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        """
        Initialize optimizer builder.
        
        Args:
            model: Model to optimize
            lr: Learning rate
            weight_decay: Weight decay coefficient
            betas: Adam betas
            eps: Adam epsilon
        """
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

    def build(self) -> torch.optim.Optimizer:
        """
        Build optimizer with proper weight decay.
        
        Applies weight decay only to non-bias and non-LayerNorm parameters.
        """
        param_dict = {name: param for name, param in self.model.named_parameters()}
        param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        nodecay_params = []
        
        for name, param in param_dict.items():
            # Skip weight decay for bias and normalization layers
            if param.dim() >= 2 and 'norm' not in name.lower() and 'bias' not in name.lower():
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(optim_groups, lr=self.lr, betas=self.betas, eps=self.eps)


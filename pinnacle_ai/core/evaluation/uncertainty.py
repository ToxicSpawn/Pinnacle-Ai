"""
Uncertainty Estimation
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Uncertainty estimation using Monte Carlo dropout."""
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: Model to estimate uncertainty for
            dropout_rate: Dropout rate for MC sampling
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout in all layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active during eval
    
    def monte_carlo_dropout(
        self,
        input_ids: Tensor,
        n_samples: int = 10,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            input_ids: Input token IDs
            n_samples: Number of MC samples
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with mean, std, and uncertainty
        """
        self.model.train()  # Enable dropout
        
        outputs = []
        with torch.no_grad():
            for _ in range(n_samples):
                if attention_mask is not None:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    out = self.model(input_ids=input_ids)
                
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                
                outputs.append(logits)
        
        # Stack and compute statistics
        outputs = torch.stack(outputs)  # [n_samples, batch, seq, vocab]
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        uncertainty = std.mean(dim=-1)  # Average over vocabulary
        
        return {
            "mean": mean,
            "std": std,
            "uncertainty": uncertainty,
        }
    
    def ensemble_uncertainty(
        self,
        models: list,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Estimate uncertainty using model ensemble.
        
        Args:
            models: List of models
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with mean, std, and uncertainty
        """
        outputs = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                if attention_mask is not None:
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    out = model(input_ids=input_ids)
                
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                
                outputs.append(logits)
        
        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        uncertainty = std.mean(dim=-1)
        
        return {
            "mean": mean,
            "std": std,
            "uncertainty": uncertainty,
        }


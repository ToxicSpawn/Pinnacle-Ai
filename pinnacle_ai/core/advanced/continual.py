"""
Continual Learning with Elastic Weight Consolidation (EWC)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, List, Optional
import copy
import logging

logger = logging.getLogger(__name__)


class ContinualLearner:
    """Continual learner with experience replay and EWC."""
    
    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 1000,
        ewc_lambda: float = 0.1,
    ):
        """
        Initialize continual learner.
        
        Args:
            model: Model to train
            memory_size: Size of experience replay buffer
            ewc_lambda: EWC regularization strength
        """
        self.model = model
        self.memory_size = memory_size
        self.ewc_lambda = ewc_lambda
        self.memory: List[Tensor] = []
        
        # Store initial parameters for EWC
        self.initial_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
        # Fisher information matrix
        self.fisher: Dict[str, Tensor] = {}
    
    def learn(self, new_data: Tensor, task_id: int) -> Tensor:
        """
        Learn from new data with continual learning.
        
        Args:
            new_data: New training data
            task_id: Task identifier
            
        Returns:
            Combined loss
        """
        # Add to memory
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append(new_data)
        
        # Compute EWC loss
        ewc_loss = self._compute_ewc_loss()
        
        # Task loss
        outputs = self.model(new_data)
        if isinstance(outputs, tuple):
            task_loss = outputs[1] if len(outputs) > 1 else outputs[0]
        elif hasattr(outputs, 'loss'):
            task_loss = outputs.loss
        else:
            task_loss = outputs
        
        # Combined loss
        total_loss = task_loss + ewc_loss
        
        return total_loss
    
    def _compute_ewc_loss(self) -> Tensor:
        """
        Compute Elastic Weight Consolidation loss.
        
        Returns:
            EWC loss tensor
        """
        if not self.memory or not self.fisher:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.initial_params:
                fisher = self.fisher[name]
                initial = self.initial_params[name]
                ewc_loss += (fisher * (param - initial).pow(2)).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def update_fisher(self, data: Tensor):
        """
        Update Fisher information matrix.
        
        Args:
            data: Data to compute Fisher on
        """
        self.model.train()
        
        # Zero gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # Compute gradients
        outputs = self.model(data)
        if isinstance(outputs, tuple):
            loss = outputs[1] if len(outputs) > 1 else outputs[0]
        elif hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            loss = outputs
        
        loss.backward()
        
        # Update Fisher
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.fisher:
                    self.fisher[name] = torch.zeros_like(param)
                self.fisher[name] += param.grad.pow(2)
        
        # Normalize
        num_samples = len(self.memory)
        if num_samples > 0:
            for name in self.fisher:
                self.fisher[name] /= num_samples


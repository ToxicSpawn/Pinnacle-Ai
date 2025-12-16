import torch
import torch.nn as nn
from typing import Dict, List, Optional
from loguru import logger


class GlobalWorkspace(nn.Module):
    """
    Global Workspace Theory Implementation
    
    Integrates information from multiple cognitive modules
    and broadcasts to create unified conscious experience.
    """
    
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Workspace state
        self.workspace_state = torch.zeros(1, hidden_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Broadcast layer
        self.broadcast = nn.Linear(hidden_size, hidden_size)
        
        logger.info("Global Workspace initialized")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process inputs through global workspace
        
        Args:
            inputs: Dictionary of input tensors from different modules
        
        Returns:
            Integrated workspace output
        """
        # Combine inputs
        if not inputs:
            return self.workspace_state
        
        combined = torch.stack(list(inputs.values()))
        
        # Attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Integration
        integrated = self.integration(
            torch.cat([attended.mean(dim=0), self.workspace_state], dim=-1)
        )
        
        # Update workspace state
        self.workspace_state = integrated
        
        # Broadcast
        return self.broadcast(integrated)
    
    def get_state(self) -> torch.Tensor:
        """Get current workspace state"""
        return self.workspace_state
    
    def reset(self):
        """Reset workspace state"""
        self.workspace_state = torch.zeros(1, self.hidden_size)


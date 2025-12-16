"""
Federated Learning Support
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from typing import Dict, Any, List, Optional
import copy
import logging

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """Federated learning trainer."""
    
    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 10,
        local_epochs: int = 3,
    ):
        """
        Initialize federated trainer.
        
        Args:
            model: Global model
            num_clients: Number of clients
            local_epochs: Number of local training epochs per round
        """
        self.model = model
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.optimizer = None
    
    def train(self, global_epochs: int = 10, client_data_fn: Optional[callable] = None):
        """
        Perform federated training.
        
        Args:
            global_epochs: Number of global training rounds
            client_data_fn: Function to get client data (client_id -> data)
        """
        if client_data_fn is None:
            logger.warning("No client data function provided. Using placeholder.")
            client_data_fn = lambda client_id: torch.randn(10, 100)  # Placeholder
        
        for global_epoch in range(global_epochs):
            logger.info(f"Global epoch {global_epoch + 1}/{global_epochs}")
            
            # Client training
            client_updates = []
            for client_id in range(self.num_clients):
                client_data = client_data_fn(client_id)
                client_update = self._client_train(client_id, client_data)
                client_updates.append(client_update)
            
            # Server aggregation (Federated Averaging)
            self._aggregate_updates(client_updates)
    
    def _client_train(self, client_id: int, data: Tensor) -> Dict[str, Tensor]:
        """
        Train client model locally.
        
        Args:
            client_id: Client identifier
            data: Client training data
            
        Returns:
            Dictionary of parameter updates
        """
        # Create local model copy
        local_model = copy.deepcopy(self.model)
        local_optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-4)
        
        # Local training
        local_model.train()
        for epoch in range(self.local_epochs):
            outputs = local_model(data)
            if isinstance(outputs, tuple):
                loss = outputs[1] if len(outputs) > 1 else outputs[0]
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs
            
            loss.backward()
            local_optimizer.step()
            local_optimizer.zero_grad()
        
        # Compute updates (difference from global model)
        updates = {}
        for (name, local_param), (_, global_param) in zip(
            local_model.named_parameters(),
            self.model.named_parameters()
        ):
            updates[name] = local_param.data - global_param.data
        
        logger.info(f"Client {client_id} training complete")
        return updates
    
    def _aggregate_updates(self, updates: List[Dict[str, Tensor]]):
        """
        Aggregate client updates using Federated Averaging.
        
        Args:
            updates: List of client updates
        """
        # Average updates
        avg_updates = {}
        for name in updates[0].keys():
            avg_updates[name] = torch.stack([update[name] for update in updates]).mean(dim=0)
        
        # Apply to global model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in avg_updates:
                    param.data += avg_updates[name]
        
        logger.info("Federated averaging complete")


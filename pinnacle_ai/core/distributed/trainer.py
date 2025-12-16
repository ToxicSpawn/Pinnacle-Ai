"""
Distributed Training System with DDP and FSDP support
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import FSDP (PyTorch 2.0+)
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    logger.warning("FSDP not available. Install PyTorch 2.0+ for FSDP support.")


class DistributedTrainer:
    """Distributed trainer with DDP and FSDP support."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        strategy: Optional[str] = None,
        mixed_precision: bool = True,
        fsdp_config: Optional[Dict] = None,
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            strategy: Distributed strategy ("ddp", "fsdp", or None)
            mixed_precision: Enable mixed precision training
            fsdp_config: FSDP configuration dictionary
        """
        self.model = model
        self.strategy = strategy
        self.mixed_precision = mixed_precision
        self.fsdp_config = fsdp_config or {}
        self.optimizer = None
        self.scaler = None
        
        if strategy is not None:
            self._setup_distributed()

    def _setup_distributed(self):
        """Setup distributed training."""
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")

        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")

        if self.strategy == "ddp":
            self.model = DDP(self.model.cuda(), device_ids=[self.rank])
            logger.info("Using DDP strategy")
        elif self.strategy == "fsdp":
            if not FSDP_AVAILABLE:
                raise RuntimeError("FSDP requires PyTorch 2.0+. Falling back to DDP.")
                self.strategy = "ddp"
                self._setup_distributed()
                return
            
            # Import here to avoid circular imports
            from pinnacle_ai.core.models.mistral import MistralDecoderLayer
            
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={MistralDecoderLayer},
            )
            
            self.model = FSDP(
                self.model.cuda(),
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                **self.fsdp_config
            )
            logger.info("Using FSDP strategy")
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set the optimizer."""
        self.optimizer = optimizer

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary with training metrics
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set. Call set_optimizer() first.")
        
        self.model.train()
        inputs = {k: v.cuda() for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            outputs = self.model(**inputs)
            if isinstance(outputs, tuple):
                loss = outputs[1] if len(outputs) > 1 else outputs[0]
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                loss = outputs

        if self.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad()
        
        return {"loss": loss.item()}

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scaler_state_dict": self.scaler.state_dict() if self.mixed_precision and self.scaler else None,
        }
        
        if self.strategy == "fsdp" and FSDP_AVAILABLE:
            # FSDP requires special checkpointing
            with FSDP.summon_full_params(self.model):
                checkpoint["model_state_dict"] = self.model.state_dict()
        else:
            # DDP or non-distributed
            if hasattr(self.model, 'module'):  # DDP wraps model
                checkpoint["model_state_dict"] = self.model.module.state_dict()
            else:
                checkpoint["model_state_dict"] = self.model.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location='cuda')
        
        if hasattr(self.model, 'module'):  # DDP
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {path}")


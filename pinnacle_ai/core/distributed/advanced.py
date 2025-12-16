"""
Advanced Distributed Training: Enhanced FSDP, Tensor Parallelism, Pipeline Parallelism
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# FSDP imports
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    logger.warning("FSDP not available. Requires PyTorch 2.0+")

# Pipeline parallelism
try:
    from torch.distributed.pipeline.sync import Pipe
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    logger.warning("Pipeline parallelism not available")


def setup_fsdp(
    model: nn.Module,
    auto_wrap_policy: Optional[callable] = None,
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    sharding_strategy: str = "FULL_SHARD",
) -> FSDP:
    """
    Setup Fully Sharded Data Parallel.
    
    Args:
        model: Model to wrap
        auto_wrap_policy: Auto wrap policy for transformer layers
        cpu_offload: Enable CPU offloading
        mixed_precision: Enable mixed precision
        sharding_strategy: Sharding strategy
        
    Returns:
        FSDP-wrapped model
    """
    if not FSDP_AVAILABLE:
        raise RuntimeError("FSDP requires PyTorch 2.0+")
    
    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    # Mixed precision policy
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    # CPU offload
    cpu_offload_policy = CPUOffload(offload_params=cpu_offload) if cpu_offload else None
    
    # Auto wrap policy
    if auto_wrap_policy is None:
        # Default: wrap transformer layers
        try:
            from pinnacle_ai.core.models.mistral import MistralDecoderLayer
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={MistralDecoderLayer}
            )
        except ImportError:
            auto_wrap_policy = None
    
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding,
        device_id=torch.cuda.current_device(),
    )


class ColumnParallelLinear(nn.Module):
    """Column-parallel linear layer for tensor parallelism."""
    
    def __init__(self, base_layer: nn.Linear, world_size: int):
        """
        Initialize column-parallel linear layer.
        
        Args:
            base_layer: Base linear layer
            world_size: Number of parallel processes
        """
        super().__init__()
        self.world_size = world_size
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features // world_size
        
        self.weight = nn.Parameter(base_layer.weight[:self.out_features].clone())
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias[:self.out_features].clone())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with all-reduce."""
        output = F.linear(x, self.weight, self.bias)
        # All-reduce across tensor parallel group
        if dist.is_initialized():
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output


class RowParallelLinear(nn.Module):
    """Row-parallel linear layer for tensor parallelism."""
    
    def __init__(self, base_layer: nn.Linear, world_size: int):
        """
        Initialize row-parallel linear layer.
        
        Args:
            base_layer: Base linear layer
            world_size: Number of parallel processes
        """
        super().__init__()
        self.world_size = world_size
        self.in_features = base_layer.in_features // world_size
        self.out_features = base_layer.out_features
        
        self.weight = nn.Parameter(base_layer.weight[:, :self.in_features].clone())
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.clone())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with all-gather."""
        # All-gather input
        if dist.is_initialized():
            gathered = [torch.zeros_like(x) for _ in range(self.world_size)]
            dist.all_gather(gathered, x)
            x = torch.cat(gathered, dim=-1)
        
        output = F.linear(x, self.weight, self.bias)
        return output


def create_pipeline(
    model: nn.Module,
    chunks: int = 8,
    checkpoint: str = "except_last",
) -> Pipe:
    """
    Create pipeline parallel model.
    
    Args:
        model: Model to parallelize
        chunks: Number of micro-batches
        checkpoint: Checkpoint strategy
        
    Returns:
        Pipeline-parallel model
    """
    if not PIPELINE_AVAILABLE:
        raise RuntimeError("Pipeline parallelism requires torch.distributed.pipeline")
    
    return Pipe(
        model,
        chunks=chunks,
        checkpoint=checkpoint,
    )


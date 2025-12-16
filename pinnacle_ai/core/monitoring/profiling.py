"""
Performance Profiling
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def profile_model(
    model: nn.Module,
    example_input: Tensor,
    num_iterations: int = 5,
    output_dir: str = "./logs",
    record_shapes: bool = True,
    profile_memory: bool = True,
) -> Dict[str, Any]:
    """
    Profile model performance.
    
    Args:
        model: Model to profile
        example_input: Example input tensor
        num_iterations: Number of iterations to profile
        output_dir: Directory for trace files
        record_shapes: Record tensor shapes
        profile_memory: Profile memory usage
        
    Returns:
        Dictionary with profiling results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if torch.cuda.is_available() else None,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_path)),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=True,
    ) as prof:
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(example_input)
            prof.step()
    
    # Get summary
    key_averages = prof.key_averages()
    
    # Print summary
    print(key_averages.table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"))
    
    # Extract metrics
    total_time = sum(event.self_cuda_time_total if torch.cuda.is_available() else event.self_cpu_time_total 
                     for event in key_averages)
    
    return {
        "total_time_us": total_time,
        "num_events": len(key_averages),
        "trace_file": str(output_path / "trace.json"),
    }


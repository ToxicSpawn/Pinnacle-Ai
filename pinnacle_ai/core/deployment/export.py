"""
Model Export: ONNX and TensorRT
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. Install TensorRT for GPU acceleration.")


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    example_input: Tensor,
    dynamic_axes: Optional[Dict] = None,
    opset_version: int = 14,
) -> bool:
    """
    Export model to ONNX format with dynamic shapes.
    
    Args:
        model: Model to export
        output_path: Output file path
        example_input: Example input tensor
        dynamic_axes: Dictionary specifying dynamic axes
        opset_version: ONNX opset version
        
    Returns:
        True if export successful
    """
    try:
        model.eval()
        
        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size", 1: "sequence_length"},
            }
        
        # Export
        torch.onnx.export(
            model,
            example_input,
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB
) -> bool:
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output engine path
        fp16: Enable FP16 precision
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes
        
    Returns:
        True if build successful
    """
    if not TENSORRT_AVAILABLE:
        logger.error("TensorRT not available")
        return False
    
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX model
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                return False
        
        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 enabled")
        
        engine = builder.build_engine(network, config)
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return False
        
        # Save engine
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved: {engine_path}")
        return True
        
    except Exception as e:
        logger.error(f"TensorRT build failed: {e}")
        return False


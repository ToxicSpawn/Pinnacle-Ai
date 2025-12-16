"""
Weights & Biases Integration
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


def setup_wandb(config: Dict[str, Any], project: str = "pinnacle-ai", name: Optional[str] = None):
    """
    Setup Weights & Biases logging.
    
    Args:
        config: Configuration dictionary
        project: W&B project name
        name: Run name (optional)
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available. Skipping W&B setup.")
        return
    
    wandb.init(
        project=project,
        config=config,
        name=name or f"mistral-{config.get('model_size', 'default')}",
    )
    logger.info("Weights & Biases initialized")


def log_metrics(metrics: Dict[str, float], step: int):
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metrics
        step: Step number
    """
    if WANDB_AVAILABLE:
        wandb.log(metrics, step=step)
    else:
        logger.debug(f"Metrics (step {step}): {metrics}")


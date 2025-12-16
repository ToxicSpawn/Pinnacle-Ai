"""
Logger - Sets up logging configuration.
"""

import logging
import sys
from typing import Dict

def setup_logging(config: Dict):
    """Setup logging configuration."""
    level = config.get("level", "INFO")
    format_str = config.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


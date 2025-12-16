"""
Optimized Data Loading Utilities.
"""

import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Pandas
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class CustomDataset:
    """Custom dataset for PyTorch."""
    
    def __init__(self, data_path: str, transform: Optional[callable] = None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file
            transform: Optional transform to apply
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas not available")
        
        self.data = pd.read_csv(data_path)
        self.transform = transform
        logger.info(f"Loaded dataset from {data_path}: {len(self.data)} samples")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """Get item by index."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Extract features and labels
        features = self.data.iloc[idx, :-1].values.astype(np.float32)
        label = self.data.iloc[idx, -1]
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
        
        return features_tensor, label_tensor


def get_data_loader(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    transform: Optional[callable] = None
) -> DataLoader:
    """
    Create optimized DataLoader.
    
    Args:
        data_path: Path to data file
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        shuffle: Shuffle data
        transform: Optional transform
        
    Returns:
        DataLoader instance
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch not available")
    
    dataset = CustomDataset(data_path, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )


def load_data(data_path: str) -> Tuple:
    """
    Load and preprocess data.
    
    Args:
        data_path: Path to data file
        
    Returns:
        Tuple of (features, labels)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("Pandas not available")
    
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


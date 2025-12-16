"""
Model Explainability and Interpretability
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Model interpreter for attention visualization and explainability."""
    
    def __init__(self, model: nn.Module, tokenizer: Any):
        """
        Initialize model interpreter.
        
        Args:
            model: Model to interpret
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def attention_visualization(self, text: str, output_attentions: bool = True) -> Dict[str, Any]:
        """
        Visualize attention patterns.
        
        Args:
            text: Input text
            output_attentions: Whether to output attention weights
            
        Returns:
            Dictionary with tokens and attention weights
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        
        self.model.eval()
        with torch.no_grad():
            # Get model outputs with attention
            if hasattr(self.model, 'forward'):
                # Try to get attention if model supports it
                try:
                    outputs = self.model(**inputs, output_attentions=output_attentions)
                    if isinstance(outputs, tuple) and len(outputs) > 1:
                        attentions = outputs[-1] if output_attentions else None
                    elif hasattr(outputs, 'attentions'):
                        attentions = outputs.attentions
                    else:
                        attentions = None
                except:
                    attentions = None
            else:
                attentions = None
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return {
            "tokens": tokens,
            "attentions": attentions,
            "input_ids": inputs["input_ids"].tolist(),
        }
    
    def feature_importance(self, text: str, target_class: int = 0) -> Dict[str, float]:
        """
        Compute feature importance using gradients.
        
        Args:
            text: Input text
            target_class: Target class for importance
            
        Returns:
            Dictionary mapping tokens to importance scores
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs.requires_grad = True
        
        self.model.train()
        outputs = self.model(**inputs)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        # Compute gradients
        logits[0, target_class].backward()
        
        # Get importance scores
        importance = inputs.grad.abs().mean(dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return dict(zip(tokens, importance[0].tolist()))


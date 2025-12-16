"""
Test Math Reasoning with Neurosymbolic Mistral
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import logging
from pinnacle_ai.core.neurosymbolic.neural_adapter import NeurosymbolicMistral
from pinnacle_ai.core.models.mistral import MistralConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sqrt2_irrational():
    """Test proof that √2 is irrational."""
    logger.info("Testing proof that √2 is irrational...")
    
    # Initialize model with small config for testing
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,  # Small for testing
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = NeurosymbolicMistral(config)
    
    # Test prompt
    prompt = "Prove that the square root of 2 is irrational"
    
    # Generate with reasoning
    result = model.generate_with_reasoning(prompt, use_symbolic=True)
    
    print("\n" + "="*80)
    print("PROOF: √2 is irrational")
    print("="*80)
    print(result)
    print("="*80 + "\n")
    
    # Test direct proof
    proof = model.prove("irrational(sqrt(2))")
    print("Direct proof result:")
    print(proof)
    print("\n")
    
    return result


def test_sqrt3_irrational():
    """Test proof that √3 is irrational."""
    logger.info("Testing proof that √3 is irrational...")
    
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = NeurosymbolicMistral(config)
    
    prompt = "Prove that the square root of 3 is irrational"
    result = model.generate_with_reasoning(prompt, use_symbolic=True)
    
    print("\n" + "="*80)
    print("PROOF: √3 is irrational")
    print("="*80)
    print(result)
    print("="*80 + "\n")
    
    return result


def test_general_proof():
    """Test general proof generation."""
    logger.info("Testing general proof generation...")
    
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = NeurosymbolicMistral(config)
    
    prompts = [
        "Prove that there are infinitely many prime numbers",
        "Prove that the sum of angles in a triangle is 180 degrees",
        "Prove that 0.999... = 1",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = model.generate_with_reasoning(prompt, use_symbolic=True)
        print(f"Result: {result[:200]}...")  # Print first 200 chars
        print("-" * 80)


if __name__ == "__main__":
    print("="*80)
    print("NEUROSYMBOLIC MATH REASONING TESTS")
    print("="*80)
    
    # Test √2 irrational
    test_sqrt2_irrational()
    
    # Test √3 irrational
    test_sqrt3_irrational()
    
    # Test general proofs
    test_general_proof()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


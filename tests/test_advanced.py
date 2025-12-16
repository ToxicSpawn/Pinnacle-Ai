"""
Advanced Tests for Self-Evolving, Quantum-Ready AI System
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import logging
from pinnacle_ai.core.neurosymbolic.neural_adapter import NeurosymbolicMistral
from pinnacle_ai.core.quantum_neuro import QuantumNeurosymbolicMistral
from pinnacle_ai.core.ai_scientist import AIScientist
from pinnacle_ai.core.self_evolving import ArchitectureEvolver
from pinnacle_ai.core.self_improving import SelfImprovingTrainer
from pinnacle_ai.core.models.mistral import MistralConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_neurosymbolic_reasoning():
    """Test neurosymbolic reasoning."""
    logger.info("Testing neurosymbolic reasoning...")
    
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = NeurosymbolicMistral(config)
    result = model.generate_with_reasoning(
        "Prove that the square root of 2 is irrational",
        use_symbolic=True
    )
    
    assert "irrational" in result.lower() or "proof" in result.lower()
    logger.info("✅ Neurosymbolic reasoning test passed")
    return True


def test_quantum_model():
    """Test quantum model."""
    logger.info("Testing quantum model...")
    
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = QuantumNeurosymbolicMistral(config, n_qubits=4)
    result = model.generate_with_reasoning(
        "Explain quantum entanglement",
        use_symbolic=True
    )
    
    assert result is not None
    logger.info("✅ Quantum model test passed")
    return True


def test_ai_scientist():
    """Test AI scientist."""
    logger.info("Testing AI scientist...")
    
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = NeurosymbolicMistral(config)
    scientist = AIScientist(model)
    results = scientist.conduct_research("neurosymbolic AI", cycles=1, verbose=False)
    
    assert len(results["hypotheses"]) == 1
    assert len(results["experiments"]) == 1
    assert results["paper"] is not None
    logger.info("✅ AI scientist test passed")
    return True


def test_architecture_evolution():
    """Test architecture evolution."""
    logger.info("Testing architecture evolution...")
    
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = NeurosymbolicMistral(config)
    evolver = ArchitectureEvolver(model, population_size=3)
    evolved_model = evolver.evolve(generations=2, verbose=False)
    
    assert evolved_model is not None
    assert evolver.best_score >= 0
    logger.info("✅ Architecture evolution test passed")
    return True


def test_self_improving():
    """Test self-improving trainer."""
    logger.info("Testing self-improving trainer...")
    
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    
    model = NeurosymbolicMistral(config)
    trainer = SelfImprovingTrainer(model)
    trainer.improve(["What is the future of AI?"], cycles=1, verbose=False)
    
    assert len(trainer.training_history) > 0
    logger.info("✅ Self-improving test passed")
    return True


if __name__ == "__main__":
    print("="*80)
    print("ADVANCED AI SYSTEM TESTS")
    print("="*80)
    
    tests = [
        ("Neurosymbolic Reasoning", test_neurosymbolic_reasoning),
        ("Quantum Model", test_quantum_model),
        ("AI Scientist", test_ai_scientist),
        ("Architecture Evolution", test_architecture_evolution),
        ("Self-Improving", test_self_improving),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True))
        except Exception as e:
            logger.error(f"{name} test failed: {e}")
            results.append((name, False))
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("="*80)
    if all_passed:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check logs above.")
    print("="*80)


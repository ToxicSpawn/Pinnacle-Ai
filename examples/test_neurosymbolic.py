"""
Example: Testing Neurosymbolic AI with Research Agent
"""

import logging
from pinnacle_ai.core.neurosymbolic.neural_adapter import NeurosymbolicMistral
from pinnacle_ai.core.models.mistral import MistralConfig
from pinnacle_ai.agents.research_agent import ResearchAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("="*80)
    print("NEUROSYMBOLIC AI + RESEARCH AGENT DEMO")
    print("="*80)
    
    # Initialize neurosymbolic model
    logger.info("Initializing Neurosymbolic Mistral model...")
    config = MistralConfig(
        vocab_size=32000,
        hidden_size=1024,  # Small for demo
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=4,
    )
    
    model = NeurosymbolicMistral(config)
    
    # Test math reasoning
    print("\n" + "="*80)
    print("1. MATH REASONING TEST")
    print("="*80)
    prompt = "Prove that the square root of 2 is irrational"
    result = model.generate_with_reasoning(prompt, use_symbolic=True)
    print(result)
    
    # Initialize research agent
    print("\n" + "="*80)
    print("2. RESEARCH AGENT TEST")
    print("="*80)
    agent = ResearchAgent(model, memory_size=100)
    
    # Generate hypothesis
    print("\nGenerating hypothesis about neurosymbolic AI...")
    hypothesis = agent.generate_hypothesis("neurosymbolic AI")
    print(f"Hypothesis: {hypothesis}")
    
    # Design experiment
    print("\nDesigning experiment...")
    experiment = agent.design_experiment(hypothesis)
    print(f"Experiment:\n{experiment[:300]}...")
    
    # Self-improvement
    print("\nRunning self-improvement cycle...")
    agent.self_improve("neurosymbolic AI", num_iterations=2)
    
    # Memory summary
    print("\nMemory Summary:")
    summary = agent.get_memory_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Complete research cycle
    print("\n" + "="*80)
    print("3. COMPLETE RESEARCH CYCLE")
    print("="*80)
    results = agent.research_cycle("autonomous AI systems", num_cycles=2)
    print(f"\nResearch Results:")
    print(f"  Topic: {results['topic']}")
    print(f"  Hypotheses generated: {len(results['hypotheses'])}")
    print(f"  Experiments designed: {len(results['experiments'])}")
    print(f"  Improvement cycles: {len(results['improvements'])}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()


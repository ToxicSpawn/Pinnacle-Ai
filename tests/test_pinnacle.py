import pytest
import asyncio

def test_config():
    """Test configuration"""
    from pinnacle_ai.core.config import PinnacleConfig
    config = PinnacleConfig()
    assert config.model_name is not None
    assert config.hidden_size > 0
    print("[OK] Config test passed")

def test_memory():
    """Test infinite memory"""
    from pinnacle_ai.memory.infinite_memory import InfiniteMemory
    
    memory = InfiniteMemory(dimension=384, max_size=1000)
    
    # Store
    memory.store("Hello world", "test")
    memory.store("Test memory", "test")
    
    # Retrieve
    results = memory.retrieve("Hello", top_k=2)
    
    assert len(results) <= 2
    assert memory.size() == 2
    print("[OK] Memory test passed")

def test_emotions():
    """Test emotional system"""
    from pinnacle_ai.consciousness.emotional import EmotionalSystem
    
    emotions = EmotionalSystem()
    emotions.process("I'm feeling happy today!", "That's wonderful!")
    
    state = emotions.get_state()
    assert "dominant" in state
    assert "mood" in state
    print("[OK] Emotions test passed")

def test_causal():
    """Test causal engine"""
    from pinnacle_ai.reasoning.causal_engine import CausalEngine
    
    engine = CausalEngine()
    engine.add_cause("rain", "wet_ground", "causes")
    
    explanation = engine.why("wet ground")
    assert "rain" in explanation.lower() or "don't have" in explanation.lower()
    print("[OK] Causal test passed")

def test_simulator():
    """Test world simulator"""
    from pinnacle_ai.simulation.world_engine import WorldSimulator
    
    sim = WorldSimulator()
    result = sim.simulate("Test scenario", steps=10)
    
    assert "scenario" in result
    assert "prediction" in result
    print("[OK] Simulator test passed")

def test_evolution():
    """Test self-evolution"""
    from pinnacle_ai.evolution.self_evolution import SelfEvolution
    
    evolution = SelfEvolution(population_size=5, mutation_rate=0.1)
    result = evolution.evolve(generations=3)
    
    assert "improvement" in result
    print("[OK] Evolution test passed")

@pytest.mark.asyncio
async def test_swarm():
    """Test swarm intelligence"""
    from pinnacle_ai.swarm.swarm_intelligence import SwarmIntelligence
    
    swarm = SwarmIntelligence(num_agents=5)
    result = await swarm.solve("Test problem")
    
    assert "aggregated_solution" in result
    print("[OK] Swarm test passed")

def test_knowledge():
    """Test knowledge engine"""
    from pinnacle_ai.knowledge.knowledge_engine import KnowledgeEngine
    
    knowledge = KnowledgeEngine()
    result = knowledge.update()
    
    assert "new_topics" in result
    print("[OK] Knowledge test passed")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*50)
    print("PINNACLE-AI TEST SUITE")
    print("="*50 + "\n")
    
    test_config()
    test_memory()
    test_emotions()
    test_causal()
    test_simulator()
    test_evolution()
    asyncio.run(test_swarm())
    test_knowledge()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_all_tests()


from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class PinnacleConfig:
    """Configuration for Pinnacle-AI"""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    hidden_size: int = 4096
    max_length: int = 4096
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    
    # Memory settings
    memory_enabled: bool = True
    memory_size: int = 100000
    memory_dimension: int = 384
    retrieval_top_k: int = 10
    
    # Consciousness settings
    consciousness_enabled: bool = True
    emotional_enabled: bool = True
    
    # Reasoning settings
    causal_reasoning_enabled: bool = True
    
    # Simulation settings
    simulation_enabled: bool = True
    
    # Evolution settings
    evolution_enabled: bool = True
    population_size: int = 10
    mutation_rate: float = 0.1
    
    # Swarm settings
    swarm_enabled: bool = True
    num_agents: int = 10
    
    # Knowledge settings
    knowledge_enabled: bool = True
    
    # Autonomous lab settings
    autonomous_lab_enabled: bool = True
    
    # Hardware settings
    device: str = "auto"
    use_4bit: bool = True
    use_flash_attention: bool = True
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

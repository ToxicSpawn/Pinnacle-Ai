"""
Quantum-Ready FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
import torch

from pinnacle_ai.core.neurosymbolic.neural_adapter import NeurosymbolicMistral
from pinnacle_ai.core.quantum_neuro import QuantumNeurosymbolicMistral
from pinnacle_ai.core.ai_scientist import AIScientist
from pinnacle_ai.core.self_evolving import ArchitectureEvolver
from pinnacle_ai.core.self_improving import SelfImprovingTrainer
from pinnacle_ai.core.models.mistral import MistralConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="Pinnacle AI Quantum-Ready API", version="0.4.0")

# Global models (initialized on startup)
classical_model = None
quantum_model = None
scientist = None
evolver = None


class Query(BaseModel):
    """Query model for generation."""
    text: str
    use_quantum: bool = False
    max_length: Optional[int] = 200


class ResearchRequest(BaseModel):
    """Request model for research."""
    question: str
    cycles: int = 3


class EvolutionRequest(BaseModel):
    """Request model for evolution."""
    generations: int = 5
    task: str = "math"


class ImprovementRequest(BaseModel):
    """Request model for self-improvement."""
    questions: List[str]
    cycles: int = 2


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global classical_model, quantum_model, scientist, evolver
    
    try:
        logger.info("Initializing models...")
        
        # Initialize config
        config = MistralConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
        )
        
        # Classical model
        classical_model = NeurosymbolicMistral(config)
        logger.info("Classical model loaded")
        
        # Quantum model
        quantum_model = QuantumNeurosymbolicMistral(config, n_qubits=4)
        logger.info("Quantum model loaded")
        
        # AI Scientist
        scientist = AIScientist(classical_model)
        logger.info("AI Scientist initialized")
        
        # Architecture Evolver
        evolver = ArchitectureEvolver(classical_model)
        logger.info("Architecture Evolver initialized")
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


@app.post("/generate")
async def generate(query: Query):
    """
    Generate text with optional quantum processing.
    
    Args:
        query: Generation query
        
    Returns:
        Generated text
    """
    if classical_model is None or quantum_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        model = quantum_model if query.use_quantum else classical_model
        
        # Generate with reasoning
        result = model.generate_with_reasoning(
            query.text,
            max_length=query.max_length or 200,
            use_symbolic=True,
        )
        
        return {
            "response": result,
            "used_quantum": query.use_quantum,
            "used_symbolic": True,
        }
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research")
async def conduct_research(request: ResearchRequest):
    """
    Conduct autonomous research.
    
    Args:
        request: Research request
        
    Returns:
        Research results
    """
    if scientist is None:
        raise HTTPException(status_code=503, detail="AI Scientist not loaded")
    
    try:
        results = scientist.conduct_research(request.question, request.cycles)
        return {
            "question": results["question"],
            "hypotheses": results["hypotheses"],
            "experiments": results["experiments"],
            "results": results["results"],
            "paper": {
                "title": results["paper"]["title"],
                "abstract": results["paper"]["abstract"],
            },
        }
    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/publish")
async def publish_paper(paper_title: str):
    """
    Publish generated paper.
    
    Args:
        paper_title: Title of paper to publish
        
    Returns:
        Publication status
    """
    if scientist is None:
        raise HTTPException(status_code=503, detail="AI Scientist not loaded")
    
    try:
        # In a real implementation, would load the paper
        paper = {
            "title": paper_title,
            "abstract": "Sample abstract",
            "sections": {},
            "references": [],
        }
        
        arxiv_id = scientist.publish_paper(paper, arxiv=True)
        return {
            "status": "published",
            "arxiv_id": arxiv_id,
            "paper_title": paper_title,
        }
    except Exception as e:
        logger.error(f"Publish error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evolve")
async def evolve_architecture(request: EvolutionRequest):
    """
    Evolve model architecture.
    
    Args:
        request: Evolution request
        
    Returns:
        Evolution results
    """
    if evolver is None:
        raise HTTPException(status_code=503, detail="Evolver not loaded")
    
    try:
        best_model = evolver.evolve(
            generations=request.generations,
            task=request.task,
            verbose=False,
        )
        
        return {
            "status": "evolved",
            "generations": request.generations,
            "task": request.task,
            "best_score": evolver.best_score,
        }
    except Exception as e:
        logger.error(f"Evolution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/improve")
async def self_improve(request: ImprovementRequest):
    """
    Self-improve the model.
    
    Args:
        request: Improvement request
        
    Returns:
        Improvement status
    """
    try:
        trainer = SelfImprovingTrainer(classical_model)
        trainer.improve(request.questions, cycles=request.cycles, verbose=False)
        
        return {
            "status": "improved",
            "questions": request.questions,
            "cycles": request.cycles,
            "training_entries": len(trainer.training_history),
        }
    except Exception as e:
        logger.error(f"Improvement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "classical_model_loaded": classical_model is not None,
        "quantum_model_loaded": quantum_model is not None,
        "scientist_loaded": scientist is not None,
        "evolver_loaded": evolver is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


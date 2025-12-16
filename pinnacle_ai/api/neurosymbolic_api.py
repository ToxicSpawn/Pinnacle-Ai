"""
FastAPI Server with Neurosymbolic Support
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
import torch

from pinnacle_ai.core.neurosymbolic.neural_adapter import NeurosymbolicMistral
from pinnacle_ai.core.models.mistral import MistralConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="Pinnacle AI Neurosymbolic API", version="0.3.0")

# Global model (loaded on startup)
model = None


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    text: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    use_symbolic: Optional[bool] = True


class ProofRequest(BaseModel):
    """Request model for proof generation."""
    goal: str


class ResearchRequest(BaseModel):
    """Request model for research agent."""
    topic: str
    num_cycles: Optional[int] = 3


class GenerationResponse(BaseModel):
    """Response model for generation."""
    response: str
    used_symbolic: bool


class ProofResponse(BaseModel):
    """Response model for proofs."""
    proof: str
    provable: bool


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model
    try:
        logger.info("Loading Neurosymbolic Mistral model...")
        
        # Use small config for API (can be changed)
        config = MistralConfig(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
        )
        
        model = NeurosymbolicMistral(config)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text with neurosymbolic reasoning."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = model.generate_with_reasoning(
            request.text,
            max_length=request.max_length,
            temperature=request.temperature,
            use_symbolic=request.use_symbolic,
        )
        
        return GenerationResponse(
            response=result,
            used_symbolic=request.use_symbolic,
        )
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prove", response_model=ProofResponse)
async def prove(request: ProofRequest):
    """Prove a goal using symbolic reasoning."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        proof = model.prove(request.goal)
        provable = "Proven" in proof or "QED" in proof
        
        return ProofResponse(
            proof=proof,
            provable=provable,
        )
    except Exception as e:
        logger.error(f"Proof error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research")
async def research(request: ResearchRequest):
    """Run research agent cycle."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        from pinnacle_ai.agents.research_agent import ResearchAgent
        
        agent = ResearchAgent(model, memory_size=100)
        results = agent.research_cycle(request.topic, num_cycles=request.num_cycles)
        
        return {
            "topic": results["topic"],
            "hypotheses": results["hypotheses"],
            "experiments": results["experiments"],
            "num_cycles": request.num_cycles,
        }
    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "neurosymbolic": True,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from loguru import logger
import asyncio

app = FastAPI(
    title="Pinnacle-AI API",
    description="The Ultimate AGI System API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI instance
ai = None


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    use_memory: Optional[bool] = True
    use_emotions: Optional[bool] = True

class GenerateResponse(BaseModel):
    response: str
    emotional_state: Optional[Dict] = None

class ThinkRequest(BaseModel):
    problem: str

class ReasonRequest(BaseModel):
    problem: str

class MemoryRequest(BaseModel):
    text: str

class RecallRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class ResearchRequest(BaseModel):
    question: str
    cycles: Optional[int] = 3

class EvolutionRequest(BaseModel):
    generations: Optional[int] = 5

class SwarmRequest(BaseModel):
    problem: str


@app.on_event("startup")
async def startup():
    """Initialize AI on startup"""
    global ai
    logger.info("Starting Pinnacle-AI...")
    
    try:
        from pinnacle_ai.core.model import PinnacleAI
        from pinnacle_ai.core.config import PinnacleConfig
        
        config = PinnacleConfig(
            use_4bit=True,
            memory_enabled=True,
            consciousness_enabled=True,
            emotional_enabled=True,
            causal_reasoning_enabled=True,
            simulation_enabled=True,
            evolution_enabled=True,
            swarm_enabled=True,
            knowledge_enabled=True,
            autonomous_lab_enabled=True
        )
        
        ai = PinnacleAI(config)
        logger.info("Pinnacle-AI loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load AI: {e}")
        # Don't raise - allow API to start without AI for testing
        ai = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Pinnacle-AI",
        "version": "1.0.0",
        "status": "operational",
        "description": "The Ultimate AGI System"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "ai_loaded": ai is not None
    }


@app.get("/status")
async def status():
    """Get detailed system status"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    return ai.get_status()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a response"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    try:
        response = ai.generate(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            use_memory=request.use_memory,
            use_emotions=request.use_emotions
        )
        
        emotional_state = ai.feel() if ai.emotions else None
        
        return GenerateResponse(
            response=response,
            emotional_state=emotional_state
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/think")
async def think(request: ThinkRequest):
    """Deep thinking on a problem"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    try:
        return ai.think(request.problem)
    except Exception as e:
        logger.error(f"Thinking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reason")
async def reason(request: ReasonRequest):
    """Step-by-step reasoning"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    try:
        response = ai.reason(request.problem)
        return {"problem": request.problem, "reasoning": response}
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/store")
async def store_memory(request: MemoryRequest):
    """Store in memory"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    return ai.remember(request.text)


@app.post("/memory/recall")
async def recall_memory(request: RecallRequest):
    """Recall from memory"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    memories = ai.recall(request.query, top_k=request.top_k)
    return {"query": request.query, "memories": memories}


@app.get("/emotions")
async def get_emotions():
    """Get emotional state"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    return ai.feel()


@app.post("/research")
async def conduct_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Conduct autonomous research"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    try:
        results = ai.research(request.question, cycles=request.cycles)
        return results
    except Exception as e:
        logger.error(f"Research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evolve")
async def evolve(request: EvolutionRequest):
    """Self-evolution"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    try:
        return ai.evolve(generations=request.generations)
    except Exception as e:
        logger.error(f"Evolution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/swarm/solve")
async def swarm_solve(request: SwarmRequest):
    """Swarm problem solving"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    try:
        return await ai.swarm_solve(request.problem)
    except Exception as e:
        logger.error(f"Swarm error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/update")
async def update_knowledge():
    """Update knowledge base"""
    if ai is None:
        raise HTTPException(status_code=503, detail="AI not loaded")
    
    return ai.update_knowledge()


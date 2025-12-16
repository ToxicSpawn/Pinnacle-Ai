"""
FastAPI backend for Pinnacle AI with comprehensive features
"""

from fastapi import FastAPI, HTTPException, Depends, status, Security, Form
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging
import time
from datetime import datetime

try:
    from src.main import PinnacleAI
    from src.security.security_manager import SecurityManager
    from src.core.performance_optimizer import PerformanceOptimizer
    PINNACLE_AVAILABLE = True
except ImportError:
    PINNACLE_AVAILABLE = False
    logging.warning("Pinnacle AI not available. API will have limited functionality.")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinnacle AI
if PINNACLE_AVAILABLE:
    pinnacle = PinnacleAI()
    security = SecurityManager()
    performance = PerformanceOptimizer(pinnacle.config)
else:
    pinnacle = None
    security = None
    performance = None

# Initialize FastAPI
app = FastAPI(
    title="Pinnacle AI API",
    description="The API for the Pinnacle AI system - The Absolute Pinnacle of Artificial Intelligence",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Dict:
    """Get current user from token"""
    if not security or not token:
        return {"sub": "anonymous", "roles": ["user"]}
    
    user = security.validate_token(token)
    if not user:
        return {"sub": "anonymous", "roles": ["user"]}
    return user

async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Dict:
    """Validate API key"""
    if not security or not api_key:
        return {"api_key": None}
    
    if not security.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return {"api_key": api_key}

# Models
class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=10000, description="The task to execute")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context for the task")
    priority: Optional[int] = Field(default=3, ge=1, le=5, description="Task priority (1-5)")
    timeout: Optional[int] = Field(default=300, ge=30, le=3600, description="Timeout in seconds")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    task: str
    result: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    learning: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ImprovementRequest(BaseModel):
    components: Optional[List[str]] = Field(default=None, description="Specific components to improve")
    strategy: Optional[str] = Field(default="balanced", description="Improvement strategy")

class SystemStatus(BaseModel):
    status: str
    components: Dict[str, Any]
    performance: Dict[str, Any]
    security: Dict[str, Any]

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pinnacle AI API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "status": "operational" if PINNACLE_AVAILABLE else "limited"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pinnacle_available": PINNACLE_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/tasks", response_model=TaskResponse)
async def execute_task(
    request: TaskRequest,
    user: Dict = Depends(get_current_user),
    api_key: Dict = Security(get_api_key)
):
    """Execute a task with Pinnacle AI"""
    if not PINNACLE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinnacle AI is not available"
        )
    
    start_time = time.time()

    try:
        # Validate input
        if not security.validate_input(request.task, "text"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid task input detected"
            )

        # Log audit event
        security.log_audit_event(
            "task_execution",
            user.get("sub", "api_user"),
            {
                "task": request.task,
                "context": request.context,
                "priority": request.priority
            }
        )

        # Optimize execution
        optimized_context = performance.optimize_execution(
            request.task,
            {**request.context, "priority": request.priority}
        )

        # Execute task
        result = pinnacle.execute_task(request.task, optimized_context)

        # Prepare response
        response = TaskResponse(
            task_id=f"task_{int(time.time())}",
            status="completed" if result.get("evaluation", {}).get("success", False) else "failed",
            task=request.task,
            result=result,
            execution=result.get("execution"),
            evaluation=result.get("evaluation"),
            learning=result.get("learning"),
            performance=result.get("performance", {
                "execution_time": time.time() - start_time,
                "optimizations_applied": optimized_context.get("optimizations", [])
            })
        )

        # Log completion
        security.log_audit_event(
            "task_completed",
            user.get("sub", "api_user"),
            {
                "task_id": response.task_id,
                "success": result.get("evaluation", {}).get("success", False),
                "quality": result.get("evaluation", {}).get("quality", 0)
            }
        )

        return response

    except Exception as e:
        logger.error(f"Task execution failed: {str(e)}")
        if security:
            security.log_audit_event(
                "task_failed",
                user.get("sub", "api_user"),
                {
                    "task": request.task,
                    "error": str(e)
                }
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
async def get_task_result(
    task_id: str,
    user: Dict = Depends(get_current_user),
    api_key: Dict = Security(get_api_key)
):
    """Get result for a previously executed task"""
    if not PINNACLE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinnacle AI is not available"
        )
    
    try:
        # In a real implementation, this would retrieve from a database
        # For this example, we'll return a placeholder
        return TaskResponse(
            task_id=task_id,
            status="completed",
            task="Example task",
            result={"message": "Task completed successfully"},
            evaluation={"success": True, "quality": 0.95, "efficiency": 0.9},
            performance={"execution_time": 2.5}
        )
    except Exception as e:
        logger.error(f"Failed to get task result: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/improve", response_model=Dict[str, Any])
async def improve_system(
    request: ImprovementRequest,
    user: Dict = Depends(get_current_user),
    api_key: Dict = Security(get_api_key)
):
    """Improve the Pinnacle AI system"""
    if not PINNACLE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinnacle AI is not available"
        )
    
    try:
        # Check if improvement is allowed
        if not pinnacle.config.get("self_evolution", {}).get("active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Self-improvement is disabled"
            )

        # Log improvement attempt
        security.log_audit_event(
            "system_improvement",
            user.get("sub", "api_user"),
            {
                "action": "initiate",
                "components": request.components,
                "strategy": request.strategy
            }
        )

        # Improve system
        improvements = pinnacle.orchestrator.improve_system()

        # Log successful improvement
        security.log_audit_event(
            "system_improvement",
            user.get("sub", "api_user"),
            {
                "action": "completed",
                "improvements": list(improvements.keys())
            }
        )

        return improvements
    except Exception as e:
        logger.error(f"System improvement failed: {str(e)}")
        if security:
            security.log_audit_event(
                "system_improvement",
                user.get("sub", "api_user"),
                {
                    "action": "failed",
                    "error": str(e)
                }
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/benchmark", response_model=Dict[str, Any])
async def run_benchmark(
    user: Dict = Depends(get_current_user),
    api_key: Dict = Security(get_api_key)
):
    """Run system benchmark"""
    if not PINNACLE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinnacle AI is not available"
        )
    
    try:
        # Log benchmark attempt
        security.log_audit_event(
            "system_benchmark",
            user.get("sub", "api_user"),
            {"action": "initiate"}
        )

        # Run benchmark
        results = pinnacle.benchmark()

        # Log benchmark completion
        security.log_audit_event(
            "system_benchmark",
            user.get("sub", "api_user"),
            {
                "action": "completed",
                "success_rate": results["summary"]["success_rate"],
                "average_quality": results["summary"]["average_quality"]
            }
        )

        return results
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        if security:
            security.log_audit_event(
                "system_benchmark",
                user.get("sub", "api_user"),
                {
                    "action": "failed",
                    "error": str(e)
                }
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/status", response_model=SystemStatus)
async def get_system_status(
    user: Dict = Depends(get_current_user),
    api_key: Dict = Security(get_api_key)
):
    """Get system status"""
    if not PINNACLE_AVAILABLE:
        return SystemStatus(
            status="limited",
            components={},
            performance={},
            security={}
        )
    
    try:
        status_data = pinnacle.orchestrator.get_system_status()
        performance_metrics = performance.get_optimization_suggestions() if performance else {}

        return SystemStatus(
            status="operational",
            components=status_data.get("components", {}),
            performance=performance_metrics,
            security=security.get_status() if security else {}
        )
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/auth/token")
async def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...)
):
    """Authenticate and get access token"""
    if not security:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Security manager not available"
        )
    
    try:
        # Simple authentication (in production, use proper user database)
        if username == "admin" and password == "password":
            access_token = security.generate_token(username, ["user", "admin"])
            return {"access_token": access_token, "token_type": "bearer"}
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/docs", response_class=HTMLResponse)
async def get_api_docs():
    """Serve API documentation"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Pinnacle AI API Documentation</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    font-family: 'Roboto', sans-serif;
                }
                iframe {
                    width: 100%;
                    height: 100vh;
                    border: none;
                }
            </style>
        </head>
        <body>
            <iframe src="/api/docs"></iframe>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=4
    )


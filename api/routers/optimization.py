# api/routers/optimization.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, List, Any, Optional, Tuple
import logging
import uuid
import time
from pydantic import BaseModel

# Create router
router = APIRouter()

# Logger
logger = logging.getLogger(__name__)

# In-memory storage for optimization jobs (replace with database in production)
optimization_jobs = {}

# Models for request/response
class OptimizationRequest(BaseModel):
    function_name: str
    dimension: int = 10
    optimizer: str = "DE"
    max_evaluations: int = 1000
    bounds: Optional[List[Tuple[float, float]]] = None
    parameters: Dict[str, Any] = {}

class OptimizationResult(BaseModel):
    id: str
    status: str
    function_name: str
    dimension: int
    optimizer: str
    max_evaluations: int
    bounds: List[Tuple[float, float]]
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None

# Background task to run optimization
def run_optimization_task(job_id: str, request: OptimizationRequest):
    try:
        # Update status to running
        optimization_jobs[job_id]["status"] = "running"
        
        # Import necessary modules
        from core.optimization import create_objective_function, create_optimizer
        import numpy as np
        
        # Create bounds if not provided
        bounds = request.bounds or [(-5, 5)] * request.dimension
        
        # Create objective function
        objective_func = create_objective_function(
            request.function_name, 
            request.dimension
        )
        
        # Create optimizer
        optimizer = create_optimizer(
            optimizer_type=request.optimizer,
            dim=request.dimension,
            bounds=bounds,
            population_size=request.parameters.get("population_size", 50),
            **{k: v for k, v in request.parameters.items() if k != "population_size"}
        )
        
        # Run optimization
        start_time = time.time()
        best_solution, best_score = optimizer.optimize(
            objective_func, 
            max_evals=request.max_evaluations,
            verbose=False
        )
        elapsed_time = time.time() - start_time
        
        # Update job with result
        optimization_jobs[job_id]["result"] = {
            "best_score": float(best_score),
            "best_solution": best_solution.tolist() if hasattr(best_solution, 'tolist') else best_solution,
            "elapsed_time": elapsed_time,
            "evaluations": request.max_evaluations,  # In a real implementation, get actual count from optimizer
            "convergence_history": [],  # In a real implementation, get from optimizer
        }
        optimization_jobs[job_id]["status"] = "completed"
        optimization_jobs[job_id]["completed_at"] = time.time()
    
    except Exception as e:
        logger.error(f"Error running optimization {job_id}: {str(e)}")
        optimization_jobs[job_id]["status"] = "failed"
        optimization_jobs[job_id]["error"] = str(e)
        optimization_jobs[job_id]["completed_at"] = time.time()

# Endpoints
@router.post("/run", response_model=OptimizationResult)
async def create_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """
    Run a new optimization
    """
    # Generate unique ID
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = {
        "id": job_id,
        "status": "pending",
        "function_name": request.function_name,
        "dimension": request.dimension,
        "optimizer": request.optimizer,
        "max_evaluations": request.max_evaluations,
        "bounds": request.bounds or [(-5, 5)] * request.dimension,
        "parameters": request.parameters,
        "created_at": time.time()
    }
    
    # Store job
    optimization_jobs[job_id] = job
    
    # Run optimization in background
    background_tasks.add_task(run_optimization_task, job_id, request)
    
    return job

@router.get("/{job_id}", response_model=OptimizationResult)
async def get_optimization(job_id: str):
    """
    Get optimization results by ID
    """
    if job_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    
    return optimization_jobs[job_id]

@router.get("/", response_model=List[OptimizationResult])
async def list_optimizations(limit: int = 10):
    """
    List recent optimization jobs
    """
    # Get most recent jobs
    recent = sorted(
        optimization_jobs.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )[:limit]
    
    return recent
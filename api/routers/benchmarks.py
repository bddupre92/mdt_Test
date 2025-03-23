# api/routers/benchmarks.py
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
import uuid
import time
from pydantic import BaseModel

# Create router
router = APIRouter()

# Logger
logger = logging.getLogger(__name__)

# In-memory storage for benchmark results (replace with database in production)
benchmark_results = {}

# Models for request/response
class BenchmarkRequest(BaseModel):
    function_name: str
    dimension: int = 10
    max_evaluations: int = 1000
    optimizers: List[str] = ["DE", "PSO", "ES", "GWO", "ACO"]
    repetitions: int = 3

class BenchmarkResult(BaseModel):
    id: str
    status: str
    function_name: str
    dimension: int
    max_evaluations: int
    optimizers: List[str]
    repetitions: int
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None

# Background task to run benchmark
def run_benchmark_task(benchmark_id: str, request: BenchmarkRequest):
    try:
        # Update status to running
        benchmark_results[benchmark_id]["status"] = "running"
        
        # Import necessary modules
        from core.optimization import create_objective_function, create_optimizer
        import numpy as np
        
        # Run benchmark
        results = {}
        for optimizer_name in request.optimizers:
            optimizer_results = []
            for _ in range(request.repetitions):
                # Create objective function
                objective_func = create_objective_function(
                    request.function_name, 
                    request.dimension
                )
                
                # Create bounds
                bounds = [(-5, 5)] * request.dimension
                
                # Create optimizer
                optimizer = create_optimizer(
                    optimizer_type=optimizer_name,
                    dim=request.dimension,
                    bounds=bounds,
                    population_size=50
                )
                
                # Run optimization
                try:
                    start_time = time.time()
                    best_solution, best_score = optimizer.optimize(
                        objective_func, 
                        max_evals=request.max_evaluations,
                        verbose=False
                    )
                    elapsed_time = time.time() - start_time
                    
                    # Add results
                    optimizer_results.append({
                        "best_score": float(best_score),
                        "elapsed_time": elapsed_time,
                        "success": True
                    })
                except Exception as e:
                    logger.error(f"Error running optimizer {optimizer_name}: {str(e)}")
                    optimizer_results.append({
                        "success": False,
                        "error": str(e)
                    })
            
            # Calculate statistics
            successful_runs = [r for r in optimizer_results if r["success"]]
            if successful_runs:
                scores = [r["best_score"] for r in successful_runs]
                times = [r["elapsed_time"] for r in successful_runs]
                results[optimizer_name] = {
                    "runs": optimizer_results,
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "mean_time": float(np.mean(times)),
                    "success_rate": len(successful_runs) / request.repetitions
                }
            else:
                results[optimizer_name] = {
                    "runs": optimizer_results,
                    "success_rate": 0,
                    "error": "All runs failed"
                }
        
        # Update benchmark results
        benchmark_results[benchmark_id]["results"] = results
        benchmark_results[benchmark_id]["status"] = "completed"
        benchmark_results[benchmark_id]["completed_at"] = time.time()
    
    except Exception as e:
        logger.error(f"Error running benchmark {benchmark_id}: {str(e)}")
        benchmark_results[benchmark_id]["status"] = "failed"
        benchmark_results[benchmark_id]["error"] = str(e)
        benchmark_results[benchmark_id]["completed_at"] = time.time()

# Endpoints
@router.post("/run", response_model=BenchmarkResult)
async def create_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """
    Run a new benchmark
    """
    # Generate unique ID
    benchmark_id = str(uuid.uuid4())
    
    # Create benchmark entry
    benchmark = {
        "id": benchmark_id,
        "status": "pending",
        "function_name": request.function_name,
        "dimension": request.dimension,
        "max_evaluations": request.max_evaluations,
        "optimizers": request.optimizers,
        "repetitions": request.repetitions,
        "created_at": time.time()
    }
    
    # Store benchmark
    benchmark_results[benchmark_id] = benchmark
    
    # Run benchmark in background
    background_tasks.add_task(run_benchmark_task, benchmark_id, request)
    
    return benchmark

@router.get("/{benchmark_id}", response_model=BenchmarkResult)
async def get_benchmark(benchmark_id: str):
    """
    Get benchmark results by ID
    """
    if benchmark_id not in benchmark_results:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    return benchmark_results[benchmark_id]

@router.get("/", response_model=List[BenchmarkResult])
async def list_benchmarks(limit: int = 10):
    """
    List recent benchmarks
    """
    # Get most recent benchmarks
    recent = sorted(
        benchmark_results.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )[:limit]
    
    return recent
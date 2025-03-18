"""
FastAPI Server

This module provides the FastAPI server for running benchmarks and accessing results.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import json

from app.core.benchmark_repository import BenchmarkRepository
from app.core.benchmark_service import BenchmarkService
from app.core.optimizer_adapters import (
    DifferentialEvolutionAdapter,
    EvolutionStrategyAdapter,
    AntColonyAdapter,
    GreyWolfAdapter,
    MetaOptimizerAdapter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize components
repository = BenchmarkRepository()
service = BenchmarkService()

# Create FastAPI app
app = FastAPI(title="Benchmark API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BenchmarkRequest(BaseModel):
    """Request model for running benchmarks."""
    function_name: str
    dimension: int = 2
    max_evaluations: int = 1000
    num_runs: int = 1
    optimizers: List[str]

class BenchmarkResponse(BaseModel):
    """Response model for benchmark results."""
    function_name: str
    dimension: int
    results: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/functions")
async def list_functions():
    """List available benchmark functions."""
    return {
        "functions": repository.list_functions()
    }

@app.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest):
    """Run benchmark with specified parameters."""
    logger.info(f"Running benchmark: {request}")
    
    try:
        # Create optimizers
        optimizers = []
        for opt_name in request.optimizers:
            if opt_name == "DifferentialEvolution":
                optimizers.append(DifferentialEvolutionAdapter())
            elif opt_name == "EvolutionStrategy":
                optimizers.append(EvolutionStrategyAdapter())
            elif opt_name == "AntColony":
                optimizers.append(AntColonyAdapter())
            elif opt_name == "GreyWolf":
                optimizers.append(GreyWolfAdapter())
            elif opt_name == "MetaOptimizer":
                optimizers.append(MetaOptimizerAdapter())
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown optimizer: {opt_name}"
                )
        
        # Run benchmark
        results = service.run_comparison(
            optimizers=optimizers,
            function_names=[request.function_name],
            dimension=request.dimension,
            max_evaluations=request.max_evaluations,
            num_runs=request.num_runs
        )
        
        # Create summary
        summary = {
            "optimizers": {},
            "convergence": {}
        }
        
        for opt_name, opt_results in results.items():
            # Calculate statistics
            fitness_values = [r.best_fitness for r in opt_results]
            summary["optimizers"][opt_name] = {
                "best_fitness": min(fitness_values),
                "mean_fitness": sum(fitness_values) / len(fitness_values),
                "evaluations": opt_results[0].function_evaluations
            }
            
            # Get convergence data from first run
            summary["convergence"][opt_name] = opt_results[0].convergence
        
        return {
            "function_name": request.function_name,
            "dimension": request.dimension,
            "results": summary
        }
        
    except Exception as e:
        logger.exception("Error running benchmark")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/results")
async def list_results():
    """List available benchmark results."""
    try:
        results = []
        for filename in os.listdir(service.results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(service.results_dir, filename), 'r') as f:
                    results.append(json.load(f))
        return {"results": results}
    except Exception as e:
        logger.exception("Error listing results")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def create_app() -> FastAPI:
    """Create FastAPI application.
    
    Returns:
        FastAPI application instance
    """
    return app 
"""
Benchmark API Router

This module provides API endpoints for running benchmarks and comparing optimizers.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import os
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

from app.core.benchmark_repository import BenchmarkRepository, BenchmarkFunction
from app.core.benchmark_service import BenchmarkService
from app.core.optimizer_adapter import OptimizerFactory, OptimizerAdapter

# Create router
router = APIRouter(
    prefix="/api/benchmarks",
    tags=["benchmarks"],
    responses={404: {"description": "Not found"}},
)

# Create benchmark service
benchmark_service = BenchmarkService(
    repository=BenchmarkRepository(),
    results_dir="benchmark_results"
)


# Models for API requests and responses
class BenchmarkRequest(BaseModel):
    """Request model for running a benchmark."""
    
    benchmark_name: str = Field(..., description="Name of the benchmark function")
    optimizer_type: str = Field(..., description="Type of optimizer to use")
    dimension: int = Field(10, description="Dimension of the problem")
    bounds: Optional[List[float]] = Field(None, description="Problem bounds [min, max]")
    max_evaluations: int = Field(1000, description="Maximum number of function evaluations")
    num_runs: int = Field(1, description="Number of benchmark runs")
    optimizer_params: Dict[str, Any] = Field({}, description="Parameters for the optimizer")


class ComparisonRequest(BaseModel):
    """Request model for comparing optimizers."""
    
    benchmark_name: str = Field(..., description="Name of the benchmark function")
    optimizer_types: List[str] = Field(..., description="Types of optimizers to compare")
    include_meta: bool = Field(False, description="Whether to include meta-optimizer in comparison")
    dimension: int = Field(10, description="Dimension of the problem")
    bounds: Optional[List[float]] = Field(None, description="Problem bounds [min, max]")
    max_evaluations: int = Field(1000, description="Maximum number of function evaluations")
    num_runs: int = Field(3, description="Number of benchmark runs per optimizer")


class BenchmarkResponse(BaseModel):
    """Response model for benchmark results."""
    
    benchmark_id: str = Field(..., description="Unique identifier for the benchmark")
    status: str = Field(..., description="Status of the benchmark")
    result_file: Optional[str] = Field(None, description="File containing benchmark results")
    message: Optional[str] = Field(None, description="Additional information")


class BenchmarkListResponse(BaseModel):
    """Response model for listing available benchmarks."""
    
    benchmarks: List[Dict[str, Any]] = Field(..., description="List of available benchmarks")


class OptimizerListResponse(BaseModel):
    """Response model for listing available optimizers."""
    
    optimizers: List[Dict[str, Any]] = Field(..., description="List of available optimizers")


@router.get("/functions", response_model=BenchmarkListResponse)
async def list_benchmark_functions():
    """List all available benchmark functions."""
    benchmarks = []
    
    for name, func in benchmark_service.repository.get_all_functions().items():
        benchmarks.append({
            "name": name,
            "dimensions": func.get_dimensions(),
            "bounds": func.get_bounds(),
            "global_minimum": func.get_global_minimum(),
            "description": func.get_description()
        })
    
    return {"benchmarks": benchmarks}


@router.get("/optimizers", response_model=OptimizerListResponse)
async def list_optimizers():
    """List all available optimizers."""
    optimizers = []
    
    # Create optimizer factory and get all optimizers
    optimizer_factory = OptimizerFactory()
    all_optimizers = optimizer_factory.create_optimizers()
    
    # Add meta-optimizer
    all_optimizers["meta"] = optimizer_factory.create_optimizer("meta")
    
    for name, optimizer in all_optimizers.items():
        optimizers.append({
            "id": name,
            "name": optimizer.name,
            "available": hasattr(optimizer, "optimizer_class") and optimizer.optimizer_class is not None
        })
    
    return {"optimizers": optimizers}


@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    """Run a benchmark with a specified optimizer."""
    try:
        # Validate benchmark function
        if not benchmark_service.repository.has_function(request.benchmark_name):
            raise HTTPException(status_code=404, detail=f"Benchmark function '{request.benchmark_name}' not found")
        
        # Create optimizer
        try:
            optimizer = OptimizerFactory.create_optimizer(
                request.optimizer_type,
                **request.optimizer_params
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Process bounds
        bounds = None
        if request.bounds is not None and len(request.bounds) == 2:
            bounds = (request.bounds[0], request.bounds[1])
        
        # Create unique benchmark ID
        import time
        benchmark_id = f"{request.benchmark_name}_{request.optimizer_type}_{int(time.time())}"
        
        # Run benchmark in background
        background_tasks.add_task(
            _run_benchmark_task,
            benchmark_service=benchmark_service,
            benchmark_id=benchmark_id,
            benchmark_name=request.benchmark_name,
            optimizer=optimizer,
            dimension=request.dimension,
            bounds=bounds,
            max_evaluations=request.max_evaluations,
            num_runs=request.num_runs
        )
        
        return {
            "benchmark_id": benchmark_id,
            "status": "running",
            "result_file": f"{benchmark_id}.json",
            "message": "Benchmark started in background"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=BenchmarkResponse)
async def compare_optimizers(request: ComparisonRequest, background_tasks: BackgroundTasks):
    """Compare multiple optimizers on a benchmark function."""
    try:
        # Validate benchmark function
        if not benchmark_service.repository.has_function(request.benchmark_name):
            raise HTTPException(status_code=404, detail=f"Benchmark function '{request.benchmark_name}' not found")
        
        # Create optimizers
        optimizers = {}
        for opt_type in request.optimizer_types:
            try:
                optimizers[opt_type] = OptimizerFactory.create_optimizer(opt_type)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Error creating optimizer '{opt_type}': {str(e)}")
        
        # Add meta-optimizer if requested
        if request.include_meta:
            meta_optimizer = OptimizerFactory.create_optimizer("meta")
            meta_optimizer.set_base_optimizers(optimizers)
            optimizers["meta"] = meta_optimizer
        
        # Process bounds
        bounds = None
        if request.bounds is not None and len(request.bounds) == 2:
            bounds = (request.bounds[0], request.bounds[1])
        
        # Create unique comparison ID
        import time
        comparison_id = f"comparison_{request.benchmark_name}_{int(time.time())}"
        
        # Run comparison in background
        background_tasks.add_task(
            _run_comparison_task,
            benchmark_service=benchmark_service,
            comparison_id=comparison_id,
            benchmark_name=request.benchmark_name,
            optimizers=optimizers,
            dimension=request.dimension,
            bounds=bounds,
            max_evaluations=request.max_evaluations,
            num_runs=request.num_runs
        )
        
        return {
            "benchmark_id": comparison_id,
            "status": "running",
            "result_file": f"{comparison_id}.json",
            "message": "Comparison started in background"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}", response_model=Dict[str, Any])
async def get_benchmark_results(result_id: str):
    """Get results of a specific benchmark run."""
    try:
        result_file = f"{result_id}.json"
        result_path = os.path.join(benchmark_service.results_dir, result_file)
        
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail=f"Result file '{result_file}' not found")
        
        with open(result_path, 'r') as f:
            results = json.load(f)
            
        return results
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", response_model=List[str])
async def list_benchmark_results():
    """List all available benchmark results."""
    try:
        results = []
        
        # Get all JSON files in results directory
        for filename in os.listdir(benchmark_service.results_dir):
            if filename.endswith(".json"):
                results.append(filename)
                
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def _run_benchmark_task(
    benchmark_service: BenchmarkService,
    benchmark_id: str,
    benchmark_name: str,
    optimizer: OptimizerAdapter,
    dimension: int,
    bounds: Optional[tuple],
    max_evaluations: int,
    num_runs: int
):
    """Run benchmark in background."""
    try:
        # Get benchmark function
        benchmark_func = benchmark_service.repository.get_function(benchmark_name)
        
        # Configure benchmark function
        benchmark_func.set_dimensions(dimension)
        if bounds is not None:
            benchmark_func.set_bounds(bounds)
        
        # Run benchmark
        benchmark_service.run_benchmark(
            benchmark_id=benchmark_id,
            benchmark_func=benchmark_func,
            optimizer=optimizer,
            max_evaluations=max_evaluations,
            num_runs=num_runs
        )
    except Exception as e:
        print(f"Error running benchmark: {e}")


async def _run_comparison_task(
    benchmark_service: BenchmarkService,
    comparison_id: str,
    benchmark_name: str,
    optimizers: Dict[str, OptimizerAdapter],
    dimension: int,
    bounds: Optional[tuple],
    max_evaluations: int,
    num_runs: int
):
    """Run optimizer comparison in background."""
    try:
        # Get benchmark function
        benchmark_func = benchmark_service.repository.get_function(benchmark_name)
        
        # Configure benchmark function
        benchmark_func.set_dimensions(dimension)
        if bounds is not None:
            benchmark_func.set_bounds(bounds)
        
        # Run comparison
        benchmark_service.run_comparison(
            comparison_id=comparison_id,
            benchmark_func=benchmark_func,
            optimizers=optimizers,
            max_evaluations=max_evaluations,
            num_runs=num_runs
        )
    except Exception as e:
        print(f"Error running comparison: {e}") 
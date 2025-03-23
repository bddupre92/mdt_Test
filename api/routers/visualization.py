# api/routers/visualization.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import logging
from pydantic import BaseModel
import numpy as np

# Create router
router = APIRouter()

# Logger
logger = logging.getLogger(__name__)

# Models for request/response
class VisualizationRequest(BaseModel):
    visualization_type: str
    data_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class VisualizationResult(BaseModel):
    visualization_type: str
    data: Dict[str, Any]
    parameters: Dict[str, Any]

# Endpoints
@router.post("/generate", response_model=VisualizationResult)
async def generate_visualization(request: VisualizationRequest):
    """
    Generate a visualization based on the request
    """
    try:
        # Check visualization type
        if request.visualization_type not in [
            "algorithm_selection", 
            "drift_analysis", 
            "optimizer_performance",
            "convergence_comparison"
        ]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported visualization type: {request.visualization_type}"
            )
        
        # Import appropriate visualization module
        if request.visualization_type == "algorithm_selection":
            from visualization.algorithm_selection_viz import generate_selection_visualization
            result_data = generate_selection_visualization(request.data_id, request.parameters)
        
        elif request.visualization_type == "drift_analysis":
            from visualization.drift_analysis import generate_drift_visualization
            result_data = generate_drift_visualization(request.data_id, request.parameters)
        
        elif request.visualization_type == "optimizer_performance":
            from visualization.optimizer_analysis import generate_performance_visualization
            result_data = generate_performance_visualization(request.data_id, request.parameters)
        
        elif request.visualization_type == "convergence_comparison":
            from visualization.dynamic_optimization_viz import generate_convergence_visualization
            result_data = generate_convergence_visualization(request.data_id, request.parameters)
        
        # Return result
        return {
            "visualization_type": request.visualization_type,
            "data": result_data,
            "parameters": request.parameters or {}
        }
    
    except ImportError as e:
        logger.error(f"Visualization module import error: {str(e)}")
        raise HTTPException(
            status_code=501, 
            detail=f"Visualization module not available: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating visualization: {str(e)}"
        )

@router.get("/optimizer-comparison", response_model=Dict[str, Any])
async def get_optimizer_comparison(
    benchmark_id: Optional[str] = None,
    function_name: Optional[str] = Query(None, description="Filter by function name"),
    dimension: Optional[int] = Query(None, description="Filter by dimension"),
    optimizers: Optional[List[str]] = Query(None, description="Filter by optimizers")
):
    """
    Get optimizer comparison visualization data
    """
    try:
        # This is a simplified implementation - in a real application, you would:
        # 1. Retrieve benchmark results from a database
        # 2. Filter based on parameters
        # 3. Process the data for visualization
        
        # For this example, we'll generate some sample data
        optimizer_names = optimizers or ["DE", "PSO", "ES", "GWO", "ACO"]
        function_names = [function_name] if function_name else ["sphere", "rosenbrock", "rastrigin"]
        dim = dimension or 10
        
        # Generate sample performance data
        performance_data = {}
        for func in function_names:
            performance_data[func] = {}
            for opt in optimizer_names:
                # Generate random performance metrics with some bias per optimizer
                base_score = {
                    "DE": 1e-8,
                    "PSO": 1e-6,
                    "ES": 1e-7,
                    "GWO": 1e-5,
                    "ACO": 1e-4
                }.get(opt, 1e-5)
                
                # Add randomness
                multiplier = np.random.uniform(0.1, 10.0)
                score = base_score * multiplier
                
                # Add some variability by function
                if func == "rosenbrock":
                    score *= 100
                elif func == "rastrigin":
                    score *= 10
                
                performance_data[func][opt] = {
                    "mean_score": score,
                    "std_score": score * 0.2,
                    "mean_time": np.random.uniform(0.5, 5.0),
                    "success_rate": np.random.uniform(0.7, 1.0)
                }
        
        # Determine best optimizer per function
        best_optimizers = {}
        for func in function_names:
            best_opt = None
            best_score = float('inf')
            
            for opt in optimizer_names:
                score = performance_data[func][opt]["mean_score"]
                if score < best_score:
                    best_score = score
                    best_opt = opt
            
            best_optimizers[func] = best_opt
        
        # Return visualization data
        return {
            "performance_data": performance_data,
            "best_optimizers": best_optimizers,
            "dimension": dim,
            "functions": function_names,
            "optimizers": optimizer_names
        }
    
    except Exception as e:
        logger.error(f"Error generating optimizer comparison: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating optimizer comparison: {str(e)}"
        )
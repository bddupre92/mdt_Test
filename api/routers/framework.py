"""
Framework Functions Router

This module provides API endpoints for running framework functions
and visualizing results.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import importlib
import inspect
import sys
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Models
class FunctionInfo(BaseModel):
    """Information about a function."""
    name: str
    doc: Optional[str] = None
    module: str
    parameters: Dict[str, Any]

class FunctionRequest(BaseModel):
    """Request to run a function."""
    module: str
    function: str
    parameters: Dict[str, Any]
    save_result: bool = True

class FunctionResponse(BaseModel):
    """Response from running a function."""
    success: bool
    message: str
    result_type: Optional[str] = None
    result: Optional[Any] = None
    output: Optional[str] = None
    visualization: Optional[str] = None
    saved_id: Optional[str] = None

class SavedRun(BaseModel):
    """A saved function run."""
    id: str
    function: str
    module: str
    timestamp: str
    parameters: Dict[str, str]

# Helper functions
def import_framework_modules():
    """Import and return available framework modules."""
    modules = {}
    
    module_names = [
        "core", "main", "visualization", "optimizers", 
        "migraine", "framework", "meta_optimizer"
    ]
    
    for name in module_names:
        try:
            if name == "main":
                # Try as a module first
                try:
                    module = importlib.import_module(name)
                    modules[name] = module
                except ImportError:
                    # Try loading from file
                    file_path = os.path.abspath("main.py")
                    if os.path.exists(file_path):
                        import imp
                        module = imp.load_source("main", file_path)
                        modules[name] = module
            else:
                module = importlib.import_module(name)
                modules[name] = module
        except ImportError:
            logger.warning(f"Module {name} not found")
        except Exception as e:
            logger.error(f"Error importing {name}: {str(e)}")
    
    return modules

def get_function_info(module, function_name):
    """Get information about a function."""
    try:
        func = getattr(module, function_name)
        signature = inspect.signature(func)
        
        parameters = {}
        for name, param in signature.parameters.items():
            if param.default is not inspect.Parameter.empty:
                parameters[name] = param.default
            else:
                parameters[name] = None
                
        return FunctionInfo(
            name=function_name,
            doc=func.__doc__,
            module=module.__name__,
            parameters=parameters
        )
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Function {function_name} not found in module {module.__name__}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting function info: {str(e)}")

def save_function_result(module_name, function_name, parameters, result):
    """Save a function result to disk."""
    try:
        # Create results directory if it doesn't exist
        results_dir = Path("framework_runs")
        results_dir.mkdir(exist_ok=True)
        
        # Generate ID and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{module_name}_{function_name}_{timestamp}"
        
        # Create filename
        filename = f"{run_id}.pkl"
        filepath = results_dir / filename
        
        # Save metadata
        metadata = {
            "id": run_id,
            "function": function_name,
            "module": module_name,
            "timestamp": timestamp,
            "parameters": {k: str(v) for k, v in parameters.items()},
            "result_file": str(filepath)
        }
        
        # Save result using pickle
        with open(filepath, "wb") as f:
            pickle.dump(result, f)
            
        # Save metadata
        meta_filepath = results_dir / f"{run_id}.json"
        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return run_id
    except Exception as e:
        logger.error(f"Error saving result: {str(e)}")
        return None

def render_result(result):
    """Render a result as JSON serializable data."""
    result_type = type(result).__name__
    visualization = None
    
    # Handle matplotlib figures
    if isinstance(result, plt.Figure):
        buf = io.BytesIO()
        result.savefig(buf, format='png')
        buf.seek(0)
        visualization = base64.b64encode(buf.read()).decode('utf-8')
        result = None
    
    # Handle pandas DataFrame
    elif isinstance(result, pd.DataFrame):
        result = result.to_dict(orient='records')
    
    # Handle other iterables
    elif hasattr(result, "__iter__") and not isinstance(result, (str, dict, list)):
        try:
            result = list(result)
        except:
            result = str(result)
    
    return result_type, result, visualization

# Endpoints
@router.get("/modules", response_model=List[str])
async def get_modules():
    """Get available framework modules."""
    modules = import_framework_modules()
    return list(modules.keys())

@router.get("/functions/{module_name}", response_model=List[str])
async def get_functions(module_name: str):
    """Get available functions in a module."""
    modules = import_framework_modules()
    
    if module_name not in modules:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    module = modules[module_name]
    
    functions = [
        name for name, _ in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_")
    ]
    
    return functions

@router.get("/function-info/{module_name}/{function_name}", response_model=FunctionInfo)
async def get_function_details(module_name: str, function_name: str):
    """Get details about a specific function."""
    modules = import_framework_modules()
    
    if module_name not in modules:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    module = modules[module_name]
    return get_function_info(module, function_name)

@router.post("/run", response_model=FunctionResponse)
async def run_function(request: FunctionRequest, background_tasks: BackgroundTasks):
    """Run a function from a framework module."""
    modules = import_framework_modules()
    
    if request.module not in modules:
        raise HTTPException(status_code=404, detail=f"Module {request.module} not found")
    
    module = modules[request.module]
    
    try:
        func = getattr(module, request.function)
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Function {request.function} not found in module {request.module}")
    
    try:
        # Capture stdout
        stdout_capture = io.StringIO()
        
        # Run function
        import contextlib
        with contextlib.redirect_stdout(stdout_capture):
            result = func(**request.parameters)
        
        # Get output
        output = stdout_capture.getvalue()
        
        # Render result
        result_type, result_data, visualization = render_result(result)
        
        # Save result if requested
        saved_id = None
        if request.save_result and result is not None:
            saved_id = save_function_result(request.module, request.function, request.parameters, result)
        
        return FunctionResponse(
            success=True,
            message=f"Successfully ran {request.function}",
            result_type=result_type,
            result=result_data,
            output=output,
            visualization=visualization,
            saved_id=saved_id
        )
    
    except Exception as e:
        logger.error(f"Error running function: {str(e)}", exc_info=True)
        return FunctionResponse(
            success=False,
            message=f"Error running function: {str(e)}",
            output=stdout_capture.getvalue() if 'stdout_capture' in locals() else None
        )

@router.get("/saved-runs", response_model=List[SavedRun])
async def get_saved_runs():
    """Get saved function runs."""
    results_dir = Path("framework_runs")
    
    if not results_dir.exists():
        return []
    
    # Find all json metadata files
    meta_files = list(results_dir.glob("*.json"))
    
    if not meta_files:
        return []
    
    # Load metadata
    runs = []
    for meta_file in meta_files:
        try:
            with open(meta_file, "r") as f:
                metadata = json.load(f)
                runs.append(SavedRun(
                    id=metadata.get("id", meta_file.stem),
                    function=metadata.get("function", "unknown"),
                    module=metadata.get("module", "unknown"),
                    timestamp=metadata.get("timestamp", "unknown"),
                    parameters=metadata.get("parameters", {})
                ))
        except Exception as e:
            logger.error(f"Error loading metadata from {meta_file}: {str(e)}")
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x.timestamp, reverse=True)
    return runs

@router.get("/saved-run/{run_id}", response_model=FunctionResponse)
async def get_saved_run(run_id: str):
    """Get a saved function run."""
    results_dir = Path("framework_runs")
    
    # Look for metadata file
    meta_file = results_dir / f"{run_id}.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    try:
        # Load metadata
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        
        # Load result
        result_file = metadata.get("result_file")
        if not result_file or not Path(result_file).exists():
            raise HTTPException(status_code=404, detail=f"Result file for run {run_id} not found")
        
        with open(result_file, "rb") as f:
            result = pickle.load(f)
        
        # Render result
        result_type, result_data, visualization = render_result(result)
        
        return FunctionResponse(
            success=True,
            message=f"Successfully loaded run {run_id}",
            result_type=result_type,
            result=result_data,
            visualization=visualization,
            saved_id=run_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading run {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading run: {str(e)}") 
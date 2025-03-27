"""
Framework Runner Component

This module provides an interactive page for running framework functions
and visualizing results directly in the dashboard.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import json
import pickle
from typing import Dict, List, Any
import sys
import os
import importlib
import inspect
from pathlib import Path
from datetime import datetime

# Import framework modules safely
def import_framework_modules(additional_modules=None):
    """Import framework modules safely, with fallbacks for missing imports.
    
    Args:
        additional_modules: Optional list of module names to import
    """
    modules = {}
    try:
        # Try to import main framework modules
        # Use importlib to dynamically import modules to avoid errors if they don't exist
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
        
        # Import core modules
        try:
            core_module = importlib.import_module("core")
            modules["core"] = core_module
        except ImportError:
            st.warning("Core module not found. Some functionality may be limited.")
        
        # Try to import main module (contains most key functions)
        try:
            # First check if it's a package
            try:
                main_module = importlib.import_module("main")
                modules["main"] = main_module
            except ImportError:
                # If not a package, load it directly as a module
                import imp
                file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../main.py"))
                if os.path.exists(file_path):
                    main_module = imp.load_source("main", file_path)
                    modules["main"] = main_module
                    st.success("Loaded main.py module successfully")
                else:
                    st.warning("Main module not found. Some main functions may be unavailable.")
        except Exception as e:
            st.warning(f"Could not import main module: {str(e)}")
        
        # Try to import visualization modules
        try:
            viz_module = importlib.import_module("visualization")
            modules["visualization"] = viz_module
        except ImportError:
            st.warning("Visualization module not found. Some functionality may be limited.")
        
        # Try to import optimizer modules
        try:
            optimizer_module = importlib.import_module("optimizers")
            modules["optimizers"] = optimizer_module
        except ImportError:
            st.warning("Optimizers module not found. Some functionality may be limited.")

        # Try to import migraine modules if they exist
        try:
            migraine_module = importlib.import_module("migraine")
            modules["migraine"] = migraine_module
        except ImportError:
            # This is optional, so we don't show a warning
            pass

        # Try to import framework modules if they exist
        try:
            framework_module = importlib.import_module("framework")
            modules["framework"] = framework_module
        except ImportError:
            # This is optional, so we don't show a warning
            pass
        
        # Import additional modules if requested
        if additional_modules:
            for module_name in additional_modules:
                try:
                    module = importlib.import_module(module_name)
                    modules[module_name] = module
                    st.success(f"Loaded module {module_name} successfully")
                except Exception as e:
                    st.warning(f"Failed to import module {module_name}: {str(e)}")
            
    except Exception as e:
        st.error(f"Error importing framework modules: {str(e)}")
        
    return modules

def get_framework_functions(modules):
    """Get available framework functions that can be run."""
    functions = {}
    
    # Define the list of allowed functions by module and name
    # Format: {module_name: [list of function names]}
    allowed_functions = {
        "core": ["run_benchmark", "train_model", "evaluate_model", "preprocess_data", "load_dataset"],
        "main": ["run_experiment", "generate_report", "compare_models", "analyze_results"],
        "visualization": ["plot_results", "generate_interactive_report", "plot_confusion_matrix", "plot_roc_curve"],
        "optimizers": ["optimize_hyperparameters", "find_best_model"],
        "migraine": ["analyze_migraine_data", "preprocess_migraine_dataset", "extract_migraine_features"],
        "framework": ["build_pipeline", "create_ensemble", "create_moe_model"]
    }
    
    # Look for allowed functions in each module
    for module_name, allowed_func_names in allowed_functions.items():
        if module_name in modules:
            module_funcs = inspect.getmembers(modules[module_name], inspect.isfunction)
            # Filter for allowed functions only
            filtered_funcs = [func for name, func in module_funcs 
                             if name in allowed_func_names]
            
            if filtered_funcs:  # Only add category if there are functions
                functions[f"{module_name.capitalize()} Functions"] = filtered_funcs
    
    # Add functions from any other explicitly added modules (from settings)
    for module_name, module in modules.items():
        if module_name not in allowed_functions.keys():
            # For custom modules, we'll be more selective
            # Only include functions that have proper docstrings
            module_funcs = inspect.getmembers(module, inspect.isfunction)
            filtered_funcs = [func for name, func in module_funcs 
                             if not name.startswith("_") and func.__doc__]
            
            if filtered_funcs:  # Only add category if there are functions
                functions[f"{module_name.capitalize()} Functions"] = filtered_funcs
    
    # If no functions were found, add some dummy functions for demonstration
    if not any(functions.values()):
        # Create a dummy module with example functions
        class DummyModule:
            @staticmethod
            def example_train_model(dataset_path, model_type="moe", epochs=10, batch_size=32):
                """Example function to train a model on the specified dataset.
                
                Args:
                    dataset_path: Path to the dataset file
                    model_type: Type of model to train (moe, ensemble, single)
                    epochs: Number of training epochs
                    batch_size: Batch size for training
                
                Returns:
                    A trained model object
                """
                return {"status": "success", "message": "Model trained successfully (example)"}
            
            @staticmethod
            def example_generate_report(results_path, output_format="html", include_plots=True):
                """Example function to generate a report from results.
                
                Args:
                    results_path: Path to the results file
                    output_format: Format for the report (html, pdf, json)
                    include_plots: Whether to include plots in the report
                
                Returns:
                    Path to the generated report
                """
                return {"status": "success", "report_path": "example_report.html"}
        
        # Add the example functions
        dummy = DummyModule()
        functions["Example Functions"] = [
            dummy.example_train_model,
            dummy.example_generate_report
        ]
    
    return functions

def get_function_parameters(func):
    """Get parameters for a function and their default values."""
    signature = inspect.signature(func)
    parameters = {}
    
    for name, param in signature.parameters.items():
        if param.default is not inspect.Parameter.empty:
            parameters[name] = param.default
        else:
            parameters[name] = None
            
    return parameters

def render_function_params(func, params=None):
    """Render UI controls for function parameters."""
    if params is None:
        params = get_function_parameters(func)
    
    user_params = {}
    
    for name, default_value in params.items():
        # Skip self parameter for class methods
        if name == "self":
            continue
            
        if isinstance(default_value, bool):
            user_params[name] = st.checkbox(f"{name}", value=default_value)
        elif isinstance(default_value, int):
            user_params[name] = st.number_input(f"{name}", value=default_value)
        elif isinstance(default_value, float):
            user_params[name] = st.number_input(f"{name}", value=default_value, format="%.5f")
        elif isinstance(default_value, str):
            user_params[name] = st.text_input(f"{name}", value=default_value)
        elif default_value is None:
            # Handle parameters with no default values
            value_type = st.selectbox(f"Type for {name}", ["String", "Number", "Boolean"])
            if value_type == "String":
                user_params[name] = st.text_input(f"{name}")
            elif value_type == "Number":
                user_params[name] = st.number_input(f"{name}", value=0.0, format="%.5f")
            else:
                user_params[name] = st.checkbox(f"{name}")
    
    return user_params

def save_result(result, function_name, params):
    """Save result to a file."""
    # Create results directory if it doesn't exist
    results_dir = Path("framework_runs")
    results_dir.mkdir(exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename based on function and timestamp
    filename = f"{function_name}_{timestamp}.pkl"
    filepath = results_dir / filename
    
    # Save metadata json
    metadata = {
        "function": function_name,
        "timestamp": timestamp,
        "parameters": {k: str(v) for k, v in params.items()},
        "result_file": str(filepath)
    }
    
    # Save result using pickle
    try:
        with open(filepath, "wb") as f:
            pickle.dump(result, f)
            
        # Save metadata
        meta_filepath = results_dir / f"{function_name}_{timestamp}.json"
        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return str(filepath)
    except Exception as e:
        st.error(f"Error saving result: {str(e)}")
        return None

def load_saved_runs():
    """Load metadata of saved runs."""
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
                runs.append(metadata)
        except:
            pass
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return runs

def load_result(result_file):
    """Load a saved result."""
    try:
        with open(result_file, "rb") as f:
            result = pickle.load(f)
        return result
    except Exception as e:
        st.error(f"Error loading result: {str(e)}")
        return None

def display_result(result):
    """Display a result based on its type."""
    if result is None:
        st.info("No result to display.")
    elif isinstance(result, plt.Figure):
        st.pyplot(result)
    elif isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif hasattr(result, "__iter__") and not isinstance(result, str):
        # Try to convert to DataFrame if iterable
        try:
            result_df = pd.DataFrame(result)
            st.dataframe(result_df)
        except:
            st.json(result)
    else:
        # Try to display as text/json
        try:
            st.json(result)
        except:
            st.text(str(result))

def render_framework_runner():
    """Render the framework runner page."""
    st.header("Framework Runner")
    st.markdown("""
    This page allows you to run various framework functions and visualize results directly in the dashboard.
    Select a function, configure parameters, and run it to see results.
    """)
    
    # Create tabs for running functions, viewing saved runs, and advanced settings
    run_tab, saved_tab, settings_tab = st.tabs(["Run Framework Functions", "View Saved Runs", "Advanced Settings"])
    
    # Store additional modules in session state
    if 'additional_modules' not in st.session_state:
        st.session_state.additional_modules = []
    
    # Advanced settings tab
    with settings_tab:
        st.subheader("Module Loading Settings")
        
        # Add a module to load
        new_module = st.text_input("Add module to load (e.g., 'meta', 'migraine_data', etc.)")
        if st.button("Add Module") and new_module:
            if new_module not in st.session_state.additional_modules:
                st.session_state.additional_modules.append(new_module)
                st.success(f"Added {new_module} to modules list")
        
        # Show currently loaded modules
        if st.session_state.additional_modules:
            st.markdown("### Currently added modules:")
            for i, module in enumerate(st.session_state.additional_modules):
                cols = st.columns([5, 1])
                cols[0].write(f"- {module}")
                if cols[1].button("Remove", key=f"remove_{i}"):
                    st.session_state.additional_modules.remove(module)
                    st.experimental_rerun()
        
        # Reload all modules button
        if st.button("Reload All Modules"):
            st.session_state.modules = import_framework_modules(st.session_state.additional_modules)
            st.success("Modules reloaded successfully")
    
    # Import framework modules (if not already loaded)
    if 'modules' not in st.session_state:
        st.session_state.modules = import_framework_modules(st.session_state.additional_modules)
    
    # Run functions tab
    with run_tab:
        modules = st.session_state.modules
        
        if not modules:
            st.error("Failed to import any framework modules. Please check your installation.")
            return
        
        # Get available functions
        functions_by_category = get_framework_functions(modules)
        
        if not functions_by_category:
            st.error("No runnable functions found in the framework.")
            return
        
        # Create tabs for different function categories
        category_tabs = st.tabs(list(functions_by_category.keys()))
        
        for i, (category, functions) in enumerate(functions_by_category.items()):
            with category_tabs[i]:
                st.subheader(category)
                
                if not functions:
                    st.info(f"No functions found in {category}")
                    continue
                
                # Function selection
                function_names = [func.__name__ for func in functions]
                selected_func_name = st.selectbox(
                    "Select function to run", 
                    function_names,
                    key=f"func_select_{category}"
                )
                
                # Find the selected function
                selected_func = next((func for func in functions if func.__name__ == selected_func_name), None)
                
                if selected_func:
                    # Display function documentation
                    if selected_func.__doc__:
                        st.markdown(f"**Description:**")
                        st.markdown(selected_func.__doc__)
                    
                    # Function parameters
                    st.markdown("### Parameters")
                    params = get_function_parameters(selected_func)
                    user_params = render_function_params(selected_func, params)
                    
                    # Save option
                    save_result_option = st.checkbox("Save results for later reference", value=True)
                    
                    # Run button
                    if st.button("Run Function", key=f"run_{category}_{selected_func_name}"):
                        with st.spinner(f"Running {selected_func_name}..."):
                            try:
                                # Capture stdout to display any printed output
                                from io import StringIO
                                import contextlib
                                
                                output = StringIO()
                                with contextlib.redirect_stdout(output):
                                    # Run the function with user parameters
                                    result = selected_func(**user_params)
                                
                                # Display any printed output
                                if output.getvalue():
                                    st.subheader("Output")
                                    st.text(output.getvalue())
                                
                                # Display results based on type
                                st.subheader("Results")
                                display_result(result)
                                
                                # Save result if requested
                                if save_result_option and result is not None:
                                    saved_path = save_result(result, selected_func_name, user_params)
                                    if saved_path:
                                        st.success(f"Result saved successfully. Access it in the 'View Saved Runs' tab.")
                                    
                            except Exception as e:
                                st.error(f"Error running function: {str(e)}")
                                st.exception(e)
    
    # Saved runs tab
    with saved_tab:
        st.subheader("Previously Saved Framework Runs")
        
        # Load saved runs
        runs = load_saved_runs()
        
        if not runs:
            st.info("No saved runs found. Run functions with the 'Save results' option enabled to save results.")
            return
        
        # Create a dataframe of saved runs
        runs_df = pd.DataFrame([
            {
                "Function": run["function"],
                "Timestamp": run["timestamp"],
                "Parameters": ", ".join([f"{k}={v}" for k, v in run["parameters"].items()])
            }
            for run in runs
        ])
        
        st.dataframe(runs_df)
        
        # Select a run to view
        selected_run_idx = st.selectbox(
            "Select a run to view", 
            range(len(runs)),
            format_func=lambda i: f"{runs[i]['function']} ({runs[i]['timestamp']})"
        )
        
        if selected_run_idx is not None:
            selected_run = runs[selected_run_idx]
            
            # Display run metadata
            st.markdown("### Run Details")
            st.markdown(f"**Function:** {selected_run['function']}")
            st.markdown(f"**Timestamp:** {selected_run['timestamp']}")
            st.markdown("**Parameters:**")
            for k, v in selected_run["parameters"].items():
                st.markdown(f"- {k}: {v}")
            
            # Load and display result
            st.markdown("### Result")
            result = load_result(selected_run["result_file"])
            display_result(result) 
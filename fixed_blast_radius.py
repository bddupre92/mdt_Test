"""
Blast Radius Analysis for migrineDT using Codegen library

This script handles symbolic link issues while leveraging Codegen's
powerful traversal capabilities to visualize function dependencies.
"""

import os
import sys
import networkx as nx
from codegen import Codebase
import tempfile
import shutil
import subprocess
from pathlib import Path

# Initialize paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "blast_radius_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Problematic paths with symbolic link loops
PROBLEM_PATHS = [
    "meta_optimizer/explainability/base_explainer.py",
    "meta_optimizer/explainability/optimizer_explainer.py",
    "meta_optimizer/explainability/shap_explainer.py"
]

# Create a temp directory for our clean copy
TEMP_DIR = tempfile.mkdtemp(prefix="blast_radius_")
print(f"Created temporary directory: {TEMP_DIR}")

# Configuration
IGNORE_EXTERNAL_MODULE_CALLS = True
IGNORE_CLASS_CALLS = False
MAX_DEPTH = 8

# Color palette for visualization
COLOR_PALETTE = {
    "StartFunction": "#9cdcfe",     # Light blue - Start Function
    "PyFunction": "#a277ff",        # Soft purple/periwinkle - PyFunction
    "PyClass": "#ffca85",           # Warm peach/orange - PyClass
    "ExternalModule": "#f694ff",    # Bright magenta/pink - ExternalModule
    "Unknown": "#cccccc"            # Gray - Unknown
}

# Key components to analyze - using fully qualified module paths to avoid ambiguity
TARGET_COMPONENTS = [
    # CLI directory components
    ("cli.main.main", "function"),
    ("cli.commands.run_optimization", "function"),
    ("cli.commands.run_explainability_analysis", "function"),
    
    # Core directory components
    ("core.optimization.run_optimization", "function"),
    ("core.meta_learning.MetaOptimizer", "class"),
    ("core.evaluation.evaluate_model", "function"),
    
    # Explainability directory components
    ("explainability.model_explainer.ShapExplainer", "class"),
    ("explainability.optimizer_explainer.OptimizerExplainer", "class"),
    ("explainability.explainer_factory.ExplainerFactory", "class"),
    
    # Migraine directory components
    ("migraine.prediction.MigrainePredictor", "class"),
    ("migraine.explainability.explain_model", "function"),
]

def create_clean_copy():
    """Create a clean copy of the codebase without problematic symbolic links
    and initialize a Git repository for Codegen compatibility
    """
    print(f"Creating clean copy of codebase in {TEMP_DIR}...")
    
    # Use rsync to copy files excluding problematic paths
    exclude_args = []
    for path in PROBLEM_PATHS:
        exclude_args.extend(["--exclude", path])
    
    # Also exclude common directories that might cause issues
    exclude_args.extend([
        "--exclude", ".git",        # Don't copy the original .git folder
        "--exclude", "__pycache__",
        "--exclude", ".venv_minimal",
        "--exclude", "venv",
        "--exclude", "env",
        "--exclude", "*.egg-info"
    ])
    
    # Run rsync command
    rsync_command = ["rsync", "-a"] + exclude_args + [f"{PROJECT_ROOT}/", TEMP_DIR]
    subprocess.run(rsync_command)
    
    # Initialize a Git repository in the temporary directory
    # This is required for Codegen to work properly
    print("Initializing Git repository in temporary directory...")
    try:
        # Change to the temporary directory
        os.chdir(TEMP_DIR)
        
        # Initialize a new Git repository
        subprocess.run(["git", "init"], check=True)
        
        # Add all files to the repository
        subprocess.run(["git", "add", "."], check=True)
        
        # Create an initial commit
        subprocess.run(["git", "config", "user.email", "blast_radius@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Blast Radius Tool"], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit for blast radius analysis"], check=True)
        
        # Change back to the original directory
        os.chdir(PROJECT_ROOT)
        
        print("Git repository initialized successfully")
    except Exception as e:
        print(f"Warning: Error initializing Git repository: {str(e)}")
        # Change back to the original directory in case of error
        os.chdir(PROJECT_ROOT)
    
    return TEMP_DIR

def generate_edge_meta(call):
    """Generate metadata for call graph edges"""
    try:
        return {
            "name": call.name,
            "file_path": getattr(call, "filepath", "unknown"),
            "start_point": getattr(call, "start_point", (0, 0)),
            "end_point": getattr(call, "end_point", (0, 0)),
            "symbol_name": "FunctionCall"
        }
    except Exception as e:
        print(f"Error generating edge metadata: {str(e)}")
        return {"name": "unknown", "error": str(e)}

def create_downstream_call_trace(src_func, G, depth=0, visited=None):
    """Creates call graph by recursively traversing function calls"""
    if visited is None:
        visited = set()
        
    # Track visited functions to prevent cycles
    func_id = getattr(src_func, "id", str(id(src_func)))
    if func_id in visited:
        return
    visited.add(func_id)
    
    # Prevent excessive recursion
    if MAX_DEPTH <= depth:
        return
    
    # External modules are not functions
    if hasattr(src_func, "__class__") and src_func.__class__.__name__ == "ExternalModule":
        return

    try:
        # Process each function call
        for call in getattr(src_func, "function_calls", []):
            try:
                # Skip self-recursive calls
                if call.name == getattr(src_func, "name", None):
                    continue
                    
                # Get called function definition
                func = getattr(call, "function_definition", None)
                if not func:
                    continue
                
                # Apply configured filters
                if hasattr(func, "__class__"):
                    func_type = func.__class__.__name__
                    if func_type == "ExternalModule" and IGNORE_EXTERNAL_MODULE_CALLS:
                        continue
                    if func_type == "Class" and IGNORE_CLASS_CALLS:
                        continue

                # Generate display name (include class for methods)
                if hasattr(func, "__class__") and func.__class__.__name__ in ["Class", "ExternalModule"]:
                    func_name = func.name
                elif hasattr(func, "is_method") and func.is_method and hasattr(func, "parent_class"):
                    func_name = f"{func.parent_class.name}.{func.name}"
                else:
                    func_name = getattr(func, "name", str(func))

                # Add node and edge with metadata
                color = COLOR_PALETTE.get(
                    getattr(func, "__class__", None).__name__ if hasattr(func, "__class__") else "Unknown", 
                    COLOR_PALETTE["Unknown"]
                )
                
                G.add_node(func, name=func_name, color=color)
                G.add_edge(src_func, func, **generate_edge_meta(call))

                # Recurse for regular functions
                if hasattr(func, "__class__") and func.__class__.__name__ == "Function":
                    create_downstream_call_trace(func, G, depth + 1, visited)
                    
            except Exception as e:
                print(f"Error processing function call: {str(e)}")
                
    except Exception as e:
        print(f"Error traversing function: {str(e)}")

def create_upstream_call_trace(target_func, G, depth=0, visited=None):
    """Creates call graph of functions that call the target function"""
    if visited is None:
        visited = set()
        
    # Track visited functions to prevent cycles
    func_id = getattr(target_func, "id", str(id(target_func)))
    if func_id in visited:
        return
    visited.add(func_id)
    
    # Prevent excessive recursion
    if MAX_DEPTH <= depth:
        return
    
    try:
        # Find all functions that call this function
        callers = getattr(target_func, "callers", [])
        if callers:
            print(f"Found {len(callers)} callers for {target_func.name}")
            
        for caller in callers:
            try:
                # Skip self-recursive calls
                if caller.name == getattr(target_func, "name", None):
                    continue
                
                # Generate display name (include class for methods)
                if hasattr(caller, "__class__") and caller.__class__.__name__ in ["Class", "ExternalModule"]:
                    caller_name = caller.name
                elif hasattr(caller, "is_method") and caller.is_method and hasattr(caller, "parent_class"):
                    caller_name = f"{caller.parent_class.name}.{caller.name}"
                else:
                    caller_name = getattr(caller, "name", str(caller))
                    
                # Add node and edge
                color = COLOR_PALETTE.get(
                    getattr(caller, "__class__", None).__name__ if hasattr(caller, "__class__") else "Unknown", 
                    COLOR_PALETTE["Unknown"]
                )
                
                G.add_node(caller, name=caller_name, color=color)
                G.add_edge(caller, target_func, name=f"calls {target_func.name}")
                
                # Recurse to find callers of this caller
                if hasattr(caller, "__class__") and caller.__class__.__name__ == "Function":
                    create_upstream_call_trace(caller, G, depth + 1, visited)
                    
            except Exception as e:
                print(f"Error processing caller: {str(e)}")
    except Exception as e:
        print(f"Error finding callers: {str(e)}")

def graph_class_methods(codebase, target_class, G=None):
    """Creates a graph visualization of all methods in a class and their call relationships
    
    Args:
        codebase: The Codegen codebase object
        target_class: The class whose methods will be graphed
        G: Optional existing graph to add to
        
    Returns:
        The graph and a boolean indicating success
    """
    if G is None:
        G = nx.DiGraph()
    
    try:
        # Add the class itself as root node with a distinctive color
        G.add_node(target_class, name=target_class.name, color=COLOR_PALETTE.get("PyClass", COLOR_PALETTE["Unknown"]))
        visited = set([target_class])
        
        # Get all methods of the class
        methods = []
        try:
            # This might vary based on the exact Codegen API version
            if hasattr(target_class, 'methods'):
                methods = target_class.methods
            elif hasattr(target_class, 'get_methods'):
                methods = target_class.get_methods()
            else:
                print(f"Could not find methods for class {target_class.name}")
        except Exception as e:
            print(f"Error getting methods for class {target_class.name}: {str(e)}")
        
        if not methods:
            print(f"No methods found for class {target_class.name}")
            return G, False
            
        # Add all methods as child nodes connected to the class
        for method in methods:
            try:
                method_name = f"{target_class.name}.{method.name}"
                G.add_node(method, name=method_name, color=COLOR_PALETTE["StartFunction"])
                visited.add(method)
                G.add_edge(target_class, method)

                # Recursively trace all downstream calls for this method
                create_downstream_call_trace(method, G)
                create_upstream_call_trace(method, G)
            except Exception as e:
                print(f"Error processing method {method.name}: {str(e)}")
        
        return G, True
    except Exception as e:
        print(f"Error graphing class methods: {str(e)}")
        return G, False

def attempt_alternative_component_lookup(codebase, component_name, component_type):
    """Attempts to find a component using alternative lookups based on project structure
    
    This function uses knowledge of the project structure to try different module paths
    when the standard lookup fails. It helps resolve ambiguity in component names.
    
    Args:
        codebase: The Codegen codebase object
        component_name: Name of the component to find
        component_type: Type of the component ('function' or 'class')
        
    Returns:
        The component if found, None otherwise
    """
    parts = component_name.split('.')
    base_name = parts[-1]
    
    # Map of known module patterns based on project structure
    module_patterns = {
        # Core functions are likely in core modules
        'run_optimization': ['core.optimization'],
        'evaluate_model': ['core.evaluation'],
        'MetaOptimizer': ['core.meta_learning'],
        
        # Explainer classes have specific locations
        'ShapExplainer': ['explainability.model_explainer'],
        'OptimizerExplainer': ['explainability.optimizer_explainer'],
        'ExplainerFactory': ['explainability.explainer_factory'],
        
        # Migraine-specific components
        'MigrainePredictor': ['migraine.prediction'],
        'explain_model': ['migraine.explainability'],
        
        # CLI components
        'run_explainability_analysis': ['cli.commands'],
        'main': ['cli.main']
    }
    
    # If the base name is in our known patterns, try those paths first
    if base_name in module_patterns:
        potential_modules = module_patterns[base_name]
        
        for module_path in potential_modules:
            try:
                if component_type == 'function':
                    component = codebase.get_function(base_name, module_path)
                    if component:
                        print(f"Found {base_name} as function in {module_path}")
                        return component
                elif component_type == 'class':
                    component = codebase.get_class(base_name, module_path)
                    if component:
                        print(f"Found {base_name} as class in {module_path}")
                        return component
            except Exception as e:
                print(f"Alternative lookup failed for {base_name} in {module_path}: {str(e)}")
    
    # Try to infer module path from common patterns
    if component_type == 'class':
        # Class naming patterns to try
        if base_name.endswith('Explainer'):
            try:
                component = codebase.get_class(base_name, 'explainability')
                if component:
                    return component
            except Exception:
                pass
            
            # Try in subdirectories of explainability
            for subdir in ['model_explainer', 'optimizer_explainer', 'drift_explainer']:
                try:
                    component = codebase.get_class(base_name, f'explainability.{subdir}')
                    if component:
                        return component
                except Exception:
                    pass
                    
        elif base_name.endswith('Predictor'):
            try:
                component = codebase.get_class(base_name, 'migraine.prediction')
                if component:
                    return component
            except Exception:
                pass
    
    # If all else fails, try a brute force approach
    common_modules = [
        'core', 'explainability', 'migraine', 'cli',
        'core.optimization', 'core.evaluation', 'core.meta_learning',
        'explainability.model_explainer', 'explainability.optimizer_explainer',
        'migraine.prediction', 'migraine.explainability',
        'cli.commands', 'cli.main'
    ]
    
    for module in common_modules:
        try:
            if component_type == 'function':
                component = codebase.get_function(base_name, module)
                if component:
                    print(f"Found {base_name} as function in {module}")
                    return component
            elif component_type == 'class':
                component = codebase.get_class(base_name, module)
                if component:
                    print(f"Found {base_name} as class in {module}")
                    return component
        except Exception:
            pass
            
    return None

def get_disambiguation_paths(base_name, context_path=None):
    """Get list of potential module paths for a component based on project structure
    
    This function uses the project structure knowledge from MEMORIEs to provide
    a prioritized list of possible module paths to try for a given component name.
    
    Args:
        base_name: The base component name (without module path)
        context_path: The original context path from the component name
        
    Returns:
        List of possible module paths to try, in priority order
    """
    # Map of known module patterns based on MEMORIEs and project structure
    component_paths = {
        # Explainer classes have specific known locations (from explainability MEMORIEs)
        'ShapExplainer': ['explainability.model_explainer', 'model_explainer', 'explainability'],
        'LimeExplainer': ['explainability.model_explainer', 'model_explainer', 'explainability'],
        'FeatureImportanceExplainer': ['explainability.model_explainer', 'model_explainer', 'explainability'],
        'BaseExplainer': ['explainability.model_explainer', 'explainability', 'model_explainer'],
        'OptimizerExplainer': ['explainability.optimizer_explainer', 'optimizer_explainer', 'explainability'],
        'ExplainerFactory': ['explainability.explainer_factory', 'explainability', 'explainer_factory'],
        
        # Migraine-specific components (from project structure MEMORY)
        'MigrainePredictor': ['migraine.prediction', 'migraine', 'prediction'],
        'explain_model': ['migraine.explainability', 'explainability', 'migraine'],
        
        # Core components (from project structure MEMORY)
        'run_optimization': ['core.optimization', 'core', 'optimization'],
        'evaluate_model': ['core.evaluation', 'core', 'evaluation'],
        'MetaOptimizer': ['core.meta_learning', 'meta_learning', 'core'],
        
        # CLI components (from project structure MEMORY)
        'run_explainability_analysis': ['cli.commands', 'commands', 'cli'],
        'main': ['cli.main', 'main', 'cli'],
    }
    
    # Start with known paths for this specific component if available
    paths_to_try = []
    if base_name in component_paths:
        paths_to_try.extend(component_paths[base_name])
    
    # If we have context path, add variations of it
    if context_path:
        # Add the full context path first
        if context_path not in paths_to_try:
            paths_to_try.insert(0, context_path)  # Higher priority
            
        # Add partial paths as fallbacks
        parts = context_path.split('.')
        if len(parts) > 1:
            # Just the last part
            if parts[-1] not in paths_to_try:
                paths_to_try.append(parts[-1])
                
            # Last two parts if available
            if len(parts) > 2 and '.'.join(parts[-2:]) not in paths_to_try:
                paths_to_try.append('.'.join(parts[-2:]))
    
    # Add heuristic module paths based on naming conventions
    if base_name.endswith('Explainer') and 'explainability' not in paths_to_try:
        paths_to_try.append('explainability')
        
    if base_name.endswith('Predictor') and 'prediction' not in paths_to_try:
        paths_to_try.append('prediction')
    
    # Always try with no module as a last resort
    paths_to_try.append('')  # Empty string means no module path
    
    # Remove duplicates while preserving order
    unique_paths = []
    for path in paths_to_try:
        if path not in unique_paths:
            unique_paths.append(path)
    
    return unique_paths

def resolve_ambiguous_component(codebase, component_name, component_type):
    """Resolves ambiguous components by trying multiple module paths
    
    This function handles component ambiguity by trying various module paths
    in a prioritized order based on project structure knowledge.
    
    Args:
        codebase: The Codegen codebase object
        component_name: Name of the component to find (may include module path)
        component_type: Type of the component ('function' or 'class')
        
    Returns:
        The resolved component if found, None otherwise
    """
    # Extract parts from the component name
    parts = component_name.split('.')
    base_name = parts[-1]
    module_path = '.'.join(parts[:-1]) if len(parts) > 1 else ''
    
    print(f"Attempting to resolve ambiguous component: {component_name}")
    
    # Try without any module path first (simplest approach from Codegen docs)
    try:
        if component_type == 'function':
            component = codebase.get_function(base_name)
            print(f"Found {base_name} as function without module path")
            return component
        elif component_type == 'class':
            component = codebase.get_class(base_name)
            print(f"Found {base_name} as class without module path")
            return component
    except Exception as e:
        # This is expected for ambiguous components
        if 'ambiguous' not in str(e).lower():
            print(f"Error during simple lookup: {str(e)}")
    
    # Get potential disambiguation paths to try
    module_paths = get_disambiguation_paths(base_name, module_path)
    print(f"Trying {len(module_paths)} possible module paths for {base_name}...")
    
    # Try each module path until we find a match
    for try_path in module_paths:
        try:
            if component_type == 'function':
                component = codebase.get_function(base_name, try_path) if try_path else codebase.get_function(base_name)
                print(f"Found {base_name} as function in module {try_path or '(none)'}")
                return component
            elif component_type == 'class':
                component = codebase.get_class(base_name, try_path) if try_path else codebase.get_class(base_name)
                print(f"Found {base_name} as class in module {try_path or '(none)'}")
                return component
        except Exception as e:
            # Only log unusual errors
            if 'ambiguous' not in str(e).lower():
                print(f"Error trying {base_name} in {try_path or '(none)'}: {str(e)}")
    
    # If all our structured attempts failed, try the existing alternative lookup strategy
    print(f"Structured disambiguation failed, trying alternative lookup for {component_name}...")
    return attempt_alternative_component_lookup(codebase, component_name, component_type)

def create_blast_radius_visualization(codebase, component, component_type):
    """Creates and saves a blast radius visualization for a component
    
    This function uses a simplified approach based on Codegen documentation:
    - For direct component lookups, it uses just the base name
    - It handles ambiguity by using resolution strategies
    
    Args:
        codebase: The Codegen codebase object
        component: Component name (possibly fully qualified)
        component_type: Type of component ('function' or 'class')
        
    Returns:
        Path to the output visualization file, or None if failed
    """
    print(f"Creating blast radius visualization for {component}...")
    
    # Initialize a new graph for this component
    G = nx.DiGraph()
    
    try:
        # Extract the base name for simplified lookup
        parts = component.split(".")
        base_name = parts[-1]
        
        # Handle the different component types
        target = None
        is_class_visualization = False
        
        # SIMPLIFIED APPROACH: Try direct lookup first with just the base name
        try:
            if component_type == "function":
                # Check if this is likely a method by the component name structure
                if len(parts) >= 3 and not parts[-2].islower():
                    # Looks like a method: module.Class.method
                    # We'll handle methods differently - first get the class
                    class_name = parts[-2]
                    func_name = parts[-1]
                    
                    try:
                        # Try to get the class first - use simplified approach
                        target_class = codebase.get_class(class_name)
                        if target_class:
                            # Get the method from the class
                            target = target_class.get_method(func_name)
                            # Add the class as a context node
                            G.add_node(target_class, 
                                    name=target_class.name,
                                    color=COLOR_PALETTE["PyClass"])
                            print(f"Found method {func_name} on class {class_name}")
                    except Exception as e:
                        print(f"Error finding class {class_name}: {str(e)}")
                
                # If not a method or if class lookup failed, try as a regular function
                if not target:
                    target = codebase.get_function(base_name)
                    print(f"Found function {base_name}")
                    
            elif component_type == "class":
                # Simple direct class lookup
                target = codebase.get_class(base_name)
                print(f"Found class {base_name}")
                is_class_visualization = True
                
        except Exception as e:
            # Handle ambiguity or other lookup errors
            print(f"Direct lookup error: {str(e)}")
            
            # Try resolving the ambiguity
            print(f"Attempting to resolve ambiguity for {component}...")
            target = resolve_ambiguous_component(codebase, component, component_type)
            if target and component_type == "class":
                is_class_visualization = True
                
        if not target:
            print(f"Could not find component: {component}")
            return None
        
        # Use special class visualization if this is a class
        if is_class_visualization:
            G, success = graph_class_methods(codebase, target, G)
            if not success:
                # If the class visualization failed, fall back to standard approach
                G.clear()
                G.add_node(target, 
                        name=target.name,
                        color=COLOR_PALETTE["PyClass"])
                print(f"Falling back to standard approach for class {target.name}...")
                create_downstream_call_trace(target, G)
                create_upstream_call_trace(target, G)
        else:
            # Add the target component as the root node
            G.add_node(target, 
                    name=target.name,
                    color=COLOR_PALETTE["StartFunction"])
            
            # Build both upstream and downstream call graphs
            print(f"Building downstream call trace for {target.name}...")
            create_downstream_call_trace(target, G)
            
            print(f"Building upstream call trace for {target.name}...")
            create_upstream_call_trace(target, G)
        
        # Generate a filename-safe version of the component name
        safe_name = component.replace('.', '_')
        output_file = os.path.join(OUTPUT_DIR, f"blast_radius_{safe_name}.html")
        
        # Render the visualization
        print(f"Visualizing graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...")
        codebase.visualize(G, output_file)
        print(f"Saved visualization to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"Error creating visualization for {component}: {str(e)}")
        return None

def create_index_html(output_files):
    """Creates an HTML index file linking to all visualizations
    
    Args:
        output_files: List of tuples (component_name, output_file_path)
    """
    index_html = os.path.join(OUTPUT_DIR, "index.html")
    
    with open(index_html, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("  <title>Blast Radius Analysis for migrineDT</title>\n")
        f.write("  <style>\n")
        f.write("    body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("    h1 { color: #333; }\n")
        f.write("    h2 { color: #555; margin-top: 30px; }\n")
        f.write("    .component { margin-bottom: 20px; padding: 10px; border-radius: 5px; }\n")
        f.write("    .component a { text-decoration: none; color: #0066cc; font-weight: bold; }\n")
        f.write("    .component a:hover { text-decoration: underline; }\n")
        f.write("    .module { background-color: #f5f5f5; padding: 10px; margin-top: 20px; border-radius: 5px; }\n")
        f.write("    .component-type { color: #777; font-style: italic; margin-left: 10px; }\n")
        f.write("    .function { background-color: #e1f5fe; }\n")
        f.write("    .class { background-color: #fff8e1; }\n")
        f.write("  </style>\n")
        f.write("</head>\n<body>\n")
        f.write("  <h1>Enhanced Blast Radius Analysis for migrineDT</h1>\n")
        f.write("  <p>This visualization shows the impact of changes to key components in the migrineDT codebase, helping identify dependencies.</p>\n")
        
        # Convert output_files to a dict for easy lookup
        output_dict = {component: path for component, path in output_files}
        
        # Group components by module
        modules = {}
        for component, component_type in TARGET_COMPONENTS:
            module = component.split('.')[0]
            if module not in modules:
                modules[module] = []
            modules[module].append((component, component_type))
        
        # Create sections for each module
        for module, components in sorted(modules.items()):
            f.write(f"  <div class='module'>\n")
            f.write(f"    <h2>{module.capitalize()} Module</h2>\n")
            
            for component, component_type in sorted(components):  # Sort components for consistent display
                # Check if this component has an output file
                if component in output_dict:
                    # Get the base filename for the link
                    base_filename = os.path.basename(output_dict[component])
                    
                    # CSS class based on component type
                    css_class = f"component {component_type}"
                    
                    f.write(f"    <div class='{css_class}'>\n")
                    f.write(f"      <h3>{component}<span class='component-type'>({component_type})</span></h3>\n")
                    f.write(f"      <a href='{base_filename}' target='_blank'>View Blast Radius</a>\n")
                    f.write(f"    </div>\n")
            
            f.write(f"  </div>\n")
        
        f.write("</body>\n</html>")
    
    print(f"\nCreated index file at: {index_html}")
    return index_html

def cleanup():
    """Clean up temporary directory"""
    print(f"Cleaning up temporary directory: {TEMP_DIR}")
    try:
        shutil.rmtree(TEMP_DIR)
    except Exception as e:
        print(f"Error cleaning up temp directory: {str(e)}")

def main():
    """Main function to run the enhanced blast radius analysis"""
    index_html = None
    try:
        # Create a clean copy of the codebase without problematic symbolic links
        clean_dir = create_clean_copy()
        
        print(f"\nInitializing Codegen with clean codebase at {clean_dir}")
        # Create a Codebase instance - initialization happens automatically when we create the object
        print("Initializing codebase...")
        codebase = Codebase(clean_dir)
        print("Codebase initialized successfully")
        
        # Track successes and failures for reporting
        successful_components = []
        failed_components = []
        output_files = []
        
        # Process each target component
        total_components = len(TARGET_COMPONENTS)
        print(f"\nProcessing {total_components} components...")
        
        for i, (component, component_type) in enumerate(TARGET_COMPONENTS, 1):
            print(f"\n{'='*80}")
            print(f"Processing [{i}/{total_components}] {component_type}: {component}")
            print(f"{'='*80}")
            
            try:
                output_file = create_blast_radius_visualization(codebase, component, component_type)
                if output_file:
                    output_files.append((component, output_file))
                    successful_components.append((component, component_type))
                    print(f"✓ Successfully created visualization for {component}")
                else:
                    failed_components.append((component, component_type))
                    print(f"✗ Failed to create visualization for {component}")
            except Exception as e:
                failed_components.append((component, component_type))
                print(f"✗ Error processing component {component}: {str(e)}")
        
        # Create an index file if we have any successful components
        if output_files:
            index_html = create_index_html(output_files)
            
            # Report success/failure stats
            print(f"\n{'='*80}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"Successfully visualized {len(successful_components)} of {total_components} components:")
            
            # Group successes by module for cleaner reporting
            by_module = {}
            for component, component_type in successful_components:
                module = component.split('.')[0]
                if module not in by_module:
                    by_module[module] = []
                by_module[module].append((component, component_type))
                
            for module, components in sorted(by_module.items()):
                print(f"\n  {module.upper()} MODULE:")
                for component, component_type in sorted(components):
                    print(f"    ✓ {component} ({component_type})")
                    
            if failed_components:
                print(f"\nFailed to visualize {len(failed_components)} components:")
                for component, component_type in sorted(failed_components):
                    print(f"    ✗ {component} ({component_type})")
            
            print(f"\n{'='*80}")
            print(f"View the results at: file://{index_html}")
            print(f"{'='*80}")
        else:
            print("\nNo components could be visualized.")
            
    except Exception as e:
        print(f"Error in blast radius analysis: {str(e)}")
    
    finally:
        # Clean up temporary directory
        cleanup()
        
    return index_html

if __name__ == "__main__":
    main()

"""
Blast Radius Analysis for migrineDT using Codegen library

This script handles symbolic link issues while leveraging Codegen's
powerful traversal capabilities to visualize function dependencies.
It creates a clean Git repository copy to work with.
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

# Key components to analyze based on the restructured project
# For each component, we now add a disambiguator key as the 4th element to help find the right component
TARGET_COMPONENTS = [
    # CLI directory components
    ("cli/main.py", "main", "function", "core_function"),
    ("cli/commands.py", "run_optimization", "function", "cli_command"),
    ("cli/commands.py", "run_explainability_analysis", "function", "cli_command"),
    
    # Core directory components
    ("core/optimization.py", "run_optimization", "function", "core_optimizer"),
    ("core/meta_learning.py", "MetaOptimizer", "class", "core_meta"),
    ("core/evaluation.py", "evaluate_model", "function", "core_eval"),
    
    # Explainability directory components
    ("explainability/model_explainer.py", "ShapExplainer", "class", "model_explainer"),
    ("explainability/optimizer_explainer.py", "OptimizerExplainer", "class", "optimizer_explainer"),
    ("explainability/explainer_factory.py", "ExplainerFactory", "class", "explainer_factory"),
    
    # Migraine directory components
    ("migraine/prediction.py", "MigrainePredictor", "class", "migraine_predictor"),
    ("migraine/explainability.py", "explain_model", "function", "migraine_explainer"),
]

def create_clean_git_copy():
    """Create a clean copy of the codebase without problematic symbolic links and initialize git"""
    print(f"Creating clean copy of codebase in {TEMP_DIR}...")
    
    # Use rsync to copy files excluding problematic paths
    exclude_args = []
    for path in PROBLEM_PATHS:
        exclude_args.extend(["--exclude", path])
    
    # Also exclude common directories that might cause issues
    exclude_args.extend([
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", ".venv_minimal",
        "--exclude", "venv",
        "--exclude", "env",
        "--exclude", "*.egg-info"
    ])
    
    # Run rsync command
    rsync_command = ["rsync", "-a"] + exclude_args + [f"{PROJECT_ROOT}/", TEMP_DIR]
    print(f"Running command: {' '.join(rsync_command)}")
    subprocess.run(rsync_command)
    
    # Initialize git repository
    print("Initializing git repository in clean copy...")
    subprocess.run(["git", "init"], cwd=TEMP_DIR)
    subprocess.run(["git", "config", "user.email", "blast_radius@example.com"], cwd=TEMP_DIR)
    subprocess.run(["git", "config", "user.name", "Blast Radius Analysis"], cwd=TEMP_DIR)
    
    # Add all files and commit
    subprocess.run(["git", "add", "."], cwd=TEMP_DIR)
    subprocess.run(["git", "commit", "-m", "Initial commit for blast radius analysis"], cwd=TEMP_DIR)
    
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

def attempt_alternative_component_lookup(codebase, component_name, component_type, file_path, disambiguator=None):
    """Attempts a more aggressive approach to find components using partial module paths and a modified search approach"""
    print(f"Using alternative lookup approach for {component_type} {component_name} in {file_path} with disambiguator: {disambiguator}")
    
    try:
        # If we have a disambiguator, we'll use it to refine our approach
        if disambiguator:
            # Some common patterns based on disambiguator prefixes
            if disambiguator == "core_function":
                print("Using core_function disambiguation strategy")
                # For main entry points, try a direct lookup in the cli module
                try:
                    if component_type == "function":
                        return codebase.get_function(component_name, "cli.main")
                except Exception as e:
                    print(f"core_function strategy failed: {str(e)}")
            
            elif disambiguator.startswith("cli_"):
                print("Using cli disambiguation strategy")
                # CLI components are likely in the cli module
                try:
                    if component_type == "function":
                        return codebase.get_function(component_name, "cli.commands")
                except Exception as e:
                    print(f"cli strategy failed: {str(e)}")
            
            elif disambiguator.startswith("core_"):
                print("Using core disambiguation strategy")
                # Try the specific core module
                try:
                    module_part = disambiguator.replace("core_", "")
                    if module_part == "optimizer" and component_type == "function":
                        return codebase.get_function(component_name, "core.optimization")
                    elif module_part == "meta" and component_type == "class":
                        return codebase.get_class(component_name, "core.meta_learning")
                    elif module_part == "eval" and component_type == "function":
                        return codebase.get_function(component_name, "core.evaluation")
                except Exception as e:
                    print(f"core strategy failed: {str(e)}")
            
            elif disambiguator.endswith("_explainer"):
                print("Using explainer disambiguation strategy")
                # Try the specific explainability module
                try:
                    if disambiguator == "model_explainer" and component_type == "class":
                        return codebase.get_class(component_name, "explainability.model_explainer")
                    elif disambiguator == "optimizer_explainer" and component_type == "class":
                        return codebase.get_class(component_name, "explainability.optimizer_explainer")
                    elif disambiguator == "explainer_factory" and component_type == "class":
                        return codebase.get_class(component_name, "explainability.explainer_factory")
                    elif disambiguator == "migraine_explainer" and component_type == "function":
                        return codebase.get_function(component_name, "migraine.explainability")
                except Exception as e:
                    print(f"explainer strategy failed: {str(e)}")
            
            elif disambiguator == "migraine_predictor":
                print("Using migraine_predictor disambiguation strategy")
                try:
                    if component_type == "class":
                        return codebase.get_class(component_name, "migraine.prediction")
                except Exception as e:
                    print(f"migraine_predictor strategy failed: {str(e)}")
        
        # If disambiguator approach failed or no disambiguator provided, try the generic approach
        # Try a variety of module path formats
        module_variations = []
        
        # Original path converted to module format
        module_path = file_path.replace('/', '.').replace('.py', '')
        module_variations.append(module_path)
        
        # Just the filename without extension as module
        simple_module = os.path.basename(file_path).replace('.py', '')
        module_variations.append(simple_module)
        
        # Try with parent directory + filename
        parts = file_path.split('/')
        if len(parts) >= 2:
            parent_and_file = '.'.join(parts[-2:]).replace('.py', '')
            module_variations.append(parent_and_file)
        
        # Try with just the component name as the module name
        module_variations.append(component_name.lower())
        
        # Try common module prefixes if in common directories
        for common_prefix in ['core', 'explainability', 'migraine', 'cli', 'utils']:
            if common_prefix in file_path:
                module_variations.append(f"{common_prefix}.{simple_module}")
                
        # Add empty string to try global lookup
        module_variations.append('')
        
        # Try each module path variation
        for module_var in module_variations:
            try:
                if component_type == "function":
                    if module_var:
                        target = codebase.get_function(component_name, module_var)
                    else:
                        target = codebase.get_function(component_name)
                    print(f"Found function {component_name} with module variation: {module_var}")
                    return target
                elif component_type == "class":
                    if module_var:
                        target = codebase.get_class(component_name, module_var)
                    else:
                        target = codebase.get_class(component_name)
                    print(f"Found class {component_name} with module variation: {module_var}")
                    return target
            except Exception as e:
                if "ambiguous" in str(e).lower():
                    print(f"Module variation {module_var} resulted in ambiguity")
                    continue
                else:
                    print(f"Module variation {module_var} failed: {str(e)}")
        
        # All attempts failed
        print(f"All lookup methods failed for {component_type} {component_name} in {file_path}")
        return None
        
    except Exception as e:
        print(f"Error in alternative component lookup: {str(e)}")
        return None

def create_blast_radius_visualization(codebase, file_path, component_name, component_type, disambiguator=None):
    """Creates and saves a blast radius visualization for a component using direct lookups"""
    print(f"\nCreating blast radius visualization for {file_path}:{component_name}...")
    
    # Initialize a new graph for this component
    G = nx.DiGraph()
    
    try:
        # Find the target component using direct lookups
        target = None
        component_id = f"{file_path}:{component_name}"
        
        # Extract module path from file path
        # Convert file path to module format
        module_path = file_path.replace('/', '.').replace('.py', '')
        print(f"Looking for {component_type} {component_name} in module {module_path}")
        
        # First try with module path
        try:
            if component_type == "function":
                target = codebase.get_function(component_name, module_path)
                print(f"Found function {component_name} in {module_path}")
            elif component_type == "class":
                target = codebase.get_class(component_name, module_path)
                print(f"Found class {component_name} in {module_path}")
        except Exception as e:
            error_msg = str(e)
            print(f"Error finding {component_type} {component_name} in {module_path}: {error_msg}")
            
                # If we couldn't find the component with direct approaches, try our alternative lookup
            target = attempt_alternative_component_lookup(codebase, component_name, component_type, file_path, disambiguator)
        
        if not target:
            print(f"Could not find component: {component_name} in {file_path}")
            return None
        
        print(f"Found target component: {target.name}")
        
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
        safe_name = component_id.replace(':', '_').replace('/', '_').replace('.', '_')
        output_file = os.path.join(OUTPUT_DIR, f"blast_radius_{safe_name}.html")
        
        # Render the visualization
        print(f"Visualizing graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...")
        codebase.visualize(G, output_file)
        print(f"Saved visualization to {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"Error creating visualization for {file_path}:{component_name}: {str(e)}")
        return None

def create_index_html(output_files):
    """Creates an HTML index file linking to all visualizations"""
    index_html = os.path.join(OUTPUT_DIR, "index.html")
    
    with open(index_html, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("  <title>Blast Radius Analysis for migrineDT</title>\n")
        f.write("  <style>\n")
        f.write("    body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("    h1 { color: #333; }\n")
        f.write("    h2 { color: #555; margin-top: 30px; }\n")
        f.write("    .component { margin-bottom: 20px; }\n")
        f.write("    .component a { text-decoration: none; color: #0066cc; }\n")
        f.write("    .component a:hover { text-decoration: underline; }\n")
        f.write("    .module { background-color: #f5f5f5; padding: 10px; margin-top: 20px; }\n")
        f.write("  </style>\n")
        f.write("</head>\n<body>\n")
        f.write("  <h1>Blast Radius Analysis for migrineDT</h1>\n")
        f.write("  <p>This visualization shows the impact of changes to key components in the migrineDT codebase.</p>\n")
        
        # Group components by module
        modules = {}
        for file_path, component_name, _, _ in TARGET_COMPONENTS:
            module = file_path.split('/')[0]
            component_id = f"{file_path}:{component_name}"
            if module not in modules:
                modules[module] = []
            modules[module].append(component_id)
        
        # Create sections for each module
        for module, components in modules.items():
            f.write(f"  <div class='module'>\n")
            f.write(f"    <h2>{module.capitalize()} Module</h2>\n")
            
            for component in components:
                if component in output_files and output_files[component]:
                    # Get the base filename for the link
                    base_filename = os.path.basename(output_files[component])
                    
                    f.write(f"    <div class='component'>\n")
                    f.write(f"      <h3>{component}</h3>\n")
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
    """Main function to run the blast radius analysis"""
    try:
        # Create a clean copy of the codebase with git initialized
        clean_dir = create_clean_git_copy()
        
        # Create dummy files for the problematic symbolic links
        for problem_path in PROBLEM_PATHS:
            full_path = os.path.join(clean_dir, problem_path)
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # Create a dummy file
            with open(full_path, 'w') as f:
                f.write("# Dummy file created for blast radius analysis\n")
        
        # Commit these files to the repository
        subprocess.run(["git", "add", "."], cwd=clean_dir)
        subprocess.run(["git", "commit", "-m", "Add dummy files for symbolic links"], cwd=clean_dir)
        
        print(f"Initializing Codegen with clean codebase at {clean_dir}")
        codebase = Codebase(clean_dir)
        
        # Codebase is automatically initialized during creation
        print("Codebase initialized successfully")
        
        # Process each target component
        output_files = {}
        for file_path, component_name, component_type, disambiguator in TARGET_COMPONENTS:
            try:
                component_id = f"{file_path}:{component_name}"
                output_file = create_blast_radius_visualization(codebase, file_path, component_name, component_type, disambiguator)
                output_files[component_id] = output_file
            except Exception as e:
                print(f"Error processing component {file_path}:{component_name}: {str(e)}")
        
        # Create an index file
        index_html = create_index_html(output_files)
        
        print(f"\nAnalysis complete. View the results at: file://{index_html}")
        
    except Exception as e:
        print(f"Error in blast radius analysis: {str(e)}")
    
    finally:
        # Clean up temporary directory
        cleanup()

if __name__ == "__main__":
    main()

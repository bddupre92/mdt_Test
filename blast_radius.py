import codegen
from codegen import Codebase
import networkx as nx

@codegen.function("visualize-project-blast-radius")
def run(codebase: Codebase):
    # Create directed graph
    G = nx.DiGraph()
    
    # Key components to analyze from your main.py
    key_classes = [
        "MetaOptimizer",
        "MetaLearner",
        "ModelFactory",
        "ExplainerFactory",
        "OptimizerFactory",
        "DriftAnalyzer",
        "OptimizerAnalyzer",
        "FrameworkEvaluator"
    ]
    
    # Add core nodes
    for class_name in key_classes:
        target_class = codebase.get_class(class_name)
        if target_class:
            G.add_node(target_class, color="#4CAF50")  # Green for core classes
            
            # Analyze dependencies
            for usage in target_class.usages:
                usage_symbol = usage.usage_symbol
                
                # Color code based on type
                if usage_symbol.__class__.__name__ == "Function":
                    color = "#2196F3"  # Blue for functions
                elif usage_symbol.__class__.__name__ == "Class":
                    color = "#FFC107"  # Yellow for classes
                else:
                    color = "#9C27B0"  # Purple for other symbols
                
                G.add_node(usage_symbol, color=color)
                G.add_edge(target_class, usage_symbol)
    
    # The visualization will be available on codegen.sh
    print("View the visualization on codegen.sh!")

if __name__ == "__main__":
    codebase = Codebase.from_local("/Users/blair.dupre/Documents/migrineDT/mdt_Test/main.py")  # Point to your repository root
    run(codebase)
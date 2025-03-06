"""
algorithm_selection_demo.py
---------------------------
A standalone demo script to demonstrate the algorithm selection visualization capabilities.
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the visualization
try:
    from visualization.algorithm_selection_viz import AlgorithmSelectionVisualizer
    ALGORITHM_VIZ_AVAILABLE = True
except ImportError:
    ALGORITHM_VIZ_AVAILABLE = False
    logging.warning("Algorithm selection visualization not available")

def run_algorithm_selection_demo():
    """Run a demonstration of algorithm selection visualization."""
    if not ALGORITHM_VIZ_AVAILABLE:
        logging.error("Algorithm selection visualization is not available.")
        return {"success": False, "error": "Algorithm selection visualization not available"}

    # Create a directory for visualizations
    viz_dir = 'results/algorithm_selection_demo'
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize the visualizer
    visualizer = AlgorithmSelectionVisualizer(save_dir=viz_dir)
    
    # Define test functions for context
    test_functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley']
    optimizers = ['DE', 'ES', 'GWO', 'ACO', 'DE-Adaptive', 'ES-Adaptive']
    
    # Create sample data
    print("Creating sample algorithm selections for demonstration...")
    for func_name in test_functions:
        print(f"Processing {func_name}...")
        
        # For demonstration, manually create algorithm selections
        for i in range(1, 21):  # Create 20 selections per function
            # Randomly select an optimizer
            optimizer = np.random.choice(optimizers)
            score = 100 - i * 5  # Fake improvement in score
            
            # Record the selection
            visualizer.record_selection(
                iteration=i,
                optimizer=optimizer,
                problem_type=func_name,
                score=score,
                context={"function_name": func_name, "phase": "optimization"}
            )
            print(f"  Recorded selection of {optimizer} for iteration {i}")
    
    # Generate all types of plots
    print("Generating algorithm selection visualizations...")
    
    print("Generating frequency visualization...")
    visualizer.plot_selection_frequency()
    
    print("Generating timeline visualization...")
    visualizer.plot_selection_timeline()
    
    print("Generating problem distribution visualization...")
    visualizer.plot_problem_distribution()
    
    print("Generating performance comparison visualization...")
    visualizer.plot_performance_comparison()
    
    print("Generating phase selection visualization...")
    visualizer.plot_phase_selection()
    
    print("Generating summary dashboard...")
    visualizer.create_summary_dashboard()
    
    print("Generating interactive visualizations...")
    visualizer.interactive_selection_timeline()
    visualizer.interactive_dashboard()
    
    print("Algorithm selection demo completed successfully.")
    print(f"Visualizations saved to: {viz_dir}")
    
    return {"success": True, "visualizations_path": viz_dir}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Algorithm Selection Visualization Demo")
    parser.add_argument("--viz-dir", type=str, default="results/algorithm_selection_demo",
                        help="Directory to save visualization files")
    parser.add_argument("--plots", nargs="+", 
                        choices=["frequency", "timeline", "problem", "performance", "phase", "dashboard", "interactive"],
                        default=["frequency", "timeline", "dashboard", "interactive"],
                        help="List of plot types to generate")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_algorithm_selection_demo()

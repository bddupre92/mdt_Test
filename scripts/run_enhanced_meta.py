#!/usr/bin/env python3
"""
Run Enhanced Meta-Learning with CLI arguments
"""
import os
import argparse
import logging
import json
import numpy as np
import shutil
from core.meta_learning import run_enhanced_meta_learning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Helper class for handling numpy types in JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def main():
    """
    Run enhanced meta-learning with command-line arguments matching main_v2.py
    """
    # Create directory for results if it doesn't exist
    results_dir = 'results/enhanced_meta_main'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create arguments similar to what would be used in main_v2.py
    args = argparse.Namespace()
    
    # Meta-learning specific parameters
    args.dimension = 10  # Problem dimension
    args.use_ml_selection = True  # Use ML-based selection
    args.extract_features = True  # Extract problem features
    args.visualize = True  # Enable visualization
    args.enhanced_meta = True  # Flag for enhanced meta-learning
    
    # Enhanced meta-learning parameters
    args.meta_method = 'random'  # Meta-learning method (random, ucb, thompson, bayesian)
    args.meta_surrogate = 'rf'   # Surrogate model (gp, rf, mlp)
    args.meta_selection = 'greedy'  # Selection strategy (greedy, probability)
    args.meta_exploration = 0.2  # Exploration parameter
    args.meta_history_weight = 0.5  # History weight
    
    # Optimization parameters
    args.max_evals = 1000  # Maximum evaluations per function
    args.n_parallel = 2  # Number of parallel runs
    args.budget_per_iteration = 50  # Budget per iteration
    
    # Output parameters
    args.save_dir = results_dir  # Directory to save results
    
    # Run enhanced meta-learning
    print("Starting enhanced meta-learning with standard CLI arguments...")
    results = run_enhanced_meta_learning(args)
    
    # Ensure results are properly saved
    try:
        # Check if visualizations were generated
        default_viz_dir = 'results/enhanced_meta/visualizations'
        if os.path.exists(default_viz_dir) and len(os.listdir(default_viz_dir)) > 0:
            # Copy visualizations to our custom dir
            for viz_file in os.listdir(default_viz_dir):
                source_path = os.path.join(default_viz_dir, viz_file)
                dest_path = os.path.join(viz_dir, viz_file)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, dest_path)
            print(f"Copied visualization files to {viz_dir}")
        
        # Check if results files exist and copy them
        default_results_dir = 'results/enhanced_meta'
        for filename in ['enhanced_meta_results.json', 'enhanced_meta_selections.json', 'problem_features.json']:
            source_path = os.path.join(default_results_dir, filename)
            if os.path.exists(source_path):
                dest_path = os.path.join(results_dir, filename)
                shutil.copy2(source_path, dest_path)
                print(f"Copied {filename} to {results_dir}")
                
        # Create a summary file
        summary_file = os.path.join(results_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'execution_time': results.get('execution_time', 0),
                'best_algorithm': results.get('best_algorithm', 'Unknown'),
                'completed_problems': results.get('completed_problems', 0),
                'total_problems': results.get('total_problems', 0),
                'parameters': {
                    'dimension': args.dimension,
                    'max_evals': args.max_evals,
                    'use_ml_selection': args.use_ml_selection,
                    'extract_features': args.extract_features
                }
            }, f, indent=2, cls=NumpyEncoder)
        print(f"Summary saved to {summary_file}")
            
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
    
    # Print summary
    print("\nMeta-Learning Summary:")
    print("=====================")
    
    if 'best_algorithm' in results:
        print(f"Overall best algorithm: {results['best_algorithm']}")
    
    if 'algorithm_selections' in results:
        print("\nBest algorithm per function:")
        for func, selections in results['algorithm_selections'].items():
            best_algo = max(selections.items(), key=lambda x: x[1])[0]
            count = selections[best_algo]
            print(f"  {func}: {best_algo} ({count} selections)")
    
    print(f"\nResults saved to {args.save_dir}")
    
    return 0

if __name__ == "__main__":
    main() 
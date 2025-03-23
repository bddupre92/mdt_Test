import argparse
import logging
from core.meta_learning import run_enhanced_meta_learning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create arguments
args = argparse.Namespace()
args.dimension = 10
args.visualize = True
args.use_ml_selection = True
args.extract_features = True
args.max_evals = 500  # Reduce evaluations for faster execution
args.save_dir = 'results/enhanced_meta'
args.n_parallel = 2
args.budget_per_iteration = 50

# Run enhanced meta-learning
print("Starting enhanced meta-learning...")
results = run_enhanced_meta_learning(args)
print("Meta-learning completed successfully!")
print(f"Results saved in {args.save_dir}") 
"""
Main execution module for the meta_optimizer package.
This allows running the package directly with 'python -m meta_optimizer'.
"""
import os
import sys

# Add the root directory to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Import and run the main function from the root project's main.py
from main import main

if __name__ == "__main__":
    main()

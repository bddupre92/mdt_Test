import os
import sys
import logging
from pathlib import Path
import matplotlib
import numpy as np
import random
import torch

from utils.logging_config import setup_logging

def setup_environment(seed=None, use_gpu=False, log_level=logging.INFO):
    """
    Configure the application environment including:
    - Logging setup
    - Directory creation
    - Matplotlib backend configuration
    - Random seed initialization for reproducibility
    - GPU configuration if available
    
    Parameters:
    -----------
    seed : int, optional
        Random seed for reproducibility
    use_gpu : bool, optional
        Whether to use GPU if available
    log_level : int, optional
        Logging level
    
    Returns:
    --------
    dict
        Environment information
    """
    # Set up logging
    setup_logging(level=log_level)
    
    # Create necessary directories
    directories = [
        'results', 
        'results/plots', 
        'results/data', 
        'results/explainability',
        'results/drift',
        'results/performance',
        'results/benchmarks',
        'results/meta_learning',
        'results/enhanced_meta',
        'results/enhanced_meta/visualizations',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
    
    # Set matplotlib backend and interactive mode
    try:
        current_backend = matplotlib.get_backend()
        logging.info(f"Current backend before setting: {current_backend}")
        
        # Try to use TkAgg backend for interactive plotting
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.ion()  # Enable interactive mode
        
        logging.info(f"Backend after forcing to TkAgg: {matplotlib.get_backend()}")
        logging.info(f"Interactive mode after plt.ion(): {plt.isinteractive()}")
    except Exception as e:
        logging.warning(f"Could not set matplotlib backend: {e}")
    
    # Set random seeds for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Random seed set to {seed}")
    
    # Configure GPU if requested and available
    device = "cpu"
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        if use_gpu:
            logging.warning("GPU requested but not available, using CPU")
        else:
            logging.info("Using CPU")
    
    # Return environment information
    return {
        "directories": directories,
        "matplotlib_backend": matplotlib.get_backend(),
        "device": device,
        "seed": seed,
        "python_version": sys.version
    }

def check_dependencies():
    """
    Check that all required dependencies are installed
    
    Returns:
    --------
    bool
        True if all dependencies are available
    """
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scikit-learn', 'torch', 'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                # For scikit-learn, we need to import sklearn
                import sklearn
                logging.debug(f"Successfully imported {package} (as sklearn)")
            else:
                __import__(package)
                logging.debug(f"Successfully imported {package}")
        except ImportError as e:
            logging.error(f"Error importing {package}: {str(e)}")
            missing.append(package)
    
    if missing:
        logging.error(f"Missing required packages: {', '.join(missing)}")
        return False
    
    return True

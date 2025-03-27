"""
Check MoE Framework Dependencies

This script checks if all required MoE framework dependencies are available
and properly configured before starting the dashboard.
"""

import os
import sys
import importlib
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

REQUIRED_MODULES = [
    'moe_framework.workflow.moe_pipeline',
    'moe_framework.gating.gating_network',
    'moe_framework.experts.base_expert',
    'moe_framework.integration.pipeline_connector',
    'moe_framework.execution.execution_pipeline',
    'moe_framework.interfaces.pipeline'
]

REQUIRED_PACKAGES = [
    'streamlit',
    'pandas',
    'numpy',
    'plotly',
    'networkx',
    'torch'
]

def check_package(package_name: str) -> bool:
    """Check if a Python package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_moe_module(module_path: str) -> bool:
    """Check if a MoE framework module is available."""
    try:
        importlib.import_module(module_path)
        logger.info(f"✅ Module {module_path} is available")
        return True
    except ImportError as e:
        logger.error(f"❌ Module {module_path} is not available: {str(e)}")
        return False

def find_meta_module() -> Optional[str]:
    """Find the meta module directory."""
    current_dir = os.getcwd()
    potential_paths = [
        os.path.join(current_dir, 'meta_optimizer'),
        os.path.join(current_dir, 'meta'),
        os.path.join(os.path.dirname(current_dir), 'meta_optimizer'),
        os.path.join(os.path.dirname(current_dir), 'meta')
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            logger.info(f"Added potential meta module path: {path}")
            sys.path.append(path)
            return path
    return None

def check_dependencies() -> Dict[str, bool]:
    """
    Check all required dependencies for the MoE framework dashboard.
    
    Returns:
        Dictionary with check results
    """
    results = {
        'packages': True,
        'moe_modules': True,
        'meta_module': False
    }
    
    # Check required packages
    for package in REQUIRED_PACKAGES:
        if not check_package(package):
            logger.error(f"❌ Required package {package} is not installed")
            results['packages'] = False
    
    # Try to find and add meta module to path
    meta_path = find_meta_module()
    results['meta_module'] = meta_path is not None
    
    # Check MoE framework modules
    for module in REQUIRED_MODULES:
        if not check_moe_module(module):
            results['moe_modules'] = False
    
    # Log overall status
    if all(results.values()):
        logger.info("✅ All dependencies are satisfied")
    else:
        logger.warning("⚠️ Some dependencies are missing")
        
    return results

def main():
    """Run dependency checks and report results."""
    results = check_dependencies()
    
    if not results['packages']:
        logger.error("Missing required Python packages. Please run: pip install -r requirements.txt")
        sys.exit(1)
        
    if not results['moe_modules']:
        logger.error("Missing required MoE framework modules. Please check installation")
        sys.exit(1)
        
    if not results['meta_module']:
        logger.warning("Meta module not found. Some features may be limited")
    
    logger.info("Dependency check completed successfully")

if __name__ == "__main__":
    main() 
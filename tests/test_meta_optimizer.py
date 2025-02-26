"""
Test suite for meta-optimizer functionality.
"""
import pytest
import numpy as np
from typing import Dict, Any
import logging

from meta.meta_optimizer import MetaOptimizer
from optimizers.optimizer_factory import create_optimizers
from optimizers.base_optimizer import BaseOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def test_config():
    """Test configuration."""
    dim = 5  # Reduced dimension
    bounds = [(-5.12, 5.12)] * dim
    return dim, bounds

@pytest.fixture
def meta_optimizer(test_config):
    """Create meta-optimizer instance."""
    dim, bounds = test_config
    # Create optimizers using factory
    all_optimizers = create_optimizers(dim=dim, bounds=bounds, include_meta=False)
    # Select a subset for testing
    optimizers = {
        'DE': all_optimizers['DE (Standard)'],
        'GWO': all_optimizers['GWO (Standard)']
    }
    return MetaOptimizer(dim=dim, bounds=bounds, optimizers=optimizers, n_parallel=1)

@pytest.fixture
def test_objective():
    """Simple test objective function."""
    def objective(x):
        x = np.asarray(x)  # Convert input to numpy array
        return np.sum(x**2)  # Simple sphere function
    return objective

def test_meta_optimizer_initialization(meta_optimizer, test_config):
    """Test meta-optimizer initialization."""
    dim, bounds = test_config
    assert meta_optimizer is not None
    assert meta_optimizer.dim == dim
    assert len(meta_optimizer.optimizers) == 2
    assert all(isinstance(opt, BaseOptimizer) for opt in meta_optimizer.optimizers.values())

def test_optimization(meta_optimizer, test_objective):
    """Test basic optimization."""
    solution = meta_optimizer.optimize(
        objective_func=test_objective,
        max_evals=50  # Minimal evaluations for testing
    )
    
    assert solution is not None
    assert isinstance(solution, np.ndarray)
    assert len(solution) == meta_optimizer.dim
    assert all(-5.12 <= x <= 5.12 for x in solution)

def test_get_best_solution(meta_optimizer, test_objective):
    """Test getting best solution."""
    solution = meta_optimizer.optimize(
        objective_func=test_objective,
        max_evals=50
    )
    
    assert solution is not None
    assert isinstance(solution, np.ndarray)
    assert len(solution) == meta_optimizer.dim

def test_optimization_history(meta_optimizer, test_objective):
    """Test optimization history."""
    meta_optimizer.optimize(
        objective_func=test_objective,
        max_evals=50
    )
    
    history = meta_optimizer.history
    assert history is not None
    assert len(history.records) > 0

"""
test_functions.py
----------------
Standard benchmark functions for optimization algorithms.
Includes both classical test functions and ML-specific objectives.
"""

import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn

@dataclass
class TestFunction:
    name: str
    func: Callable
    dim: int
    bounds: List[Tuple[float, float]]
    global_minimum: float = 0.0
    characteristics: dict = None
    
    def __call__(self, x):
        return self.func(x)

class ClassicalTestFunctions:
    @staticmethod
    def sphere(x) -> float:
        """Sphere function (continuous, convex, unimodal)"""
        x = np.asarray(x)
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x) -> float:
        """Rosenbrock function (continuous, non-convex, unimodal)"""
        x = np.asarray(x)
        return np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def rastrigin(x) -> float:
        """Rastrigin function (continuous, non-convex, multimodal)"""
        x = np.asarray(x)
        return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    @staticmethod
    def ackley(x) -> float:
        """Ackley function (continuous, non-convex, multimodal)"""
        x = np.asarray(x)
        a, b, c = 20, 0.2, 2*np.pi
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c*x))
        return -a * np.exp(-b*np.sqrt(sum_sq/d)) - np.exp(sum_cos/d) + a + np.exp(1)
    
    @staticmethod
    def griewank(x) -> float:
        """Griewank function (continuous, non-convex, multimodal)"""
        x = np.asarray(x)
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
        return 1 + sum_sq - prod_cos
    
    @staticmethod
    def levy(x) -> float:
        """Levy function (continuous, non-convex, multimodal)"""
        x = np.asarray(x)
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
        term3 = (w[-1]-1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
        return term1 + term2 + term3

class MLTestFunctions:
    @staticmethod
    def create_nn_training_objective(
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            device: str = 'cpu',
            max_epochs: int = 10
        ) -> Callable:
        """Create objective function for neural network training"""
        
        def objective(params: np.ndarray) -> float:
            # Map parameters to model
            idx = 0
            for p in model.parameters():
                num_params = np.prod(p.shape)
                p.data = torch.tensor(
                    params[idx:idx+num_params].reshape(p.shape),
                    device=device
                )
                idx += num_params
            
            # Training loop
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters())
            best_val_loss = float('inf')
            
            for epoch in range(max_epochs):
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, targets).item()
                val_loss /= len(val_loader)
                best_val_loss = min(best_val_loss, val_loss)
            
            return best_val_loss
        
        return objective
    
    @staticmethod
    def create_sklearn_cv_objective(
            estimator: BaseEstimator,
            X: np.ndarray,
            y: np.ndarray,
            param_ranges: dict,
            cv_splits: int = 5,
            metric: Callable = None
        ) -> Callable:
        """Create objective function for sklearn model cross-validation"""
        
        from sklearn.model_selection import cross_val_score
        if metric is None:
            from sklearn.metrics import accuracy_score
            metric = accuracy_score
        
        def objective(params: np.ndarray) -> float:
            # Map parameters to estimator hyperparameters
            param_dict = {}
            idx = 0
            for name, (low, high) in param_ranges.items():
                param_dict[name] = params[idx] * (high - low) + low
                idx += 1
            
            estimator.set_params(**param_dict)
            scores = cross_val_score(
                estimator, X, y,
                cv=cv_splits,
                scoring=metric
            )
            return -np.mean(scores)  # Negative because we minimize
        
        return objective

# Dictionary of test functions with their characteristics
TEST_FUNCTIONS = {
    'sphere': lambda dim, bounds: TestFunction(
        name='Sphere',
        func=ClassicalTestFunctions.sphere,
        dim=dim,
        bounds=bounds,
        characteristics={
            'convex': True,
            'separable': True,
            'multimodal': False,
            'continuous': True
        }
    ),
    'rosenbrock': lambda dim, bounds: TestFunction(
        name='Rosenbrock',
        func=ClassicalTestFunctions.rosenbrock,
        dim=dim,
        bounds=bounds,
        characteristics={
            'convex': False,
            'separable': False,
            'multimodal': False,
            'continuous': True
        }
    ),
    'rastrigin': lambda dim, bounds: TestFunction(
        name='Rastrigin',
        func=ClassicalTestFunctions.rastrigin,
        dim=dim,
        bounds=bounds,
        characteristics={
            'convex': False,
            'separable': True,
            'multimodal': True,
            'continuous': True
        }
    ),
    'ackley': lambda dim, bounds: TestFunction(
        name='Ackley',
        func=ClassicalTestFunctions.ackley,
        dim=dim,
        bounds=bounds,
        characteristics={
            'convex': False,
            'separable': False,
            'multimodal': True,
            'continuous': True
        }
    ),
    'griewank': lambda dim, bounds: TestFunction(
        name='Griewank',
        func=ClassicalTestFunctions.griewank,
        dim=dim,
        bounds=bounds,
        characteristics={
            'convex': False,
            'separable': False,
            'multimodal': True,
            'continuous': True
        }
    ),
    'levy': lambda dim, bounds: TestFunction(
        name='Levy',
        func=ClassicalTestFunctions.levy,
        dim=dim,
        bounds=bounds,
        characteristics={
            'convex': False,
            'separable': False,
            'multimodal': True,
            'continuous': True
        }
    )
}

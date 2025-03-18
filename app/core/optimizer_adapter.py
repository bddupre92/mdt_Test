"""
Optimizer Adapter for Integration with Existing Optimizers

This module provides adapters for integrating with existing optimizer
implementations, providing a consistent interface for the benchmark system.
"""

import importlib
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

class OptimizerAdapter:
    """Base class for optimizer adapters."""
    
    def __init__(self, name: str):
        """Initialize optimizer adapter.
        
        Args:
            name: Name of the optimizer
        """
        self.name = name
    
    def optimize(self, 
                function: Callable, 
                max_evaluations: int = 1000, 
                callback: Optional[Callable] = None,
                bounds: Optional[Tuple[float, float]] = None,
                dimension: int = 10,
                **kwargs) -> Dict[str, Any]:
        """Optimize a function.
        
        Args:
            function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            callback: Optional callback function
            bounds: Function bounds as (min, max)
            dimension: Problem dimension
            kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement optimize")


class DifferentialEvolutionAdapter(OptimizerAdapter):
    """Adapter for Differential Evolution optimizer."""
    
    def __init__(self, **kwargs):
        """Initialize DE adapter.
        
        Args:
            kwargs: Additional parameters for DE
        """
        super().__init__("DifferentialEvolution")
        self.params = kwargs
        
        # Try to import the DE optimizer
        try:
            de_module = importlib.import_module("optimizers.differential_evolution")
            self.optimizer_class = getattr(de_module, "DifferentialEvolution")
            logger.info("Successfully imported DifferentialEvolution optimizer")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import DifferentialEvolution optimizer: {e}")
            self.optimizer_class = None
    
    def optimize(self, 
                function: Callable, 
                max_evaluations: int = 1000, 
                callback: Optional[Callable] = None,
                bounds: Optional[Tuple[float, float]] = None,
                dimension: int = 10,
                **kwargs) -> Dict[str, Any]:
        """Optimize a function using Differential Evolution.
        
        Args:
            function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            callback: Optional callback function
            bounds: Function bounds as (min, max)
            dimension: Problem dimension
            kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        if self.optimizer_class is None:
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": "DifferentialEvolution optimizer not available"
            }
        
        # Create optimizer instance
        optimizer_params = {**self.params, **kwargs}
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = (-5.0, 5.0)
        
        # Create optimizer
        optimizer = self.optimizer_class(
            dimension=dimension,
            bounds=[bounds[0], bounds[1]],
            **optimizer_params
        )
        
        # Prepare callback wrapper
        def callback_wrapper(iteration, best_fitness, best_position, num_evaluations, **kwargs):
            """Wrap optimizer callback to match benchmark service format."""
            if callback:
                should_terminate = callback(
                    iteration=iteration,
                    fitness=best_fitness,
                    position=best_position,
                    evaluations=num_evaluations,
                    **kwargs
                )
                return should_terminate
            return False
        
        # Run optimization
        try:
            result = optimizer.optimize(
                function,
                max_evaluations=max_evaluations,
                callback=callback_wrapper if callback else None
            )
            
            # Format result for benchmark service
            return {
                "best_fitness": float(result.get("best_fitness", float('inf'))),
                "best_position": result.get("best_position", np.zeros(dimension)),
                "evaluations": result.get("evaluations", 0),
                "success": True,
                "convergence": result.get("convergence", [])
            }
        
        except Exception as e:
            logger.exception(f"Error running DifferentialEvolution: {e}")
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": str(e)
            }


class EvolutionStrategyAdapter(OptimizerAdapter):
    """Adapter for Evolution Strategy optimizer."""
    
    def __init__(self, **kwargs):
        """Initialize ES adapter.
        
        Args:
            kwargs: Additional parameters for ES
        """
        super().__init__("EvolutionStrategy")
        self.params = kwargs
        
        # Try to import the ES optimizer
        try:
            es_module = importlib.import_module("optimizers.evolution_strategy")
            self.optimizer_class = getattr(es_module, "EvolutionStrategy")
            logger.info("Successfully imported EvolutionStrategy optimizer")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import EvolutionStrategy optimizer: {e}")
            self.optimizer_class = None
    
    def optimize(self, 
                function: Callable, 
                max_evaluations: int = 1000, 
                callback: Optional[Callable] = None,
                bounds: Optional[Tuple[float, float]] = None,
                dimension: int = 10,
                **kwargs) -> Dict[str, Any]:
        """Optimize a function using Evolution Strategy.
        
        Args:
            function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            callback: Optional callback function
            bounds: Function bounds as (min, max)
            dimension: Problem dimension
            kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        if self.optimizer_class is None:
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": "EvolutionStrategy optimizer not available"
            }
        
        # Create optimizer instance
        optimizer_params = {**self.params, **kwargs}
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = (-5.0, 5.0)
        
        # Create optimizer
        optimizer = self.optimizer_class(
            dimension=dimension,
            bounds=[bounds[0], bounds[1]],
            **optimizer_params
        )
        
        # Prepare callback wrapper
        def callback_wrapper(iteration, best_fitness, best_position, num_evaluations, **kwargs):
            """Wrap optimizer callback to match benchmark service format."""
            if callback:
                should_terminate = callback(
                    iteration=iteration,
                    fitness=best_fitness,
                    position=best_position,
                    evaluations=num_evaluations,
                    **kwargs
                )
                return should_terminate
            return False
        
        # Run optimization
        try:
            result = optimizer.optimize(
                function,
                max_evaluations=max_evaluations,
                callback=callback_wrapper if callback else None
            )
            
            # Format result for benchmark service
            return {
                "best_fitness": float(result.get("best_fitness", float('inf'))),
                "best_position": result.get("best_position", np.zeros(dimension)),
                "evaluations": result.get("evaluations", 0),
                "success": True,
                "convergence": result.get("convergence", [])
            }
        
        except Exception as e:
            logger.exception(f"Error running EvolutionStrategy: {e}")
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": str(e)
            }


class AntColonyOptimizerAdapter(OptimizerAdapter):
    """Adapter for Ant Colony Optimization."""
    
    def __init__(self, **kwargs):
        """Initialize ACO adapter.
        
        Args:
            kwargs: Additional parameters for ACO
        """
        super().__init__("AntColonyOptimization")
        self.params = kwargs
        
        # Try to import the ACO optimizer
        try:
            aco_module = importlib.import_module("optimizers.aco")
            self.optimizer_class = getattr(aco_module, "AntColonyOptimization")
            logger.info("Successfully imported AntColonyOptimization optimizer")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import AntColonyOptimization optimizer: {e}")
            self.optimizer_class = None
    
    def optimize(self, 
                function: Callable, 
                max_evaluations: int = 1000, 
                callback: Optional[Callable] = None,
                bounds: Optional[Tuple[float, float]] = None,
                dimension: int = 10,
                **kwargs) -> Dict[str, Any]:
        """Optimize a function using Ant Colony Optimization.
        
        Args:
            function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            callback: Optional callback function
            bounds: Function bounds as (min, max)
            dimension: Problem dimension
            kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        # Implementation similar to DifferentialEvolutionAdapter
        if self.optimizer_class is None:
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": "AntColonyOptimization optimizer not available"
            }
        
        # Create optimizer instance
        optimizer_params = {**self.params, **kwargs}
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = (-5.0, 5.0)
        
        # Create optimizer
        optimizer = self.optimizer_class(
            dimension=dimension,
            bounds=[bounds[0], bounds[1]],
            **optimizer_params
        )
        
        # Prepare callback wrapper
        def callback_wrapper(iteration, best_fitness, best_position, num_evaluations, **kwargs):
            """Wrap optimizer callback to match benchmark service format."""
            if callback:
                should_terminate = callback(
                    iteration=iteration,
                    fitness=best_fitness,
                    position=best_position,
                    evaluations=num_evaluations,
                    **kwargs
                )
                return should_terminate
            return False
        
        # Run optimization
        try:
            result = optimizer.optimize(
                function,
                max_evaluations=max_evaluations,
                callback=callback_wrapper if callback else None
            )
            
            # Format result for benchmark service
            return {
                "best_fitness": float(result.get("best_fitness", float('inf'))),
                "best_position": result.get("best_position", np.zeros(dimension)),
                "evaluations": result.get("evaluations", 0),
                "success": True,
                "convergence": result.get("convergence", [])
            }
        
        except Exception as e:
            logger.exception(f"Error running AntColonyOptimization: {e}")
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": str(e)
            }


class GreyWolfOptimizerAdapter(OptimizerAdapter):
    """Adapter for Grey Wolf Optimizer."""
    
    def __init__(self, **kwargs):
        """Initialize GWO adapter.
        
        Args:
            kwargs: Additional parameters for GWO
        """
        super().__init__("GreyWolfOptimizer")
        self.params = kwargs
        
        # Try to import the GWO optimizer
        try:
            gwo_module = importlib.import_module("optimizers.gwo")
            self.optimizer_class = getattr(gwo_module, "GreyWolfOptimizer")
            logger.info("Successfully imported GreyWolfOptimizer optimizer")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import GreyWolfOptimizer optimizer: {e}")
            self.optimizer_class = None
    
    def optimize(self, 
                function: Callable, 
                max_evaluations: int = 1000, 
                callback: Optional[Callable] = None,
                bounds: Optional[Tuple[float, float]] = None,
                dimension: int = 10,
                **kwargs) -> Dict[str, Any]:
        """Optimize a function using Grey Wolf Optimizer.
        
        Args:
            function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            callback: Optional callback function
            bounds: Function bounds as (min, max)
            dimension: Problem dimension
            kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        # Implementation similar to DifferentialEvolutionAdapter
        if self.optimizer_class is None:
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": "GreyWolfOptimizer optimizer not available"
            }
        
        # Create optimizer instance
        optimizer_params = {**self.params, **kwargs}
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = (-5.0, 5.0)
        
        # Create optimizer
        optimizer = self.optimizer_class(
            dimension=dimension,
            bounds=[bounds[0], bounds[1]],
            **optimizer_params
        )
        
        # Prepare callback wrapper
        def callback_wrapper(iteration, best_fitness, best_position, num_evaluations, **kwargs):
            """Wrap optimizer callback to match benchmark service format."""
            if callback:
                should_terminate = callback(
                    iteration=iteration,
                    fitness=best_fitness,
                    position=best_position,
                    evaluations=num_evaluations,
                    **kwargs
                )
                return should_terminate
            return False
        
        # Run optimization
        try:
            result = optimizer.optimize(
                function,
                max_evaluations=max_evaluations,
                callback=callback_wrapper if callback else None
            )
            
            # Format result for benchmark service
            return {
                "best_fitness": float(result.get("best_fitness", float('inf'))),
                "best_position": result.get("best_position", np.zeros(dimension)),
                "evaluations": result.get("evaluations", 0),
                "success": True,
                "convergence": result.get("convergence", [])
            }
        
        except Exception as e:
            logger.exception(f"Error running GreyWolfOptimizer: {e}")
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": str(e)
            }


class MetaOptimizerAdapter(OptimizerAdapter):
    """Adapter for Meta Optimizer."""
    
    def __init__(self, **kwargs):
        """Initialize Meta Optimizer adapter.
        
        Args:
            kwargs: Additional parameters for Meta Optimizer
        """
        super().__init__("MetaOptimizer")
        self.params = kwargs
        self.base_optimizers = {}
        self.selected_optimizers = []
        
        # Try to import the Meta Optimizer
        try:
            meta_module = importlib.import_module("meta.meta_optimizer")
            self.optimizer_class = getattr(meta_module, "MetaOptimizer")
            logger.info("Successfully imported MetaOptimizer")
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import MetaOptimizer: {e}")
            self.optimizer_class = None
    
    def set_base_optimizers(self, optimizers: Dict[str, OptimizerAdapter]):
        """Set base optimizers for the meta-optimizer.
        
        Args:
            optimizers: Dictionary of optimizer adapters keyed by ID
        """
        self.base_optimizers = optimizers
    
    def get_selected_optimizers(self) -> List[str]:
        """Get list of selected optimizers.
        
        Returns:
            List of selected optimizer IDs
        """
        return self.selected_optimizers
    
    def optimize(self, 
                function: Callable, 
                max_evaluations: int = 1000, 
                callback: Optional[Callable] = None,
                bounds: Optional[Tuple[float, float]] = None,
                dimension: int = 10,
                **kwargs) -> Dict[str, Any]:
        """Optimize a function using Meta Optimizer.
        
        Args:
            function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            callback: Optional callback function
            bounds: Function bounds as (min, max)
            dimension: Problem dimension
            kwargs: Additional parameters
            
        Returns:
            Dictionary with optimization results
        """
        if self.optimizer_class is None:
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": "MetaOptimizer not available"
            }
        
        # Create optimizer instance
        optimizer_params = {**self.params, **kwargs}
        
        # Set default bounds if not provided
        if bounds is None:
            bounds = (-5.0, 5.0)
        
        # Create optimizer
        optimizer = self.optimizer_class(
            dimension=dimension,
            bounds=[bounds[0], bounds[1]],
            **optimizer_params
        )
        
        # Set base optimizers if available
        if hasattr(optimizer, "set_optimizers") and self.base_optimizers:
            # Extract actual optimizer instances from adapters
            optimizer_instances = {}
            for optimizer_id, adapter in self.base_optimizers.items():
                if hasattr(adapter, "optimizer_class") and adapter.optimizer_class is not None:
                    optimizer_instances[optimizer_id] = adapter.optimizer_class(
                        dimension=dimension,
                        bounds=[bounds[0], bounds[1]],
                        **adapter.params
                    )
            
            optimizer.set_optimizers(optimizer_instances)
        
        # Prepare callback wrapper
        def callback_wrapper(iteration, best_fitness, best_position, num_evaluations, **kwargs):
            """Wrap optimizer callback to match benchmark service format."""
            # Track selected optimizers if available
            if hasattr(optimizer, "get_selected_optimizer"):
                selected = optimizer.get_selected_optimizer()
                if selected:
                    self.selected_optimizers.append(selected)
            
            if callback:
                should_terminate = callback(
                    iteration=iteration,
                    fitness=best_fitness,
                    position=best_position,
                    evaluations=num_evaluations,
                    **kwargs
                )
                return should_terminate
            return False
        
        # Run optimization
        try:
            result = optimizer.optimize(
                function,
                max_evaluations=max_evaluations,
                callback=callback_wrapper if callback else None
            )
            
            # Format result for benchmark service
            formatted_result = {
                "best_fitness": float(result.get("best_fitness", float('inf'))),
                "best_position": result.get("best_position", np.zeros(dimension)),
                "evaluations": result.get("evaluations", 0),
                "success": True,
                "convergence": result.get("convergence", [])
            }
            
            # Add selected optimizers if available
            if self.selected_optimizers:
                formatted_result["selected_optimizers"] = self.selected_optimizers
            
            return formatted_result
        
        except Exception as e:
            logger.exception(f"Error running MetaOptimizer: {e}")
            return {
                "best_fitness": float('inf'),
                "best_position": np.zeros(dimension),
                "evaluations": 0,
                "success": False,
                "error": str(e)
            }


class OptimizerFactory:
    """Factory for creating optimizer adapters."""
    
    @staticmethod
    def create_optimizer(optimizer_type: str, **kwargs) -> OptimizerAdapter:
        """Create optimizer adapter by type.
        
        Args:
            optimizer_type: Type of optimizer
            kwargs: Additional parameters for optimizer
            
        Returns:
            Optimizer adapter instance
            
        Raises:
            ValueError: If optimizer type is unknown
        """
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type in ["de", "differential_evolution"]:
            return DifferentialEvolutionAdapter(**kwargs)
        elif optimizer_type in ["es", "evolution_strategy"]:
            return EvolutionStrategyAdapter(**kwargs)
        elif optimizer_type in ["aco", "ant_colony"]:
            return AntColonyOptimizerAdapter(**kwargs)
        elif optimizer_type in ["gwo", "grey_wolf"]:
            return GreyWolfOptimizerAdapter(**kwargs)
        elif optimizer_type in ["meta", "meta_optimizer"]:
            return MetaOptimizerAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_optimizers() -> Dict[str, OptimizerAdapter]:
        """Create all available optimizer adapters.
        
        Returns:
            Dictionary of optimizer adapters keyed by ID
        """
        return {
            "de": DifferentialEvolutionAdapter(),
            "es": EvolutionStrategyAdapter(),
            "aco": AntColonyOptimizerAdapter(),
            "gwo": GreyWolfOptimizerAdapter()
        } 
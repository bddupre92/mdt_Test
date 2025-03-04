"""
Optimizer state tracking module for monitoring optimization progress and behavior.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class OptimizerState:
    """Data class for storing optimizer state information."""
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict = field(default_factory=dict)
    fitness: float = float('inf')
    generation: int = 0
    gradient: Optional[np.ndarray] = None
    landscape_metrics: Dict = field(default_factory=dict)
    convergence_metrics: Dict = field(default_factory=dict)

class OptimizerStateTracker:
    """Tracks and analyzes optimizer state over time."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.state_history: List[OptimizerState] = []
        self.parameter_history: Dict[str, List[float]] = {}
        self.fitness_history: List[float] = []
        self.gradient_history: List[np.ndarray] = []
        
    def update_state(self, 
                    parameters: Dict,
                    fitness: float,
                    generation: int,
                    gradient: Optional[np.ndarray] = None) -> None:
        """Update optimizer state with new information."""
        # Calculate landscape metrics
        landscape_metrics = self._analyze_landscape(parameters, fitness, gradient)
        
        # Calculate convergence metrics
        convergence_metrics = self._analyze_convergence(fitness)
        
        # Create new state
        state = OptimizerState(
            parameters=parameters,
            fitness=fitness,
            generation=generation,
            gradient=gradient,
            landscape_metrics=landscape_metrics,
            convergence_metrics=convergence_metrics
        )
        
        # Update histories
        self.state_history.append(state)
        self._update_parameter_history(parameters)
        self.fitness_history.append(fitness)
        if gradient is not None:
            self.gradient_history.append(gradient)
            
        # Trim histories if needed
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
            self.fitness_history.pop(0)
            if gradient is not None:
                self.gradient_history.pop(0)
    
    def _update_parameter_history(self, parameters: Dict) -> None:
        """Update parameter adaptation history."""
        for param_name, value in parameters.items():
            if param_name not in self.parameter_history:
                self.parameter_history[param_name] = []
            self.parameter_history[param_name].append(value)
            if len(self.parameter_history[param_name]) > self.window_size:
                self.parameter_history[param_name].pop(0)
    
    def _analyze_landscape(self, 
                         parameters: Dict,
                         fitness: float,
                         gradient: Optional[np.ndarray]) -> Dict:
        """Analyze optimization landscape characteristics."""
        metrics = {}
        
        # Calculate ruggedness using fitness history
        if len(self.fitness_history) > 1:
            fitness_changes = np.diff(self.fitness_history)
            metrics['ruggedness'] = np.std(fitness_changes)
            metrics['modality'] = len([1 for fc in fitness_changes if fc < 0])
        
        # Estimate gradient properties if available
        if gradient is not None and len(self.gradient_history) > 0:
            prev_gradient = self.gradient_history[-1]
            metrics['gradient_variance'] = np.var(gradient)
            metrics['gradient_alignment'] = np.dot(gradient, prev_gradient) / (
                np.linalg.norm(gradient) * np.linalg.norm(prev_gradient)
            )
        
        return metrics
    
    def _analyze_convergence(self, current_fitness: float) -> Dict:
        """Analyze convergence metrics."""
        metrics = {}
        
        if len(self.fitness_history) > 1:
            # Calculate improvement rate
            improvements = np.diff(self.fitness_history)
            metrics['improvement_rate'] = np.mean(improvements < 0)
            
            # Calculate convergence speed
            metrics['convergence_speed'] = abs(
                np.mean(improvements) / np.std(improvements)
                if np.std(improvements) > 0 else 0
            )
            
            # Check for stagnation
            recent_improvements = improvements[-min(10, len(improvements)):]
            metrics['stagnating'] = all(abs(imp) < 1e-6 for imp in recent_improvements)
            
        return metrics
    
    def get_state_summary(self) -> Dict:
        """Get summary of current optimization state."""
        if not self.state_history:
            return {}
            
        current_state = self.state_history[-1]
        
        return {
            'current_fitness': current_state.fitness,
            'generation': current_state.generation,
            'landscape_metrics': current_state.landscape_metrics,
            'convergence_metrics': current_state.convergence_metrics,
            'parameter_adaptation': {
                name: {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for name, values in self.parameter_history.items()
            }
        }

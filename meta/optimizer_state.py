"""
optimizer_state.py
----------------
State management for optimizers.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class OptimizerState:
    """Store optimizer state information"""
    evaluations: int
    runtime: float
    history: List[Tuple[int, float]]  # (evaluation_count, best_score)
    success_rate: Optional[float] = None
    diversity_history: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            'evaluations': self.evaluations,
            'runtime': self.runtime,
            'history': self.history,
            'success_rate': self.success_rate,
            'diversity_history': self.diversity_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizerState':
        """Create state from dictionary."""
        return cls(**data)

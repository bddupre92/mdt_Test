"""
Module for tracking and analyzing optimization history.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import os
from collections import defaultdict


class OptimizationHistory:
    """Class for tracking optimization history and learning from past performance."""
    
    def __init__(self, history_file: Optional[str] = None):
        """
        Initialize optimization history.
        
        Args:
            history_file: Optional path to save/load history
        """
        self.records = []
        self.history_file = history_file
        if history_file and Path(history_file).exists():
            self.load_history()
    
    def _convert_to_native_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_native_types(obj.tolist())
        elif isinstance(obj, dict):
            return {k: self._convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(x) for x in obj]
        return obj
            
    def add_record(self, features: Dict[str, float], optimizer: str, 
                  performance: float, success: bool = False) -> None:
        """
        Add a new optimization record.
        
        Args:
            features: Problem features dictionary
            optimizer: Name of optimizer used
            performance: Final performance achieved
            success: Whether optimization was successful
        """
        record = {
            'features': self._convert_to_native_types(features),
            'optimizer': optimizer,
            'performance': float(performance),
            'success': bool(success)
        }
        self.records.append(record)
        
        # Save after each update
        if self.history_file:
            self.save_history()
    
    def find_similar_problems(self, features: Dict[str, float], 
                            n_neighbors: int = 3) -> Optional[List[Tuple[float, Dict]]]:
        """
        Find similar problems based on features.
        
        Args:
            features: Problem features to match
            n_neighbors: Number of similar problems to return
            
        Returns:
            List of (similarity_score, record) tuples for most similar problems
        """
        if not self.records:
            return None
            
        # Calculate similarity scores
        similarities = []
        for record in self.records:
            sim_score = self._calculate_similarity(features, record['features'])
            similarities.append((sim_score, record))
            
        # Get top N similar problems
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:n_neighbors]
    
    def get_optimizer_performance(self, optimizer: str) -> Dict[str, float]:
        """
        Get performance statistics for a specific optimizer.
        
        Args:
            optimizer: Name of optimizer
            
        Returns:
            Dictionary with performance statistics
        """
        performances = [r['performance'] for r in self.records 
                       if r['optimizer'] == optimizer]
        
        if not performances:
            return {'mean': 0.0, 'std': 0.0, 'success_rate': 0.0}
            
        return {
            'mean': float(np.mean(performances)),
            'std': float(np.std(performances)),
            'success_rate': sum(1 for r in self.records 
                              if r['optimizer'] == optimizer and r['success']) / len(performances)
        }
    
    def save_history(self) -> None:
        """Save history to file."""
        if not self.history_file:
            return
            
        # Ensure directory exists
        Path(self.history_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.history_file, 'w') as f:
            json.dump([self._convert_to_native_types(record) for record in self.records], f, indent=2)
    
    def load_history(self) -> None:
        """Load history from file."""
        if not self.history_file or not os.path.exists(self.history_file):
            return
            
        with open(self.history_file, 'r') as f:
            data = json.load(f)
            self.records = data
    
    def _calculate_similarity(self, features1: Dict[str, float], 
                            features2: Dict[str, float]) -> float:
        """
        Calculate similarity score between two feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
            
        # Calculate normalized Euclidean distance for common features
        squared_diff = 0.0
        for feature in common_features:
            # Normalize values to [0, 1] range assuming reasonable bounds
            v1 = min(1.0, max(0.0, float(features1[feature])))
            v2 = min(1.0, max(0.0, float(features2[feature])))
            squared_diff += (v1 - v2) ** 2
            
        distance = np.sqrt(squared_diff / len(common_features))
        
        # Convert distance to similarity score
        return 1.0 / (1.0 + distance)

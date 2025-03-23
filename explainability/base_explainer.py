"""
base_explainer.py
----------------
Base class for all explainability methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json


class BaseExplainer(ABC):
    """Abstract base class for all explainability methods"""
    
    def __init__(self, name: str, model=None, feature_names: Optional[List[str]] = None):
        """
        Initialize base explainer
        
        Args:
            name: Name of the explainer
            model: Pre-trained model to explain
            feature_names: List of feature names
        """
        self.name = name
        self.model = model
        self.feature_names = feature_names
        self.explanation_data = {}
        self.supported_plot_types = []
        self.last_explanation = None
    
    @abstractmethod
    def explain(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Generate explanation for the provided data
        
        Args:
            X: Input features to explain
            y: Optional target values
            **kwargs: Additional parameters for specific explainers
            
        Returns:
            Dictionary containing explanation data
        """
        pass
    
    @abstractmethod
    def plot(self, plot_type: str = 'summary', **kwargs) -> plt.Figure:
        """
        Create visualization of the explanation
        
        Args:
            plot_type: Type of plot to generate
            **kwargs: Additional parameters for specific plot types
            
        Returns:
            Matplotlib figure object
        """
        pass
    
    def set_model(self, model):
        """Set model to explain"""
        self.model = model
        return self
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names"""
        self.feature_names = feature_names
        return self
    
    def get_supported_plot_types(self) -> List[str]:
        """Get list of supported plot types"""
        return self.supported_plot_types
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from explanation
        
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not self.last_explanation:
            raise ValueError("No explanation available. Run explain() first.")
        
        return self.last_explanation.get('feature_importance', {})
    
    def save_explanation(self, filepath: Path):
        """
        Save explanation data to file
        
        Args:
            filepath: Path to save explanation to
        """
        if not self.last_explanation:
            raise ValueError("No explanation available. Run explain() first.")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in self.last_explanation.items():
            if key == 'explanations' and hasattr(value, '__iter__'):
                # Skip LIME explanation objects - they can't be serialized
                serializable_data[key] = "LIME explanation objects (not serializable)"
                continue
            elif key == 'shap_values':
                # Handle SHAP values specially
                if isinstance(value, list):
                    # For multi-class models
                    serializable_data[key] = [arr.tolist() if isinstance(arr, np.ndarray) else str(arr) for arr in value]
                elif isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = str(value)
                continue
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, dict):
                # Handle nested dictionaries
                serializable_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_dict[k] = v.tolist()
                    elif hasattr(v, 'tolist'):
                        serializable_dict[k] = v.tolist()
                    elif hasattr(v, '__class__'):
                        # Convert complex objects to string representation
                        serializable_dict[k] = str(v)
                    else:
                        serializable_dict[k] = v
                serializable_data[key] = serializable_dict
            elif hasattr(value, '__class__') and value.__class__.__name__ == 'Explanation':
                # Handle LIME Explanation objects
                try:
                    serializable_data[key] = {
                        'intercept': float(value.intercept[0]) if hasattr(value, 'intercept') else 0.0,
                        'score': float(value.score) if hasattr(value, 'score') else 0.0,
                        'local_exp': {str(k): [(int(i), float(v)) for i, v in exp] 
                                     for k, exp in value.local_exp.items()} if hasattr(value, 'local_exp') else {},
                        'feature_names': value.domain_mapper.feature_names if hasattr(value, 'domain_mapper') and 
                                        hasattr(value.domain_mapper, 'feature_names') else []
                    }
                except Exception as e:
                    # If serialization fails, store a string representation
                    serializable_data[key] = f"LIME explanation (not serializable): {str(e)}"
            elif hasattr(value, 'tolist'):
                # Handle other array-like objects
                serializable_data[key] = value.tolist()
            elif hasattr(value, '__dict__'):
                # Handle custom objects by converting to dict
                try:
                    serializable_data[key] = {k: v for k, v in value.__dict__.items() 
                                            if not k.startswith('_') and not callable(v)}
                except:
                    serializable_data[key] = str(value)
            else:
                serializable_data[key] = value
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save explanation to {filepath}: {str(e)}")
            # Create a simplified version with just the feature importance
            simplified_data = {
                'feature_importance': serializable_data.get('feature_importance', {}),
                'error': f"Full explanation could not be serialized: {str(e)}"
            }
            with open(filepath, 'w') as f:
                json.dump(simplified_data, f, indent=2)
    
    def load_explanation(self, filepath: str):
        """
        Load explanation data from file
        
        Args:
            filepath: Path to load the explanation data from
        """
        import json
        
        with open(filepath, 'r') as f:
            self.last_explanation = json.load(f)
            
        # Convert lists back to numpy arrays
        for key, value in self.last_explanation.items():
            if isinstance(value, list):
                self.last_explanation[key] = np.array(value)
            elif isinstance(value, dict):
                self.last_explanation[key] = {
                    k: np.array(v) if isinstance(v, list) else v
                    for k, v in value.items()
                }
        
        return self.last_explanation

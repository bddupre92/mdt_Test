"""
Model management utilities for migraine prediction models.
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, TypeVar, Union

# Type variable for generic class methods
T = TypeVar('T')

class ModelManager:
    """
    Handles the saving and loading of migraine prediction models.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelManager.
        
        Args:
            model_dir: Directory to store models in
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.default_model_path = self.model_dir / "default_model.json"
        
        # Create default model config if it doesn't exist
        if not self.default_model_path.exists():
            with open(self.default_model_path, 'w') as f:
                json.dump({"default_model": None}, f)
    
    def save_model(self, model: Any, name: str, description: str = "", make_default: bool = False, metadata: Dict[str, Any] = None) -> str:
        """
        Save a model to disk.
        
        Args:
            model: The model to save
            name: Name of the model
            description: Optional description
            make_default: Whether to make this the default model
            metadata: Additional metadata to store with the model
            
        Returns:
            Model ID
        """
        # Generate ID
        model_id = f"{name}_{int(time.time())}"
        
        # Create model directory
        model_path = self.model_dir / model_id
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model using pickle
        model_file = model_path / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        base_metadata = {
            "id": model_id,
            "name": name,
            "description": description,
            "created_at": time.time(),
            "version": "1.0.0"
        }
        
        # Merge custom metadata if provided
        if metadata:
            base_metadata.update(metadata)
        
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(base_metadata, f)
        
        # Set as default if requested
        if make_default:
            self._set_default_model(model_id)
        
        return model_id
    
    def load_model(self, model_id: Optional[str] = None) -> tuple[Any, Dict[str, Any]]:
        """
        Load a model from disk.
        
        Args:
            model_id: ID of the model to load, or None for default
            
        Returns:
            Tuple of (loaded model, metadata dictionary)
        """
        # If no model ID provided, use default
        if model_id is None:
            model_id = self._get_default_model()
            if model_id is None:
                raise ValueError("No default model found")
        
        # Load model from pickle
        model_path = self.model_dir / model_id
        model_file = model_path / "model.pkl"
        metadata_file = model_path / "metadata.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        
        # Load the model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load the metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "id": model_id,
                "name": "Unknown",
                "description": "",
                "created_at": 0,
                "version": "unknown"
            }
        
        return model, metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models.
        
        Returns:
            List of model metadata dictionaries
        """
        models = []
        default_model = self._get_default_model()
        
        # Iterate through model directories
        for path in self.model_dir.iterdir():
            if path.is_dir():
                metadata_file = path / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        metadata["is_default"] = metadata["id"] == default_model
                        models.append(metadata)
        
        # Sort by creation time
        return sorted(models, key=lambda m: m["created_at"], reverse=True)
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        model_path = self.model_dir / model_id
        
        if not model_path.exists():
            return False
        
        # Remove model directory
        for file in model_path.iterdir():
            file.unlink()
        model_path.rmdir()
        
        # Update default model if needed
        default_model = self._get_default_model()
        if default_model == model_id:
            self._set_default_model(None)
        
        return True
    
    def get_model_version(self, model_id: Optional[str] = None) -> str:
        """
        Get the version of a model.
        
        Args:
            model_id: ID of the model, or None for default
            
        Returns:
            Model version string
        """
        if model_id is None:
            model_id = self._get_default_model()
            if model_id is None:
                return "no_model"
        
        metadata_file = self.model_dir / model_id / "metadata.json"
        
        if not metadata_file.exists():
            return "unknown"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            return metadata.get("version", "unknown")
    
    def _get_default_model(self) -> Optional[str]:
        """Get the default model ID."""
        if not self.default_model_path.exists():
            return None
        
        with open(self.default_model_path, 'r') as f:
            data = json.load(f)
            return data.get("default_model")
    
    def _set_default_model(self, model_id: Optional[str]):
        """Set the default model ID."""
        with open(self.default_model_path, 'w') as f:
            json.dump({"default_model": model_id}, f)

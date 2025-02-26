"""
model_manager.py
--------------
Manages saving and loading of MetaOptimizer models.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import shutil
from .meta_optimizer import MetaOptimizer
from .optimization_history import OptimizationHistory
from optimizers.optimizer_factory import create_optimizers

class ModelManager:
    """Manages saving and loading of MetaOptimizer models."""
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize ModelManager.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.config_file = self.base_path / "model_config.json"
        self.load_config()
        
    def load_config(self):
        """Load or create configuration file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'models': {},
                'default_model': None
            }
            self.save_config()
            
    def save_config(self):
        """Save configuration file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def save_model(self, 
                  model: MetaOptimizer, 
                  name: str,
                  description: str = "",
                  make_default: bool = False) -> str:
        """
        Save a MetaOptimizer model.
        
        Args:
            model: MetaOptimizer instance
            name: Name for the model
            description: Optional description
            make_default: Whether to make this the default model
            
        Returns:
            Model ID
        """
        # Create unique model ID
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"{name}_{timestamp}"
        model_dir = self.base_path / model_id
        
        # If directory exists, try with a different timestamp
        counter = 1
        while model_dir.exists():
            model_id = f"{name}_{timestamp}_{counter}"
            model_dir = self.base_path / model_id
            counter += 1
            
        model_dir.mkdir(parents=True)
        
        # Save model state
        model_state = {
            'dim': model.dim,
            'bounds': model.bounds,
            'n_parallel': model.n_parallel,
            'history_file': str(model_dir / 'history.json'),
            'selection_file': str(model_dir / 'selection.json')
        }
        
        # Save optimizer states
        optimizer_states = {}
        for opt_name, opt in model.optimizers.items():
            if hasattr(opt, 'get_state'):
                state = opt.get_state()
                optimizer_states[opt_name] = state.to_dict()
        
        # Save model metadata
        metadata = {
            'id': model_id,
            'name': name,
            'description': description,
            'created_at': datetime.datetime.now().isoformat(),
            'model_state': model_state,
            'optimizer_states': optimizer_states
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save history and selection data
        if hasattr(model, 'history'):
            with open(model_dir / 'history.json', 'w') as f:
                json.dump(model.history.to_dict(), f, indent=2)
                
        if hasattr(model, 'selection_history'):
            with open(model_dir / 'selection.json', 'w') as f:
                json.dump(model.selection_history.to_dict(), f, indent=2)
        
        # Make default if requested
        if make_default:
            default_path = self.base_path / 'default'
            try:
                if default_path.exists():
                    if default_path.is_symlink():
                        default_path.unlink()
                    else:
                        shutil.rmtree(default_path)
                default_path.symlink_to(model_dir)
            except FileExistsError:
                # If we can't create the symlink, try to force remove and retry
                try:
                    os.remove(str(default_path))
                    default_path.symlink_to(model_dir)
                except (OSError, FileExistsError) as e:
                    print(f"Warning: Could not create default symlink: {e}")
            
        # Update config
        self.config['models'][model_id] = {
            'name': name,
            'description': description,
            'created': datetime.datetime.now().isoformat(),
            'path': str(model_dir)
        }
        
        if make_default or not self.config['default_model']:
            self.config['default_model'] = model_id
            
        self.save_config()
        return model_id
    
    def load_model(self, model_id: Optional[str] = None) -> MetaOptimizer:
        """
        Load a MetaOptimizer model.
        
        Args:
            model_id: ID of model to load, or None for default
            
        Returns:
            MetaOptimizer instance
        """
        # Get model directory
        if model_id is None:
            model_dir = self.base_path / 'default'
            if not model_dir.exists():
                raise ValueError("No default model found")
            if model_dir.is_symlink():
                model_dir = model_dir.resolve()
        else:
            model_dir = self.base_path / model_id
            if not model_dir.exists():
                raise ValueError(f"Model {model_id} not found")
                
        # Load metadata
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Create optimizers
        optimizers = create_optimizers(
            dim=metadata['model_state']['dim'],
            bounds=metadata['model_state']['bounds']
        )
            
        # Create new model instance
        model = MetaOptimizer(
            dim=metadata['model_state']['dim'],
            bounds=metadata['model_state']['bounds'],
            optimizers=optimizers,
            n_parallel=metadata['model_state']['n_parallel']
        )
        
        # Load history if present
        if (model_dir / 'history.json').exists():
            with open(model_dir / 'history.json', 'r') as f:
                model.history = OptimizationHistory.from_dict(json.load(f))
                
        # Load selection history if present
        if (model_dir / 'selection.json').exists():
            with open(model_dir / 'selection.json', 'r') as f:
                model.selection_history = OptimizationHistory.from_dict(json.load(f))
                
        # Restore optimizer states
        for opt_name, opt_state in metadata['optimizer_states'].items():
            if opt_name in model.optimizers and hasattr(model.optimizers[opt_name], 'set_state'):
                state = OptimizerState.from_dict(opt_state)
                model.optimizers[opt_name].set_state(state)
                
        return model
    
    def list_models(self) -> Dict[str, Any]:
        """List all saved models."""
        return {
            model_id: info
            for model_id, info in self.config['models'].items()
        }
    
    def delete_model(self, model_id: str):
        """Delete a saved model."""
        if model_id not in self.config['models']:
            raise ValueError(f"Model {model_id} not found")
            
        # Remove model files
        model_dir = Path(self.config['models'][model_id]['path'])
        for file in model_dir.glob('*'):
            file.unlink()
        model_dir.rmdir()
        
        # Update config
        if self.config['default_model'] == model_id:
            self.config['default_model'] = None
        del self.config['models'][model_id]
        self.save_config()

    def get_model_version(self) -> str:
        """
        Get current model version.
        
        Returns:
            Model version string
        """
        default_path = self.base_path / 'default'
        if not default_path.exists():
            raise ValueError("No default model found")
            
        if default_path.is_symlink():
            target = default_path.resolve()
            return target.name
            
        raise ValueError("Default model path is not a symlink")

"""
Tests for the State Management components.

This test suite covers:
1. StateManager interface implementation
2. FileSystemStateManager functionality
3. Checkpoint saving and loading
4. Error handling for state operations
"""

import os
import json
import shutil
import tempfile
import pytest
from unittest.mock import Mock, patch

from moe_framework.persistence.state_manager import (
    StateManager,
    FileSystemStateManager
)


class TestFileSystemStateManager:
    """Tests for the FileSystemStateManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = FileSystemStateManager(
            checkpoint_dir=self.temp_dir,
            max_checkpoints=3
        )
        
        # Sample state for testing
        self.test_state = {
            "pipeline_id": "test_pipeline",
            "experts": {
                "expert1": {"trained": True},
                "expert2": {"trained": False}
            },
            "gating_network": {"weights": {"expert1": 0.7, "expert2": 0.3}},
            "metadata": {
                "timestamp": "2025-03-25T09:00:00",
                "version": "1.0.0"
            }
        }
    
    def teardown_method(self):
        """Tear down test fixtures after each test method."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_state(self):
        """Test basic save and load functionality."""
        # Save state
        checkpoint_path = self.state_manager.save_state(self.test_state)
        
        # Verify file exists
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.startswith(self.temp_dir)
        
        # Load state
        loaded_state = self.state_manager.load_state(checkpoint_path)
        
        # Verify state matches
        assert loaded_state == self.test_state
    
    def test_checkpoint_naming_convention(self):
        """Test that checkpoints follow expected naming convention."""
        # Save state
        checkpoint_path = self.state_manager.save_state(self.test_state)
        
        # Verify naming convention
        filename = os.path.basename(checkpoint_path)
        assert filename.startswith("checkpoint_")
        assert filename.endswith(".json")
        
        # Should contain a timestamp
        parts = filename.split("_")
        assert len(parts) > 2  # At least "checkpoint_TIMESTAMP.json"
    
    def test_max_checkpoints_limit(self):
        """Test that only max_checkpoints are kept."""
        # Create more than max_checkpoints
        paths = []
        for i in range(5):  # More than max_checkpoints (3)
            state = self.test_state.copy()
            state["iteration"] = i
            path = self.state_manager.save_state(state)
            paths.append(path)
        
        # Verify only the most recent max_checkpoints exist
        existing_checkpoints = [f for f in os.listdir(self.temp_dir) 
                              if f.startswith("checkpoint_") and f.endswith(".json")]
        
        assert len(existing_checkpoints) == 3  # max_checkpoints
        
        # Verify the oldest checkpoints were removed
        assert not os.path.exists(paths[0])
        assert not os.path.exists(paths[1])
        assert os.path.exists(paths[2])
        assert os.path.exists(paths[3])
        assert os.path.exists(paths[4])
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading a non-existent checkpoint."""
        non_existent_path = os.path.join(self.temp_dir, "non_existent_checkpoint.json")
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            self.state_manager.load_state(non_existent_path)
    
    def test_save_to_nonexistent_directory(self):
        """Test saving to a non-existent directory."""
        # Setup state manager with non-existent directory
        non_existent_dir = os.path.join(self.temp_dir, "non_existent_subdir")
        state_manager = FileSystemStateManager(
            checkpoint_dir=non_existent_dir,
            max_checkpoints=3
        )
        
        # Should create the directory and save successfully
        checkpoint_path = state_manager.save_state(self.test_state)
        
        # Verify
        assert os.path.exists(non_existent_dir)
        assert os.path.exists(checkpoint_path)
    
    def test_save_invalid_state(self):
        """Test saving a state that can't be serialized to JSON."""
        # Create a state with a non-serializable object
        invalid_state = {
            "pipeline_id": "test_pipeline",
            "non_serializable": set([1, 2, 3])  # Sets are not JSON serializable
        }
        
        # Should raise TypeError
        with pytest.raises(TypeError):
            self.state_manager.save_state(invalid_state)
    
    def test_load_corrupted_checkpoint(self):
        """Test loading a corrupted checkpoint file."""
        # Create a corrupted checkpoint file
        corrupted_path = os.path.join(self.temp_dir, "corrupted_checkpoint.json")
        with open(corrupted_path, 'w') as f:
            f.write("{This is not valid JSON")
        
        # Should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            self.state_manager.load_state(corrupted_path)
    
    def test_list_checkpoints(self):
        """Test listing available checkpoints."""
        # Create several checkpoints
        for i in range(3):
            state = self.test_state.copy()
            state["iteration"] = i
            self.state_manager.save_state(state)
        
        # List checkpoints
        checkpoints = self.state_manager.list_checkpoints()
        
        # Verify
        assert len(checkpoints) == 3
        for checkpoint in checkpoints:
            assert os.path.exists(checkpoint)
            assert checkpoint.startswith(self.temp_dir)
            assert checkpoint.endswith(".json")
    
    def test_get_latest_checkpoint(self):
        """Test getting the latest checkpoint."""
        # Create several checkpoints with delays to ensure different timestamps
        paths = []
        for i in range(3):
            state = self.test_state.copy()
            state["iteration"] = i
            path = self.state_manager.save_state(state)
            paths.append(path)
        
        # Get latest checkpoint
        latest = self.state_manager.get_latest_checkpoint()
        
        # Verify it matches the last one created
        assert latest == paths[-1]
    
    def test_get_latest_checkpoint_empty(self):
        """Test getting the latest checkpoint when none exist."""
        # Don't create any checkpoints
        latest = self.state_manager.get_latest_checkpoint()
        
        # Should return None
        assert latest is None
    
    def test_delete_checkpoint(self):
        """Test deleting a specific checkpoint."""
        # Create a checkpoint
        checkpoint_path = self.state_manager.save_state(self.test_state)
        
        # Verify it exists
        assert os.path.exists(checkpoint_path)
        
        # Delete it
        self.state_manager.delete_checkpoint(checkpoint_path)
        
        # Verify it's gone
        assert not os.path.exists(checkpoint_path)
    
    def test_delete_all_checkpoints(self):
        """Test deleting all checkpoints."""
        # Create several checkpoints
        for i in range(3):
            state = self.test_state.copy()
            state["iteration"] = i
            self.state_manager.save_state(state)
        
        # Verify checkpoints exist
        assert len(os.listdir(self.temp_dir)) > 0
        
        # Delete all checkpoints
        self.state_manager.delete_all_checkpoints()
        
        # Verify all checkpoint files are gone
        checkpoint_files = [f for f in os.listdir(self.temp_dir) 
                           if f.startswith("checkpoint_") and f.endswith(".json")]
        assert len(checkpoint_files) == 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

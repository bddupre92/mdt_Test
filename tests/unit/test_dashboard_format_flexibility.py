"""
Unit tests for dashboard format flexibility.

These tests verify that our dashboard components can handle different data formats,
including both dictionary and object access patterns.
"""

import os
import sys
import unittest
from pathlib import Path
import json
from types import SimpleNamespace

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import components to test
from app.ui.components.performance_views.helpers import safe_get_metric, safe_has_metric


class TestDashboardFormatFlexibility(unittest.TestCase):
    """Test cases to verify dashboard components can handle different data formats."""
    
    @staticmethod
    def are_equivalent(a, b):
        """Check if a dictionary and an object are functionally equivalent.
        
        This handles comparisons between dict and SimpleNamespace objects recursively.
        
        Args:
            a: First object (usually a dict)
            b: Second object (usually a SimpleNamespace)
            
        Returns:
            True if objects contain equivalent data, False otherwise
        """
        # Handle None values
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
            
        # Handle SimpleNamespace objects
        if hasattr(a, '__dict__') and not isinstance(a, type):
            a = a.__dict__
        if hasattr(b, '__dict__') and not isinstance(b, type):
            b = b.__dict__
            
        # Now both should be dictionaries
        if not isinstance(a, dict) or not isinstance(b, dict):
            return a == b
            
        # Check if dictionaries have the same keys
        if set(a.keys()) != set(b.keys()):
            return False
            
        # Check if all values are equivalent
        for key in a:
            if isinstance(a[key], dict) or hasattr(a[key], '__dict__'):
                # Recursive check for nested structures
                if not TestDashboardFormatFlexibility.are_equivalent(a[key], b[key]):
                    return False
            elif isinstance(a[key], (list, tuple)) and isinstance(b[key], (list, tuple)):
                # Compare lists/tuples element by element
                if len(a[key]) != len(b[key]):
                    return False
                for i in range(len(a[key])):
                    if not TestDashboardFormatFlexibility.are_equivalent(a[key][i], b[key][i]):
                        return False
            elif a[key] != b[key]:
                return False
                
        return True

    def setUp(self):
        """Set up test fixtures."""
        # Create test data in different formats
        self.dict_format = {
            "rmse": 0.135,
            "mae": 0.109,
            "r2": 0.821,
            "nested": {
                "value": 42
            }
        }
        
        # Convert dict to object with attributes
        self.obj_format = SimpleNamespace(**self.dict_format)
        self.obj_format.nested = SimpleNamespace(**self.dict_format["nested"])
        
        # Load the test checkpoint files
        self.checkpoint_dir = os.path.join(project_root, "checkpoints", "dev")
        self.checkpoint_files = {}
        
        if os.path.exists(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.checkpoint_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            self.checkpoint_files[filename] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {filename}: {str(e)}")

    def test_safe_get_metric(self):
        """Test that safe_get_metric handles both dict and object access."""
        # Test dictionary access
        self.assertEqual(safe_get_metric(self.dict_format, "rmse"), 0.135)
        self.assertEqual(safe_get_metric(self.dict_format, "mae"), 0.109)
        self.assertEqual(safe_get_metric(self.dict_format, "r2"), 0.821)
        self.assertEqual(safe_get_metric(self.dict_format, "nested.value"), 42)
        
        # Test default value for missing keys
        self.assertEqual(safe_get_metric(self.dict_format, "missing", default=100), 100)
        
        # Test object attribute access
        self.assertEqual(safe_get_metric(self.obj_format, "rmse"), 0.135)
        self.assertEqual(safe_get_metric(self.obj_format, "mae"), 0.109)
        self.assertEqual(safe_get_metric(self.obj_format, "r2"), 0.821)
        self.assertEqual(safe_get_metric(self.obj_format, "nested.value"), 42)
        
        # Test default value for missing attributes
        self.assertEqual(safe_get_metric(self.obj_format, "missing", default=100), 100)

    def test_component_helper_functions(self):
        """Test that all component helper functions work correctly."""
        # Test safe_get_metric and safe_has_metric
        
        # Test dictionary access
        self.assertEqual(safe_get_metric(self.dict_format, "rmse"), 0.135)
        self.assertEqual(safe_get_metric(self.dict_format, "nested.value"), 42)
        self.assertTrue(safe_has_metric(self.dict_format, "rmse"))
        self.assertTrue(safe_has_metric(self.dict_format, "nested.value"))
        self.assertFalse(safe_has_metric(self.dict_format, "missing"))
        
        # Test object attribute access
        self.assertEqual(safe_get_metric(self.obj_format, "rmse"), 0.135)
        self.assertEqual(safe_get_metric(self.obj_format, "nested.value"), 42)
        self.assertTrue(safe_has_metric(self.obj_format, "rmse"))
        self.assertTrue(safe_has_metric(self.obj_format, "nested.value"))
        self.assertFalse(safe_has_metric(self.obj_format, "missing"))

    def test_checkpoint_formats(self):
        """Test that all checkpoint formats can be accessed with our helpers."""
        for filename, data in self.checkpoint_files.items():
            print(f"Testing access patterns for {filename}")
            
            # For each checkpoint, try to access common metrics if they exist
            metrics_to_test = [
                "overall_rmse", "overall_mae", "overall_r2",  # Flat format
                "performance_metrics.overall.rmse",           # Nested format
                "performance_metrics.expert_benchmarks"       # Complex nested
            ]
            
            for metric_path in metrics_to_test:
                # This should not raise exceptions even if the metric doesn't exist
                value = safe_get_metric(data, metric_path, default="NOT_FOUND")
                print(f"  {metric_path} = {value}")
                
                # Also test as object
                obj_data = json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))
                obj_value = safe_get_metric(obj_data, metric_path, default="NOT_FOUND")
                print(f"  (as object) {metric_path} = {obj_value}")
                
                # Both access patterns should yield the same result, but for complex objects,
                # we need to compare them differently
                if value == "NOT_FOUND" or obj_value == "NOT_FOUND":
                    # Simple case: both should be NOT_FOUND or neither should be
                    self.assertEqual(value, obj_value)
                elif isinstance(value, dict):
                    # For dictionaries and objects, compare content equivalence
                    # rather than using direct equality
                    self.assertTrue(self.are_equivalent(value, obj_value))
                else:
                    # For simple values, direct comparison should work
                    self.assertEqual(value, obj_value)


if __name__ == "__main__":
    unittest.main()

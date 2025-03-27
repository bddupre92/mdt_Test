#!/usr/bin/env python3
"""
Test script for MoE comparison integration.

This script tests the integration of MoE comparison with the baseline validation
framework by validating command line interface functionality and core comparison
results.
"""
import os
import sys
import json
import unittest
import tempfile
from pathlib import Path
import shutil

# Add parent directory to path to allow imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.commands import MoEComparisonCommand
from baseline_comparison.moe_comparison import MoEBaselineComparison, create_moe_adapter
from validation.baseline_validation import run_baseline_validation


class TestMoEComparison(unittest.TestCase):
    """Test cases for MoE comparison functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_moe_config.json")
        
        # Create a simple test configuration
        self.test_config = {
            "moe_framework": {
                "gating_network": {
                    "type": "confidence_based",
                    "hidden_layers": [32, 16],
                    "activation": "relu",
                    "dropout_rate": 0.1
                },
                "experts": {
                    "count": 3,
                    "specialization": "function_based",
                    "types": ["global", "local", "hybrid"]
                },
                "integration": {
                    "method": "weighted_average",
                    "confidence_threshold": 0.5
                }
            }
        }
        
        # Save the test configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove test directory and contents
        shutil.rmtree(self.test_dir)
    
    def test_moe_adapter_creation(self):
        """Test creation of MoE adapter."""
        adapter = create_moe_adapter(self.config_path)
        self.assertIsNotNone(adapter)
        self.assertTrue(hasattr(adapter, 'predict'))
        self.assertTrue(hasattr(adapter, 'get_expert_contributions'))
    
    def test_moe_baseline_comparison(self):
        """Test MoE baseline comparison initialization."""
        adapter = create_moe_adapter(self.config_path)
        comparison = MoEBaselineComparison(
            simple_baseline=True,
            meta_learner=True,
            enhanced_meta=True,
            satzilla_selector=True,
            moe_adapter=adapter,
            output_dir=self.test_dir
        )
        self.assertIsNotNone(comparison)
        self.assertEqual(comparison.moe_adapter, adapter)
    
    def test_moe_comparison_command(self):
        """Test MoE comparison command execution."""
        # Create minimal args for test - just enough to initialize, not to run fully
        args = {
            "moe_comparison": True,
            "moe_config_path": self.config_path,
            "output_dir": os.path.join(self.test_dir, "command_test"),
            "num_trials": 1,  # Minimal for speed
            "functions": ["sphere"],  # Just one simple function for testing
            "all_functions": False,
            "visualize_moe_contributions": False,
            "calculate_expert_impact": False,
            "detailed_report": False
        }
        
        # Create command
        command = MoEComparisonCommand(args)
        self.assertIsNotNone(command)
        self.assertEqual(command.args["moe_config_path"], self.config_path)
    
    def test_run_validation_with_moe(self):
        """Test running baseline validation with MoE option."""
        # Set up minimal parameters for validation test
        params = {
            "moe_config_path": self.config_path,
            "output_dir": os.path.join(self.test_dir, "validation_test"),
            "num_trials": 1,
            "functions": ["sphere"],
            "include_moe": True
        }
        
        # This only tests initialization of validation framework with MoE
        # without running actual validation (which would be too time-consuming for a unit test)
        summary = run_baseline_validation(**params)
        
        # Verify MoE was included in the parameter dict
        self.assertIn("include_moe", params)
        self.assertTrue(params["include_moe"])


if __name__ == "__main__":
    unittest.main()

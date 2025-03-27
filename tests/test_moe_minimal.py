#!/usr/bin/env python3
"""
Minimal test script for MoE comparison functionality.

This script tests the core functionality of the MoE comparison
without requiring the full CLI integration.
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

# Import the core comparison functionality directly
from baseline_comparison.moe_comparison import (
    MoEBaselineComparison,
    create_moe_adapter
)

def create_test_config():
    """Create a simple test configuration for MoE."""
    config = {
        "moe_framework": {
            "gating_network": {
                "type": "confidence_based",
                "hidden_layers": [32, 16],
                "activation": "relu"
            },
            "experts": {
                "count": 3,
                "specialization": "function_based"
            },
            "integration": {
                "method": "weighted_average",
                "confidence_threshold": 0.5
            }
        }
    }
    
    # Save config to temporary file
    config_path = os.path.join(parent_dir, "tests", "temp_moe_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def test_moe_comparison():
    """Run a minimal test of MoE comparison functionality."""
    print("Creating test configuration...")
    config_path = create_test_config()
    
    try:
        print("Testing MoE adapter creation...")
        adapter = create_moe_adapter(config_path)
        print(f"MoE adapter created: {adapter is not None}")
        
        print("\nTesting MoE baseline comparison initialization...")
        comparison = MoEBaselineComparison(
            simple_baseline=True,
            meta_learner=True,
            enhanced_meta=False,  # Limiting to reduce test complexity
            satzilla_selector=False,  # Limiting to reduce test complexity
            moe_adapter=adapter,
            output_dir=os.path.join(parent_dir, "tests", "moe_test_output")
        )
        print(f"MoE baseline comparison created: {comparison is not None}")
        
        print("\nMoE comparison core functionality test passed!")
        return True
    except Exception as e:
        print(f"Error during MoE comparison test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    print("==== Testing MoE Comparison Core Functionality ====\n")
    success = test_moe_comparison()
    sys.exit(0 if success else 1)

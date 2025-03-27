#!/usr/bin/env python3
"""
Test script for MoE CLI integration.

This script validates the CLI command structure for MoE integration
without executing the actual MoE functionality.
"""
import os
import sys
import unittest
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

# Import directly from commands.py to avoid any circular import issues
commands_path = os.path.join(parent_dir, 'cli', 'commands.py')
import importlib.util
spec = importlib.util.spec_from_file_location("commands_module", commands_path)
commands_module = importlib.util.module_from_spec(spec)
# We're not executing the module to avoid dependency issues
# spec.loader.exec_module(commands_module)

# Import just the argument parser
from cli.argument_parser import create_parser

class TestMoECLIIntegration(unittest.TestCase):
    """Test cases for MoE CLI command integration."""
    
    def test_argument_parser_has_moe_comparison(self):
        """Test that the argument parser has the moe_comparison command."""
        parser = create_parser()
        
        # Get all available subcommands
        subparsers_action = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers_action = action
                break
        
        if subparsers_action is None:
            self.fail("No subparsers found in the argument parser")
        
        # Get the choices (commands) available in the subparser
        choices = subparsers_action.choices
        
        # Check if the moe_comparison command is available
        self.assertIn('moe_comparison', choices, 
                     "moe_comparison command not found in the CLI parser")
        
        # Verify some of the expected arguments for the moe_comparison command
        moe_parser = choices['moe_comparison']
        
        # Helper function to check if an argument exists
        def has_argument(parser, arg_name):
            for action in parser._actions:
                if action.dest == arg_name:
                    return True
            return False
        
        # Check for important MoE comparison arguments
        self.assertTrue(has_argument(moe_parser, 'moe_config_path'), 
                       "moe_config_path argument not found")
        self.assertTrue(has_argument(moe_parser, 'output_dir'), 
                       "output_dir argument not found")
        self.assertTrue(has_argument(moe_parser, 'functions'), 
                       "functions argument not found")
    
    def test_moe_command_definition(self):
        """
        Test that the MoEComparisonCommand class is properly defined.
        
        Note: This test only verifies the existence of the class and methods,
        not their functionality, to avoid dependency issues.
        """
        # Check if the commands.py file exists
        self.assertTrue(os.path.exists(commands_path), 
                       f"commands.py file not found at {commands_path}")
        
        # Read the file content to check for class definition
        with open(commands_path, 'r') as f:
            content = f.read()
        
        # Check for MoEComparisonCommand class definition
        self.assertIn("class MoEComparisonCommand", content,
                     "MoEComparisonCommand class not defined in commands.py")
        
        # Check for execute method
        self.assertIn("def execute(self)", content,
                     "execute method not found in commands.py")
        
        # Check for MoE-specific code elements
        self.assertIn("MoEBaselineComparison", content,
                     "MoEBaselineComparison not used in commands.py")
        self.assertIn("create_moe_adapter", content,
                     "create_moe_adapter not used in commands.py")
        
        print("MoEComparisonCommand class is properly defined in commands.py")

if __name__ == "__main__":
    print("==== Testing MoE CLI Integration ====\n")
    unittest.main()

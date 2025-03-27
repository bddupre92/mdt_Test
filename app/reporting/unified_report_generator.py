"""
Unified Report Generator

This module provides a centralized mechanism for generating interactive reports
that can be used by both the command-line tools and the Streamlit dashboard.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
import datetime
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedReportGenerator:
    """
    Unified report generator that can be used by both command-line tools and the Streamlit dashboard.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory for storing reports (default: results/reports)
        """
        self.output_dir = output_dir or os.path.join("results", "reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Import all report modules dynamically
        self.report_modules = {}
        self.available_modules = []
        
        # Only try to load modules that exist
        modules_dir = os.path.join(os.path.dirname(__file__), "modules")
        if os.path.exists(modules_dir):
            for file in os.listdir(modules_dir):
                if file.endswith(".py") and not file.startswith("__"):
                    report_type = file[:-3]  # Remove .py extension
                    try:
                        spec = importlib.util.spec_from_file_location(
                            report_type,
                            os.path.join(modules_dir, file)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        self.report_modules[report_type] = module
                        self.available_modules.append(report_type)
                        logger.info(f"Successfully imported {report_type} module")
                    except Exception as e:
                        logger.warning(f"Error importing {report_type} module: {e}")
                        continue
        
        if not self.report_modules:
            logger.warning("No report modules found in modules directory")
            # Don't raise error, allow dashboard to show default options
    
    def generate_report(self, test_results, results_dir=None, return_html=False, report_type="interactive", include_sections=None):
        """
        Generate a report based on test results.
        
        Args:
            test_results: Dictionary of test results
            results_dir: Directory with result files (default: self.output_dir)
            return_html: If True, return the HTML content instead of writing to a file
            report_type: Type of report to generate (interactive, summary, etc.)
            include_sections: List of report sections to include (default: all available)
            
        Returns:
            Path to the generated report or HTML content if return_html=True
        """
        results_dir = results_dir or self.output_dir
        
        # Default to interactive report if no specific type is requested
        if report_type == "interactive" and "moe_interactive_report" in self.report_modules:
            # If specific sections are requested, we need to modify the test_results
            if include_sections:
                # Make a copy to avoid modifying the original
                modified_test_results = test_results.copy()
                modified_test_results["include_sections"] = include_sections
                return self.report_modules["moe_interactive_report"].generate_interactive_report(
                    modified_test_results, results_dir, return_html
                )
            else:
                return self.report_modules["moe_interactive_report"].generate_interactive_report(
                    test_results, results_dir, return_html
                )
        elif report_type in self.report_modules:
            # If the report type matches a specific module, use that module's generator
            generator_name = f"generate_{report_type}_section"
            if hasattr(self.report_modules[report_type], generator_name):
                generator = getattr(self.report_modules[report_type], generator_name)
                html_content = generator(test_results, results_dir)
                
                if return_html:
                    return "\n".join(html_content)
                else:
                    # Generate a file with just this report section
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    reports_dir = os.path.join(results_dir, "reports")
                    os.makedirs(reports_dir, exist_ok=True)
                    report_path = os.path.join(reports_dir, f"{report_type}_{timestamp}.html")
                    
                    with open(report_path, "w") as f:
                        f.write("\n".join(html_content))
                    
                    return report_path
            else:
                logger.warning(f"No generator function found for {report_type}, defaulting to interactive")
                return self.report_modules["moe_interactive_report"].generate_interactive_report(
                    test_results, results_dir, return_html
                )
        else:
            logger.warning(f"Unknown report type: {report_type}, defaulting to interactive")
            if "moe_interactive_report" in self.report_modules:
                return self.report_modules["moe_interactive_report"].generate_interactive_report(
                    test_results, results_dir, return_html
                )
            else:
                raise ValueError("Interactive report module not available")
    
    def get_all_reports(self, directory=None):
        """
        Get all reports in the specified directory.
        
        Args:
            directory: Directory to search for reports (default: self.output_dir)
            
        Returns:
            List of report paths
        """
        directory = directory or self.output_dir
        reports = []
        
        # Walk through the directory and find all HTML reports
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".html") and "interactive_report" in file:
                    reports.append(os.path.join(root, file))
        
        return sorted(reports, key=os.path.getmtime, reverse=True)
    
    def run_validation_and_generate_report(self, validation_type, args_dict, include_sections=None):
        """
        Run validation and generate a report.
        
        Args:
            validation_type: Type of validation to run (moe, real_data, etc.)
            args_dict: Dictionary of arguments for the validation
            include_sections: List of report sections to include (default: all available)
            
        Returns:
            Dictionary with validation results and report path
        """
        try:
            # Add include_sections to args_dict if provided
            if include_sections:
                args_dict["include_sections"] = include_sections
                
            if validation_type == "moe":
                # Import dynamically to avoid circular imports
                from core.moe_validation import run_moe_validation
                result = run_moe_validation(args_dict)
            elif validation_type == "real_data":
                # Import dynamically to avoid circular imports
                from core.real_data_validation import run_real_data_validation
                result = run_real_data_validation(args_dict)
            else:
                logger.error(f"Unknown validation type: {validation_type}")
                return {"success": False, "message": f"Unknown validation type: {validation_type}"}
            
            return result
        except Exception as e:
            logger.error(f"Error running validation: {e}")
            return {"success": False, "message": f"Error running validation: {e}"}
        
    def get_available_report_sections(self):
        """
        Get a list of all available report sections.
        
        Returns:
            List of available report section names
        """
        sections = []
        for module_name, module in self.report_modules.items():
            # Convert module_name to a more readable format
            if module_name == "moe_interactive_report":
                continue  # Skip this as it's the container for all reports
                
            # Convert snake_case to Title Case and remove _report suffix
            section_name = module_name.replace("_report", "")
            section_name = " ".join(word.capitalize() for word in section_name.split("_"))
            sections.append({
                "id": module_name,
                "name": section_name
            })
            
        return sorted(sections, key=lambda x: x["name"])


# Create a singleton instance for global use
report_generator = UnifiedReportGenerator()

def get_report_generator():
    """Get the global report generator instance."""
    return report_generator

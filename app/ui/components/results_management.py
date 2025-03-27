"""
Results Management System for MoE Framework

This component provides a Streamlit interface for managing, comparing, and exporting
results from the MoE framework. It integrates with the existing interactive report system
and adds comparative analysis tools and export capabilities.
"""

import os
import sys
import glob
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import datetime
import shutil
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import base64
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the interactive report generator
from tests.moe_interactive_report import generate_interactive_report


class ResultsManagementSystem:
    """
    Results Management System for the MoE framework.
    
    This class provides functionality for:
    1. Listing and loading existing reports
    2. Comparative analysis between different runs
    3. Export capabilities for reports in various formats
    4. Integration with the existing HTML report system
    """
    
    def __init__(self, results_base_dir: str = None):
        """
        Initialize the Results Management System.
        
        Args:
            results_base_dir: Base directory for results. If None, uses default location.
        """
        if results_base_dir is None:
            self.results_base_dir = os.path.join(project_root, 'results')
        else:
            self.results_base_dir = results_base_dir
            
        # Ensure the results directory exists
        os.makedirs(self.results_base_dir, exist_ok=True)
        
        # Create reports directory if it doesn't exist
        self.reports_dir = os.path.join(self.results_base_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Create exports directory if it doesn't exist
        self.exports_dir = os.path.join(self.results_base_dir, 'exports')
        os.makedirs(self.exports_dir, exist_ok=True)
        
        logger.info(f"Results Management System initialized with base directory: {self.results_base_dir}")
    
    def get_available_reports(self) -> List[Dict[str, Any]]:
        """
        Get a list of available reports.
        
        Returns:
            List of dictionaries with report information
        """
        reports = []
        
        # Look for HTML reports in the reports directory
        html_files = glob.glob(os.path.join(self.reports_dir, "*.html"))
        
        for html_file in html_files:
            filename = os.path.basename(html_file)
            # Extract timestamp from filename (format: interactive_report_YYYY-MM-DD_HH-MM-SS.html)
            match = re.search(r'interactive_report_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.html', filename)
            
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                    
                    # Look for metadata file
                    metadata_file = os.path.join(self.reports_dir, f"metadata_{timestamp_str}.json")
                    metadata = {}
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    reports.append({
                        'filename': filename,
                        'path': html_file,
                        'timestamp': timestamp,
                        'timestamp_str': timestamp_str,
                        'metadata': metadata
                    })
                except Exception as e:
                    logger.warning(f"Error parsing timestamp from filename {filename}: {e}")
        
        # Sort reports by timestamp (newest first)
        reports.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return reports
    
    def get_report_metrics(self, report_path: str) -> Dict[str, Any]:
        """
        Extract key metrics from a report.
        
        Args:
            report_path: Path to the report HTML file
            
        Returns:
            Dictionary of key metrics
        """
        # Look for a corresponding metrics JSON file
        metrics_path = report_path.replace('.html', '_metrics.json')
        
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metrics from {metrics_path}: {e}")
        
        # If no metrics file exists, return empty dict
        return {}
    
    def compare_reports(self, report_paths: List[str]) -> Dict[str, Any]:
        """
        Compare metrics from multiple reports.
        
        Args:
            report_paths: List of paths to report HTML files
            
        Returns:
            Dictionary with comparative metrics
        """
        comparison = {
            'reports': [],
            'metrics': {},
            'improvements': {}
        }
        
        # Get metrics for each report
        for report_path in report_paths:
            report_name = os.path.basename(report_path)
            metrics = self.get_report_metrics(report_path)
            
            if metrics:
                comparison['reports'].append({
                    'name': report_name,
                    'path': report_path,
                    'metrics': metrics
                })
        
        # If we have at least two reports with metrics, calculate improvements
        if len(comparison['reports']) >= 2:
            # Get common metrics across all reports
            common_metrics = set.intersection(*[set(report['metrics'].keys()) for report in comparison['reports']])
            
            # For each common metric, create a comparison
            for metric in common_metrics:
                comparison['metrics'][metric] = [report['metrics'][metric] for report in comparison['reports']]
                
                # Calculate improvement between first and last report
                if isinstance(comparison['metrics'][metric][0], (int, float)) and isinstance(comparison['metrics'][metric][-1], (int, float)):
                    first_value = comparison['metrics'][metric][0]
                    last_value = comparison['metrics'][metric][-1]
                    
                    if first_value != 0:
                        improvement = (last_value - first_value) / first_value * 100
                        comparison['improvements'][metric] = improvement
        
        return comparison
    
    def export_report(self, report_path: str, format: str = 'html') -> str:
        """
        Export a report in the specified format.
        
        Args:
            report_path: Path to the report HTML file
            format: Export format ('html', 'pdf', 'csv')
            
        Returns:
            Path to the exported file
        """
        report_name = os.path.basename(report_path)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if format == 'html':
            # Simply copy the HTML file
            export_path = os.path.join(self.exports_dir, f"{report_name.replace('.html', '')}_{timestamp}.html")
            shutil.copy2(report_path, export_path)
            return export_path
        
        elif format == 'pdf':
            # For PDF export, we need to convert HTML to PDF
            # This could be done with libraries like weasyprint or pdfkit
            # For now, we'll just create a placeholder
            export_path = os.path.join(self.exports_dir, f"{report_name.replace('.html', '')}_{timestamp}.pdf")
            
            # Placeholder for PDF conversion
            with open(export_path, 'w') as f:
                f.write("PDF export placeholder")
            
            logger.warning("PDF export is not fully implemented yet")
            return export_path
        
        elif format == 'csv':
            # Extract metrics and save as CSV
            metrics = self.get_report_metrics(report_path)
            export_path = os.path.join(self.exports_dir, f"{report_name.replace('.html', '')}_{timestamp}.csv")
            
            if metrics:
                # Convert metrics to DataFrame and save as CSV
                df = pd.DataFrame([metrics])
                df.to_csv(export_path, index=False)
            else:
                # Create empty CSV with header
                with open(export_path, 'w') as f:
                    f.write("No metrics available for export")
            
            return export_path
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def generate_new_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate a new interactive report.
        
        Args:
            test_results: Dictionary of test results
            
        Returns:
            Path to the generated report
        """
        # Call the existing interactive report generator
        report_path = generate_interactive_report(test_results, self.results_base_dir)
        
        # Extract timestamp from the report path
        match = re.search(r'interactive_report_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.html', os.path.basename(report_path))
        if match:
            timestamp_str = match.group(1)
            
            # Save metadata
            metadata = {
                'generated_at': datetime.datetime.now().isoformat(),
                'test_results_summary': {
                    'num_models': len(test_results.get('models', [])),
                    'num_datasets': len(test_results.get('datasets', [])),
                    'num_metrics': len(test_results.get('metrics', []))
                }
            }
            
            metadata_path = os.path.join(self.reports_dir, f"metadata_{timestamp_str}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Extract and save key metrics
            metrics = {}
            
            # Extract model performance metrics if available
            if 'model_performance' in test_results:
                metrics.update(test_results['model_performance'])
            
            # Save metrics
            metrics_path = report_path.replace('.html', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return report_path


def render_results_management_ui():
    """
    Render the Results Management System UI using Streamlit.
    """
    st.title("Results Management System")
    
    # Initialize the Results Management System
    results_system = ResultsManagementSystem()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Reports Explorer", "Comparative Analysis", "Export Tools"])
    
    with tab1:
        st.header("Reports Explorer")
        
        # Get available reports
        reports = results_system.get_available_reports()
        
        if not reports:
            st.info("No reports found. Generate a new report to get started.")
        else:
            # Display reports in a table
            report_data = []
            for report in reports:
                report_data.append({
                    "Report Name": report['filename'],
                    "Date": report['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    "Metadata": ", ".join([f"{k}: {v}" for k, v in report['metadata'].items()]) if report['metadata'] else "No metadata"
                })
            
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df)
            
            # Select a report to view
            selected_report_index = st.selectbox(
                "Select a report to view:",
                options=range(len(reports)),
                format_func=lambda i: reports[i]['filename']
            )
            
            if selected_report_index is not None:
                selected_report = reports[selected_report_index]
                
                # Display report details
                st.subheader(f"Report: {selected_report['filename']}")
                st.write(f"Generated on: {selected_report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Display metadata if available
                if selected_report['metadata']:
                    st.json(selected_report['metadata'])
                
                # Get metrics for this report
                metrics = results_system.get_report_metrics(selected_report['path'])
                if metrics:
                    st.subheader("Key Metrics")
                    st.json(metrics)
                
                # Button to open the report in a new tab
                if st.button("Open Report in Browser"):
                    # Use webbrowser module to open the file
                    webbrowser.open_new_tab(f"file://{selected_report['path']}")
    
    with tab2:
        st.header("Comparative Analysis")
        
        # Get available reports
        reports = results_system.get_available_reports()
        
        if len(reports) < 2:
            st.info("Need at least two reports for comparative analysis. Generate more reports to use this feature.")
        else:
            # Multi-select for reports to compare
            selected_report_indices = st.multiselect(
                "Select reports to compare:",
                options=range(len(reports)),
                format_func=lambda i: reports[i]['filename']
            )
            
            if len(selected_report_indices) >= 2:
                selected_reports = [reports[i] for i in selected_report_indices]
                selected_report_paths = [report['path'] for report in selected_reports]
                
                # Compare the selected reports
                comparison = results_system.compare_reports(selected_report_paths)
                
                # Display comparison results
                if comparison['metrics']:
                    st.subheader("Metrics Comparison")
                    
                    # Create a DataFrame for the metrics
                    metrics_data = {}
                    for i, report in enumerate(selected_reports):
                        report_name = report['filename']
                        metrics_data[report_name] = {}
                        
                        for metric, values in comparison['metrics'].items():
                            if i < len(values):
                                metrics_data[report_name][metric] = values[i]
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df)
                    
                    # Display improvements
                    if comparison['improvements']:
                        st.subheader("Improvements")
                        
                        improvements_data = []
                        for metric, improvement in comparison['improvements'].items():
                            improvements_data.append({
                                "Metric": metric,
                                "Improvement (%)": f"{improvement:.2f}%",
                                "Direction": "Increase" if improvement > 0 else "Decrease"
                            })
                        
                        improvements_df = pd.DataFrame(improvements_data)
                        st.dataframe(improvements_df)
                        
                        # Create a bar chart of improvements
                        fig = px.bar(
                            improvements_data,
                            x="Metric",
                            y="Improvement (%)",
                            color="Direction",
                            title="Metric Improvements Between Reports"
                        )
                        st.plotly_chart(fig)
                else:
                    st.warning("No common metrics found for comparison.")
    
    with tab3:
        st.header("Export Tools")
        
        # Get available reports
        reports = results_system.get_available_reports()
        
        if not reports:
            st.info("No reports found. Generate a new report to get started.")
        else:
            # Select a report to export
            selected_report_index = st.selectbox(
                "Select a report to export:",
                options=range(len(reports)),
                format_func=lambda i: reports[i]['filename'],
                key="export_report_select"
            )
            
            if selected_report_index is not None:
                selected_report = reports[selected_report_index]
                
                # Select export format
                export_format = st.selectbox(
                    "Select export format:",
                    options=["html", "pdf", "csv"],
                    format_func=lambda f: f.upper()
                )
                
                # Button to export the report
                if st.button("Export Report"):
                    with st.spinner(f"Exporting report as {export_format.upper()}..."):
                        try:
                            export_path = results_system.export_report(
                                selected_report['path'],
                                format=export_format
                            )
                            
                            st.success(f"Report exported successfully to: {export_path}")
                            
                            # Provide a download link
                            if os.path.exists(export_path):
                                with open(export_path, "rb") as file:
                                    file_contents = file.read()
                                    
                                b64_contents = base64.b64encode(file_contents).decode()
                                download_filename = os.path.basename(export_path)
                                
                                href = f'<a href="data:application/{export_format};base64,{b64_contents}" download="{download_filename}">Download {export_format.upper()} File</a>'
                                st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error exporting report: {str(e)}")


if __name__ == "__main__":
    # This allows running the component directly for testing
    render_results_management_ui()

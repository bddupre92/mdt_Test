"""
Data Configuration Connector for Performance Analysis

This module provides functions to connect data configurations with performance results,
enabling traceable links between data preprocessing choices and model performance.
"""

import streamlit as st
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_data_config(config_id):
    """
    Load a data configuration by ID.
    
    Args:
        config_id: ID of the data configuration to load
        
    Returns:
        Dictionary containing the data configuration, or None if not found
    """
    # Define the base directory for template storage
    templates_dir = os.environ.get(
        "MOE_TEMPLATES_DIR", 
        os.path.join(Path.home(), ".moe_framework", "templates")
    )
    
    # Try to find the configuration file
    config_path = os.path.join(templates_dir, f"{config_id}.json")
    
    if not os.path.exists(config_path):
        # Try searching in subdirectories
        for root, dirs, files in os.walk(templates_dir):
            for filename in files:
                if filename == f"{config_id}.json":
                    config_path = os.path.join(root, filename)
                    break
    
    # Load the configuration if found
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data configuration: {str(e)}")
            return None
    else:
        return None

def render_data_config_connector(config_id):
    """
    Render the data configuration connector UI.
    
    Args:
        config_id: ID of the data configuration to display
    """
    if not config_id:
        st.info("No data configuration linked to this analysis")
        return
    
    st.markdown("## Data Configuration Connection")
    
    with st.expander("Data Configuration Details", expanded=False):
        st.markdown(f"**Configuration ID**: `{config_id}`")
        
        # Load the data configuration
        config = load_data_config(config_id)
        
        if config:
            # Display the configuration details
            st.markdown("### Configuration Summary")
            
            # Display basic info
            if "name" in config:
                st.markdown(f"**Name**: {config['name']}")
            if "description" in config:
                st.markdown(f"**Description**: {config['description']}")
            if "created_at" in config:
                st.markdown(f"**Created**: {config['created_at']}")
            
            # Display pipeline operations if available
            if "pipeline" in config and isinstance(config["pipeline"], list):
                st.markdown("### Preprocessing Pipeline")
                
                for i, operation in enumerate(config["pipeline"]):
                    operation_type = operation.get("type", "Unknown")
                    st.markdown(f"**Step {i+1}**: {operation_type}")
                    
                    # Display parameters if available
                    if "parameters" in operation:
                        params = operation["parameters"]
                        param_text = ", ".join([f"{k}: {v}" for k, v in params.items()])
                        st.markdown(f"Parameters: {param_text}")
            
            # Display quality metrics if available
            if "quality_metrics" in config:
                st.markdown("### Data Quality Metrics")
                
                metrics = config["quality_metrics"]
                if isinstance(metrics, dict):
                    # Create a formatted display of metrics
                    for category, category_metrics in metrics.items():
                        st.markdown(f"**{category}**")
                        if isinstance(category_metrics, dict):
                            for metric_name, value in category_metrics.items():
                                st.markdown(f"- {metric_name}: {value}")
                        else:
                            st.markdown(f"- Value: {category_metrics}")
                else:
                    st.markdown(f"Quality Score: {metrics}")
            
            # Option to view full configuration
            with st.expander("View Full Configuration JSON"):
                st.json(config)
            
            # Add link to data configuration dashboard
            st.markdown("[Open in Data Configuration Dashboard](../data_configuration?config_id={})".format(config_id))
        else:
            st.warning(f"Data configuration with ID '{config_id}' not found")
            st.markdown("This may happen if the configuration was deleted or if it's stored in a location not accessible to the dashboard.")
    
    # Horizontal line to separate from the rest of the content
    st.markdown("---")

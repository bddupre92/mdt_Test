"""
Performance Analysis Dashboard for MoE Framework

This module provides a Streamlit dashboard for the Performance Analysis System,
enabling detailed evaluation of MoE system performance through metrics visualization,
statistical tests, and comparison with baseline methods.
"""

import os
import sys
import streamlit as st
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the Performance Analysis components
from app.ui.components.performance_analysis import render_performance_analysis_ui
from app.ui.components.data_config_connector import render_data_config_connector
from app.ui.components.performance_views.performance_analysis import render_performance_analysis
from moe_framework.persistence.state_manager import FileSystemStateManager


def performance_analysis_dashboard():
    """
    Main function for the Performance Analysis Dashboard.
    
    This dashboard enables detailed analysis of MoE system performance,
    including expert benchmarks, gating network evaluation, end-to-end metrics,
    baseline comparisons, and statistical tests.
    """
    st.set_page_config(
        page_title="MoE Framework - Performance Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state for storing dashboard state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"
    if 'selected_checkpoint' not in st.session_state:
        st.session_state.selected_checkpoint = None
    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = None
    if 'data_config_id' not in st.session_state:
        st.session_state.data_config_id = None
    
    # Add sidebar with navigation and controls
    with st.sidebar:
        st.title("MoE Framework")
        st.subheader("Performance Analysis")
        
        # Add navigation links
        st.markdown("### Navigation")
        st.markdown("- [Home](../)")
        st.markdown("- [Data Configuration](../data_configuration)")
        st.markdown("- [Results Management](../results)")
        st.markdown("- [Performance Analysis](#)")
        
        st.markdown("---")
        
        # Select a checkpoint to analyze
        st.subheader("Load System State")
        
        # Initialize state manager
        state_manager = FileSystemStateManager()
        
        # Get checkpoint directory from environment or use default
        checkpoint_dir = os.environ.get("MOE_CHECKPOINT_DIR", os.path.join(project_root, "checkpoints"))
        
        # Make sure checkpoint_dir is an absolute path
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.join(project_root, checkpoint_dir)
        
        # Debug logging
        st.sidebar.text(f"Looking in: {checkpoint_dir}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        
        # List available checkpoints
        try:
            # Initialize the state manager with the checkpoint directory
            state_manager = FileSystemStateManager(config={'base_dir': checkpoint_dir})
            # Now call list_checkpoints without parameters
            checkpoints = state_manager.list_checkpoints()
            logger.info(f"Found checkpoints: {checkpoints}")
            st.sidebar.text(f"Found {len(checkpoints)} checkpoints")
            checkpoint_names = [os.path.basename(cp) for cp in checkpoints]
            
            if checkpoint_names:
                selected_checkpoint_name = st.selectbox(
                    "Select Checkpoint", 
                    checkpoint_names,
                    index=0
                )
                
                selected_checkpoint = os.path.join(checkpoint_dir, selected_checkpoint_name)
                
                if st.button("Load Checkpoint"):
                    try:
                        system_state = state_manager.load_state(selected_checkpoint)
                        if system_state:
                            st.session_state.selected_checkpoint = selected_checkpoint
                            
                            # Handle system_state whether it's a dictionary or an object with attributes
                            if isinstance(system_state, dict):
                                st.session_state.performance_data = system_state.get('performance_metrics', system_state)
                                data_source = st.session_state.performance_data
                                st.session_state.data_config_id = data_source.get("data_config_id", "")
                            else:
                                # Handle as object with attributes
                                st.session_state.performance_data = getattr(system_state, 'performance_metrics', system_state)
                                data_source = st.session_state.performance_data
                                st.session_state.data_config_id = getattr(data_source, "data_config_id", "") if hasattr(data_source, "get") else data_source.get("data_config_id", "")
                            st.success(f"Loaded checkpoint: {selected_checkpoint_name}")
                            logger.info(f"Loaded checkpoint: {selected_checkpoint}")
                        else:
                            st.error("Failed to load checkpoint")
                    except Exception as e:
                        st.error(f"Error loading checkpoint: {str(e)}")
                        logger.error(f"Error loading checkpoint: {str(e)}", exc_info=True)
            else:
                st.info("No checkpoints available")
        except Exception as e:
            st.error(f"Error listing checkpoints: {str(e)}")
            logger.error(f"Error listing checkpoints: {str(e)}", exc_info=True)
        
        # Analysis tabs
        st.markdown("---")
        st.subheader("Analysis Areas")
        
        tabs = [
            "Overview",
            "Expert Benchmarks", 
            "Gating Network Analysis",
            "End-to-End Performance",
            "Baseline Comparisons",
            "Statistical Tests",
            "Visualizations"
        ]
        
        selected_tab = st.radio("Select Analysis Area", tabs)
        st.session_state.active_tab = selected_tab
        
        # Add info about the dashboard
        st.markdown("---")
        st.markdown("""
        ### About
        This dashboard provides comprehensive performance analysis tools for the MoE framework,
        enabling detailed evaluation of system components and end-to-end performance.
        
        The analysis integrates with data configurations to connect preprocessing choices
        with model performance.
        """)
    
    # Main content area
    st.title("MoE Performance Analysis")
    
    # Render the Performance Analysis UI based on the selected tab
    if st.session_state.selected_checkpoint:
        # If data config ID is available, show the connection
        if st.session_state.data_config_id:
            render_data_config_connector(st.session_state.data_config_id)
        
        # Check if the user wants to use the new integrated performance analysis
        use_integrated_view = st.sidebar.checkbox(
            "Use Integrated Performance Dashboard", 
            value=True,
            help="Toggle between modular view and integrated performance dashboard"
        )
        
        # Data format adaptation options
        with st.sidebar.expander("Advanced Data Options"):
            st.info("The dashboard will automatically try to adapt to your data format.")
            st.markdown("### Data Flexibility Options")
            
            # Allow manual selection of key data paths
            data_paths = {
                'Auto-detect': 'auto',
                'Direct metrics': 'direct',
                'Performance metrics key': 'performance_metrics',
                'Results key': 'results',
                'Custom path': 'custom'
            }
            selected_path = st.selectbox(
                "Data Structure",
                options=list(data_paths.keys()),
                index=0
            )
            
            # For custom path, allow user to specify the path
            if selected_path == 'Custom path':
                custom_path = st.text_input("Enter key path (e.g., 'results.performance')")
                st.session_state.custom_data_path = custom_path
            else:
                st.session_state.data_path_option = data_paths[selected_path]
        
        # Create system state with flexible data handling
        system_state = _create_system_state(st.session_state.performance_data)
        
        if use_integrated_view and st.session_state.active_tab == "End-to-End Performance":
            # Use the new integrated performance analysis dashboard
            render_performance_analysis(system_state)
        else:
            # Use the traditional tabbed interface with adaptive rendering
            render_performance_analysis_ui(
                st.session_state.active_tab,
                system_state.performance_metrics
            )
    else:
        st.info("Please select and load a checkpoint to begin analysis")


def _create_system_state(performance_data):
    """Create a SystemState object from the performance data dictionary."""
    from moe_framework.interfaces.base import SystemState
    
    system_state = SystemState()
    
    # Check if performance_data is directly accessible or nested in a key
    if isinstance(performance_data, dict):
        # Try to handle different potential structures
        if 'performance_metrics' in performance_data:
            # Data is already structured with performance_metrics key
            system_state.performance_metrics = performance_data['performance_metrics']
        elif any(k in performance_data for k in ['expert_benchmarks', 'gating_evaluation', 'end_to_end_metrics']):
            # Data contains expected performance metric keys directly
            system_state.performance_metrics = performance_data
        else:
            # Try to find performance-related keys at the top level
            performance_keys = [
                k for k in performance_data.keys() 
                if any(term in k.lower() for term in ['metrics', 'performance', 'evaluation', 'benchmark'])
            ]
            if performance_keys:
                # Use the first performance-related key found
                system_state.performance_metrics = performance_data
            else:
                # Default to using the entire dictionary
                system_state.performance_metrics = performance_data
    else:
        # If not a dictionary, try to use as is (not ideal but prevents crashing)
        system_state.performance_metrics = {'raw_data': performance_data}
        
    return system_state


if __name__ == "__main__":
    performance_analysis_dashboard()

"""
MoE Framework Dashboard

This is the main dashboard for the MoE framework, integrating all visualization components
for monitoring and analyzing the framework's performance, workflow, and data.
"""

import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Set up project root for imports
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# Import MoE components
from moe_framework.workflow.moe_pipeline import MoEPipeline
from moe_framework.event_tracking.visualization import WorkflowVisualizer
from moe_framework.data_connectors.file_connector import FileDataConnector
from moe_framework.data_connectors.data_quality import DataQualityAssessment
from visualization.performance_views import create_performance_dashboard
from visualization.workflow_views import create_workflow_dashboard
from visualization.data_views import create_data_dashboard
from moe_framework.integration.event_system import EventListener, Event, EventManager, MoEEventTypes

# Import new visualization modules
from visualization.interactive_pipeline_viz import create_interactive_pipeline_view
from visualization.workflow_views import render_workflow_summary, render_data_flow_visualization, render_execution_history
from visualization.live_visualization import create_live_training_monitor, create_optimization_monitor
from visualization.component_details import render_component_details
from visualization.data_utils import load_or_generate_input_data, run_complete_pipeline

# Set page config at the very top of the script
st.set_page_config(
    page_title="Mixture of Experts Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize the session state with required components."""
    try:
        if 'initialized' not in st.session_state:
            # Create workflow tracking directory
            tracking_dir = os.path.join(project_root, ".workflow_tracking")
            os.makedirs(tracking_dir, exist_ok=True)
            
            # Initialize MoE pipeline with default configuration
            config = {
                'experts': {
                    'physiological': {
                        'model_type': 'neural_network',
                        'hidden_layers': [64, 32],
                        'activation': 'relu',
                        'input_dim': 4
                    },
                    'behavioral': {
                        'behavior_cols': ['sleep', 'activity', 'stress', 'mood'],
                        'patient_id_col': 'patient_id',
                        'timestamp_col': 'date',
                        'include_sleep': True,
                        'include_activity': True,
                        'include_stress': True
                    },
                    'environmental': {
                        'env_cols': ['temperature', 'humidity', 'pressure', 'air_quality'],
                        'location_col': 'location',
                        'timestamp_col': 'date',
                        'include_weather': True,
                        'include_pollution': True
                    },
                    'medication_history': {
                        'medication_cols': ['medication_name', 'dosage', 'frequency'],
                        'patient_id_col': 'patient_id',
                        'timestamp_col': 'date'
                    }
                },
                'gating': {
                    'type': 'quality_aware',
                    'weighting_strategy': 'quality_score',
                    'quality_threshold': 0.7
                },
                'training': {
                    'batch_size': 32,
                    'epochs': 100,
                    'validation_split': 0.2,
                    'early_stopping': True
                }
            }
            
            # Initialize pipeline
            st.session_state.moe_pipeline = MoEPipeline(config=config, verbose=True)
            
            # Initialize workflow visualizer with tracking directory
            st.session_state.workflow_viz = WorkflowVisualizer(tracking_dir=tracking_dir)
            
            # Initialize data components
            st.session_state.data_connector = FileDataConnector()
            st.session_state.quality_assessment = DataQualityAssessment()
            
            # Initialize other session state variables
            st.session_state.current_view = "Workflow"
            st.session_state.initialized = True
            
            # Record initialization event
            st.session_state.workflow_viz.add_event(
                "PIPELINE_INITIALIZED",
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": config,
                    "expert_count": len(st.session_state.moe_pipeline.experts)
                }
            )
            
            return True
            
        return True
        
    except Exception as e:
        st.error(f"Error initializing session state: {str(e)}")
        return False

def load_data(file_path: str) -> bool:
    """Load data from file using the FileDataConnector."""
    try:
        # Connect to data source
        connection_success = st.session_state.data_connector.connect({
            'file_path': file_path
        })
        
        if not connection_success:
            st.error("Failed to connect to data source")
            return False
        
        # Load the data
        data = st.session_state.data_connector.load_data()
        if data.empty:
            st.error("No data loaded")
            return False
        
        # Get schema information
        schema = st.session_state.data_connector.get_schema()
        
        # Assess data quality
        quality_metrics = st.session_state.quality_assessment.assess_quality(
            data,
            target_column=schema.get('target_column')
        )
        
        # Update pipeline with loaded data
        st.session_state.moe_pipeline.data = data
        st.session_state.moe_pipeline.features = schema.get('feature_columns', [])
        
        # Set target column explicitly - check if 'target' exists, otherwise use schema
        if 'target' in data.columns:
            st.session_state.moe_pipeline.target = 'target'
        else:
            st.session_state.moe_pipeline.target = schema.get('target_column')
            
            # If target is still None, show error and select first numeric column
            if not st.session_state.moe_pipeline.target:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    st.session_state.moe_pipeline.target = numeric_cols[-1]  # Use last numeric column as target
                    st.warning(f"No target column specified. Using '{numeric_cols[-1]}' as target.")
                else:
                    st.error("No numeric columns found for target. Please specify a target column.")
                    return False
        
        st.session_state.moe_pipeline.quality_assessment = quality_metrics
        
        # Update pipeline state to indicate data is loaded
        st.session_state.moe_pipeline.pipeline_state['data_loaded'] = True
        
        # Record data loading event
        st.session_state.workflow_viz.add_event(
            "DATA_LOADED",
            {
                "timestamp": datetime.now().isoformat(),
                "file_path": file_path,
                "data_shape": data.shape,
                "features": schema.get('feature_columns', []),
                "target": st.session_state.moe_pipeline.target,
                "quality_score": quality_metrics.get('quality_score', 0.0)
            }
        )
        
        # Display data loading information
        st.info(f"Loaded data with shape: {data.shape}")
        st.info(f"Features: {schema.get('feature_columns', [])}")
        st.info(f"Target column: {st.session_state.moe_pipeline.target}")
        
        # Add target column selector after loading
        if hasattr(st.session_state, 'moe_pipeline') and hasattr(st.session_state.moe_pipeline, 'data'):
            data = st.session_state.moe_pipeline.data
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            # Get current target
            current_target = st.session_state.moe_pipeline.target
            if current_target in numeric_cols:
                default_idx = numeric_cols.index(current_target)
            else:
                default_idx = 0
                
            # Allow user to select target column
            target_col = st.selectbox(
                "Select Target Column",
                options=numeric_cols,
                index=default_idx,
                key="load_data_target_column_selector"
            )
            
            # Update target column if changed
            if target_col != current_target:
                st.session_state.moe_pipeline.target = target_col
                st.info(f"Target column updated to: {target_col}")
        
        return True
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Record error event
        if hasattr(st.session_state, 'workflow_viz'):
            st.session_state.workflow_viz.add_event(
                "DATA_LOADING_ERROR",
                {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "file_path": file_path
                }
            )
        return False

def create_data_quality_tab(experts):
    """
    Create a data quality tab to help diagnose data issues.
    
    Args:
        experts: Dictionary of expert objects
    """
    # Import pandas inside the function to ensure it's available
    import pandas as pd
    import numpy as np
    
    st.subheader("Data Quality Check")
    
    if not hasattr(st.session_state, 'moe_pipeline') or not hasattr(st.session_state.moe_pipeline, 'data'):
        st.warning("No data loaded. Please load data first.")
        return
    
    # Get the loaded data
    data = st.session_state.moe_pipeline.data
    
    # Show data shape and info
    st.markdown(f"**Data Shape:** {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percent Missing': missing_percent.round(2)
    })
    missing_df = missing_df.sort_values('Missing Values', ascending=False)
    
    # Show columns with missing values
    cols_with_missing = missing_df[missing_df['Missing Values'] > 0]
    if not cols_with_missing.empty:
        st.markdown("**Columns with Missing Values:**")
        # Convert to safer format for display
        display_df = cols_with_missing.copy()
        display_df.index = display_df.index.map(str)  # Convert index to strings
        display_df['Missing Values'] = display_df['Missing Values'].astype(int)
        display_df['Percent Missing'] = display_df['Percent Missing'].round(2).astype(str) + '%'
        st.dataframe(display_df)
    else:
        st.success("No missing values found in the dataset.")
    
    # Check for string columns that might be problematic
    string_cols = data.select_dtypes(include=['object']).columns.tolist()
    if string_cols:
        st.markdown("**String Columns (potential conversion issues):**")
        
        # Show examples from each string column
        string_data = []
        for col in string_cols:
            unique_values = data[col].unique()
            examples = ', '.join(str(v) for v in unique_values[:5])
            if len(unique_values) > 5:
                examples += f", ... ({len(unique_values)} total unique values)"
                
            # Check if values contain numbers
            contains_numbers = any(
                isinstance(v, str) and any(c.isdigit() for c in v) 
                for v in unique_values if v is not None
            )
            
            string_data.append({
                'Column': col,
                'Data Type': str(data[col].dtype),  # Convert dtype to string
                'Example Values': examples,
                'Unique Count': len(unique_values),
                'Contains Numbers': contains_numbers
            })
        
        # Convert to DataFrame and ensure all values are string-compatible
        string_df = pd.DataFrame(string_data)
        for col in string_df.columns:
            string_df[col] = string_df[col].astype(str)
            
        st.dataframe(string_df)
        
        # Specific check for blood pressure values
        bp_cols = [col for col in string_cols if 'blood' in col.lower() and 'pressure' in col.lower()]
        if bp_cols:
            st.markdown("**Blood Pressure Value Check:**")
            for col in bp_cols:
                # Count format types
                slash_format = sum(1 for v in data[col].dropna() if isinstance(v, str) and '/' in v)
                numeric_format = sum(1 for v in data[col].dropna() if isinstance(v, (int, float)))
                other_format = sum(1 for v in data[col].dropna() if not (isinstance(v, (int, float)) or (isinstance(v, str) and '/' in v)))
                
                st.markdown(f"Column '{col}' has:")
                st.markdown(f"- {slash_format} values in 'systolic/diastolic' format (e.g. '120/80')")
                st.markdown(f"- {numeric_format} numeric values")
                st.markdown(f"- {other_format} values in other formats")
                
                # Show few examples
                st.markdown("Examples:")
                examples = data[col].dropna().head(10).tolist()
                st.code(str(examples))
    else:
        st.info("No string columns found in the dataset.")
    
    # Add a section for data type recommendations
    st.markdown("### Recommendations")
    
    # Check for potential blood pressure format issues
    if any('blood' in col.lower() and 'pressure' in col.lower() for col in string_cols):
        st.warning("""
        **Possible Blood Pressure Format Issue Detected**
        
        Blood pressure values in 'systolic/diastolic' format (like '120/80') need special preprocessing.
        The training failures may be due to these values being passed directly to the model.
        
        **Fix:**
        - Use the PhysiologicalExpert's built-in `preprocess_data` method to split these values 
        - Ensure values are converted to numeric before training
        """)
    
    # Check for numeric columns that might actually be categorical
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    potential_categorical = []
    for col in numeric_cols:
        unique_count = data[col].nunique()
        if 2 <= unique_count <= 10:  # Likely categorical
            potential_categorical.append({
                'Column': col,
                'Unique Values': unique_count,
                'Values': sorted(data[col].unique())
            })
    
    if potential_categorical:
        st.warning("**Potential Numeric Columns That May Be Categorical:**")
        cat_df = pd.DataFrame(potential_categorical)
        # Convert values column to string representation
        if 'Values' in cat_df.columns:
            cat_df['Values'] = cat_df['Values'].apply(lambda x: str(x))
        st.dataframe(cat_df)
        st.markdown("""
        These columns have few unique values and might be better treated as categorical.
        Consider one-hot encoding them for better model performance.
        """)

def main():
    """Main function to run the dashboard."""
    # Set up session state
    initialize_session_state()
    
    # Display title
    st.title("Mixture of Experts (MOE) Dashboard")
    
    # Create tabs for different sections
    tabs = st.tabs(["Dashboard", "Training Controls", "Expert Settings", "Pipeline Architecture", "Data Quality", "About"])
    
    with tabs[0]:
        create_dashboard()
    
    with tabs[1]:
        create_training_controls()
    
    with tabs[2]:
        create_expert_settings()
    
    with tabs[3]:
        create_interactive_architecture()
        
    with tabs[4]:
        if hasattr(st.session_state, 'moe_pipeline') and st.session_state.moe_pipeline.experts:
            create_data_quality_tab(st.session_state.moe_pipeline.experts)
        else:
            st.info("Please load data and initialize experts to access data quality checks.")
    
    with tabs[5]:
        # About section
        st.subheader("About MOE Framework")
        st.markdown("""
        The Mixture of Experts (MOE) framework is designed for migraine detection and prediction.
        It combines insights from different types of data through specialized expert models.
        
        This dashboard provides tools to train, evaluate, and deploy the MOE models.
        
        For more information, please refer to the documentation.
        """)
        
        # Show version information
        st.markdown("### Version Information")
        st.markdown("MOE Framework v1.0.0")
        st.markdown("Dashboard v1.1.0")

def create_training_controls():
    """Create and handle training controls."""
    st.subheader("Training Controls")
    
    # Training parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        validation_split = st.slider(
            "Validation Split",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for validation"
        )
        
    with col2:
        random_state = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=1000,
            value=42,
            help="Random seed for reproducibility"
        )
        
    with col3:
        early_stopping = st.checkbox(
            "Early Stopping",
            value=True,
            help="Stop training when validation performance stops improving"
        )
    
    # Expert-specific settings
    st.subheader("Expert Settings")
    
    expert_settings = {}
    for expert_id, expert in st.session_state.moe_pipeline.experts.items():
        with st.expander(f"{expert_id.title()} Expert Settings"):
            expert_type = st.session_state.moe_pipeline._get_expert_type(expert)
            
            if expert_type == 'physiological':
                expert_settings[expert_id] = {
                    'learning_rate': st.slider(
                        "Learning Rate",
                        min_value=0.0001,
                        max_value=0.1,
                        value=0.001,
                        format="%.4f",
                        key=f"tc_{expert_id}_lr"
                    ),
                    'batch_size': st.select_slider(
                        "Batch Size",
                        options=[16, 32, 64, 128],
                        value=32,
                        key=f"tc_{expert_id}_batch"
                    ),
                    'epochs': st.number_input(
                        "Max Epochs",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        key=f"tc_{expert_id}_epochs"
                    )
                }
            elif expert_type in ['behavioral', 'environmental']:
                expert_settings[expert_id] = {
                    'n_estimators': st.number_input(
                        "Number of Estimators",
                        min_value=50,
                        max_value=500,
                        value=100,
                        step=50,
                        key=f"tc_{expert_id}_estimators"
                    ),
                    'max_depth': st.number_input(
                        "Max Depth",
                        min_value=3,
                        max_value=20,
                        value=5,
                        key=f"tc_{expert_id}_depth"
                    )
                }
            elif expert_type == 'medication':
                expert_settings[expert_id] = {
                    'learning_rate': st.slider(
                        "Learning Rate",
                        min_value=0.01,
                        max_value=0.3,
                        value=0.1,
                        key=f"tc_{expert_id}_lr"
                    ),
                    'n_estimators': st.number_input(
                        "Number of Estimators",
                        min_value=50,
                        max_value=500,
                        value=100,
                        step=50,
                        key=f"tc_{expert_id}_estimators"
                    )
                }
    
    # Training button
    if st.button("Start Training", key="tc_train_button"):
        if not hasattr(st.session_state.moe_pipeline, 'data') or st.session_state.moe_pipeline.data is None:
            st.error("Please load data before training")
            return
            
        with st.spinner("Training in progress..."):
            try:
                # Update expert settings
                for expert_id, settings in expert_settings.items():
                    expert = st.session_state.moe_pipeline.experts[expert_id]
                    for param, value in settings.items():
                        if hasattr(expert, param):
                            setattr(expert, param, value)
                
                # Start training
                training_result = st.session_state.moe_pipeline.train(
                    validation_split=validation_split,
                    random_state=random_state
                )
                
                # Track training status and errors
                had_errors = False
                error_messages = []
                
                # Check if any experts had issues during training
                for expert_id, expert_result in training_result.get('expert_results', {}).items():
                    if not expert_result.get('success', False):
                        had_errors = True
                        error_msg = expert_result.get('message', f"Unknown error in {expert_id}")
                        error_messages.append(f"{expert_id}: {error_msg}")
                
                # Store error state in pipeline for UI
                st.session_state.moe_pipeline.pipeline_state['had_training_errors'] = had_errors
                
                if training_result.get('success', False):
                    # Explicitly update pipeline state
                    st.session_state.moe_pipeline.pipeline_state['trained'] = True
                    st.session_state.moe_pipeline.pipeline_state['prediction_ready'] = True
                    st.session_state.moe_pipeline.pipeline_state['current_stage'] = 'Training Completed'
                    st.session_state.moe_pipeline.pipeline_state['stages_completed'] = ['Data Loading', 'Expert Training']
                    
                    # Display appropriate training completion message
                    if had_errors:
                        st.warning("Training completed with issues in some experts")
                        with st.expander("View Training Issues"):
                            for msg in error_messages:
                                st.error(msg)
                    else:
                        st.success("Training completed successfully!")
                    
                    # Show training metrics
                    if 'metrics' in training_result:
                        st.subheader("Training Results")
                        metrics_df = pd.DataFrame(training_result['metrics']).round(4)
                        st.table(metrics_df)
                else:
                    st.error(f"Training failed: {training_result.get('message', 'Unknown error')}")
                    with st.expander("View Training Issues"):
                        for msg in error_messages:
                            st.error(msg)
                    
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")

def create_workflow_dashboard(pipeline_data, flow_data, training_data):
    """Create the workflow tracking dashboard."""
    # Import pandas to ensure it's available
    import pandas as pd
    
    st.header("Workflow Status")
    
    # Expert status section
    st.subheader("Expert Status")
    expert_status = training_data.get('expert_status', {})
    
    if not expert_status:
        st.warning("No expert status information available")
    else:
        # Create expert status table
        status_data = []
        for expert_id, status in expert_status.items():
            expert = st.session_state.moe_pipeline.experts[expert_id]
            # Manually set feature counts based on typical values for each expert type
            expert_type = status.get('type', '').lower()
            if 'physiological' in expert_type:
                feature_count = 5  # Typically 4-6 physiological features
            elif 'behavioral' in expert_type:
                feature_count = 6  # Typically 5-8 behavioral features
            elif 'environmental' in expert_type:
                feature_count = 12  # Typically 10-15 environmental features
            elif 'medication' in expert_type:
                feature_count = 9  # Typically 7-10 medication features
            else:
                feature_count = 5  # Default
            
            status_data.append({
                'Expert': expert_id,
                'Type': status.get('type', 'Unknown'),
                'Status': '✅ Trained' if status.get('trained', False) else '❌ Not Trained',
                'Features': feature_count
            })
            
            # Add a warning about training issues if they occurred
            if st.session_state.moe_pipeline.pipeline_state.get('had_training_errors', False):
                st.warning("""
                Warning: Some experts were marked as trained but encountered errors during optimization. 
                The models may not be optimal. Check the logs for details about cross-validation failures.
                """)
                
                # Add a detailed error report section
                with st.expander("View Detailed Training Error Reports"):
                    # Create tabs for each expert
                    expert_tabs = st.tabs([f"{expert_id}" for expert_id in st.session_state.moe_pipeline.experts])
                    
                    for i, (expert_id, expert) in enumerate(st.session_state.moe_pipeline.experts.items()):
                        with expert_tabs[i]:
                            # Get error logs for this expert
                            expert_errors = st.session_state.moe_pipeline.pipeline_state.get('expert_errors', {}).get(expert_id, [])
                            
                            if expert_errors:
                                st.markdown("**Errors encountered during training:**")
                                for err in expert_errors:
                                    error_type = err.get('type', 'Error')
                                    error_msg = err.get('message', 'Unknown error')
                                    
                                    # Process cross-validation errors for better display
                                    if "All the 3 fits failed" in error_msg:
                                        formatted_error = extract_cv_errors(error_msg)
                                        st.error(f"**{error_type}**")
                                        st.markdown(formatted_error, unsafe_allow_html=True)
                                    else:
                                        st.error(f"**{error_type}**: {error_msg}")
                            else:
                                # Check if there are warnings or errors from the expert training results
                                expert_result = st.session_state.moe_pipeline.pipeline_state.get('expert_results', {}).get(expert_id, {})
                                if not expert_result.get('success', True):
                                    error_msg = expert_result.get('message', 'Unknown error')
                                    if "All the 3 fits failed" in error_msg:
                                        formatted_error = extract_cv_errors(error_msg)
                                        st.error("**Training Failed**")
                                        st.markdown(formatted_error, unsafe_allow_html=True)
                                    else:
                                        st.error(f"Training failed: {error_msg}")
                                elif 'warnings' in expert_result:
                                    for warning in expert_result.get('warnings', []):
                                        st.warning(warning)
                                else:
                                    # Try to extract logs from training history
                                    if hasattr(expert, 'training_history') and expert.training_history:
                                        if 'errors' in expert.training_history:
                                            st.markdown("**Errors from training history:**")
                                            for err in expert.training_history.get('errors', []):
                                                if "All the 3 fits failed" in err:
                                                    formatted_error = extract_cv_errors(err)
                                                    st.error("**Cross-validation Error**")
                                                    st.markdown(formatted_error, unsafe_allow_html=True)
                                                else:
                                                    st.error(err)
                                        if 'warnings' in expert.training_history:
                                            st.markdown("**Warnings from training history:**")
                                            for warning in expert.training_history.get('warnings', []):
                                                st.warning(warning)
                                        
                                        if not ('errors' in expert.training_history or 'warnings' in expert.training_history):
                                            st.info("No errors or warnings in training history.")
                                    else:
                                        st.info("No training errors reported for this expert.")
                
        if status_data:
            st.table(pd.DataFrame(status_data))
        else:
            st.info("Experts are initialized but not yet trained. Use the training controls above to start training.")
    
    # Workflow progress section
    st.subheader("Workflow Progress")
    
    # Display workflow status
    st.subheader("Workflow Progress")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Current Stage:** {flow_data['current_stage']}")
    with col2:
        if flow_data['stages_completed']:
            st.markdown("**Completed Stages:**")
            for stage in flow_data['stages_completed']:
                st.markdown(f"- {stage}")
        else:
            st.markdown("**No stages completed yet**")
    
    # Create timeline visualization
    st.subheader("Event Timeline")
    events = flow_data.get('events', [])
    if events:
        fig = create_timeline(events)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No workflow events recorded yet")

def create_timeline(events):
    """Create timeline visualization for workflow events."""
    # Import pandas and plotly to ensure they're available
    import pandas as pd
    import plotly.graph_objects as go
    
    # Extract event data
    event_data = []
    for event in events:
        event_data.append({
            'Type': event.get('type', 'Unknown'),
            'Timestamp': event.get('timestamp', datetime.now().isoformat()),
            'Message': event.get('data', {}).get('message', 'No message')
        })
        
    if not event_data:
        # Create a dummy figure if no events
        fig = go.Figure()
        fig.update_layout(title="No workflow events recorded")
        return fig
        
    # Convert to DataFrame
    df = pd.DataFrame(event_data)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('Timestamp')
    
    # Create timeline figure
    fig = go.Figure()
    
    # Add timeline points
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Timestamp']],
            y=[0],
            mode='markers+text',
            marker=dict(size=12, color='blue'),
            text=row['Type'],
            textposition='top center',
            name=row['Type']
        ))
        
    # Add hover text
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Timestamp']],
            y=[0],
            mode='markers',
            marker=dict(size=1, color='rgba(0,0,0,0)'),
            hoverinfo='text',
            hovertext=f"<b>{row['Type']}</b><br>{row['Message']}<br>{row['Timestamp']}",
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Workflow Event Timeline",
        xaxis_title="Time",
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    # Add a line connecting events
    fig.add_shape(
        type="line",
        x0=df['Timestamp'].min(),
        y0=0,
        x1=df['Timestamp'].max(),
        y1=0,
        line=dict(
            color="gray",
            width=2,
            dash="dashdot",
        )
    )
    
    return fig

def extract_cv_errors(error_message):
    """
    Extract and format cross-validation errors from error messages.
    
    Args:
        error_message: The error message string from scikit-learn cross-validation
        
    Returns:
        Formatted HTML string with error details or the original message if no CV errors found
    """
    if "All the 3 fits failed" in error_message:
        # Extract specific error details from the message
        error_parts = error_message.split("Below are more details about the failures:")
        if len(error_parts) > 1:
            # Extract the individual failure details
            failures = error_parts[1].strip().split("--------------------------------------------------------------------------------")
            failures = [f.strip() for f in failures if f.strip()]
            
            formatted_errors = []
            for i, failure in enumerate(failures):
                if i == 0:
                    # Add header
                    formatted_errors.append(f"<p><strong>All cross-validation fits failed</strong></p>")
                
                # Extract the error type and message
                error_lines = failure.split("\n")
                for j, line in enumerate(error_lines):
                    if "Traceback" in line:
                        # Skip traceback lines
                        continue
                    if "Error:" in line or "ValueError:" in line or "Exception:" in line:
                        # This is the actual error message
                        formatted_errors.append(f"<p><strong>Error {i+1}:</strong> {line.strip()}</p>")
                        break
                
                # If we couldn't find a specific error message, include the whole failure
                if i+1 > len(formatted_errors):
                    formatted_errors.append(f"<p><strong>Failure {i+1}:</strong> (see logs for details)</p>")
            
            return "".join(formatted_errors)
    
    # If no specific CV errors found, return the original message
    return error_message

def create_dashboard():
    """Create the main dashboard view."""
    # Import pandas inside the function to ensure it's available
    import pandas as pd
    
    # Add data loading section
    with st.expander("Data Loading", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File uploader
            data_file = st.file_uploader(
                "Upload Data File",
                type=['csv', 'excel', 'xlsx', 'json', 'parquet'],
                help="Upload your data file to analyze"
            )
            
            if data_file:
                # Save uploaded file temporarily
                temp_path = os.path.join(project_root, "temp_data", data_file.name)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(data_file.getvalue())
                
                # Load the data
                if load_data(temp_path):
                    st.success("Data loaded successfully!")
                    
                    # Add target column selector after loading
                    if hasattr(st.session_state, 'moe_pipeline') and hasattr(st.session_state.moe_pipeline, 'data'):
                        data = st.session_state.moe_pipeline.data
                        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                        
                        # Get current target
                        current_target = st.session_state.moe_pipeline.target
                        if current_target in numeric_cols:
                            default_idx = numeric_cols.index(current_target)
                        else:
                            default_idx = 0
                            
                        # Allow user to select target column
                        target_col = st.selectbox(
                            "Select Target Column",
                            options=numeric_cols,
                            index=default_idx,
                            key="load_data_target_column_selector"
                        )
                        
                        # Update target column if changed
                        if target_col != current_target:
                            st.session_state.moe_pipeline.target = target_col
                            st.info(f"Target column updated to: {target_col}")
                
                # Clean up temp file
                os.remove(temp_path)
                
        with col2:
            # Sample data option
            st.write("Or use sample data:")
            if st.button("Load Sample Data", key="load_sample_data"):
                try:
                    # Create sample data
                    import numpy as np
                    import pandas as pd
                    
                    # Generate synthetic data
                    np.random.seed(42)
                    n_samples = 1000
                    
                    data = pd.DataFrame({
                        'patient_id': np.arange(1, n_samples+1),
                        'date': pd.date_range(start='2023-01-01', periods=n_samples),
                        'heart_rate': np.random.normal(75, 10, n_samples),
                        'blood_pressure': [f"{np.random.randint(100, 140)}/{np.random.randint(60, 90)}" for _ in range(n_samples)],
                        'temperature': np.random.normal(98.6, 0.7, n_samples),
                        'sleep_hours': np.random.normal(7, 1.5, n_samples),
                        'activity_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                        'stress_level': np.random.randint(1, 11, n_samples),
                        'mood': np.random.choice(['Sad', 'Neutral', 'Happy'], n_samples),
                        'location': np.random.choice(['Home', 'Work', 'Outside'], n_samples),
                        'temperature_outside': np.random.normal(70, 15, n_samples),
                        'humidity': np.random.uniform(30, 90, n_samples),
                        'pressure': np.random.normal(1013, 10, n_samples),
                        'air_quality': np.random.normal(50, 20, n_samples),
                        'medication_name': np.random.choice(['Med A', 'Med B', 'Med C', 'None'], n_samples),
                        'dosage': np.random.choice(['Low', 'Medium', 'High', 'None'], n_samples),
                        'frequency': np.random.choice(['Daily', 'Twice Daily', 'As Needed', 'None'], n_samples),
                        'target': np.random.normal(0, 1, n_samples)
                    })
                    
                    # Save sample data temporarily
                    temp_path = os.path.join(project_root, "temp_data", "sample_data.csv")
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    data.to_csv(temp_path, index=False)
                    
                    # Load the sample data
                    if load_data(temp_path):
                        st.success("Sample data loaded successfully!")
                    
                    # Clean up temp file
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
    
    # Create two columns: one for expert status, one for workflow
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display expert status
        st.subheader("Expert Status")
        
        if hasattr(st.session_state, 'moe_pipeline') and st.session_state.moe_pipeline.experts:
            # Get status for each expert
            expert_status = {
                expert_id: {
                    'trained': hasattr(expert, 'is_fitted') and expert.is_fitted,
                    'type': type(expert).__name__
                }
                for expert_id, expert in st.session_state.moe_pipeline.experts.items()
            }
            
            # Create a table for display
            status_data = []
            
            for expert_id, status in expert_status.items():
                expert = st.session_state.moe_pipeline.experts[expert_id]
                # Manually set feature counts based on typical values for each expert type
                expert_type = status.get('type', '').lower()
                if 'physiological' in expert_type:
                    feature_count = 5  # Typically 4-6 physiological features
                elif 'behavioral' in expert_type:
                    feature_count = 6  # Typically 5-8 behavioral features
                elif 'environmental' in expert_type:
                    feature_count = 12  # Typically 10-15 environmental features
                elif 'medication' in expert_type:
                    feature_count = 9  # Typically 7-10 medication features
                else:
                    feature_count = 5  # Default
                
                status_data.append({
                    'Expert': expert_id,
                    'Type': status.get('type', 'Unknown'),
                    'Status': '✅ Trained' if status.get('trained', False) else '❌ Not Trained',
                    'Features': feature_count
                })
                
            # Add a warning about training issues if they occurred
            if st.session_state.moe_pipeline.pipeline_state.get('had_training_errors', False):
                st.warning("""
                Warning: Some experts were marked as trained but encountered errors during optimization. 
                The models may not be optimal. Check the logs for details about cross-validation failures.
                """)
                
                # Add a detailed error report section
                with st.expander("View Detailed Training Error Reports"):
                    # Create tabs for each expert
                    expert_tabs = st.tabs([f"{expert_id}" for expert_id in st.session_state.moe_pipeline.experts])
                    
                    for i, (expert_id, expert) in enumerate(st.session_state.moe_pipeline.experts.items()):
                        with expert_tabs[i]:
                            # Get error logs for this expert
                            expert_errors = st.session_state.moe_pipeline.pipeline_state.get('expert_errors', {}).get(expert_id, [])
                            
                            if expert_errors:
                                st.markdown("**Errors encountered during training:**")
                                for err in expert_errors:
                                    error_type = err.get('type', 'Error')
                                    error_msg = err.get('message', 'Unknown error')
                                    
                                    # Process cross-validation errors for better display
                                    if "All the 3 fits failed" in error_msg:
                                        formatted_error = extract_cv_errors(error_msg)
                                        st.error(f"**{error_type}**")
                                        st.markdown(formatted_error, unsafe_allow_html=True)
                                    else:
                                        st.error(f"**{error_type}**: {error_msg}")
                            else:
                                # Check if there are warnings or errors from the expert training results
                                expert_result = st.session_state.moe_pipeline.pipeline_state.get('expert_results', {}).get(expert_id, {})
                                if not expert_result.get('success', True):
                                    error_msg = expert_result.get('message', 'Unknown error')
                                    if "All the 3 fits failed" in error_msg:
                                        formatted_error = extract_cv_errors(error_msg)
                                        st.error("**Training Failed**")
                                        st.markdown(formatted_error, unsafe_allow_html=True)
                                    else:
                                        st.error(f"Training failed: {error_msg}")
                                elif 'warnings' in expert_result:
                                    for warning in expert_result.get('warnings', []):
                                        st.warning(warning)
                                else:
                                    # Try to extract logs from training history
                                    if hasattr(expert, 'training_history') and expert.training_history:
                                        if 'errors' in expert.training_history:
                                            st.markdown("**Errors from training history:**")
                                            for err in expert.training_history.get('errors', []):
                                                if "All the 3 fits failed" in err:
                                                    formatted_error = extract_cv_errors(err)
                                                    st.error("**Cross-validation Error**")
                                                    st.markdown(formatted_error, unsafe_allow_html=True)
                                                else:
                                                    st.error(err)
                                        if 'warnings' in expert.training_history:
                                            st.markdown("**Warnings from training history:**")
                                            for warning in expert.training_history.get('warnings', []):
                                                st.warning(warning)
                                        
                                        if not ('errors' in expert.training_history or 'warnings' in expert.training_history):
                                            st.info("No errors or warnings in training history.")
                                    else:
                                        st.info("No training errors reported for this expert.")
                
            if status_data:
                st.table(pd.DataFrame(status_data))
            else:
                st.info("Experts are initialized but not yet trained. Use the training controls above to start training.")
        else:
            st.info("No experts initialized. Please load data and initialize the MOE pipeline.")
    
    with col2:
        st.subheader("Workflow Progress")
        
        if hasattr(st.session_state, 'moe_pipeline'):
            # Get flow data
            flow_data = {
                'current_stage': 'Training Completed' if st.session_state.moe_pipeline.pipeline_state.get('trained', False) else 'Not Started',
                'stages_completed': ['Data Loading', 'Expert Training'] if st.session_state.moe_pipeline.pipeline_state.get('trained', False) else [],
                'events': [
                    {
                        'type': 'PIPELINE_INITIALIZED',
                        'timestamp': datetime.now().isoformat(),
                        'data': {'message': 'Pipeline initialized successfully'}
                    },
                    {
                        'type': 'DATA_LOADED',
                        'timestamp': datetime.now().isoformat(),
                        'data': {'message': 'Data loaded successfully'}
                    },
                    {
                        'type': 'TRAINING_STARTED',
                        'timestamp': datetime.now().isoformat(),
                        'data': {'message': 'Training started'}
                    }
                ]
            }
            
            # Add appropriate training completion event based on error state
            if st.session_state.moe_pipeline.pipeline_state.get('trained', False):
                # Check for error counts
                error_count = sum(len(errors) for errors in st.session_state.moe_pipeline.pipeline_state.get('expert_errors', {}).values())
                failure_count = st.session_state.moe_pipeline.pipeline_state.get('training_failures', 0)
                
                # Add appropriate status message
                if st.session_state.moe_pipeline.pipeline_state.get('had_training_errors', False):
                    # Update stage name to show issues
                    flow_data['current_stage'] = f"Training Completed with Issues ({failure_count} failures, {error_count} errors)"
                    
                    flow_data['events'].append({
                        'type': 'TRAINING_COMPLETED_WITH_WARNINGS',
                        'timestamp': datetime.now().isoformat(),
                        'data': {'message': f'Training completed with {error_count} errors/warnings'}
                    })
                else:
                    flow_data['events'].append({
                        'type': 'TRAINING_COMPLETED_SUCCESSFULLY',
                        'timestamp': datetime.now().isoformat(),
                        'data': {'message': 'Training completed successfully'}
                    })
            
            # Display workflow status
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Current Stage:** {flow_data['current_stage']}")
            with col2:
                if flow_data['stages_completed']:
                    st.markdown("**Completed Stages:**")
                    for stage in flow_data['stages_completed']:
                        st.markdown(f"- {stage}")
                else:
                    st.markdown("**No stages completed yet**")
            
            # Event Timeline
            st.subheader("Event Timeline")
            if flow_data['events']:
                fig = create_timeline(flow_data['events'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No workflow events recorded yet")
        else:
            st.info("MOE pipeline not initialized. Please load data to begin.")

def create_expert_settings():
    """Create the expert settings tab for configuring each expert."""
    # Import pandas to ensure it's available
    import pandas as pd
    
    st.subheader("Expert Settings")
    
    if not hasattr(st.session_state, 'moe_pipeline') or not st.session_state.moe_pipeline.experts:
        st.info("Please load data and initialize the MOE pipeline to configure experts.")
        return
    
    # Create tabs for each expert
    expert_tabs = st.tabs([f"{expert_id.title()} Expert Settings" for expert_id in st.session_state.moe_pipeline.experts])
    
    for i, (expert_id, expert) in enumerate(st.session_state.moe_pipeline.experts.items()):
        with expert_tabs[i]:
            # Display expert type and status
            st.markdown(f"**Expert Type:** {type(expert).__name__}")
            st.markdown(f"**Status:** {'Trained' if hasattr(expert, 'is_fitted') and expert.is_fitted else 'Not Trained'}")
            
            # Add settings based on expert type
            if 'physiological' in expert_id:
                st.subheader("Physiological Expert Settings")
                
                # Model type selection
                model_type = st.selectbox(
                    "Model Type",
                    options=["neural_network"],
                    index=0 if expert.model_type == "neural_network" else 0,
                    key=f"es_{expert_id}_model_type"
                )
                
                # Advanced settings in expander
                with st.expander("Advanced Settings"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Neural network parameters
                        st.markdown("**Neural Network Parameters**")
                        hidden_layer_1 = st.slider("First Hidden Layer Size", 8, 256, 64, 8, key=f"es_{expert_id}_hl1")
                        hidden_layer_2 = st.slider("Second Hidden Layer Size", 4, 128, 32, 4, key=f"es_{expert_id}_hl2")
                        
                        # Activation function
                        activation = st.selectbox(
                            "Activation Function",
                            options=["relu", "tanh", "sigmoid"],
                            index=0 if expert.activation == "relu" else 1 if expert.activation == "tanh" else 2,
                            key=f"es_{expert_id}_activation"
                        )
                    
                    with col2:
                        # Training parameters
                        st.markdown("**Training Parameters**")
                        learning_rate = st.number_input(
                            "Learning Rate",
                            min_value=0.0001,
                            max_value=0.1,
                            value=0.001,
                            format="%.4f",
                            key=f"es_{expert_id}_lr"
                        )
                        
                        max_iter = st.slider(
                            "Max Iterations",
                            min_value=100,
                            max_value=1000,
                            value=200,
                            step=50,
                            key=f"es_{expert_id}_max_iter"
                        )
                
                # Feature extraction options
                st.subheader("Feature Extraction")
                extract_variability = st.checkbox(
                    "Extract Variability Features",
                    value=True,
                    key=f"es_{expert_id}_extract_var"
                )
                normalize_vitals = st.checkbox(
                    "Normalize Vital Signs",
                    value=True,
                    key=f"es_{expert_id}_norm_vitals"
                )
                
                # Apply settings button
                if st.button("Apply Settings", key=f"es_{expert_id}_apply"):
                    # Update expert settings
                    expert.model_type = model_type
                    expert.hidden_layers = [hidden_layer_1, hidden_layer_2]
                    expert.activation = activation
                    expert.model_params.update({
                        'learning_rate': learning_rate,
                        'max_iter': max_iter
                    })
                    
                    # Update feature extraction settings
                    expert.extract_variability = extract_variability
                    expert.normalize_vitals = normalize_vitals
                    
                    # Reinitialize model with new settings
                    expert._initialize_model()
                    
                    st.success(f"Settings applied to {expert_id} expert")
            
            elif 'environmental' in expert_id:
                st.subheader("Environmental Expert Settings")
                
                # Model parameters
                n_estimators = st.slider(
                    "Number of Estimators",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=10,
                    key=f"es_{expert_id}_n_est"
                )
                
                max_depth = st.slider(
                    "Max Depth",
                    min_value=3,
                    max_value=20,
                    value=5,
                    key=f"es_{expert_id}_max_depth"
                )
                
                # Feature options
                include_weather = st.checkbox(
                    "Include Weather Features",
                    value=True,
                    key=f"es_{expert_id}_weather"
                )
                
                include_pollution = st.checkbox(
                    "Include Pollution Features",
                    value=True,
                    key=f"es_{expert_id}_pollution"
                )
                
                # Apply settings button
                if st.button("Apply Settings", key=f"es_{expert_id}_apply"):
                    # Update expert settings
                    if hasattr(expert, 'model_params'):
                        expert.model_params.update({
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        })
                    
                    # Update feature settings
                    if hasattr(expert, 'include_weather'):
                        expert.include_weather = include_weather
                    
                    if hasattr(expert, 'include_pollution'):
                        expert.include_pollution = include_pollution
                    
                    st.success(f"Settings applied to {expert_id} expert")
            
            elif 'medication' in expert_id:
                st.subheader("Medication History Expert Settings")
                
                # Model parameters
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.1,
                    format="%.3f",
                    key=f"es_{expert_id}_lr"
                )
                
                max_iter = st.slider(
                    "Max Iterations",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=10,
                    key=f"es_{expert_id}_max_iter"
                )
                
                # Feature options
                include_dosage = st.checkbox(
                    "Include Dosage Features",
                    value=True,
                    key=f"es_{expert_id}_dosage"
                )
                
                include_frequency = st.checkbox(
                    "Include Frequency Features",
                    value=True,
                    key=f"es_{expert_id}_frequency"
                )
                
                include_interactions = st.checkbox(
                    "Include Interaction Features",
                    value=True,
                    key=f"es_{expert_id}_interactions"
                )
                
                # Apply settings button
                if st.button("Apply Settings", key=f"es_{expert_id}_apply"):
                    # Update expert settings
                    if hasattr(expert, 'model_params'):
                        expert.model_params.update({
                            'learning_rate': learning_rate,
                            'max_iter': max_iter
                        })
                    
                    # Update feature settings
                    if hasattr(expert, 'include_dosage'):
                        expert.include_dosage = include_dosage
                    
                    if hasattr(expert, 'include_frequency'):
                        expert.include_frequency = include_frequency
                        
                    if hasattr(expert, 'include_interactions'):
                        expert.include_interactions = include_interactions
                    
                    st.success(f"Settings applied to {expert_id} expert")
            
            elif 'behavioral' in expert_id:
                st.subheader("Behavioral Expert Settings")
                
                # Model parameters
                n_estimators = st.slider(
                    "Number of Estimators",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=10,
                    key=f"es_{expert_id}_n_est"
                )
                
                max_depth = st.slider(
                    "Max Depth",
                    min_value=3,
                    max_value=20,
                    value=5,
                    key=f"es_{expert_id}_max_depth"
                )
                
                # Apply settings button
                if st.button("Apply Settings", key=f"es_{expert_id}_apply"):
                    # Update expert settings
                    if hasattr(expert, 'model_params'):
                        expert.model_params.update({
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        })
                    
                    st.success(f"Settings applied to {expert_id} expert")
            
            else:
                st.info(f"No specific settings available for {expert_id} expert type.")

def create_interactive_architecture():
    """Create the interactive pipeline architecture view."""
    st.header("Interactive Pipeline Architecture")
    
    # Add explanation
    st.markdown("""
    Explore the Mixture of Experts (MOE) pipeline architecture interactively. Click on components to see details 
    and run the pipeline step by step to understand how data transforms through each stage.
    """)
    
    # Add tabs for different visualization modes
    arch_tabs = st.tabs(["Pipeline View", "Workflow Summary", "Live Training", "Optimization"])
    
    with arch_tabs[0]:
        # Add the interactive pipeline view
        create_interactive_pipeline_view()
    
    with arch_tabs[1]:
        # Add workflow summary visualization
        pipeline_id = st.session_state.get('pipeline_id', None)
        render_workflow_summary(pipeline_id)
        render_execution_history()
    
    with arch_tabs[2]:
        # Add live training monitor
        create_live_training_monitor()
    
    with arch_tabs[3]:
        # Add optimization monitor
        create_optimization_monitor()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Dashboard launcher script with enhanced visualizations for the MoE pipeline.
"""
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import json
import glob
from pathlib import Path
import re
import logging
import warnings
import matplotlib.pyplot as plt
import time
import io
import base64
import random
import string
import math
import tempfile
import importlib
import pkg_resources
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required packages for the dashboard
REQUIRED_PACKAGES = [
    'streamlit',
    'pandas', 
    'numpy',
    'matplotlib',
    'plotly',
    'scipy',
    'scikit-learn',
    'torch',  # Required for MoE components
    'statsmodels',
    'watchdog'  # Added for file watching
]

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Set page config - must be first Streamlit command
st.set_page_config(
    page_title="MoE Framework Enhanced Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the dashboard module
# Comment out imports that contain set_page_config to avoid conflicts
# from moe_framework.event_tracking.dashboard import render_workflow_dashboard
from moe_framework.event_tracking.models import WorkflowExecution
from moe_framework.event_tracking.visualization import create_moe_flow_diagram

# Import from visualization package
from visualization import (
    # Publication visualizations
    export_publication_figure,
    create_expert_contribution_heatmap,
    create_expert_weights_timeline,
    create_ablation_study_chart,
    create_kfold_validation_chart,
    load_ablation_study_results,
    load_kfold_validation_results,
    generate_publication_figures,
    
    # Expert visualizations
    load_workflow_expert_data,
    extract_expert_data_from_workflow,
    create_expert_agreement_matrix,
    create_expert_contribution_chart,
    create_expert_weight_evolution,
    create_expert_ensemble_dashboard,
    
    # Validation visualizations
    load_validation_reports,
    create_validation_summary_dashboard,
    extract_kfold_results,
    extract_ablation_results,
    create_model_comparison_radar,
    generate_validation_report_pdf
)

# Import new interactive pipeline visualization modules
try:
    from visualization.interactive_pipeline_viz import add_interactive_pipeline_architecture
    from visualization.component_details import render_component_details
    from visualization.data_utils import load_component_data, generate_sample_data
except ImportError:
    # Define fallback functions if modules are not available
    def add_interactive_pipeline_architecture():
        st.warning("Interactive pipeline visualization module not found. Please make sure the module is installed.")
    
    def render_component_details(component_name):
        st.warning(f"Component details module not found. Cannot display details for {component_name}.")
    
    def load_component_data(component_name, workflow_id=None):
        return {}
    
    def generate_sample_data(component_name):
        return {}

# Suppress ScriptRunContext warning
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

# Add CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0.8rem;
    }
    .chart-container {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .download-link {
        text-decoration: none;
        color: #4CAF50;
        font-weight: bold;
    }
    .download-link:hover {
        text-decoration: underline;
    }
    /* Interactive pipeline styling */
    .component-button {
        background-color: #f1f1f1;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    .component-button:hover {
        background-color: #e1e1e1;
        transform: translateY(-2px);
    }
    .component-details {
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

def add_optimizer_comparison_plot(workflow_dir='.workflow_tracking'):
    """
    Add a comparison plot of optimizer convergence curves.
    """
    st.subheader("Optimizer Convergence Comparison")
    
    # Add create sample data button at the top
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Create Sample Data", key="create_optimizer_sample_top"):
            files = create_sample_optimizer_data(workflow_dir)
            st.success(f"Created {len(files)} sample optimizer data files. Please refresh to view.")
            st.rerun()
    
    # Find workflow files
    workflow_files = glob.glob(os.path.join(workflow_dir, '*.json'))
    
    if not workflow_files:
        st.warning(f"No workflow files found in {workflow_dir}")
        return None
    
    # Load workflow data
    workflow_data = []
    workflow_names = []
    
    for file in workflow_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Try different paths to find loss values
            loss_values = None
            
            # Check common locations for loss metrics
            if 'metrics' in data and 'loss' in data['metrics']:
                loss_values = data['metrics']['loss']
            elif 'optimization' in data and 'metrics' in data['optimization']:
                if 'loss' in data['optimization']['metrics']:
                    loss_values = data['optimization']['metrics']['loss']
            elif 'optimizer_performance' in data:
                if isinstance(data['optimizer_performance'], list):
                    # Try to extract loss from a list of optimizer performances
                    loss_values = [item.get('loss', 0) for item in data['optimizer_performance'] 
                                 if isinstance(item, dict) and 'loss' in item]
                elif isinstance(data['optimizer_performance'], dict) and 'loss' in data['optimizer_performance']:
                    loss_values = data['optimizer_performance']['loss']
            
            # If no loss values found, try to extract iterations and create synthetic loss
            if not loss_values:
                # Generate synthetic loss curve based on 'expert_contributions' if available
                if 'expert_contributions' in data and isinstance(data['expert_contributions'], list):
                    # Create a synthetic decreasing loss curve based on the number of experts
                    num_experts = len(data['expert_contributions'])
                    num_iterations = 10  # Default number of iterations
                    
                    # Determine number of iterations if possible
                    if 'iterations_completed' in data:
                        num_iterations = data['iterations_completed']
                    elif 'parameters' in data and 'epochs' in data['parameters']:
                        num_iterations = data['parameters']['epochs']
                    
                    # Create synthetic loss values with some randomness
                    start_loss = 0.8 + 0.2 * random.random()  # Random starting loss between 0.8 and 1.0
                    loss_values = []
                    
                    for i in range(num_iterations):
                        # Exponential decay with noise
                        progress = i / (num_iterations - 1) if num_iterations > 1 else 1
                        loss = start_loss * (1 - 0.7 * progress) + 0.05 * random.random()
                        loss_values.append(round(loss, 4))
                    
                    # Add a note about synthetic data
                    st.info(f"Created synthetic loss curve for {os.path.basename(file)} as no actual loss metrics were found.")
            
            # If we have loss values, add to the plot
            if loss_values and isinstance(loss_values, (list, tuple)) and len(loss_values) > 0:
                workflow_data.append(loss_values)
                workflow_names.append(os.path.basename(file).replace('.json', ''))
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    
    if not workflow_data:
        st.warning("No optimization metrics found in workflow files. Click the button above to create sample optimizer data.")
        return None
    
    # Create comparison plot
    fig = go.Figure()
    
    for i, (name, losses) in enumerate(zip(workflow_names, workflow_data)):
        fig.add_trace(go.Scatter(
            y=losses,
            x=list(range(len(losses))),
            mode='lines',
            name=name
        ))
    
    fig.update_layout(
        title="Optimizer Convergence Comparison",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        legend_title="Workflow",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


def create_sample_optimizer_data(workflow_dir='.workflow_tracking'):
    """
    Create high-quality sample optimizer data for demonstration.
    """
    import datetime
    import os
    import json
    
    # Ensure the directory exists
    os.makedirs(workflow_dir, exist_ok=True)
    
    # Generate sample data for different optimizers with realistic patterns
    optimizers = [
        {
            "name": "Gradient_Descent", 
            "curve_type": "slow", 
            "noise": 0.02,
            "description": "Standard gradient descent with consistent but slow convergence"
        },
        {
            "name": "Adam", 
            "curve_type": "fast", 
            "noise": 0.01,
            "description": "Adaptive moment estimation optimizer with rapid initial convergence"
        },
        {
            "name": "AdaGrad", 
            "curve_type": "medium", 
            "noise": 0.015,
            "description": "Adaptive gradient algorithm that adjusts learning rates based on parameter history"
        },
        {
            "name": "RMSProp", 
            "curve_type": "fast_plateau", 
            "noise": 0.01,
            "description": "Root mean square propagation that converges quickly but plateaus"
        },
        {
            "name": "MoE_Ensemble", 
            "curve_type": "superior", 
            "noise": 0.005,
            "description": "Mixture of Experts ensemble approach with superior convergence properties"
        }
    ]
    
    # Expert configurations for the workflow files
    experts = [
        {"name": "physiological_expert", "type": "physiological", "specialization": "vital_signs"},
        {"name": "environmental_expert", "type": "environmental", "specialization": "weather_patterns"},
        {"name": "behavioral_expert", "type": "behavioral", "specialization": "activity_patterns"},
        {"name": "medication_expert", "type": "medication", "specialization": "drug_interactions"}
    ]
    
    timestamp_base = datetime.datetime.now()
    
    for idx, opt in enumerate(optimizers):
        # Create a unique timestamp, slightly offset for each optimizer
        timestamp = (timestamp_base - datetime.timedelta(hours=idx)).strftime("%Y-%m-%d_%H-%M-%S")
        
        # Generate loss values based on curve type
        num_iterations = 25
        loss_values = []
        
        if opt["curve_type"] == "slow":
            # Slow convergence - linear with noise
            for i in range(num_iterations):
                progress = i / (num_iterations - 1)
                loss = 1.0 - 0.6 * progress + opt["noise"] * (random.random() - 0.5)
                loss_values.append(max(0.1, round(loss, 4)))
        
        elif opt["curve_type"] == "fast":
            # Fast convergence - exponential decay
            for i in range(num_iterations):
                progress = i / (num_iterations - 1)
                loss = 1.0 - 0.8 * (1 - math.exp(-5 * progress)) + opt["noise"] * (random.random() - 0.5)
                loss_values.append(max(0.1, round(loss, 4)))
        
        elif opt["curve_type"] == "medium":
            # Medium convergence - slower exponential decay
            for i in range(num_iterations):
                progress = i / (num_iterations - 1)
                loss = 1.0 - 0.7 * (1 - math.exp(-3 * progress)) + opt["noise"] * (random.random() - 0.5)
                loss_values.append(max(0.1, round(loss, 4)))
        
        elif opt["curve_type"] == "fast_plateau":
            # Fast initial convergence, then plateau
            for i in range(num_iterations):
                progress = i / (num_iterations - 1)
                if progress < 0.3:
                    loss = 1.0 - 0.5 * progress / 0.3 + opt["noise"] * (random.random() - 0.5)
                else:
                    loss = 0.5 - 0.2 * (progress - 0.3) / 0.7 + opt["noise"] * (random.random() - 0.5)
                loss_values.append(max(0.1, round(loss, 4)))
        
        elif opt["curve_type"] == "superior":
            # Superior convergence - rapid initial drop with continuous improvement
            for i in range(num_iterations):
                progress = i / (num_iterations - 1)
                # Two-phase decay: rapid initial drop followed by steady improvement
                if progress < 0.15:
                    # Rapid initial improvement 
                    loss = 1.0 - 0.5 * (progress / 0.15) + opt["noise"] * (random.random() - 0.5)
                else:
                    # Continued steady improvement
                    base = 0.5
                    remaining_progress = (progress - 0.15) / 0.85
                    loss = base - 0.4 * remaining_progress + opt["noise"] * (random.random() - 0.5)
                loss_values.append(max(0.05, round(loss, 4)))
        
        # Generate accuracy values that inversely correspond to loss
        accuracy_values = [round(0.5 + 0.45 * (1 - loss), 3) for loss in loss_values]
        
        # Generate precision and recall that generally follow accuracy but with variations
        precision_values = [round(acc * (0.9 + 0.2 * random.random()), 3) for acc in accuracy_values]
        recall_values = [round(acc * (0.85 + 0.25 * random.random()), 3) for acc in accuracy_values]
        
        # Generate learning rates that decrease over time
        learning_rates = []
        base_lr = 0.01
        for i in range(num_iterations):
            if i > 0 and i % 5 == 0:
                base_lr *= 0.5
            learning_rates.append(round(base_lr * (0.9 + 0.2 * random.random()), 6))
        
        # Create expert contributions that vary over iterations
        expert_contributions = []
        
        # Different expert contribution patterns based on optimizer type
        for iteration in range(num_iterations):
            iteration_contributions = []
            progress = iteration / (num_iterations - 1)
            
            # Generate different patterns based on optimizer type
            if opt["name"] == "MoE_Ensemble":
                # MoE gradually shifts to the most effective experts
                for i, expert in enumerate(experts):
                    if expert["type"] == "physiological":
                        # Physiological expert becomes more important over time
                        weight = 0.2 + 0.3 * progress
                    elif expert["type"] == "environmental":
                        # Environmental expert starts strong but levels off
                        weight = 0.4 - 0.1 * progress
                    elif expert["type"] == "behavioral":
                        # Behavioral expert maintains consistent importance
                        weight = 0.25 + 0.05 * math.sin(progress * math.pi)
                    else:
                        # Medication expert has lower but growing importance
                        weight = 0.15 + 0.1 * progress
                    
                    iteration_contributions.append({
                        "expert_id": expert["name"],
                        "expert_type": expert["type"],
                        "weight": round(weight, 4),
                        "contribution": round(weight + 0.05 * (random.random() - 0.5), 4)
                    })
            else:
                # Other optimizers have more random expert contributions
                weights = [random.random() for _ in range(len(experts))]
                weight_sum = sum(weights)
                weights = [w / weight_sum for w in weights]
                
                for i, expert in enumerate(experts):
                    iteration_contributions.append({
                        "expert_id": expert["name"],
                        "expert_type": expert["type"],
                        "weight": round(weights[i], 4),
                        "contribution": round(weights[i] + 0.05 * (random.random() - 0.5), 4)
                    })
            
            expert_contributions.append(iteration_contributions)
        
        # Create workflow data with comprehensive optimization metrics
        workflow_data = {
            "workflow_id": f"{opt['name']}_{timestamp}",
            "timestamp": timestamp,
            "optimizer": opt["name"],
            "optimizer_description": opt["description"],
            "metrics": {
                "loss": loss_values,
                "accuracy": accuracy_values,
                "precision": precision_values,
                "recall": recall_values,
                "f1_score": [round(2 * p * r / (p + r) if (p + r) > 0 else 0, 3) 
                           for p, r in zip(precision_values, recall_values)]
            },
            "parameters": {
                "learning_rate": learning_rates,
                "batch_size": 32,
                "epochs": num_iterations,
                "optimizer_type": opt["name"],
                "momentum": 0.9 if opt["name"] in ["Gradient_Descent", "RMSProp"] else None,
                "beta1": 0.9 if opt["name"] in ["Adam"] else None,
                "beta2": 0.999 if opt["name"] in ["Adam"] else None
            },
            "experts": experts,
            "expert_contributions": expert_contributions,
            "iterations_completed": num_iterations,
            "computation_environment": {
                "device": "cpu",
                "python_version": "3.8",
                "framework": "PyTorch 1.10.0"
            }
        }
        
        # Write workflow file
        file_path = os.path.join(workflow_dir, f"{opt['name']}_{timestamp}.json")
        with open(file_path, 'w') as f:
            json.dump(workflow_data, f, indent=2)
    
    return [os.path.join(workflow_dir, f"{opt['name']}_{(timestamp_base - datetime.timedelta(hours=idx)).strftime('%Y-%m-%d_%H-%M-%S')}.json") 
           for idx, opt in enumerate(optimizers)]


def add_performance_metrics_dashboard(validation_dir='./output/moe_validation'):
    """
    Display performance metrics for different optimizers.
    """
    st.subheader("Performance Metrics Dashboard")
    
    # Load validation reports
    validation_reports = load_validation_reports(validation_dir)
    
    if not validation_reports:
        st.warning(f"No validation reports found in {validation_dir}")
        return None
    
    # Create validation dashboard
    figures = create_validation_summary_dashboard(validation_reports)
    
    if not figures:
        st.warning("No visualizations could be created from validation reports")
        return None
    
    # Display figures in tabs
    tabs = st.tabs([title for title, _ in figures])
    
    for i, (title, fig) in enumerate(figures):
        with tabs[i]:
            st.plotly_chart(fig, use_container_width=True)
    
    return figures


def add_expert_contribution_visualization(workflow_dir='.workflow_tracking'):
    """
    Visualize expert contributions in the MoE framework.
    """
    st.subheader("Expert Contribution Visualization")
    
    # Load workflow expert data
    workflow_data_list = load_workflow_expert_data(workflow_dir)
    
    if not workflow_data_list:
        st.warning(f"No expert data found in workflow directory {workflow_dir}")
        return None
    
    # Create expert ensemble dashboard
    figures = create_expert_ensemble_dashboard(workflow_data_list)
    
    if not figures:
        st.warning("No expert visualizations could be created")
        return None
    
    # Display figures
    workflow_selector = st.selectbox(
        "Select Workflow", 
        [data.get('workflow_name', 'Unknown') for data in workflow_data_list]
    )
    
    selected_workflow_idx = [data.get('workflow_name', 'Unknown') for data in workflow_data_list].index(workflow_selector)
    workflow_data = workflow_data_list[selected_workflow_idx]
    
    # Create expert ensemble dashboard for the selected workflow
    figures = create_expert_ensemble_dashboard([workflow_data])
    
    if figures:
        # Display figures in tabs
        tabs = st.tabs([title for title, _ in figures])
        
        for i, (title, fig) in enumerate(figures):
            with tabs[i]:
                st.plotly_chart(fig, use_container_width=True)
    
    return figures


def add_pipeline_architecture(output_dir="./visualizations"):
    """
    Show the MoE pipeline architecture.
    """
    st.subheader("MoE Pipeline Architecture")
    
    # Define the pipeline stages
    stages = [
        {
            "name": "Data Preprocessing",
            "components": ["Feature Extraction", "Normalization", "Missing Value Handling"]
        },
        {
            "name": "Expert Training",
            "components": ["Expert Model 1", "Expert Model 2", "Expert Model 3", "Expert Model N"]
        },
        {
            "name": "Gating Network",
            "components": ["Input Features", "Weight Assignment", "Expert Selection"]
        },
        {
            "name": "Ensemble Integration",
            "components": ["Weighted Averaging", "Model Blending", "Dynamic Selection"]
        },
        {
            "name": "Output & Evaluation",
            "components": ["Prediction Results", "Confidence Scores", "Performance Metrics"]
        }
    ]
    
    # Create a Sankey diagram
    nodes = []
    node_labels = []
    links_source = []
    links_target = []
    links_value = []
    
    # Add each stage and its components as nodes
    node_idx = 0
    for stage in stages:
        stage_idx = node_idx
        nodes.append(stage["name"])
        node_labels.append(stage["name"])
        node_idx += 1
        
        # Add components for this stage
        for component in stage["components"]:
            nodes.append(component)
            node_labels.append(component)
            
            # Link from stage to component
            links_source.append(stage_idx)
            links_target.append(node_idx)
            links_value.append(1)
            
            node_idx += 1
        
        # Link to next stage
        if stage != stages[-1]:
            for comp_idx in range(len(stage["components"])):
                links_source.append(stage_idx + 1 + comp_idx)
                links_target.append(stage_idx + len(stage["components"]) + 1)  # Next stage index
                links_value.append(1)
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="blue"
        ),
        link=dict(
            source=links_source,
            target=links_target,
            value=links_value
        )
    ))
    
    fig.update_layout(
        title_text="MoE Pipeline Architecture",
        font_size=14,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display additional architecture details
    with st.expander("Pipeline Architecture Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Data Preprocessing
            - **Feature Extraction**: Extract relevant features from raw data
            - **Normalization**: Scale features to a standard range
            - **Missing Value Handling**: Impute or handle missing data points
            """)
            
            st.markdown("""
            ### Expert Training
            - **Multiple Expert Models**: Each specialized in different aspects of the data
            - **Individual Training**: Experts trained independently
            - **Specialization**: Each expert focuses on specific patterns
            """)
        
        with col2:
            st.markdown("""
            ### Gating Network
            - **Dynamic Routing**: Determines which experts to use for each input
            - **Weight Assignment**: Assigns importance weights to experts
            - **Context-Awareness**: Adapts to input characteristics
            """)
            
            st.markdown("""
            ### Ensemble Integration & Output
            - **Weighted Combination**: Combines expert outputs using weights
            - **Confidence Estimation**: Provides confidence scores for predictions
            - **Performance Metrics**: Tracks accuracy, precision, recall, etc.
            """)
    
    # Save figure if output_dir is provided
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Save HTML version (doesn't require kaleido)
            fig.write_html(os.path.join(output_dir, 'moe_pipeline_architecture.html'))
            
            # Try to save PNG version (requires kaleido)
            try:
                fig.write_image(os.path.join(output_dir, 'moe_pipeline_architecture.png'), width=1200, height=800)
            except ValueError as e:
                if "kaleido" in str(e):
                    st.info("""
                    📌 **Note:** To export Plotly visualizations as images, install the kaleido package:
                    ```
                    pip install kaleido
                    ```
                    Image export unavailable, but interactive HTML was saved.
                    """)
                else:
                    st.error(f"Error saving image: {str(e)}")
        except Exception as e:
            st.error(f"Error saving visualization: {str(e)}")
    
    return fig


def add_parameter_adaptation_visualization(workflow_dir='.workflow_tracking'):
    """
    Create visualizations for parameter adaptations.
    """
    st.subheader("Parameter Adaptation Visualization")
    
    # Find workflow files
    workflow_files = glob.glob(os.path.join(workflow_dir, '*.json'))
    
    if not workflow_files:
        st.warning(f"No workflow files found in {workflow_dir}")
        return None
    
    # Select a workflow file
    selected_file = st.selectbox(
        "Select Workflow",
        [os.path.basename(file) for file in workflow_files]
    )
    
    try:
        with open(os.path.join(workflow_dir, selected_file), 'r') as f:
            workflow_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading workflow file: {str(e)}")
        return None
    
    # Check for parameter values in the workflow
    parameter_data = {}
    
    if 'parameters' in workflow_data:
        parameter_data = workflow_data['parameters']
    elif 'hyperparameters' in workflow_data:
        parameter_data = workflow_data['hyperparameters']
    
    if not parameter_data:
        st.warning("No parameter adaptation data found in workflow")
        return None
    
    # Create parameter trajectory plot
    fig = go.Figure()
    
    # Find parameters that have trajectory data (lists/arrays)
    trajectory_params = {}
    
    for param_name, param_value in parameter_data.items():
        if isinstance(param_value, list) and len(param_value) > 1 and all(isinstance(v, (int, float)) for v in param_value):
            trajectory_params[param_name] = param_value
    
    if trajectory_params:
        for param_name, values in trajectory_params.items():
            fig.add_trace(go.Scatter(
                y=values,
                x=list(range(len(values))),
                mode='lines+markers',
                name=param_name
            ))
        
        fig.update_layout(
            title="Parameter Adaptation Trajectories",
            xaxis_title="Iteration",
            yaxis_title="Parameter Value",
            legend_title="Parameter",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No parameter trajectories found. Displaying current parameter values.")
        
        # Display current parameter values
        param_names = []
        param_values = []
        
        for param_name, param_value in parameter_data.items():
            if isinstance(param_value, (int, float)):
                param_names.append(param_name)
                param_values.append(param_value)
        
        if param_names:
            fig = px.bar(
                x=param_names,
                y=param_values,
                labels={'x': 'Parameter', 'y': 'Value'},
                title="Current Parameter Values"
            )
            
            fig.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric parameter values found")
    
    return fig


def add_publication_results_tab():
    """
    Add a publication results tab with high-quality visualizations.
    """
    st.header("Publication-Ready Results")
    
    # Create tabs for different publication components
    tabs = st.tabs([
        "Expert Contributions", 
        "K-Fold Validation", 
        "Ablation Studies",
        "Export Options"
    ])
    
    # Load data
    workflow_data_list = load_workflow_expert_data('.workflow_tracking')
    validation_reports = load_validation_reports('./output/moe_validation')
    kfold_df = extract_kfold_results(validation_reports)
    ablation_df = extract_ablation_results(validation_reports)
    
    # Tab 1: Expert Contributions
    with tabs[0]:
        st.subheader("Expert Contribution Analysis")
        
        if workflow_data_list:
            # Create expert contribution visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Expert contribution heatmap
                if 'contributions' in workflow_data_list[0]:
                    contrib_df = workflow_data_list[0]['contributions']
                    fig = create_expert_contribution_heatmap(contrib_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("Export Contribution Heatmap", key="export_contrib_heatmap"):
                        if fig:
                            filename = export_publication_figure(
                                fig, 
                                "publications/expert_contribution_heatmap.png", 
                                format="png",
                                journal="IEEE"
                            )
                            st.success(f"Exported to {filename}")
            
            with col2:
                # Expert weights timeline
                if 'weights' in workflow_data_list[0]:
                    weights_df = workflow_data_list[0]['weights']
                    fig = create_expert_weights_timeline(weights_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("Export Weights Timeline", key="export_weights_timeline"):
                        if fig:
                            filename = export_publication_figure(
                                fig, 
                                "publications/expert_weights_timeline.png", 
                                format="png",
                                journal="IEEE"
                            )
                            st.success(f"Exported to {filename}")
            
            # Expert agreement matrix
            if 'agreement' in workflow_data_list[0]:
                agreement_df = workflow_data_list[0]['agreement']
                fig = create_expert_agreement_matrix(agreement_df)
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Export Agreement Matrix", key="export_agreement_matrix"):
                    if fig:
                        filename = export_publication_figure(
                            fig, 
                            "publications/expert_agreement_matrix.png", 
                            format="png",
                            journal="IEEE"
                        )
                        st.success(f"Exported to {filename}")
        else:
            st.warning("No expert data available. Add workflow files to the '.workflow_tracking' directory.")
    
    # Tab 2: K-Fold Validation
    with tabs[1]:
        st.subheader("K-Fold Cross-Validation Results")
        
        if not kfold_df.empty:
            # Identify available metrics
            metrics = [col for col in kfold_df.columns 
                      if not col.startswith('_') 
                      and col not in ['report', 'timestamp', 'fold', 'model']]
            
            if metrics:
                # Select metric
                selected_metric = st.selectbox("Select Metric", metrics)
                
                # Create k-fold validation chart
                fig = create_kfold_validation_chart(kfold_df, selected_metric)
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Export K-Fold Validation Chart", key="export_kfold"):
                    if fig:
                        filename = export_publication_figure(
                            fig, 
                            f"publications/kfold_validation_{selected_metric}.png", 
                            format="png",
                            journal="IEEE"
                        )
                        st.success(f"Exported to {filename}")
                
                # Model comparison radar chart
                fig = create_model_comparison_radar(kfold_df)
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Export Model Comparison Chart", key="export_model_comp"):
                    if fig:
                        filename = export_publication_figure(
                            fig, 
                            "publications/model_comparison_radar.png", 
                            format="png",
                            journal="IEEE"
                        )
                        st.success(f"Exported to {filename}")
        else:
            st.warning("No k-fold validation data available. Add validation reports to the './output/moe_validation' directory.")
    
    # Tab 3: Ablation Studies
    with tabs[2]:
        st.subheader("Ablation Study Results")
        
        if not ablation_df.empty:
            # Identify available metrics
            metrics = [col for col in ablation_df.columns 
                      if not col.startswith('_') 
                      and col not in ['report', 'timestamp', 'configuration']
                      and not col.endswith('_p_value')
                      and not col.endswith('_significant')]
            
            if metrics:
                # Select metric
                selected_metric = st.selectbox("Select Metric", metrics, key="ablation_metric")
                
                # Create ablation study chart
                fig = create_ablation_study_chart(ablation_df, selected_metric)
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Export Ablation Study Chart", key="export_ablation"):
                    if fig:
                        filename = export_publication_figure(
                            fig, 
                            f"publications/ablation_study_{selected_metric}.png", 
                            format="png",
                            journal="IEEE"
                        )
                        st.success(f"Exported to {filename}")
        else:
            st.warning("No ablation study data available. Add validation reports to the './output/moe_validation' directory.")
    
    # Tab 4: Export Options
    with tabs[3]:
        st.subheader("Export Publication Materials")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Export All Figures")
            
            if st.button("Generate All Publication Figures", key="generate_all"):
                # Prepare data dictionary
                data_dict = {}
                
                if workflow_data_list:
                    if 'contributions' in workflow_data_list[0]:
                        data_dict['expert_contributions'] = workflow_data_list[0]['contributions']
                    
                    if 'weights' in workflow_data_list[0]:
                        data_dict['expert_weights'] = workflow_data_list[0]['weights']
                
                if not ablation_df.empty:
                    data_dict['ablation_study'] = ablation_df
                    data_dict['ablation_metric'] = metrics[0] if metrics else 'RMSE'
                
                if not kfold_df.empty:
                    data_dict['kfold_validation'] = kfold_df
                
                # Generate figures
                if data_dict:
                    saved_paths = generate_publication_figures(data_dict, output_dir="./publications")
                    
                    st.success(f"Generated {len(saved_paths)} publication figures.")
                    
                    for path in saved_paths:
                        st.markdown(f"- {os.path.basename(path)}")
                else:
                    st.warning("No data available for generating figures.")
        
        with col2:
            st.markdown("### Generate PDF Report")
            
            if validation_reports:
                if st.button("Generate Validation PDF Report", key="generate_pdf"):
                    output_file = generate_validation_report_pdf(validation_reports, output_file="./publications/validation_report.pdf")
                    
                    st.success(f"Generated PDF report: {output_file}")
                    
                    # Provide download link
                    with open(output_file, "rb") as f:
                        pdf_bytes = f.read()
                    
                    b64_pdf = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="validation_report.pdf" class="download-link">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("No validation reports available for PDF generation.")
            
            st.markdown("### Figure Settings")
            journal_format = st.selectbox(
                "Journal Format",
                ["IEEE", "APA", "Generic"],
                index=0
            )
            
            st.markdown(f"Using **{journal_format}** formatting for exported figures.")


def add_validation_report_viewer(validation_dir='./output/moe_validation'):
    """
    Add a validation report viewer for HTML reports.
    """
    st.header("Validation Report Viewer")
    
    # Find HTML reports
    html_reports = []
    for root, _, files in os.walk(validation_dir):
        for file in files:
            if file.endswith('.html'):
                html_reports.append(os.path.join(root, file))
    
    if not html_reports:
        st.warning(f"No HTML validation reports found in {validation_dir}")
        return
    
    # Sort reports by modification time (newest first)
    html_reports.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Select a report
    selected_report = st.selectbox(
        "Select Validation Report",
        [os.path.basename(report) for report in html_reports]
    )
    
    # Display the selected report
    report_path = [report for report in html_reports if os.path.basename(report) == selected_report][0]
    
    try:
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        # Modify links in the HTML content to make them work
        # This is necessary for reports that reference local resources
        report_dir = os.path.dirname(report_path)
        modified_html = re.sub(
            r'(src|href)="(?!http)([^"]+)"',
            lambda m: f'{m.group(1)}="{os.path.join(report_dir, m.group(2))}"',
            html_content
        )
        
        # Display the report in an iframe
        st.components.v1.html(modified_html, height=800, scrolling=True)
        
        # Provide a download link
        with open(report_path, "rb") as f:
            html_bytes = f.read()
        
        b64_html = base64.b64encode(html_bytes).decode()
        href = f'<a href="data:text/html;base64,{b64_html}" download="{selected_report}" class="download-link">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error displaying validation report: {str(e)}")


def create_expert_comparison_viz():
    """
    Create expert performance comparison visualization for publication.
    """
    st.subheader("Expert Performance Comparison")
    
    # Check if we have real data
    if 'pipeline_data' in st.session_state and st.session_state.pipeline_data:
        if 'expert_training' in st.session_state.pipeline_data:
            component_data = st.session_state.pipeline_data['expert_training']
            from visualization.component_details import render_expert_training_visualization
            render_expert_training_visualization(component_data)
            return
    
    # Display placeholder for missing data
    st.info("No expert data found. Please run the MoE pipeline to generate real expert data.")
    
    # Add button to run pipeline
    if st.button("Run Pipeline to Generate Expert Data"):
        with st.spinner("Running pipeline to generate expert data..."):
            from visualization.data_utils import load_or_generate_input_data, process_data_through_pipeline
            
            # Generate sample data
            input_data = load_or_generate_input_data(synthetic=True)
            
            if input_data is not None:
                # Process through pipeline up to expert training
                results = process_data_through_pipeline(input_data, up_to_component="expert_training")
                
                # Store in session state
                if 'pipeline_data' not in st.session_state:
                    st.session_state.pipeline_data = {}
                
                st.session_state.pipeline_data.update(results)
                st.session_state.input_data = input_data
                
                st.success("Pipeline executed successfully up to expert training!")
                st.rerun()

def create_ablation_study_viz():
    """
    Create ablation study visualization for publication.
    """
    st.subheader("Ablation Study Results")
    
    # Check if we have real data
    validation_reports = load_validation_reports('./output/moe_validation')
    ablation_df = extract_ablation_results(validation_reports)
    
    if not ablation_df.empty:
        # Identify available metrics
        metrics = [col for col in ablation_df.columns 
                  if not col.startswith('_') 
                  and col not in ['report', 'timestamp', 'configuration']
                  and not col.endswith('_p_value')
                  and not col.endswith('_significant')]
        
        if metrics:
            # Select metric
            selected_metric = st.selectbox("Select Metric for Ablation Analysis", metrics, key="ablation_metrics")
            
            # Create ablation study chart
            fig = create_ablation_study_chart(ablation_df, selected_metric)
            st.plotly_chart(fig, use_container_width=True)
            return
    
    # Display placeholder for missing data
    st.info("No ablation study data found. Please run validations to generate ablation study data.")
    
    # Add option to create sample data
    if st.button("Generate Sample Ablation Data"):
        from visualization.data_utils import generate_sample_validation_files
        with st.spinner("Generating sample ablation data..."):
            success = generate_sample_validation_files()
            if success:
                st.success("✅ Sample validation files generated successfully!")
                st.rerun()
            else:
                st.error("Failed to generate sample validation files.")

def create_kfold_validation_viz():
    """
    Create k-fold validation visualization for publication.
    """
    st.subheader("K-Fold Cross-Validation Results")
    
    # Check if we have real data
    validation_reports = load_validation_reports('./output/moe_validation')
    kfold_df = extract_kfold_results(validation_reports)
    
    if not kfold_df.empty:
        # Identify available metrics
        metrics = [col for col in kfold_df.columns 
                  if not col.startswith('_') 
                  and col not in ['report', 'timestamp', 'fold', 'model']]
        
        if metrics:
            # Select metric
            selected_metric = st.selectbox("Select Metric for K-Fold Analysis", metrics)
            
            # Create k-fold validation chart
            fig = create_kfold_validation_chart(kfold_df, selected_metric)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison radar chart
            st.subheader("Model Comparison Across Folds")
            fig = create_model_comparison_radar(kfold_df)
            st.plotly_chart(fig, use_container_width=True)
            return
    
    # Display placeholder for missing data
    st.info("No k-fold validation data found. Please run validations to generate k-fold data.")
    
    # Add option to create sample data
    if st.button("Generate Sample K-Fold Data"):
        from visualization.data_utils import generate_sample_validation_files
        with st.spinner("Generating sample k-fold data..."):
            success = generate_sample_validation_files()
            if success:
                st.success("✅ Sample validation files generated successfully!")
                st.rerun()
            else:
                st.error("Failed to generate sample validation files.")

def create_hyperparameter_sensitivity_viz():
    """
    Create hyperparameter sensitivity visualization for publication.
    """
    st.subheader("Hyperparameter Sensitivity Analysis")
    
    # Check if we have real data
    if 'pipeline_data' in st.session_state and st.session_state.pipeline_data:
        if 'hyperparameter_sensitivity' in st.session_state.pipeline_data:
            sensitivity_data = st.session_state.pipeline_data['hyperparameter_sensitivity']
            
            # Create visualization based on data
            if isinstance(sensitivity_data, dict) and sensitivity_data:
                # Extract parameters and metrics
                param_names = list(sensitivity_data.keys())
                
                # Let user select parameter to visualize
                selected_param = st.selectbox("Select hyperparameter:", param_names)
                
                if selected_param in sensitivity_data:
                    param_data = sensitivity_data[selected_param]
                    
                    if isinstance(param_data, dict) and 'values' in param_data and 'metrics' in param_data:
                        values = param_data['values']
                        metrics = param_data['metrics']
                        
                        # Create visualization
                        if isinstance(metrics, dict):
                            metric_names = list(metrics.keys())
                            selected_metric = st.selectbox("Select metric:", metric_names)
                            
                            if selected_metric in metrics:
                                metric_values = metrics[selected_metric]
                                
                                # Create line chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=values,
                                    y=metric_values,
                                    mode='lines+markers'
                                ))
                                
                                fig.update_layout(
                                    title=f"Impact of {selected_param} on {selected_metric}",
                                    xaxis_title=selected_param,
                                    yaxis_title=selected_metric
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                return
    
    # Display placeholder for missing data
    st.info("No hyperparameter sensitivity data found. Please run sensitivity analysis to generate data.")
    
    # Add option to create sample data
    if st.button("Generate Sample Sensitivity Data"):
        with st.spinner("Generating sample hyperparameter sensitivity data..."):
            # Create sample data
            sample_data = {
                "learning_rate": {
                    "values": [0.001, 0.01, 0.05, 0.1, 0.5],
                    "metrics": {
                        "accuracy": [0.82, 0.87, 0.85, 0.80, 0.72],
                        "loss": [0.48, 0.35, 0.38, 0.52, 0.75]
                    }
                },
                "batch_size": {
                    "values": [16, 32, 64, 128, 256],
                    "metrics": {
                        "accuracy": [0.83, 0.86, 0.88, 0.87, 0.85],
                        "loss": [0.42, 0.38, 0.36, 0.37, 0.40]
                    }
                },
                "num_experts": {
                    "values": [2, 3, 4, 5, 6],
                    "metrics": {
                        "accuracy": [0.82, 0.85, 0.88, 0.89, 0.89],
                        "loss": [0.45, 0.40, 0.35, 0.34, 0.34]
                    }
                }
            }
            
            # Store in session state
            if 'pipeline_data' not in st.session_state:
                st.session_state.pipeline_data = {}
            
            st.session_state.pipeline_data['hyperparameter_sensitivity'] = sample_data
            st.success("Sample hyperparameter sensitivity data generated!")
            st.rerun()

def add_validation_metrics_viz(validation_dir='./output/moe_validation'):
    """
    Display validation metrics from validation reports.
    """
    st.subheader("Validation Metrics")
    
    # Find validation reports
    validation_reports = load_validation_reports(validation_dir)
    
    if validation_reports:
        # Create summary dashboard
        figures = create_validation_summary_dashboard(validation_reports)
        
        if figures:
            # Display figures in tabs
            tabs = st.tabs([title for title, _ in figures])
            
            for i, (title, fig) in enumerate(figures):
                with tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualizations could be created from validation reports.")
    else:
        st.info("No validation reports found.")
        
        # Add button to generate sample data
        if st.button("Generate Sample Validation Data"):
            from visualization.data_utils import generate_sample_validation_files
            with st.spinner("Generating sample validation data..."):
                success = generate_sample_validation_files()
                if success:
                    st.success("✅ Sample validation files generated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to generate sample validation files.")

def check_dependencies():
    """Check if all required packages are installed."""
    missing = []
    
    for package in REQUIRED_PACKAGES:
        try:
            pkg_resources.get_distribution(package)
        except pkg_resources.DistributionNotFound:
            missing.append(package)
    
    return missing

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_moe_modules():
    """Check if the MoE framework modules are available, prioritizing improved integration."""
    # Try to import improved integration first
    try:
        from visualization.improved_moe_integration import MOE_MODULES_AVAILABLE as IMPROVED_AVAILABLE
        if IMPROVED_AVAILABLE:
            st.sidebar.success("✅ Using improved MoE integration")
            return True, "improved"
    except ImportError:
        st.sidebar.warning("⚠️ Improved MoE integration not available")
        
    # Try original integration if improved is not available
    try:
        from visualization.moe_integration import MOE_MODULES_AVAILABLE as ORIGINAL_AVAILABLE
        if ORIGINAL_AVAILABLE:
            st.sidebar.success("✅ Using original MoE integration")
            return True, "original"
    except ImportError:
        st.sidebar.warning("⚠️ Original MoE integration not available")
    
    return False, None

def setup_demo_data():
    """Setup demo data for the dashboard if not already present."""
    if 'demo_data_setup' not in st.session_state:
        st.session_state.demo_data_setup = True
        st.session_state.demo_mode = True
        st.session_state.use_real_data = False
        
        # Try to load or generate sample data
        try:
            from visualization.data_utils import load_or_generate_input_data
            sample_data = load_or_generate_input_data(synthetic=True)
            st.session_state.sample_data = sample_data
            logger.info("Demo data setup complete")
        except Exception as e:
            logger.error(f"Error setting up demo data: {e}")
            st.error("Error setting up demo data. Some features may not work correctly.")

def enhance_dashboard():
    """Main function to enhance the MoE dashboard with advanced visualizations"""
    # Ensure that we're using the global os module
    import os as os_module  # Rename to avoid shadowing
    
    # Initialize MoE framework integration
    integration_success = False
    try:
        # First make sure we have all required dependencies
        required_packages = ['tqdm', 'numpy', 'pandas', 'matplotlib', 'networkx', 'seaborn', 'torch']
        missing_packages = check_dependencies()
        
        if missing_packages:
            st.sidebar.warning(f"⚠️ Missing required packages: {', '.join(missing_packages)}")
            st.sidebar.info("Install them with: pip install " + " ".join(missing_packages))
        else:
            # Try to import and initialize the improved MoE integration
            try:
                from visualization.improved_moe_integration import integrate_with_dashboard, MOE_MODULES_AVAILABLE
                integration_success = integrate_with_dashboard()
                if integration_success:
                    st.sidebar.success("✅ Using real MoE framework components")
                else:
                    st.sidebar.warning("⚠️ Using fallback MoE implementations")
            except (ImportError, NameError) as e:
                st.sidebar.warning(f"⚠️ Improved MoE integration error: {str(e)}")
                st.sidebar.info("Trying original integration...")
                
                # Try the original integration if improved fails
                try:
                    from visualization.moe_integration import replace_fallback_components, MOE_MODULES_AVAILABLE
                    integration_success = replace_fallback_components()
                    if integration_success:
                        st.sidebar.success("✅ Using original MoE integration")
                    else:
                        st.sidebar.warning("⚠️ Using fallback MoE implementations")
                except (ImportError, NameError) as e:
                    st.sidebar.warning(f"⚠️ Original MoE integration error: {str(e)}")
                    st.sidebar.info("Using fallback implementations.")
    except Exception as e:
        st.sidebar.error(f"Error initializing MoE integration: {str(e)}")
        st.sidebar.info("Using fallback implementations.")
    
    st.sidebar.title("MoE Dashboard")
    
    # Check for workflow and validation directories
    workflow_dir_exists = os_module.path.exists("./output/workflows") and len(os_module.listdir("./output/workflows")) > 0
    validation_dir_exists = os_module.path.exists("./output/moe_validation") and len(os_module.listdir("./output/moe_validation")) > 0
    
    # Add sample data generation if needed
    if not workflow_dir_exists or not validation_dir_exists:
        st.sidebar.warning("Missing data folders. You can generate sample data for demonstration.")
        generate_sample = st.sidebar.button("Generate Sample Data")
        
        if generate_sample:
            with st.spinner("Generating sample data..."):
                # Generate workflow files
                if not workflow_dir_exists:
                    from visualization.data_utils import generate_sample_workflow_files
                    success = generate_sample_workflow_files()
                    if success:
                        st.sidebar.success("✅ Sample workflow files generated successfully!")
                    else:
                        st.sidebar.error("Failed to generate sample workflow files.")
                
                # Generate validation files
                if not validation_dir_exists:
                    from visualization.data_utils import generate_sample_validation_files
                    success = generate_sample_validation_files()
                    if success:
                        st.sidebar.success("✅ Sample validation files generated successfully!")
                    else:
                        st.sidebar.error("Failed to generate sample validation files.")
    
    # Add MoE Pipeline Runner section to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("MoE Pipeline Runner")
    st.sidebar.markdown("""
        Run the MoE pipeline with real data and visualize the outputs from each stage.
        This will provide actual insights into how your data flows through the pipeline.
    """)
    
    # Import the necessary modules for pipeline execution
    try:
        from visualization.data_utils import load_or_generate_input_data, run_complete_pipeline
        
        # Determine data source for pipeline
        data_source = st.sidebar.radio(
            "Select data source for MoE pipeline:",
            ["Use synthetic data", "Upload CSV file"]
        )
        
        if data_source == "Use synthetic data":
            num_samples = st.sidebar.slider("Number of synthetic samples:", 100, 5000, 1000)
            
            if st.sidebar.button("Run Pipeline with Synthetic Data"):
                with st.spinner("Running MoE pipeline with synthetic data..."):
                    # Generate synthetic data
                    input_data = load_or_generate_input_data(synthetic=True, num_samples=num_samples)
                    
                    if input_data is not None:
                        # Run the pipeline
                        results, pipeline_id = run_complete_pipeline(input_data)
                        st.session_state.input_data = input_data
                        st.session_state.pipeline_results = results
                        st.session_state.pipeline_id = pipeline_id
                        
                        st.sidebar.success(f"✅ Pipeline executed successfully! ID: {pipeline_id}")
                        st.sidebar.markdown("Go to the **Interactive Pipeline** tab to explore the results.")
        else:
            uploaded_file = st.sidebar.file_uploader("Upload CSV file:", type=["csv"])
            
            if uploaded_file is not None:
                if st.sidebar.button("Run Pipeline with Uploaded Data"):
                    with st.spinner("Running MoE pipeline with uploaded data..."):
                        # Save to temp file then load
                        import tempfile
                        # Use the renamed os module to avoid conflict
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name
                        
                        input_data = load_or_generate_input_data(file_path=temp_path)
                        
                        # Clean up temp file
                        os_module.unlink(temp_path)
                        
                        if input_data is not None:
                            # Run the pipeline
                            results, pipeline_id = run_complete_pipeline(input_data)
                            st.session_state.input_data = input_data
                            st.session_state.pipeline_results = results
                            st.session_state.pipeline_id = pipeline_id
                            
                            st.sidebar.success(f"✅ Pipeline executed successfully! ID: {pipeline_id}")
                            st.sidebar.markdown("Go to the **Interactive Pipeline** tab to explore the results.")
    except Exception as e:
        st.sidebar.error(f"Error setting up pipeline runner: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # Main tabs for different dashboard sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Interactive Pipeline", 
        "Pipeline Overview", 
        "Performance Metrics",
        "Expert Contributions",
        "Parameter Adaptation",
        "Validation Reports",
        "Publication Results"
    ])
    
    # Tab 1: Interactive Pipeline
    with tab1:
        from visualization.interactive_pipeline_viz import add_interactive_pipeline_architecture
        add_interactive_pipeline_architecture()
    
    # Tab 2: Pipeline Overview
    with tab2:
        st.header("Pipeline Overview")
        
        # Create a placeholder for an overview diagram
        st.subheader("MoE Pipeline Architecture")
        add_pipeline_architecture()
        
        # Add a brief explanation of the pipeline components
        st.subheader("Pipeline Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Preprocessing**")
            st.markdown("Cleans and transforms raw input data.")
            
            st.markdown("**Feature Extraction**")
            st.markdown("Extracts relevant features from preprocessed data.")
            
            st.markdown("**Missing Data Handling**")
            st.markdown("Applies specialized techniques for missing values.")
            
            st.markdown("**Expert Training**")
            st.markdown("Trains specialized models for different aspects of the data.")
        
        with col2:
            st.markdown("**Gating Network**")
            st.markdown("Determines which expert should handle each input.")
            
            st.markdown("**MoE Integration**")
            st.markdown("Combines outputs from multiple experts based on gating.")
            
            st.markdown("**Output Generation**")
            st.markdown("Produces final predictions and results.")
    
    # Tab 3: Performance Metrics
    with tab3:
        st.header("Performance Metrics")
        
        # Check for workflow files
        if workflow_dir_exists:
            # Get workflow files
            workflow_files = os_module.listdir("./output/workflows")
            workflow_files = [f for f in workflow_files if f.endswith(".json")]
            
            if workflow_files:
                # Let user select a workflow
                selected_workflow = st.selectbox(
                    "Select a workflow:",
                    workflow_files,
                    format_func=lambda x: x.replace("workflow_", "").replace(".json", "")
                )
                
                # Display optimizer comparison
                add_optimizer_comparison_plot(f"./output/workflows/{selected_workflow}")
            else:
                st.info("No workflow files found. Run the pipeline to generate workflow data.")
        else:
            st.info("No workflow directory found. Generate sample data or run the pipeline to create it.")
    
    # Tab 4: Expert Contributions
    with tab4:
        st.header("Expert Contributions")
        
        try:
            # Try to add expert contributions visualization
            add_expert_contribution_visualization()
        except Exception as e:
            st.error(f"Error displaying expert contributions: {str(e)}")
            st.info("Please make sure you have the necessary data or generate sample data.")
    
    # Tab 5: Parameter Adaptation
    with tab5:
        st.header("Parameter Adaptation Visualization")
        
        if workflow_dir_exists:
            # Get workflow files
            workflow_files = os_module.listdir("./output/workflows")
            workflow_files = [f for f in workflow_files if f.endswith(".json")]
            
            if workflow_files:
                # Let user select a workflow
                selected_workflow = st.selectbox(
                    "Select a workflow file:",
                    workflow_files,
                    format_func=lambda x: x.replace("workflow_", "").replace(".json", ""),
                    key="param_adapt_workflow"
                )
                
                # Display parameter adaptation visualization
                add_parameter_adaptation_visualization(f"./output/workflows/{selected_workflow}")
            else:
                st.info("No workflow files found. Run the pipeline to generate workflow data.")
        else:
            st.info("No workflow directory found. Generate sample data or run the pipeline to create it.")
    
    # Tab 6: Validation Reports
    with tab6:
        st.header("Validation Reports")
        
        if validation_dir_exists:
            # Display validation metrics
            try:
                add_validation_metrics_viz()
            except Exception as e:
                st.error(f"Error displaying validation metrics: {str(e)}")
                st.info("Please make sure you have the necessary validation data or generate sample data.")
        else:
            st.info("No validation directory found. Generate sample data or run the pipeline to create it.")
    
    # Tab 7: Publication Results
    with tab7:
        st.header("Publication-Ready Results")
        
        st.markdown("""
        This section generates publication-quality visualizations that can be directly used
        in research papers or presentations. Select the type of visualization you want to generate.
        """)
        
        # Create selection for visualization type
        viz_type = st.selectbox(
            "Select visualization type:",
            [
                "Expert Performance Comparison",
                "Ablation Study Results",
                "K-Fold Cross-Validation Results",
                "Hyperparameter Sensitivity Analysis"
            ]
        )
        
        # Add placeholder visualizations
        if viz_type == "Expert Performance Comparison":
            create_expert_comparison_viz()
        elif viz_type == "Ablation Study Results":
            create_ablation_study_viz()
        elif viz_type == "K-Fold Cross-Validation Results":
            create_kfold_validation_viz()
        elif viz_type == "Hyperparameter Sensitivity Analysis":
            create_hyperparameter_sensitivity_viz()
    
    # Add a footer
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == "__main__":
    enhance_dashboard() 
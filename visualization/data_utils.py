"""
Data Utilities Module.

This module provides utility functions for loading and processing data 
for interactive pipeline visualization components.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import streamlit as st
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import string

# Add parent directory to path to import MoE modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import MoE modules
try:
    from moe.pipeline import Pipeline
    from moe.data_preprocessing import DataPreprocessor
    from moe.feature_extraction import FeatureExtractor
    from moe.missing_data import MissingDataHandler
    from moe.expert_training import ExpertTrainer
    from moe.gating_network import GatingNetwork
    from moe.moe_integration import MoEIntegrator
    from moe.output_generator import OutputGenerator
    MOE_MODULES_AVAILABLE = True
except ImportError:
    # Define fallback classes to avoid errors when modules are not available
    class BaseFallback:
        def __init__(self, *args, **kwargs):
            self.metrics = {}
            print(f"Created fallback instance of {self.__class__.__name__}")
            
        def process(self, data, *args, **kwargs):
            # Add some basic metrics
            self.metrics = {
                "processing_time": round(random.uniform(0.1, 2.0), 3),
                "success_rate": round(random.uniform(0.85, 0.99), 3)
            }
            
            # Add component-specific metrics
            if hasattr(self, 'add_metrics'):
                self.add_metrics(data)
                
            # Return processed data (no actual processing in fallback)
            if isinstance(data, pd.DataFrame):
                # Make a copy to simulate processing
                processed = data.copy()
                if len(processed.columns) >= 2:
                    # Add a new column to show some change
                    col_name = f"{self.__class__.__name__}_processed"
                    processed[col_name] = np.random.random(len(processed))
                return processed
            return data
    
    class DataPreprocessor(BaseFallback):
        def add_metrics(self, data):
            if isinstance(data, pd.DataFrame):
                self.metrics.update({
                    "rows_processed": len(data),
                    "columns_processed": len(data.columns),
                    "missing_values_detected": int(data.isna().sum().sum()),
                    "outliers_removed": int(random.uniform(0, len(data) * 0.05))
                })
    
    class FeatureExtractor(BaseFallback):
        def add_metrics(self, data):
            if isinstance(data, pd.DataFrame):
                self.metrics.update({
                    "features_extracted": len(data.columns) + int(random.uniform(1, 3)),
                    "feature_importance_score": round(random.uniform(0.7, 0.9), 2),
                    "dimensionality_reduction": round(random.uniform(0.1, 0.3), 2)
                })
    
    class MissingDataHandler(BaseFallback):
        def add_metrics(self, data):
            if isinstance(data, pd.DataFrame):
                missing = data.isna().sum().sum()
                self.metrics.update({
                    "missing_values_before": int(missing),
                    "missing_values_after": 0,
                    "imputation_accuracy": round(random.uniform(0.8, 0.95), 2),
                    "imputation_method": "MICE" if random.random() > 0.5 else "KNN"
                })
    
    class ExpertTrainer(BaseFallback):
        def add_metrics(self, data):
            self.metrics.update({
                "num_experts": int(random.uniform(3, 5)),
                "training_accuracy": round(random.uniform(0.75, 0.95), 3),
                "validation_accuracy": round(random.uniform(0.7, 0.9), 3),
                "training_time": round(random.uniform(10, 60), 2)
            })
            
        def process(self, data):
            super().process(data)
            # Return synthetic expert models
            num_experts = self.metrics["num_experts"]
            return {
                f"expert_{i}": {
                    "accuracy": round(random.uniform(0.75, 0.95), 3),
                    "f1_score": round(random.uniform(0.7, 0.93), 3),
                    "specialization": random.choice(["time_series", "categorical", "numerical"])
                } for i in range(num_experts)
            }
    
    class GatingNetwork(BaseFallback):
        def add_metrics(self, data):
            self.metrics.update({
                "routing_accuracy": round(random.uniform(0.8, 0.95), 3),
                "confidence": round(random.uniform(0.75, 0.9), 3),
                "entropy": round(random.uniform(0.1, 0.5), 3)
            })
            
        def process(self, data):
            super().process(data)
            # Return synthetic gating weights
            expert_weights = {}
            if isinstance(data, pd.DataFrame):
                num_samples = min(10, len(data))
                num_experts = int(random.uniform(3, 5))
                
                for i in range(num_samples):
                    # Generate random weights that sum to 1
                    weights = np.random.random(num_experts)
                    weights = weights / weights.sum()
                    
                    expert_weights[f"sample_{i}"] = {
                        f"expert_{j}": round(weights[j], 3) for j in range(num_experts)
                    }
            
            return expert_weights
    
    class MoEIntegrator(BaseFallback):
        def add_metrics(self, data):
            self.metrics.update({
                "ensemble_improvement": round(random.uniform(0.05, 0.15), 3),
                "integration_method": random.choice(["weighted_average", "stacking", "boosting"]),
                "integration_time": round(random.uniform(0.1, 2.0), 3)
            })
            
        def process(self, *args, **kwargs):
            """Handle either two separate parameters or a combined parameter dictionary"""
            # Add basic metrics
            self.metrics = {
                "processing_time": round(random.uniform(0.1, 2.0), 3),
                "success_rate": round(random.uniform(0.85, 0.99), 3),
                "ensemble_improvement": round(random.uniform(0.05, 0.15), 3),
                "integration_method": random.choice(["weighted_average", "stacking", "boosting"]),
                "integration_time": round(random.uniform(0.1, 2.0), 3)
            }
            
            # Extract expert_models and gating_weights from args or kwargs
            expert_models = None
            gating_weights = None
            
            if len(args) == 2:
                # Two separate parameters: process(expert_models, gating_weights)
                expert_models, gating_weights = args
            elif len(args) == 1 and isinstance(args[0], dict):
                # One combined parameter: process({"expert_models": ..., "gating_weights": ...})
                combined = args[0]
                expert_models = combined.get("expert_models", {})
                gating_weights = combined.get("gating_weights", {})
            
            # If still not found, check kwargs
            if expert_models is None and "expert_models" in kwargs:
                expert_models = kwargs["expert_models"]
            if gating_weights is None and "gating_weights" in kwargs:
                gating_weights = kwargs["gating_weights"]
            
            # Generate synthetic integration result
            num_experts = 0
            if isinstance(expert_models, dict):
                num_experts = len(expert_models)
            
            # Create dummy output with integrated prediction
            integrated_output = {
                "predictions": [round(random.uniform(0, 1), 3) for _ in range(10)],
                "expert_contributions": {
                    f"expert_{i}": round(random.uniform(0, 1), 3) for i in range(max(3, num_experts))
                },
                "confidence": round(random.uniform(0.7, 0.95), 3)
            }
            
            return integrated_output
    
    class OutputGenerator(BaseFallback):
        def add_metrics(self, data):
            self.metrics.update({
                "final_accuracy": round(random.uniform(0.85, 0.97), 3),
                "f1_score": round(random.uniform(0.83, 0.96), 3),
                "processing_time": round(random.uniform(0.05, 0.5), 3)
            })
    
    class Pipeline(BaseFallback):
        def __init__(self, **components):
            self.data_preprocessor = components.get('data_preprocessor', DataPreprocessor())
            self.feature_extractor = components.get('feature_extractor', FeatureExtractor())
            self.missing_data_handler = components.get('missing_data_handler', MissingDataHandler())
            self.expert_trainer = components.get('expert_trainer', ExpertTrainer())
            self.gating_network = components.get('gating_network', GatingNetwork())
            self.moe_integrator = components.get('moe_integrator', MoEIntegrator())
            self.output_generator = components.get('output_generator', OutputGenerator())
            print("Created fallback Pipeline with all components")
    
    MOE_MODULES_AVAILABLE = False

# Define constants for data paths
WORKFLOW_TRACKING_DIR = ".workflow_tracking"
VALIDATION_REPORTS_DIR = "./output/moe_validation"
SAMPLE_DATA_DIR = "./sample_data"
PIPELINE_DATA_DIR = "./pipeline_data"
MOE_CONFIG_PATH = "./moe_config.json"

# Store pipeline data in session state to persist between steps
if 'pipeline_data' not in st.session_state:
    st.session_state.pipeline_data = {}

def initialize_pipeline(config_path=MOE_CONFIG_PATH):
    """
    Initialize the MoE pipeline using configuration.
    
    Args:
        config_path: Path to the MoE configuration file
        
    Returns:
        Pipeline object if successful, None otherwise
    """
    if not MOE_MODULES_AVAILABLE:
        st.warning("MoE modules are not available. Using fallback implementation.")
    
    # Load configuration if exists
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            config = {}
    
    # Initialize pipeline components
    try:
        # Create component instances with config
        data_preprocessor = DataPreprocessor(**(config.get('data_preprocessing', {})))
        feature_extractor = FeatureExtractor(**(config.get('feature_extraction', {})))
        missing_data_handler = MissingDataHandler(**(config.get('missing_data_handling', {})))
        expert_trainer = ExpertTrainer(**(config.get('expert_training', {})))
        gating_network = GatingNetwork(**(config.get('gating_network', {})))
        moe_integrator = MoEIntegrator(**(config.get('moe_integration', {})))
        output_generator = OutputGenerator(**(config.get('output_generation', {})))
        
        # Create and return pipeline
        pipeline = Pipeline(
            data_preprocessor=data_preprocessor,
            feature_extractor=feature_extractor,
            missing_data_handler=missing_data_handler,
            expert_trainer=expert_trainer,
            gating_network=gating_network,
            moe_integrator=moe_integrator,
            output_generator=output_generator
        )
        
        return pipeline
    except Exception as e:
        st.error(f"Error initializing pipeline: {e}")
        return None

def process_data_through_pipeline(input_data, up_to_component=None, config_path=MOE_CONFIG_PATH):
    """
    Process data through the MoE pipeline up to a specific component.
    
    Args:
        input_data: Input data to process
        up_to_component: Name of the component to process up to (or None for full pipeline)
        config_path: Path to the MoE configuration file
        
    Returns:
        Dict: Results from each pipeline stage
    """
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = initialize_pipeline(config_path)
    if not pipeline:
        print("Failed to initialize pipeline")
        return {}
    
    # Initialize results dictionary
    results = {
        'input_data': input_data
    }
    
    # Define processing order
    components = [
        'data_preprocessing',
        'feature_extraction',
        'missing_data_handling',
        'expert_training',
        'gating_network',
        'moe_integration',
        'output_generation'
    ]
    
    try:
        # Process data through each component
        current_data = input_data
        
        for component in components:
            print(f"Processing component: {component}")
            
            if component == 'data_preprocessing':
                print("  Running data preprocessor...")
                processed_data = pipeline.data_preprocessor.process(current_data)
                results['data_preprocessing'] = {
                    'input': current_data,
                    'output': processed_data,
                    'metrics': getattr(pipeline.data_preprocessor, 'metrics', {})
                }
                current_data = processed_data
            
            elif component == 'feature_extraction':
                print("  Running feature extractor...")
                extracted_features = pipeline.feature_extractor.process(current_data)
                results['feature_extraction'] = {
                    'input': current_data,
                    'output': extracted_features,
                    'metrics': getattr(pipeline.feature_extractor, 'metrics', {})
                }
                current_data = extracted_features
            
            elif component == 'missing_data_handling':
                print("  Running missing data handler...")
                handled_data = pipeline.missing_data_handler.process(current_data)
                results['missing_data_handling'] = {
                    'input': current_data,
                    'output': handled_data,
                    'metrics': getattr(pipeline.missing_data_handler, 'metrics', {})
                }
                current_data = handled_data
            
            elif component == 'expert_training':
                # Save current data for gating network
                expert_input = current_data
                
                print("  Running expert trainer...")
                trained_experts = pipeline.expert_trainer.process(current_data)
                results['expert_training'] = {
                    'input': current_data,
                    'output': trained_experts,
                    'metrics': getattr(pipeline.expert_trainer, 'metrics', {})
                }
                
                # Expert training might return models rather than transformed data
                # Keep current_data unchanged for parallel gating network
                results['expert_models'] = trained_experts
            
            elif component == 'gating_network':
                # Use the same input as expert training
                print("  Running gating network...")
                gating_result = pipeline.gating_network.process(expert_input)
                results['gating_network'] = {
                    'input': expert_input,
                    'output': gating_result,
                    'metrics': getattr(pipeline.gating_network, 'metrics', {})
                }
                
                # Gating result might be weights rather than transformed data
                results['gating_weights'] = gating_result
            
            elif component == 'moe_integration':
                # Integration needs both expert models and gating weights
                print("  Running MoE integrator...")
                
                # Debug information
                print(f"    Expert models type: {type(results.get('expert_models', {}))}")
                print(f"    Gating weights type: {type(results.get('gating_weights', {}))}")
                
                # Handle different parameter formats
                try:
                    # First try with two separate parameters
                    integration_result = pipeline.moe_integrator.process(
                        results.get('expert_models', {}),
                        results.get('gating_weights', {})
                    )
                except TypeError as te:
                    print(f"    Error with two parameters: {te}")
                    try:
                        # Try with a single combined parameter
                        integration_result = pipeline.moe_integrator.process({
                            'expert_models': results.get('expert_models', {}),
                            'gating_weights': results.get('gating_weights', {})
                        })
                    except Exception as e2:
                        print(f"    Error with combined parameter: {e2}")
                        # Fallback to simple processing
                        print("    Using fallback integration")
                        integration_result = {
                            'integrated_output': current_data,
                            'expert_weights': results.get('gating_weights', {}),
                            'metrics': {
                                'integration_success': False,
                                'fallback_used': True
                            }
                        }
                
                results['moe_integration'] = {
                    'input': {
                        'expert_models': results.get('expert_models', {}),
                        'gating_weights': results.get('gating_weights', {})
                    },
                    'output': integration_result,
                    'metrics': getattr(pipeline.moe_integrator, 'metrics', {})
                }
                current_data = integration_result
            
            elif component == 'output_generation':
                print("  Running output generator...")
                final_output = pipeline.output_generator.process(current_data)
                results['output_generation'] = {
                    'input': current_data,
                    'output': final_output,
                    'metrics': getattr(pipeline.output_generator, 'metrics', {})
                }
                current_data = final_output
            
            # Stop if we've reached the requested component
            if component == up_to_component:
                print(f"Stopped at component: {up_to_component}")
                break
        
        # Save final results
        results['final_output'] = current_data
        print("Pipeline processing completed successfully")
        
        return results
    
    except Exception as e:
        import traceback
        print(f"Error processing data through pipeline: {e}")
        print(traceback.format_exc())
        st.error(f"Error processing data through pipeline: {e}")
        return results

def load_or_generate_input_data(file_path=None, synthetic=False, num_samples=1000):
    """
    Load input data from file or generate synthetic data.
    
    Args:
        file_path: Path to input data file (CSV, JSON)
        synthetic: Whether to generate synthetic data
        num_samples: Number of synthetic samples to generate
        
    Returns:
        DataFrame or Dict: Input data
    """
    if file_path and os.path.exists(file_path):
        # Load data from file
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.csv':
                return pd.read_csv(file_path)
            elif ext == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                st.error(f"Unsupported file format: {ext}")
                return None
        except Exception as e:
            st.error(f"Error loading input data: {e}")
            return None
    
    elif synthetic:
        # Generate synthetic data
        np.random.seed(42)  # For reproducibility
        
        # Create a dataset with various column types
        data = {
            'id': list(range(1, num_samples + 1)),
            'age': np.random.normal(45, 15, num_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], num_samples),
            'heart_rate': np.random.normal(75, 12, num_samples),
            'blood_pressure_systolic': np.random.normal(120, 15, num_samples),
            'blood_pressure_diastolic': np.random.normal(80, 10, num_samples),
            'cholesterol': np.random.normal(200, 30, num_samples),
            'glucose': np.random.normal(100, 25, num_samples),
            'bmi': np.random.normal(25, 5, num_samples),
            'smoker': np.random.choice([True, False], num_samples, p=[0.25, 0.75]),
            'exercise_hours_week': np.random.gamma(2, 2, num_samples),
            'sleep_hours': np.random.normal(7, 1.5, num_samples)
        }
        
        # Add some categorical features
        data['risk_category'] = np.random.choice(['Low', 'Medium', 'High'], num_samples, p=[0.6, 0.3, 0.1])
        data['diet_quality'] = np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], num_samples)
        
        # Add a target variable (e.g., disease status)
        # Higher probability of disease with age, cholesterol, etc.
        disease_prob = 0.1 + 0.3 * (
            (data['age'] > 50).astype(float) +
            (data['cholesterol'] > 240).astype(float) +
            (data['glucose'] > 120).astype(float) +
            (data['bmi'] > 30).astype(float) +
            data['smoker'].astype(float)
        ) / 5
        
        data['disease_status'] = np.random.binomial(1, np.clip(disease_prob, 0, 1))
        
        # Add missing values
        features_with_missing = ['heart_rate', 'cholesterol', 'glucose', 'bmi', 'exercise_hours_week', 'sleep_hours']
        for feature in features_with_missing:
            missing_mask = np.random.choice(
                [True, False], 
                num_samples, 
                p=[0.05, 0.95]  # 5% missing values
            )
            data[feature] = np.where(missing_mask, np.nan, data[feature])
        
        return pd.DataFrame(data)
    
    else:
        st.error("No input data provided and synthetic data generation disabled.")
        return None

def load_component_data_from_pipeline(component_name, pipeline_id=None):
    """
    Load data for a specific component from the pipeline's saved results.
    
    Args:
        component_name: Name of the component to load data for
        pipeline_id: ID of the pipeline run to use, or None for the current/latest run
        
    Returns:
        Dict: Component data or empty dict if not found
    """
    # If pipeline_id is None, use the most recent data from session state
    if pipeline_id is None:
        if 'pipeline_data' in st.session_state and component_name in st.session_state.pipeline_data:
            return st.session_state.pipeline_data.get(component_name, {})
    
    # If we have a specific pipeline_id, try to load from disk
    if pipeline_id:
        pipeline_file = os.path.join(PIPELINE_DATA_DIR, f"pipeline_{pipeline_id}.json")
        if os.path.exists(pipeline_file):
            try:
                with open(pipeline_file, 'r') as f:
                    pipeline_data = json.load(f)
                    return pipeline_data.get(component_name, {})
            except Exception as e:
                st.error(f"Error loading pipeline data: {e}")
    
    # If we couldn't find the data, return an empty dict
    return {}

def save_pipeline_results(results, pipeline_id=None):
    """
    Save pipeline results to a file.
    
    Args:
        results: Results dictionary to save
        pipeline_id: Optional pipeline ID (auto-generated if None)
        
    Returns:
        str: Pipeline ID
    """
    # Generate ID if not provided
    if not pipeline_id:
        pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    
    # Ensure pipeline data directory exists
    os.makedirs(PIPELINE_DATA_DIR, exist_ok=True)
    
    # Save results to file
    try:
        # Convert DataFrames to dictionaries for JSON serialization
        serializable_results = {}
        
        for component, data in results.items():
            if isinstance(data, dict):
                component_data = {}
                
                for key, value in data.items():
                    if isinstance(value, pd.DataFrame):
                        # Convert DataFrame to dict
                        component_data[key] = value.to_dict(orient='records')
                    elif isinstance(value, np.ndarray):
                        # Convert numpy array to list
                        component_data[key] = value.tolist()
                    else:
                        component_data[key] = value
                
                serializable_results[component] = component_data
            elif isinstance(data, pd.DataFrame):
                # Top-level DataFrame
                serializable_results[component] = data.to_dict(orient='records')
            else:
                serializable_results[component] = data
        
        # Add metadata
        serializable_results["_metadata"] = {
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "moe_modules_available": MOE_MODULES_AVAILABLE
        }
        
        # Save to file
        with open(os.path.join(PIPELINE_DATA_DIR, f"pipeline_{pipeline_id}.json"), 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        return pipeline_id
    except Exception as e:
        st.error(f"Error saving pipeline results: {str(e)}")
        return pipeline_id

def run_complete_pipeline(input_data=None, config_path=MOE_CONFIG_PATH):
    """
    Run the complete MoE pipeline and return results.
    
    Args:
        input_data: Input data to process (generated if None)
        config_path: Path to the MoE configuration file
        
    Returns:
        tuple: (results, pipeline_id)
    """
    # Generate input data if not provided
    if input_data is None:
        input_data = load_or_generate_input_data(synthetic=True)
        if input_data is None:
            return {}, None
    
    # Process through pipeline
    try:
        # Process data through all components
        results = process_data_through_pipeline(input_data, up_to_component=None, config_path=config_path)
        
        # Save results
        pipeline_id = save_pipeline_results(results)
        
        # Store in session state
        st.session_state.pipeline_data = results
        
        return results, pipeline_id
    except Exception as e:
        st.error(f"Error processing data through pipeline: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {}, None

def load_component_data(component_name: str, workflow_id: Optional[str] = None) -> Dict:
    """
    Load data for a specific pipeline component, either from a workflow, pipeline results,
    or fallback to sample data.
    
    Args:
        component_name: Name of the component to load data for
        workflow_id: ID of the workflow to use, or None for latest workflow
        
    Returns:
        Dict: Component data or empty dict if not found
    """
    # Default empty data
    component_data = {
        "id": workflow_id or "latest",
        "component": component_name,
        "metrics": {},
        "visualizations": {},
        "data_samples": {}
    }
    
    # First try to get data from pipeline results
    pipeline_data = load_component_data_from_pipeline(component_name)
    if pipeline_data:
        component_data.update(pipeline_data)
        component_data["_source"] = "pipeline"
        return component_data
    
    # If not in pipeline, try to load from workflow data
    workflow = None
    if workflow_id:
        workflow = load_workflow_by_id(workflow_id)
    else:
        workflow = load_latest_workflow()
    
    if workflow:
        # Extract component data from workflow if available
        components = workflow.get("components", {})
        if component_name in components:
            component_data.update(components[component_name])
            component_data["_source"] = "workflow"
            component_data["workflow_id"] = workflow.get("id")
            return component_data
    
    # If not found in workflow, check for sample data
    sample_data_file = os.path.join(SAMPLE_DATA_DIR, f"{component_name}_sample.json")
    
    if os.path.exists(sample_data_file):
        try:
            with open(sample_data_file, "r") as f:
                sample_data = json.load(f)
                component_data.update(sample_data)
                component_data["_source"] = "sample"
                return component_data
        except Exception as e:
            print(f"Error loading sample data file {sample_data_file}: {e}")
    
    # If no data found, return empty component data
    component_data["_source"] = "empty"
    return component_data

# Keep remaining utility functions

@st.cache_data
def load_all_workflows() -> List[Dict]:
    """
    Load all available workflow tracking data.
    
    Returns:
        List[Dict]: List of workflow data dictionaries
    """
    workflows = []
    
    # Check if workflow tracking directory exists
    if not os.path.exists(WORKFLOW_TRACKING_DIR):
        return workflows
    
    # Find all workflow JSON files
    workflow_files = glob.glob(os.path.join(WORKFLOW_TRACKING_DIR, "workflow_*.json"))
    
    # Load each workflow file
    for workflow_file in workflow_files:
        try:
            with open(workflow_file, "r") as f:
                workflow_data = json.load(f)
                
                # Add file path for reference
                workflow_data["_file_path"] = workflow_file
                
                # Extract workflow ID from filename if not in data
                if "id" not in workflow_data:
                    filename = os.path.basename(workflow_file)
                    workflow_id = filename.replace("workflow_", "").replace(".json", "")
                    workflow_data["id"] = workflow_id
                
                workflows.append(workflow_data)
        except Exception as e:
            print(f"Error loading workflow file {workflow_file}: {e}")
    
    # Sort workflows by timestamp if available
    workflows.sort(key=lambda w: w.get("timestamp", ""), reverse=True)
    
    return workflows

@st.cache_data
def load_workflow_by_id(workflow_id: str) -> Optional[Dict]:
    """
    Load a specific workflow by its ID.
    
    Args:
        workflow_id: ID of the workflow to load
        
    Returns:
        Dict or None: Workflow data if found, None otherwise
    """
    # Check for direct file match
    workflow_file = os.path.join(WORKFLOW_TRACKING_DIR, f"workflow_{workflow_id}.json")
    
    if os.path.exists(workflow_file):
        try:
            with open(workflow_file, "r") as f:
                workflow_data = json.load(f)
                
                # Add file path for reference
                workflow_data["_file_path"] = workflow_file
                
                return workflow_data
        except Exception as e:
            print(f"Error loading workflow file {workflow_file}: {e}")
            return None
    
    # If direct file not found, search all workflows
    all_workflows = load_all_workflows()
    for workflow in all_workflows:
        if workflow.get("id") == workflow_id:
            return workflow
    
    return None

@st.cache_data
def load_latest_workflow() -> Optional[Dict]:
    """
    Load the most recent workflow data.
    
    Returns:
        Dict or None: Most recent workflow data if available, None otherwise
    """
    all_workflows = load_all_workflows()
    
    if all_workflows:
        return all_workflows[0]  # Already sorted by timestamp (newest first)
    
    return None

@st.cache_data
def load_validation_reports() -> List[Dict]:
    """
    Load all available validation reports.
    
    Returns:
        List[Dict]: List of validation report metadata
    """
    reports = []
    
    # Check if validation reports directory exists
    if not os.path.exists(VALIDATION_REPORTS_DIR):
        return reports
    
    # Find all HTML report files
    report_files = glob.glob(os.path.join(VALIDATION_REPORTS_DIR, "*.html"))
    
    # Extract metadata from filenames
    for report_file in report_files:
        filename = os.path.basename(report_file)
        
        # Extract metadata from filename pattern (e.g., "report_2023-01-01_12-34-56.html")
        parts = filename.replace(".html", "").split("_")
        
        report_type = parts[0] if len(parts) > 0 else "unknown"
        
        # Extract timestamp if present
        timestamp = None
        if len(parts) > 1:
            date_part = parts[1] if len(parts) > 1 else ""
            time_part = parts[2] if len(parts) > 2 else ""
            
            if date_part and time_part:
                timestamp = f"{date_part}_{time_part}"
        
        reports.append({
            "filename": filename,
            "path": report_file,
            "type": report_type,
            "timestamp": timestamp
        })
    
    # Sort reports by timestamp if available
    reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    
    return reports

def check_workflow_files():
    """Check if workflow files exist in the specified directory."""
    workflow_dir = os.path.join(".", "output", "workflows")
    return os.path.exists(workflow_dir) and len(os.listdir(workflow_dir)) > 0

def check_validation_files():
    """Check if validation files exist in the specified directory."""
    validation_dir = os.path.join(".", "output", "moe_validation")
    return os.path.exists(validation_dir) and len(os.listdir(validation_dir)) > 0

def generate_sample_workflow_files():
    """Generate sample workflow files for demonstration purposes."""
    output_dir = os.path.join(".", "output", "workflows")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a few sample workflow files
    for i in range(3):
        workflow_id = f"workflow_{generate_random_uuid()}"
        timestamp = datetime.now() - timedelta(days=i)
        
        # Create sample workflow data
        workflow_data = {
            "workflow_id": workflow_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed",
            "optimizer": {
                "name": ["SGD", "Adam", "RMSprop"][i % 3],
                "learning_rate": [0.01, 0.001, 0.0005][i % 3],
                "iterations": 100,
                "convergence_history": [
                    {"iteration": j, "loss": 1.0 * (0.95 ** j) + 0.1 * np.random.rand()}
                    for j in range(1, 101)
                ]
            },
            "experts": {
                f"expert_{j}": {
                    "accuracy": round(0.8 + 0.1 * np.random.rand(), 2),
                    "specialization": ["time_series", "categorical", "numerical"][j % 3],
                    "training_time": round(10 + 5 * np.random.rand(), 1)
                }
                for j in range(3)
            },
            "parameter_adaptation": {
                "learning_rate": [
                    {"iteration": k*10, "value": 0.01 * (0.9 ** k)}
                    for k in range(10)
                ],
                "dropout_rate": [
                    {"iteration": k*10, "value": min(0.5, 0.2 + 0.03 * k)}
                    for k in range(10)
                ],
                "batch_size": [
                    {"iteration": k*10, "value": int(32 * (1 + 0.2 * k))}
                    for k in range(10)
                ]
            }
        }
        
        # Save to file
        with open(os.path.join(output_dir, f"{workflow_id}.json"), "w") as f:
            json.dump(workflow_data, f, indent=2)
    
    return True

def generate_sample_validation_files():
    """Generate sample validation files for demonstration purposes."""
    output_dir = os.path.join(".", "output", "moe_validation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample validation results
    validation_data = {
        "validation_id": generate_random_uuid(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "accuracy": 0.88,
            "precision": 0.86,
            "recall": 0.89,
            "f1_score": 0.87,
            "auc": 0.92
        },
        "confusion_matrix": {
            "true_positive": 42,
            "false_positive": 7,
            "true_negative": 38,
            "false_negative": 5
        },
        "expert_contributions": {
            "expert_1": 0.45,
            "expert_2": 0.35,
            "expert_3": 0.20
        },
        "feature_importance": {
            "feature_1": 0.25,
            "feature_2": 0.20,
            "feature_3": 0.18,
            "feature_4": 0.15,
            "feature_5": 0.12,
            "feature_6": 0.10
        }
    }
    
    # Save to file
    with open(os.path.join(output_dir, "validation_results.json"), "w") as f:
        json.dump(validation_data, f, indent=2)
    
    return True

def generate_random_uuid():
    """Generate a random UUID-like string."""
    return "-".join([
        "".join([random.choice("0123456789abcdef") for _ in range(8)]),
        "".join([random.choice("0123456789abcdef") for _ in range(4)]),
        "".join([random.choice("0123456789abcdef") for _ in range(4)]),
        "".join([random.choice("0123456789abcdef") for _ in range(4)]),
        "".join([random.choice("0123456789abcdef") for _ in range(12)])
    ])

if __name__ == "__main__":
    # For testing this module independently
    import streamlit as st
    
    st.set_page_config(layout="wide", page_title="Data Utilities")
    
    st.title("Data Utilities Test")
    
    # Test workflow loading
    st.header("Workflow Data")
    workflows = load_all_workflows()
    
    if workflows:
        st.write(f"Found {len(workflows)} workflows")
        workflow_ids = [w.get("id") for w in workflows]
        selected_workflow = st.selectbox("Select a workflow", workflow_ids)
        
        if selected_workflow:
            workflow = load_workflow_by_id(selected_workflow)
            st.json(workflow)
    else:
        st.write("No workflows found.")
        
        # Test sample data generation
        st.header("Sample Data Generation")
        component_name = st.selectbox(
            "Select a component",
            [
                "data_preprocessing",
                "feature_extraction", 
                "missing_data_handling",
                "expert_training",
                "gating_network",
                "moe_integration",
                "output_generation"
            ]
        )
        
        if st.button("Generate Sample Data"):
            sample_data = generate_component_sample_data(component_name)
            st.json(sample_data) 
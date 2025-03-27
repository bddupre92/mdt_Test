"""
Improved MoE Framework Integration Module

This module provides enhanced integration between the dashboard and the MoE framework,
using better error handling and compatibility layers to ensure proper initialization
of MoE components.
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Fix for torch.__path__._path issue
# This prevents Streamlit from inspecting torch modules deeply, which can cause errors
import torch
if hasattr(torch, '__path__') and hasattr(torch.__path__, '_path'):
    # Block attribute access to prevent Streamlit watcher from inspecting these modules
    orig_getattr = torch.__path__.__class__.__getattr__
    def safe_getattr(self, name):
        if name == '_path':
            return []
        return orig_getattr(self, name)
    torch.__path__.__class__.__getattr__ = safe_getattr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to track module availability - needed by the dashboard
MOE_MODULES_AVAILABLE = False
FRAMEWORK_MODULES = {}

# Try to import our expert adapter for properly initializing MoE components
try:
    from visualization.expert_adapter import initialize_moe_pipeline, get_default_moe_config
    HAS_EXPERT_ADAPTER = True
    logger.info("Using expert adapter for MoE pipeline initialization")
except ImportError:
    HAS_EXPERT_ADAPTER = False
    logger.warning("Expert adapter not available. Pipeline initialization may fail.")

# Add project root to path to help with imports
try:
    # Get the project root (assuming this module is in the visualization directory)
    project_root = Path(__file__).parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        logger.info(f"Added project root to path: {project_root}")

    # Try to find potential meta module paths
    potential_meta_paths = [
        os.path.join(project_root, "meta_optimizer"),
        os.path.join(project_root, "moe_framework", "meta"),
        os.path.join(project_root, "moe_framework", "meta_optimizer")
    ]
    
    for path in potential_meta_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            logger.info(f"Added potential meta module path: {path}")
except Exception as e:
    logger.warning(f"Error setting up import paths: {str(e)}")


def check_framework_availability():
    """
    Check which MoE framework modules are available.
    
    Returns:
        Dict[str, bool]: Dictionary of module names and their availability
    """
    global MOE_MODULES_AVAILABLE, FRAMEWORK_MODULES
    
    modules = {
        "moe_framework.workflow.moe_pipeline": False,
        "moe_framework.gating.gating_network": False,
        "moe_framework.experts.base_expert": False,
        "moe_framework.integration.pipeline_connector": False,
        "moe_framework.execution.execution_pipeline": False,
        "moe_framework.interfaces.pipeline": False
    }
    
    for module_name in modules:
        try:
            __import__(module_name)
            modules[module_name] = True
            logger.info(f"✅ Module {module_name} is available")
        except ImportError:
            logger.warning(f"❌ Module {module_name} is not available")
    
    min_required = all([
        modules["moe_framework.workflow.moe_pipeline"],
        modules["moe_framework.gating.gating_network"],
        modules["moe_framework.experts.base_expert"]
    ])
    
    if min_required:
        logger.info("✅ Minimum required modules are available")
        MOE_MODULES_AVAILABLE = True
    else:
        logger.warning("❌ Some required modules are missing")
        MOE_MODULES_AVAILABLE = False
    
    # Store modules in global variable for other parts of the dashboard to use
    FRAMEWORK_MODULES = modules
    
    # Also try to import specific classes
    classes_available = {
        "MoEPipeline": False,
        "GatingNetwork": False,
        "BaseExpert": False,
        "IntegrationConnector": False,
        "ExecutionPipeline": False,
        "Pipeline": False
    }
    
    try:
        from moe_framework.workflow.moe_pipeline import MoEPipeline
        classes_available["MoEPipeline"] = True
        logger.info("Successfully imported MoEPipeline")
    except ImportError:
        pass
    
    try:
        from moe_framework.gating.gating_network import GatingNetwork
        classes_available["GatingNetwork"] = True
        logger.info("Successfully imported GatingNetwork")
    except ImportError:
        pass
    
    try:
        from moe_framework.experts.base_expert import BaseExpert
        classes_available["BaseExpert"] = True
        logger.info("Successfully imported BaseExpert")
    except ImportError:
        pass
    
    try:
        from moe_framework.integration.pipeline_connector import IntegrationConnector
        classes_available["IntegrationConnector"] = True
        logger.info("Successfully imported IntegrationConnector")
    except ImportError:
        pass
    
    try:
        from moe_framework.execution.execution_pipeline import ExecutionPipeline
        classes_available["ExecutionPipeline"] = True
        logger.info("Successfully imported ExecutionPipeline")
    except ImportError:
        pass
    
    try:
        from moe_framework.interfaces.pipeline import Pipeline
        classes_available["Pipeline"] = True
        logger.info("Successfully imported Pipeline interfaces")
    except ImportError:
        pass

    # Recheck module availability after imports to ensure consistent reporting
    for module_name in modules:
        try:
            __import__(module_name)
            modules[module_name] = True
        except ImportError:
            modules[module_name] = False

    min_required = all([
        modules["moe_framework.workflow.moe_pipeline"],
        modules["moe_framework.gating.gating_network"],
        modules["moe_framework.experts.base_expert"]
    ])
    
    # Update global variables
    MOE_MODULES_AVAILABLE = min_required
    FRAMEWORK_MODULES = modules
    
    logger.info(f"MoE framework modules available: {min_required}")
    
    return modules, min_required


# Add process method to GatingNetwork if it's missing
def patch_gating_network():
    """Add the process method to GatingNetwork if it's missing."""
    try:
        from moe_framework.gating.gating_network import GatingNetwork
        
        # Check if process method already exists
        if not hasattr(GatingNetwork, 'process'):
            # Add process method that delegates to weight_experts
            def process(self, data):
                """Process data through the gating network."""
                # Check if weight_experts exists
                if hasattr(self, 'weight_experts'):
                    return self.weight_experts(data)
                else:
                    # Fallback implementation
                    logger.warning("GatingNetwork has no weight_experts method, using fallback")
                    return {
                        'weights': {f'expert_{i}': 1.0/len(self.experts) for i in range(len(self.experts))},
                        'confidence': 0.8,
                        'explanation': "Equal weights assigned (fallback)"
                    }
            
            # Add the method to the class
            GatingNetwork.process = process
            logger.info("Added process method to GatingNetwork")
            
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not patch GatingNetwork: {str(e)}")


def initialize_pipeline(config=None, verbose=False):
    """
    Initialize a MoE pipeline with better error handling and compatibility.
    
    Args:
        config: Configuration dictionary or path to config file
        verbose: Whether to print verbose output
        
    Returns:
        Initialized MoE pipeline or None if initialization fails
    """
    # Check if MoE framework modules are available
    modules, min_required = check_framework_availability()
    
    if not min_required:
        logger.warning("Cannot initialize MoE pipeline, required modules are missing")
        return None
    
    # Patch the GatingNetwork class to ensure it has the process method
    patch_gating_network()
    
    logger.info("Creating MoEPipeline instance...")

    # Ensure config is a dictionary, not a string
    if isinstance(config, str):
        if os.path.exists(config):
            try:
                with open(config, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading config file {config}: {e}")
                config = {}
        else:
            # If it's a string but not a file path, assume it's a config name
            config = {"config_name": config}

    try:
        # Use expert adapter if available for better parameter handling
        if HAS_EXPERT_ADAPTER:
            # Initialize pipeline with adapter
            pipeline = initialize_moe_pipeline(config, verbose=verbose)
            if pipeline is not None:
                logger.info("Successfully initialized MoE pipeline with adapter")
                return pipeline
            else:
                logger.warning("Failed to initialize pipeline with adapter")
        
        # Fall back to direct initialization if adapter fails or is not available
        try:
            from moe_framework.workflow.moe_pipeline import MoEPipeline
            # Try new style initialization first
            pipeline = MoEPipeline(config=config, verbose=verbose)
            logger.info("Successfully initialized MoE pipeline with new style")
            return pipeline
        except TypeError:
            # If that fails, try old style initialization
            from moe_framework.workflow.moe_pipeline import MoEPipeline
            pipeline = MoEPipeline(verbose=verbose)
            logger.info("Successfully initialized MoE pipeline with old style")
            return pipeline
        
    except Exception as e:
        logger.error(f"Error initializing MoE pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Failed to initialize pipeline.")
        return None


# Special compatibility class for the dashboard
class HybridMoEPipeline:
    """
    Compatibility layer for the dashboard that combines real MoE components
    with fallback implementations.
    """
    
    def __init__(self, config=None, verbose=False):
        self.config = config
        self.verbose = verbose
        self.pipeline = initialize_pipeline(config, verbose)
        
        # Extract components from the real pipeline if available
        if self.pipeline:
            # Try to get components from the real pipeline
            self.gating_network = getattr(self.pipeline, 'gating_network', None)
            self.data_preprocessor = getattr(self.pipeline, 'data_preprocessor', None)
            self.feature_extractor = getattr(self.pipeline, 'feature_extractor', None)
            self.missing_data_handler = getattr(self.pipeline, 'missing_data_handler', None)
            self.expert_trainer = getattr(self.pipeline, 'expert_trainer', None)
            self.moe_integrator = getattr(self.pipeline, 'moe_integrator', None)
            self.output_generator = getattr(self.pipeline, 'output_generator', None)
        else:
            # Create fallback implementations
            from visualization.data_utils import (
                DataPreprocessor, FeatureExtractor, MissingDataHandler,
                ExpertTrainer, GatingNetwork, MoEIntegrator, OutputGenerator
            )
            self.data_preprocessor = DataPreprocessor()
            self.feature_extractor = FeatureExtractor()
            self.missing_data_handler = MissingDataHandler()
            self.expert_trainer = ExpertTrainer()
            self.gating_network = GatingNetwork()
            self.moe_integrator = MoEIntegrator()
            self.output_generator = OutputGenerator()
            logger.warning("Created fallback pipeline with mock components")
        
    def process(self, data):
        """Process data through the pipeline with fallback support."""
        if self.pipeline is None:
            logger.warning("Using fallback implementation as no pipeline is available")
            # Fallback implementation
            return self._fallback_process(data)
        
        try:
            # Try to use the real pipeline's process method
            if hasattr(self.pipeline, 'process'):
                return self.pipeline.process(data)
            else:
                # Manual processing through pipeline components
                return self._fallback_process(data)
        except Exception as e:
            logger.error(f"Error processing data through real pipeline: {str(e)}")
            # Fallback to our manual implementation
            return self._fallback_process(data)
    
    def _fallback_process(self, data):
        """Manual pipeline processing when real pipeline is unavailable."""
        try:
            # Process data through all components
            logger.info("Processing data through fallback pipeline")
            
            # Data preprocessing
            processed_data = self.data_preprocessor.process(data)
            
            # Feature extraction
            features = self.feature_extractor.process(processed_data)
            
            # Missing data handling
            clean_data = self.missing_data_handler.process(features)
            
            # Expert training
            experts = self.expert_trainer.process(clean_data)
            
            # Gating network
            weights = self.gating_network.process(clean_data)
            
            # MoE integration
            integrated = self.moe_integrator.process(experts, weights)
            
            # Output generation
            output = self.output_generator.process(integrated)
            
            # Return a dictionary with results from each stage
            result = {
                'input': data,
                'processed_data': processed_data,
                'features': features,
                'clean_data': clean_data,
                'experts': experts,
                'weights': weights,
                'integrated': integrated,
                'output': output,
                'success': True
            }
            
            # Add metrics if available
            metrics = {}
            for component_name in ['data_preprocessor', 'feature_extractor', 
                                   'missing_data_handler', 'expert_trainer',
                                   'gating_network', 'moe_integrator', 
                                   'output_generator']:
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'metrics'):
                    metrics[component_name] = component.metrics
                
                result['metrics'] = metrics
        except Exception as e:
            logger.error(f"Error in fallback processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            result = {
                'input': data,
                'output': data,
                'success': False,
                'error': str(e)
            }
        
        return result


def process_data(pipeline, input_data):
    """
    Process data through the MoE pipeline.
    
    Args:
        pipeline: Initialized MoE pipeline
        input_data: Input data to process
        
    Returns:
        Dictionary with processed data and results
    """
    if pipeline is None:
        logger.warning("Cannot process data, pipeline is not initialized")
        return {
            'success': False,
            'error': 'Pipeline not initialized',
            'data': None,
            'predictions': None,
            'metrics': None
        }
    
    try:
        # Process the data through the pipeline
        processed_data = pipeline.preprocess_data(input_data)
        logger.info("Successfully preprocessed data")
        
        # Train the pipeline
        pipeline.train(processed_data, target_column='migraine')
        logger.info("Successfully trained pipeline")
        
        # Make predictions
        predictions = pipeline.predict(processed_data)
        logger.info("Successfully made predictions")
        
        # Evaluate the pipeline
        metrics = pipeline.evaluate(processed_data, target_column='migraine')
        logger.info("Successfully evaluated pipeline")
        
        return {
            'success': True,
            'data': processed_data,
            'predictions': predictions,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error processing data through MoE pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Partial results if available
        result = {
            'success': False,
            'error': str(e),
            'data': None,
            'predictions': None,
            'metrics': None
        }
        
        # Include partial results if available
        try:
            if 'processed_data' in locals():
                result['data'] = processed_data
            if 'predictions' in locals():
                result['predictions'] = predictions
            if 'metrics' in locals():
                result['metrics'] = metrics
        except:
            pass
        
        return result


def run_moe_pipeline(input_data=None):
    """
    Run the complete MoE pipeline on provided input data.
    
    Args:
        input_data: Input data to process, generated if None
        
    Returns:
        Dictionary with results
    """
    logger.info("Running real MoE pipeline from improved integration module")
    
    # Generate synthetic data if input_data is None
    if input_data is None:
        input_data = load_sample_data()
        logger.info("Generated synthetic data for pipeline testing")
    
    # Initialize a hybrid pipeline for better compatibility
    pipeline = HybridMoEPipeline(verbose=True)
    
    # Process the data
    results = pipeline.process(input_data)
    
    # Generate a unique pipeline ID
    import uuid
    pipeline_id = str(uuid.uuid4())[:8]
    
    if isinstance(results, dict):
        results['pipeline_id'] = pipeline_id
    else:
        results = {
            'success': True,
            'data': input_data,
            'predictions': None,
            'metrics': None,
            'pipeline_id': pipeline_id
        }
    
    return results


def load_sample_data():
    """
    Generate synthetic data for testing the MoE pipeline.
    
    Returns:
        DataFrame with synthetic data
    """
    # Generate a synthetic dataset with features for each expert
    np.random.seed(42)
    n_samples = 100
    
    # Create a DataFrame with patient IDs and dates
    dates = pd.date_range(start='2023-01-01', periods=n_samples)
    patient_ids = np.random.randint(1000, 9999, n_samples)
    
    data = pd.DataFrame({
        'patient_id': patient_ids,
        'date': dates,
        'location': np.random.choice(['New York', 'Boston', 'Chicago', 'San Francisco'], n_samples),
        
        # Physiological features
        'heart_rate': np.random.normal(75, 10, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'temperature': np.random.normal(98.6, 0.5, n_samples),
        
        # Environmental features
        'env_temperature': np.random.normal(72, 8, n_samples),
        'humidity': np.random.normal(50, 15, n_samples),
        'air_quality': np.random.normal(35, 10, n_samples),
        
        # Behavioral features
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'activity_level': np.random.normal(6, 2, n_samples),
        'stress_level': np.random.normal(5, 2, n_samples),
        
        # Medication features
        'medication_name': np.random.choice(['Sumatriptan', 'Rizatriptan', 'Eletriptan', 'None'], n_samples),
        'dosage': np.random.choice([25, 50, 100, 0], n_samples),
        'frequency': np.random.choice([1, 2, 3, 0], n_samples),
        
        # Target variable
        'migraine': np.random.binomial(1, 0.3, n_samples)
    })
    
    return data


def integrate_with_dashboard():
    """
    Integrate the MoE framework with the dashboard by replacing fallback implementations
    with real MoE components.
    
    Returns:
        True if integration was successful, False otherwise
    """
    # Check if MoE framework modules are available
    modules, available = check_framework_availability()
    
    if not available:
        logger.warning("Cannot integrate MoE framework with dashboard, required modules are missing")
        return False
    
    try:
        # Import data_utils module
        import visualization.data_utils as data_utils
        
        # Define enhanced versions of data_utils functions that use real MoE components
        
        def enhanced_initialize_pipeline(config_path=None):
            """Enhanced version of data_utils.initialize_pipeline using real MoE components"""
            logger.info("Using real MoE pipeline from improved integration module")
            return HybridMoEPipeline(config_path, verbose=True)
        
        def enhanced_run_complete_pipeline(input_data=None, config_path=None):
            """Enhanced version of data_utils.run_complete_pipeline using real MoE components"""
            logger.info("Using real MoE pipeline from improved integration module")
            
            # Create a hybrid pipeline for better compatibility
            pipeline = HybridMoEPipeline(config_path, verbose=True)
            
            if input_data is None:
                input_data = load_sample_data()
            
            # Process the data
            try:
                results = pipeline.process(input_data)
                # Generate a unique pipeline ID
                import uuid
                pipeline_id = str(uuid.uuid4())[:8]
                
                # Return results in the expected format
                return results, pipeline_id
            except Exception as e:
                logger.error(f"Error running MoE pipeline: {str(e)}")
                logger.warning("Falling back to data_utils implementation")
                return data_utils.run_complete_pipeline(input_data, config_path)
        
        # Replace the data_utils functions with enhanced versions
        data_utils.initialize_pipeline = enhanced_initialize_pipeline
        data_utils.run_complete_pipeline = enhanced_run_complete_pipeline
        data_utils.MOE_MODULES_AVAILABLE = MOE_MODULES_AVAILABLE
        
        logger.info("Successfully integrated MoE framework with dashboard")
        return True
        
    except Exception as e:
        logger.error(f"Error integrating MoE framework with dashboard: {str(e)}")
        return False


# Automatically check framework availability
modules, available = check_framework_availability()

# Automatically integrate with dashboard when this module is imported
integration_successful = integrate_with_dashboard() 
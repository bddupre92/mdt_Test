"""
Example demonstrating the fully integrated MoE pipeline with event system, 
integration layer, and state management capabilities.

This example shows:
1. Setting up the MoEPipeline with all integrated components
2. Registering custom event listeners
3. Training and prediction with event emission
4. Checkpointing and restoring the pipeline state
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required components
from moe_framework.workflow.moe_pipeline import MoEPipeline
from moe_framework.integration.event_system import Event, MoEEventTypes
from moe_framework.integration.integration_layer import WeightedAverageIntegration, AdaptiveIntegration
from moe_framework.persistence.state_manager import FileSystemStateManager

# Define a custom event listener class for demonstration
class PipelineEventLogger:
    """Custom event logger for pipeline events."""
    
    def __init__(self, log_file_path=None):
        self.events = []
        self.log_file_path = log_file_path
        
    def on_training_started(self, event):
        """Handle training started events."""
        msg = f"Training started: {event.data}"
        logger.info(msg)
        self.events.append(("TRAINING_STARTED", event.data, datetime.now()))
        self._write_to_log(msg)
        
    def on_training_completed(self, event):
        """Handle training completed events."""
        msg = f"Training completed: {event.data}"
        logger.info(msg)
        self.events.append(("TRAINING_COMPLETED", event.data, datetime.now()))
        self._write_to_log(msg)
    
    def on_prediction_completed(self, event):
        """Handle prediction completed events."""
        msg = f"Prediction completed: {event.data}"
        logger.info(msg)
        self.events.append(("PREDICTION_COMPLETED", event.data, datetime.now()))
        self._write_to_log(msg)
    
    def on_checkpoint_completed(self, event):
        """Handle checkpoint completed events."""
        msg = f"Checkpoint completed: {event.data}"
        logger.info(msg)
        self.events.append(("CHECKPOINT_COMPLETED", event.data, datetime.now()))
        self._write_to_log(msg)
    
    def _write_to_log(self, message):
        """Write message to log file if path is provided."""
        if self.log_file_path:
            with open(self.log_file_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
    
    def get_event_summary(self):
        """Get a summary of all captured events."""
        return {
            'event_count': len(self.events),
            'event_types': list(set(e[0] for e in self.events)),
            'first_event': self.events[0] if self.events else None,
            'last_event': self.events[-1] if self.events else None
        }

def run_integrated_pipeline_example():
    """Run the integrated pipeline example."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "../output/integrated_example")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize log file for event logger
    log_file_path = os.path.join(output_dir, "event_log.txt")
    with open(log_file_path, 'w') as f:
        f.write(f"Event log created at {datetime.now().isoformat()}\n")
    
    # Create pipeline configuration
    config = {
        'output_dir': output_dir,
        'environment': 'dev',
        'experts': {
            'physiological': {
                'type': 'physiological',
                'params': {
                    'model_type': 'linear',
                    'feature_subset': ['heart_rate', 'blood_pressure']
                }
            },
            'behavioral': {
                'type': 'behavioral',
                'params': {
                    'model_type': 'forest',
                    'feature_subset': ['sleep_hours', 'activity_level']
                }
            },
            'environmental': {
                'type': 'environmental',
                'params': {
                    'model_type': 'boosted',
                    'feature_subset': ['weather', 'location']
                }
            }
        },
        'gating_network': {
            'type': 'quality_aware',
            'params': {
                'quality_impact': 0.7,
                'drift_threshold': 0.2
            }
        },
        'meta_learner': {
            'enable_personalization': True,
            'method': 'bayesian',
            'memory_storage_dir': os.path.join(output_dir, 'patient_memory'),
            'exploration_factor': 0.2
        },
        'integration': {
            'method': 'adaptive',
            'params': {
                'quality_threshold': 0.6,
                'confidence_weight': 0.8
            }
        },
        'state_management': {
            'checkpoint_dir': os.path.join(output_dir, 'checkpoints'),
            'auto_checkpoint': True,
            'max_checkpoints': 5
        }
    }
    
    # Initialize pipeline with the configuration
    pipeline = MoEPipeline(config, verbose=True)
    
    # Create event logger and register with the pipeline's event manager
    event_logger = PipelineEventLogger(log_file_path)
    
    # Register custom event listeners
    pipeline.event_manager.register_listener(
        MoEEventTypes.TRAINING_STARTED, 
        event_logger.on_training_started
    )
    
    pipeline.event_manager.register_listener(
        MoEEventTypes.TRAINING_COMPLETED, 
        event_logger.on_training_completed
    )
    
    pipeline.event_manager.register_listener(
        MoEEventTypes.PREDICTION_COMPLETED, 
        event_logger.on_prediction_completed
    )
    
    pipeline.event_manager.register_listener(
        MoEEventTypes.CHECKPOINT_COMPLETED, 
        event_logger.on_checkpoint_completed
    )
    
    # Generate synthetic data for demo
    logger.info("Generating synthetic data for demonstration...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'heart_rate': np.random.normal(70, 10, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'activity_level': np.random.normal(5, 2, n_samples),
        'weather': np.random.normal(25, 5, n_samples),
        'location': np.random.normal(0, 1, n_samples),
        'migraine_intensity': np.random.normal(3, 2, n_samples)
    }
    
    # Create DataFrame and save to file
    df = pd.DataFrame(data)
    data_path = os.path.join(output_dir, 'synthetic_data.csv')
    df.to_csv(data_path, index=False)
    logger.info(f"Saved synthetic data to {data_path}")
    
    # Load data into pipeline
    logger.info("Loading data into pipeline...")
    load_result = pipeline.load_data(data_path, target_column='migraine_intensity')
    logger.info(f"Data loading result: {load_result['message']}")
    
    # Set patient context for demonstration
    pipeline.set_patient("patient_123")
    
    # Train the pipeline
    logger.info("Training the MoE pipeline...")
    train_result = pipeline.train(validation_split=0.3)
    logger.info(f"Training result: {train_result['message']}")
    
    # Make a prediction with the pipeline
    logger.info("Making prediction with trained pipeline...")
    prediction_result = pipeline.predict(use_loaded_data=True)
    logger.info(f"Prediction result: {prediction_result['message']}")
    
    # Check if a checkpoint was created during training
    if pipeline.pipeline_state.get('checkpoint_available', False):
        checkpoint_path = pipeline.pipeline_state.get('last_checkpoint_path')
        logger.info(f"Checkpoint available at: {checkpoint_path}")
        
        # Demonstrate checkpoint restoration
        logger.info("Creating a new pipeline instance and restoring from checkpoint...")
        new_pipeline = MoEPipeline(config, verbose=True)
        restore_result = new_pipeline.load_checkpoint(checkpoint_path)
        
        if restore_result:
            logger.info("Successfully restored pipeline from checkpoint")
            
            # Make prediction with restored pipeline
            restored_prediction = new_pipeline.predict(use_loaded_data=True)
            logger.info(f"Restored pipeline prediction result: {restored_prediction['message']}")
            
            # Compare predictions from original and restored pipelines
            if prediction_result.get('success') and restored_prediction.get('success'):
                orig_pred = prediction_result.get('predictions')
                restored_pred = restored_prediction.get('predictions')
                
                if hasattr(orig_pred, 'shape') and hasattr(restored_pred, 'shape'):
                    match = np.allclose(orig_pred, restored_pred)
                    logger.info(f"Predictions from original and restored pipelines match: {match}")
        else:
            logger.error("Failed to restore pipeline from checkpoint")
    
    # Get event summary
    event_summary = event_logger.get_event_summary()
    logger.info(f"Event summary: {event_summary}")
    
    logger.info("Integrated pipeline example completed")
    return {
        'pipeline': pipeline,
        'event_logger': event_logger,
        'train_result': train_result,
        'prediction_result': prediction_result
    }

if __name__ == "__main__":
    run_integrated_pipeline_example()

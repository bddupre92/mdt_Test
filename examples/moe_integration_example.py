"""
Example demonstrating the integration layer with MoE pipeline.

This example shows how to use the integration layer, event system, and state manager
with the existing MoE pipeline to create a complete end-to-end workflow.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import MoE framework components
from moe_framework.workflow.moe_pipeline import MoEPipeline
from moe_framework.integration import (
    IntegrationConnector,
    AdaptiveIntegration,
    EventManager, 
    Event,
    EventListener,
    MoEEventTypes
)
from moe_framework.persistence.state_manager import FileSystemStateManager
from moe_framework.interfaces.base import PatientContext


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineEventListener(EventListener):
    """Example event listener for pipeline events."""
    
    def handle_event(self, event: Event) -> None:
        """Handle an event notification."""
        logger.info(f"Event received: {event.event_type}")
        
        if event.event_type == MoEEventTypes.PIPELINE_STARTED:
            logger.info("Pipeline execution started")
        elif event.event_type == MoEEventTypes.INTEGRATION_COMPLETED:
            logger.info(f"Integration completed with {len(event.data.get('experts_used', []))} experts")
        elif event.event_type == MoEEventTypes.CHECKPOINT_SAVED:
            logger.info(f"Checkpoint saved to: {event.data.get('path', 'unknown')}")
    
    def get_handled_event_types(self) -> List[str]:
        """Get the list of event types this listener can handle."""
        return [
            MoEEventTypes.PIPELINE_STARTED,
            MoEEventTypes.PIPELINE_COMPLETED,
            MoEEventTypes.INTEGRATION_COMPLETED,
            MoEEventTypes.CHECKPOINT_SAVED,
            MoEEventTypes.CHECKPOINT_LOADED
        ]


def main():
    """Run the MoE integration example."""
    # Create the configuration
    config = {
        'output_dir': 'outputs',
        'environment': 'dev',
        'experts': {
            'use_physiological': True,
            'use_behavioral': True,
            'use_environmental': True,
            'use_medication_history': True,
            'physiological': {
                'model_type': 'lstm',
                'input_dims': 10
            },
            'behavioral': {
                'model_type': 'random_forest',
                'n_estimators': 100
            }
        },
        'gating': {
            'strategy': 'quality_aware'
        },
        'integration': {
            'confidence_threshold': 0.65,
            'quality_threshold': 0.4
        }
    }
    
    # Create components
    event_manager = EventManager()
    
    # Register event listener
    pipeline_listener = PipelineEventListener()
    event_manager.register_listener(pipeline_listener)
    
    # Create state manager
    state_manager = FileSystemStateManager({
        'base_dir': 'checkpoints'
    })
    
    # Create integration layer
    integration_layer = AdaptiveIntegration(config.get('integration', {}))
    
    # Create integration connector
    integration_connector = IntegrationConnector(
        integration_layer=integration_layer,
        event_manager=event_manager,
        state_manager=state_manager,
        config=config
    )
    
    # Create MoE pipeline
    pipeline = MoEPipeline(config=config, verbose=True)
    
    # Emit pipeline started event
    event_manager.emit_event(Event(MoEEventTypes.PIPELINE_STARTED, {
        'config': config
    }))
    
    # Use mock data for demonstration
    sample_data_path = 'sample_data.csv'
    if not os.path.exists(sample_data_path):
        # Create sample data if it doesn't exist
        create_sample_data(sample_data_path)
    
    # Load data
    logger.info("Loading data...")
    data_result = pipeline.load_data(sample_data_path, target_column='migraine')
    
    if data_result.get('success', False):
        # Train the pipeline
        logger.info("Training models...")
        train_result = pipeline.train(validation_split=0.2)
        
        if train_result.get('success', False):
            # Generate predictions
            logger.info("Generating predictions...")
            pred_result = pipeline.predict(use_loaded_data=True)
            
            # Get expert outputs and weights
            expert_outputs = {}
            weights = {}
            
            # For demonstration, we'll create mock expert outputs and weights
            for expert_name in pipeline.experts.keys():
                expert_outputs[expert_name] = np.random.rand(10, 1)
                weights[expert_name] = np.random.rand()
            
            # Normalize weights
            weight_sum = sum(weights.values())
            weights = {k: v/weight_sum for k, v in weights.items()}
            
            # Create mock patient context
            context = PatientContext(
                patient_id="patient123",
                data_quality={
                    'physiological': 0.9,
                    'behavioral': 0.7,
                    'environmental': 0.8,
                    'medication_history': 0.6
                },
                timestamp="2025-03-25T12:00:00"
            )
            
            # Use the integration connector to integrate predictions
            logger.info("Integrating predictions...")
            integrated_prediction = integration_connector.integrate_predictions(
                expert_outputs=expert_outputs,
                weights=weights,
                context=context
            )
            
            logger.info(f"Integrated prediction shape: {integrated_prediction.shape}")
            
            # Save pipeline state as checkpoint
            logger.info("Saving checkpoint...")
            checkpoint_path = "patient123/checkpoint1"
            # In a real implementation, would pass a proper SystemState object
            # integration_connector.save_pipeline_state(pipeline.get_system_state(), checkpoint_path)
            
            # List available checkpoints
            available_checkpoints = integration_connector.get_available_checkpoints()
            if available_checkpoints:
                logger.info(f"Available checkpoints: {len(available_checkpoints)}")
            
            logger.info("Pipeline execution completed successfully")
            event_manager.emit_event(Event(MoEEventTypes.PIPELINE_COMPLETED, {
                'status': 'success'
            }))
        else:
            logger.error("Training failed")
    else:
        logger.error("Data loading failed")


def create_sample_data(file_path: str):
    """Create sample data for demonstration."""
    # Create a simple dataset with physiological, behavioral, 
    # environmental, and medication features
    n_samples = 100
    
    data = {
        # Physiological features
        'heart_rate': np.random.normal(75, 10, n_samples),
        'blood_pressure': np.random.normal(120, 15, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        
        # Behavioral features
        'stress_level': np.random.randint(1, 6, n_samples),
        'exercise_minutes': np.random.gamma(5, 10, n_samples),
        'water_intake': np.random.normal(2000, 500, n_samples),
        
        # Environmental features
        'temperature': np.random.normal(22, 5, n_samples),
        'humidity': np.random.uniform(30, 80, n_samples),
        'barometric_pressure': np.random.normal(1013, 10, n_samples),
        
        # Medication features
        'medication_adherence': np.random.uniform(0, 1, n_samples),
        'medication_dose': np.random.choice([50, 100, 150, 200], n_samples),
        
        # Target variable
        'migraine': np.random.binomial(1, 0.3, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    logger.info(f"Created sample data file: {file_path}")


if __name__ == "__main__":
    main()

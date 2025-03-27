#!/usr/bin/env python
"""
Example demonstration of the workflow tracking system with comprehensive tracking.

This example demonstrates how to use the workflow tracking system to track:
1. Workflow component executions
2. Workflow events
3. Optimizer performance
4. Expert contributions
5. Meta-learner decisions

All in the same workflow execution.
"""

import logging
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import the workflow tracking system
from moe_framework.event_tracking.workflow_tracker import WorkflowTracker
from moe_framework.event_tracking.visualization import WorkflowVisualizer
from moe_framework.event_tracking.models import (
    WorkflowComponentType, 
    ExpertContribution,
    OptimizerPerformance,
    MetaLearnerDecision,
    ComponentExecution,
    WorkflowEvent,
    WorkflowExecution
)

class ExampleMoEPipeline:
    """
    Example MoE pipeline with workflow tracking.
    """
    
    def __init__(self, name: str = "ExampleMoEPipeline"):
        """
        Initialize the example MoE pipeline.
        
        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.tracker = WorkflowTracker(output_dir="./.workflow_tracking")
        
    def run(self):
        """
        Run the example MoE pipeline with workflow tracking.
        """
        # Start tracking the workflow
        self.tracker.start_workflow(self.name)
        
        # Track different components and events
        self.load_data()
        self.train_model()
        self.make_predictions()
        self.evaluate_model()
        self.save_checkpoint()
        
        # Get workflow ID before completing it
        workflow_id = self.tracker.current_workflow.workflow_id
        
        # Set workflow results
        self.tracker.complete_workflow(
            success=True,
            results={
                "accuracy": random.uniform(0.85, 0.95),
                "execution_time": random.uniform(3.5, 5.0)
            }
        )
        
        return workflow_id
    
    def load_data(self):
        """
        Simulate loading data.
        """
        logger.info("Loading data...")
        
        # Start tracking the data loading component
        with self._component_tracker(WorkflowComponentType.DATA_LOADING):
            # Simulate data loading
            time.sleep(1)
            
            # Create and add events
            self._add_event("Loading dataset", WorkflowComponentType.DATA_LOADING)
            self._add_event("Preprocessing data", WorkflowComponentType.DATA_LOADING)
            self._add_event("Splitting train/test", WorkflowComponentType.DATA_LOADING)
            
            # Add the meta-learner decision about which dataset to use
            self.tracker.track_meta_learner_decision(
                selection_id="dataset_selection",
                selected_algorithm="TimeSeries",
                problem_features={
                    "dimensions": 10,
                    "multimodal": True,
                    "time_dependent": True,
                    "contains_nulls": False,
                    "high_cardinality": True
                },
                confidence=0.89,
                alternatives={
                    "Tabular": 0.67,
                    "Image": 0.23,
                    "Text": 0.45
                }
            )
    
    def _component_tracker(self, component_type):
        """
        Context manager for tracking components.
        
        Args:
            component_type: Type of component to track
        """
        class ComponentTracker:
            def __init__(self, tracker, component_type):
                self.tracker = tracker
                self.component_type = component_type
                self.component_id = f"{component_type.value}_{id(self)}"
                
            def __enter__(self):
                # Create component
                self.component = ComponentExecution(
                    component=self.component_type,
                    entry_time=datetime.now().isoformat()
                )
                
                # Add to workflow
                if not self.tracker.current_workflow:
                    self.tracker.start_workflow()
                    
                self.tracker.current_workflow.add_component(self.component)
                self.tracker.active_components[self.component_id] = self.component
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Complete component
                success = exc_type is None
                self.component.complete(success=success)
                
                # Remove from active components
                if self.component_id in self.tracker.active_components:
                    del self.tracker.active_components[self.component_id]
                
                return False  # Don't suppress exceptions
        
        return ComponentTracker(self.tracker, component_type)
    
    def _add_event(self, event_type, component_type):
        """Helper to add events to the workflow"""
        # Create workflow event
        event = WorkflowEvent(
            component=component_type,
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            details={},
            success=True
        )
        
        # Add event to workflow
        if self.tracker.current_workflow:
            self.tracker.current_workflow.add_event(event)
            
        return event
    
    def train_model(self):
        """
        Simulate training a model.
        """
        logger.info("Training model...")
        
        # Get the current workflow
        workflow_id = self.tracker.start_workflow(self.name)
        
        # Start tracking the model training component
        with self._component_tracker(WorkflowComponentType.EXPERT_TRAINING):
            # Simulate optimizer performances
            self._simulate_optimizers()
            
            # Simulate expert training
            self._simulate_expert_training(num_experts=10)
            
            # Create and add events
            self._add_event("Initializing model", WorkflowComponentType.EXPERT_TRAINING)
            self._add_event("Training epoch 1", WorkflowComponentType.EXPERT_TRAINING)
            self._add_event("Training epoch 2", WorkflowComponentType.EXPERT_TRAINING)
            self._add_event("Training epoch 3", WorkflowComponentType.EXPERT_TRAINING)
            
            # Add meta-learner decision for training approach
            self.tracker.track_meta_learner_decision(
                selection_id="training_approach",
                selected_algorithm="Ensemble",
                problem_features={
                    "data_size": "large",
                    "feature_count": 50,
                    "compute_resources": "high",
                    "training_time": "limited"
                },
                confidence=0.92,
                alternatives={
                    "DeepLearning": 0.85,
                    "Boosting": 0.78,
                    "SVM": 0.62
                }
            )
    
    def make_predictions(self):
        """
        Simulate making predictions.
        """
        logger.info("Making predictions...")
        
        # Start tracking the prediction component
        with self._component_tracker(WorkflowComponentType.PREDICTION):
            # Simulate prediction
            time.sleep(1)
            
            # Create and add events
            self._add_event("Generating predictions", WorkflowComponentType.PREDICTION)
            self._add_event("Aggregating expert outputs", WorkflowComponentType.PREDICTION)
            
            # Simulate expert contributions during prediction
            self._simulate_expert_predictions(num_experts=5)
            
            # Add a meta-learner decision for expert weighting
            self.tracker.track_meta_learner_decision(
                selection_id="expert_weighting",
                selected_algorithm="DynamicWeight",
                problem_features={
                    "expert_count": 5,
                    "conflicting_outputs": True,
                    "confidence_available": True
                },
                confidence=0.87,
                alternatives={
                    "EqualWeight": 0.45,
                    "PerformanceWeight": 0.76,
                    "ConfidenceWeight": 0.83
                }
            )
    
    def evaluate_model(self):
        """
        Simulate evaluating a model.
        """
        logger.info("Evaluating model...")
        
        # Start tracking the evaluation component
        with self._component_tracker(WorkflowComponentType.EVALUATION):
            # Simulate evaluation
            time.sleep(1)
            
            # Create and add events
            self._add_event("Calculating metrics", WorkflowComponentType.EVALUATION)
            self._add_event("Comparing with baseline", WorkflowComponentType.EVALUATION)
    
    def save_checkpoint(self):
        """
        Simulate saving a checkpoint.
        """
        logger.info("Saving checkpoint...")
        
        # Start tracking the checkpoint component
        with self._component_tracker(WorkflowComponentType.CHECKPOINT):
            # Simulate checkpoint
            time.sleep(0.5)
            
            # Create and add events
            self._add_event("Saving model weights", WorkflowComponentType.CHECKPOINT)
            self._add_event("Saving optimizer state", WorkflowComponentType.CHECKPOINT)
    
    def _simulate_expert_training(self, num_experts: int = 3):
        """
        Simulate training experts and track their contributions.
        
        Args:
            num_experts: Number of experts to simulate
        """
        expert_types = ["physiological_expert", "environmental_expert", "behavioral_expert", "medication_expert"]
        
        for i in range(num_experts):
            expert_type = random.choice(expert_types)
            expert_id = f"{expert_type}_{i}"
            
            # Track expert contribution
            self.tracker.track_expert_contribution(
                expert_id=expert_id,
                expert_type=expert_type,
                confidence=random.uniform(0.6, 0.9),
                weight=random.uniform(0.0, 0.7),
                prediction=random.uniform(0.3, 0.9),
                features_used=[
                    "heart_rate", "blood_pressure", "sleep_quality", "stress_level"
                ] if expert_type == "physiological_expert" else [
                    "temperature", "humidity", "barometric_pressure", "light_level"
                ] if expert_type == "environmental_expert" else [
                    "exercise_minutes", "water_intake", "screen_time", "social_activity"
                ] if expert_type == "behavioral_expert" else [
                    "medication_adherence", "medication_timing", "medication_effectiveness", "side_effects"
                ],
                feature_importance={
                    "heart_rate": 0.459, "blood_pressure": 0.040, 
                    "sleep_quality": 0.364, "stress_level": 0.137
                } if expert_type == "physiological_expert" else {
                    "temperature": 0.103, "humidity": 0.159, 
                    "barometric_pressure": 0.477, "light_level": 0.261
                } if expert_type == "environmental_expert" else {
                    "exercise_minutes": 0.385, "water_intake": 0.100, 
                    "screen_time": 0.324, "social_activity": 0.190
                } if expert_type == "behavioral_expert" else {
                    "medication_adherence": 0.589, "medication_timing": 0.174, 
                    "medication_effectiveness": 0.068, "side_effects": 0.169
                }
            )
            
            # Log event about expert training
            self._add_event(f"Trained expert: {expert_id}", WorkflowComponentType.EXPERT_TRAINING)

    def _simulate_expert_predictions(self, num_experts: int = 3):
        """
        Simulate expert predictions and track their contributions.
        
        Args:
            num_experts: Number of experts to simulate
        """
        expert_types = ["physiological_expert", "environmental_expert", "behavioral_expert", "medication_expert"]
        
        for i in range(num_experts):
            expert_type = random.choice(expert_types)
            expert_id = f"{expert_type}_pred_{i}"
            
            # Track expert contribution
            self.tracker.track_expert_contribution(
                expert_id=expert_id,
                expert_type=expert_type,
                confidence=random.uniform(0.7, 0.95),
                weight=random.uniform(0.1, 0.8),
                prediction=random.uniform(0.4, 0.95),
                features_used=[
                    "heart_rate", "blood_pressure", "sleep_quality", "stress_level"
                ] if expert_type == "physiological_expert" else [
                    "temperature", "humidity", "barometric_pressure", "light_level"
                ] if expert_type == "environmental_expert" else [
                    "exercise_minutes", "water_intake", "screen_time", "social_activity"
                ] if expert_type == "behavioral_expert" else [
                    "medication_adherence", "medication_timing", "medication_effectiveness", "side_effects"
                ],
                feature_importance={
                    "heart_rate": 0.459, "blood_pressure": 0.040, 
                    "sleep_quality": 0.364, "stress_level": 0.137
                } if expert_type == "physiological_expert" else {
                    "temperature": 0.103, "humidity": 0.159, 
                    "barometric_pressure": 0.477, "light_level": 0.261
                } if expert_type == "environmental_expert" else {
                    "exercise_minutes": 0.385, "water_intake": 0.100, 
                    "screen_time": 0.324, "social_activity": 0.190
                } if expert_type == "behavioral_expert" else {
                    "medication_adherence": 0.589, "medication_timing": 0.174, 
                    "medication_effectiveness": 0.068, "side_effects": 0.169
                }
            )
            
            # Log event about expert prediction
            self._add_event(f"Expert prediction: {expert_id}", WorkflowComponentType.PREDICTION)
    
    def _simulate_optimizers(self):
        """
        Simulate multiple optimizers and track their performance.
        """
        # Simulate Differential Evolution optimizer
        de_optimizer = self._simulate_differential_evolution()
        self.tracker.track_optimizer_performance(
            optimizer_id=de_optimizer.optimizer_id,
            optimizer_type=de_optimizer.optimizer_type,
            best_fitness=de_optimizer.best_fitness,
            convergence_curve=de_optimizer.convergence_curve,
            parameters=de_optimizer.parameters,
            diversity_history=de_optimizer.diversity_history,
            iterations=de_optimizer.iterations,
            evaluations=de_optimizer.evaluations,
            duration=de_optimizer.duration
        )
        self._add_event("Tracked DE optimizer performance", WorkflowComponentType.EXPERT_TRAINING)
        
        # Simulate Evolution Strategy optimizer
        es_optimizer = self._simulate_evolution_strategy()
        self.tracker.track_optimizer_performance(
            optimizer_id=es_optimizer.optimizer_id,
            optimizer_type=es_optimizer.optimizer_type,
            best_fitness=es_optimizer.best_fitness,
            convergence_curve=es_optimizer.convergence_curve,
            parameters=es_optimizer.parameters,
            diversity_history=es_optimizer.diversity_history,
            exploration_exploitation_ratio=es_optimizer.exploration_exploitation_ratio,
            iterations=es_optimizer.iterations,
            evaluations=es_optimizer.evaluations,
            duration=es_optimizer.duration
        )
        self._add_event("Tracked ES optimizer performance", WorkflowComponentType.EXPERT_TRAINING)
    
    def _simulate_differential_evolution(self) -> OptimizerPerformance:
        """
        Simulate a Differential Evolution optimizer and return its performance.
        
        Returns:
            OptimizerPerformance for the DE optimizer
        """
        iterations = 20
        convergence = [100.0 * np.exp(-0.1 * i) for i in range(iterations)]
        
        # Create parameter history
        f_values = [0.8 - 0.02 * i for i in range(iterations)]
        cr_values = [0.5 + 0.01 * i for i in range(iterations)]
        
        # Create diversity history
        diversity = [1.0 * np.exp(-0.05 * i) for i in range(iterations)]
        
        return OptimizerPerformance(
            optimizer_id="DE_optimizer",
            optimizer_type="differential_evolution",
            best_fitness=convergence[-1],
            convergence_curve=convergence,
            parameters={
                "F": f_values,
                "CR": cr_values
            },
            diversity_history=diversity,
            iterations=iterations,
            evaluations=iterations * 50,
            duration=1.2
        )
    
    def _simulate_evolution_strategy(self) -> OptimizerPerformance:
        """
        Simulate an Evolution Strategy optimizer and return its performance.
        
        Returns:
            OptimizerPerformance for the ES optimizer
        """
        iterations = 15
        convergence = [100.0 * np.exp(-0.12 * i) for i in range(iterations)]
        
        # Create parameter history
        sigma_values = [1.0 - 0.05 * i for i in range(iterations)]
        pop_size = [50] * iterations
        
        # Create diversity and exploration/exploitation ratio history
        diversity = [0.9 * np.exp(-0.06 * i) for i in range(iterations)]
        expl_expl_ratio = [0.7 - 0.04 * i for i in range(iterations)]
        
        return OptimizerPerformance(
            optimizer_id="ES_optimizer",
            optimizer_type="evolution_strategy",
            best_fitness=convergence[-1],
            convergence_curve=convergence,
            parameters={
                "sigma": sigma_values,
                "population_size": pop_size
            },
            diversity_history=diversity,
            exploration_exploitation_ratio=expl_expl_ratio,
            iterations=iterations,
            evaluations=iterations * 50 + 50,
            duration=0.9
        )

def run_demo():
    """
    Run the example workflow tracking demo.
    """
    logger.info("Running example workflow tracking demo")
    
    # Initialize the MoE pipeline
    pipeline = ExampleMoEPipeline()
    
    # Run the pipeline
    workflow_id = pipeline.run()
    
    # Create a visualizer
    visualizer = WorkflowVisualizer(output_dir="./visualizations")
    
    # Get the workflow that was just created
    workflow = WorkflowExecution.load(
        os.path.join("./.workflow_tracking", f"{workflow_id}.json")
    )
    
    logger.info(f"Generating visualizations for workflow: {workflow_id}")
    
    # Create visualizations
    graph_path = visualizer.visualize_workflow(workflow)
    logger.info(f"Graph visualization saved to: {graph_path}")
    
    timeline_path = visualizer.create_timeline_visualization(workflow)
    logger.info(f"Timeline visualization saved to: {timeline_path}")
    
    mermaid_path = visualizer.generate_mermaid_workflow(workflow)
    logger.info(f"Mermaid diagram saved to: {mermaid_path}")
    
    json_path = visualizer.export_workflow_json(workflow)
    logger.info(f"Workflow data exported to: {json_path}")
    
    # Generate visualizations for optimizer performance
    optimizer_visualizations = visualizer.visualize_optimizer_performance(workflow)
    if optimizer_visualizations:
        for viz_type, path in optimizer_visualizations.items():
            logger.info(f"Optimizer {viz_type} visualization saved to: {path}")
    
    # Generate visualizations for expert contributions
    expert_visualizations = visualizer.visualize_expert_contributions(workflow)
    if expert_visualizations:
        for viz_type, path in expert_visualizations.items():
            logger.info(f"Expert {viz_type} visualization saved to: {path}")
    
    # Generate visualizations for meta-learner decisions
    meta_visualizations = visualizer.visualize_meta_learner_decisions(workflow)
    if meta_visualizations:
        for viz_type, path in meta_visualizations.items():
            logger.info(f"Meta-learner {viz_type} visualization saved to: {path}")
    
    # Create MoE framework flow diagram
    from moe_framework.event_tracking.visualization import create_moe_flow_diagram
    flow_diagram_path = create_moe_flow_diagram("moe_framework_flow.png")
    logger.info(f"MoE framework flow diagram saved to: {flow_diagram_path}")
    
    logger.info("\nWorkflow tracking demo completed successfully!")
    logger.info("Visualizations have been saved to the './visualizations' directory.")
    logger.info("To launch the interactive dashboard, run: python -m moe_framework.event_tracking.dashboard")
    
    # Ask if the user wants to launch the dashboard
    launch_dashboard = input("Launch dashboard now? (y/n): ")
    if launch_dashboard.lower() == 'y':
        logger.info("Launching dashboard...")
        
        # Import and run the dashboard
        from moe_framework.event_tracking.dashboard import render_workflow_dashboard
        render_workflow_dashboard("./.workflow_tracking")

if __name__ == "__main__":
    run_demo() 
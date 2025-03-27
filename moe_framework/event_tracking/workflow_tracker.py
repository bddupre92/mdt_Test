"""
Workflow Tracker for MoE Framework

This module provides the core workflow tracking functionality for the MoE framework.
It integrates with the existing event system and adds tracking for workflow execution,
component interactions, and decision points.
"""

import os
import logging
import functools
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Type, Callable
from uuid import uuid4

from .models import (
    WorkflowEvent, 
    ComponentExecution, 
    WorkflowExecution, 
    WorkflowComponentType,
    ExpertContribution,
    OptimizerPerformance,
    MetaLearnerDecision
)

# Import event system components
from ..integration.event_system import (
    EventListener, 
    Event, 
    EventManager, 
    MoEEventTypes
)

logger = logging.getLogger(__name__)

class WorkflowTracker:
    """
    Tracks the execution of the MoE framework workflow.
    
    This class hooks into the existing event system to track the workflow
    execution and provide detailed insights into the execution flow, component
    interactions, and decision points.
    """
    
    def __init__(self, 
                 event_manager: Optional[EventManager] = None,
                 output_dir: str = "./.workflow_tracking",
                 verbose: bool = False):
        """
        Initialize the workflow tracker.
        
        Args:
            event_manager: The event manager to hook into
            output_dir: Directory to store tracking data
            verbose: Whether to log detailed tracking information
        """
        self.event_manager = event_manager
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tracking state
        self.current_workflow: Optional[WorkflowExecution] = None
        self.active_components: Dict[str, ComponentExecution] = {}
        self.tracked_methods: Dict[str, Callable] = {}
        
        # Initialize event listener if event manager is provided
        if self.event_manager:
            self._register_event_listener()
            
        logger.info(f"Initialized WorkflowTracker with output directory: {output_dir}")
    
    def _register_event_listener(self):
        """Register event listener with the event manager."""
        
        class WorkflowEventListener(EventListener):
            """Event listener for workflow tracking."""
            
            def __init__(self, tracker):
                self.tracker = tracker
                
            def handle_event(self, event: Event):
                """Handle an event."""
                self.tracker._handle_event(event)
                
            def get_handled_event_types(self):
                """Get the event types handled by this listener."""
                # Return a list of all the event type attributes from MoEEventTypes
                event_types = []
                for attr in dir(MoEEventTypes):
                    if not attr.startswith('_') and attr.isupper():
                        event_types.append(getattr(MoEEventTypes, attr))
                return event_types
        
        # Register the event listener
        self.event_manager.register_listener(WorkflowEventListener(self))
        
        if self.verbose:
            logger.info("Registered workflow event listener")
    
    def _handle_event(self, event: Event):
        """
        Handle an event from the event system.
        
        Args:
            event: The event to handle
        """
        if not self.current_workflow:
            # Start a new workflow if none is active
            self.start_workflow()
            
        # Map event type to component type
        component_type = self._map_event_to_component(event.event_type)
        
        # Create workflow event
        workflow_event = WorkflowEvent(
            component=component_type,
            event_type=event.event_type,
            timestamp=datetime.now().isoformat(),
            details=event.data,
            success=event.data.get('success', True) if isinstance(event.data, dict) else True
        )
        
        # Add event to workflow
        self.current_workflow.add_event(workflow_event)
        
        # Handle specific event types
        self._handle_specific_event(event, workflow_event)
        
        if self.verbose:
            logger.debug(f"Tracked event: {event.event_type}")
            
    def _map_event_to_component(self, event_type: str) -> WorkflowComponentType:
        """
        Map an event type to a component type.
        
        Args:
            event_type: The event type to map
            
        Returns:
            The corresponding component type
        """
        # Map MoEEventTypes to WorkflowComponentType
        event_to_component = {
            "TRAINING_STARTED": WorkflowComponentType.EXPERT_TRAINING,
            "TRAINING_COMPLETED": WorkflowComponentType.EXPERT_TRAINING,
            "EXPERT_TRAINING_STARTED": WorkflowComponentType.EXPERT_TRAINING,
            "EXPERT_TRAINING_COMPLETED": WorkflowComponentType.EXPERT_TRAINING,
            "GATING_TRAINING_STARTED": WorkflowComponentType.GATING_TRAINING,
            "GATING_TRAINING_COMPLETED": WorkflowComponentType.GATING_TRAINING,
            "PREDICTION_STARTED": WorkflowComponentType.PREDICTION,
            "PREDICTION_COMPLETED": WorkflowComponentType.PREDICTION,
            "EXPERT_PREDICTIONS_STARTED": WorkflowComponentType.PREDICTION,
            "EXPERT_PREDICTION_COMPLETED": WorkflowComponentType.PREDICTION,
            "GATING_WEIGHTS_CALCULATED": WorkflowComponentType.WEIGHT_CALCULATION,
            "INTEGRATION_STARTED": WorkflowComponentType.INTEGRATION,
            "INTEGRATION_COMPLETED": WorkflowComponentType.INTEGRATION,
            "CHECKPOINT_STARTED": WorkflowComponentType.CHECKPOINT,
            "CHECKPOINT_COMPLETED": WorkflowComponentType.CHECKPOINT,
            "CHECKPOINT_RESTORE_STARTED": WorkflowComponentType.CHECKPOINT,
            "CHECKPOINT_RESTORE_COMPLETED": WorkflowComponentType.CHECKPOINT,
            "DATA_SPLIT_COMPLETED": WorkflowComponentType.DATA_LOADING,
            "DATA_LOADED": WorkflowComponentType.DATA_LOADING,
            "QUALITY_ASSESSMENT_COMPLETED": WorkflowComponentType.QUALITY_ASSESSMENT,
            # New mappings for evolutionary algorithms
            "DE_STARTED": WorkflowComponentType.DIFFERENTIAL_EVOLUTION,
            "DE_COMPLETED": WorkflowComponentType.DIFFERENTIAL_EVOLUTION,
            "DE_ITERATION_COMPLETED": WorkflowComponentType.DIFFERENTIAL_EVOLUTION,
            "ES_STARTED": WorkflowComponentType.EVOLUTION_STRATEGY,
            "ES_COMPLETED": WorkflowComponentType.EVOLUTION_STRATEGY,
            "ES_ITERATION_COMPLETED": WorkflowComponentType.EVOLUTION_STRATEGY,
            "ACO_STARTED": WorkflowComponentType.ANT_COLONY_OPTIMIZATION,
            "ACO_COMPLETED": WorkflowComponentType.ANT_COLONY_OPTIMIZATION,
            "ACO_ITERATION_COMPLETED": WorkflowComponentType.ANT_COLONY_OPTIMIZATION,
            "GWO_STARTED": WorkflowComponentType.GREY_WOLF_OPTIMIZER,
            "GWO_COMPLETED": WorkflowComponentType.GREY_WOLF_OPTIMIZER,
            "GWO_ITERATION_COMPLETED": WorkflowComponentType.GREY_WOLF_OPTIMIZER,
            # New mappings for expert models
            "PHYSIOLOGICAL_EXPERT_STARTED": WorkflowComponentType.PHYSIOLOGICAL_EXPERT,
            "PHYSIOLOGICAL_EXPERT_COMPLETED": WorkflowComponentType.PHYSIOLOGICAL_EXPERT,
            "ENVIRONMENTAL_EXPERT_STARTED": WorkflowComponentType.ENVIRONMENTAL_EXPERT,
            "ENVIRONMENTAL_EXPERT_COMPLETED": WorkflowComponentType.ENVIRONMENTAL_EXPERT,
            "BEHAVIORAL_EXPERT_STARTED": WorkflowComponentType.BEHAVIORAL_EXPERT,
            "BEHAVIORAL_EXPERT_COMPLETED": WorkflowComponentType.BEHAVIORAL_EXPERT,
            "MEDICATION_EXPERT_STARTED": WorkflowComponentType.MEDICATION_EXPERT,
            "MEDICATION_EXPERT_COMPLETED": WorkflowComponentType.MEDICATION_EXPERT,
            # New mappings for meta-learner and gating network
            "META_LEARNER_STARTED": WorkflowComponentType.META_LEARNER,
            "META_LEARNER_COMPLETED": WorkflowComponentType.META_LEARNER,
            "META_LEARNER_DECISION_MADE": WorkflowComponentType.META_LEARNER,
            "GATING_NETWORK_STARTED": WorkflowComponentType.GATING_NETWORK,
            "GATING_NETWORK_COMPLETED": WorkflowComponentType.GATING_NETWORK,
            "GATING_NETWORK_WEIGHTS_CALCULATED": WorkflowComponentType.GATING_NETWORK,
        }
        
        # Return the mapped component type or OTHER if not found
        return WorkflowComponentType(event_to_component.get(event_type, "other"))
    
    def _handle_specific_event(self, event: Event, workflow_event: WorkflowEvent):
        """
        Handle specific event types with special tracking logic.
        
        Args:
            event: The original event
            workflow_event: The workflow event created from the original event
        """
        # Handle training events
        if event.event_type == "TRAINING_STARTED":
            # Start a new component execution for training
            component = ComponentExecution(
                component=WorkflowComponentType.EXPERT_TRAINING,
                entry_time=workflow_event.timestamp
            )
            self.current_workflow.add_component(component)
            self.active_components["training"] = component
            
        elif event.event_type == "TRAINING_COMPLETED":
            # Complete the training component
            if "training" in self.active_components:
                component = self.active_components["training"]
                component.complete(
                    success=workflow_event.success,
                    result=event.data
                )
                del self.active_components["training"]
        
        # Handle prediction events
        elif event.event_type == "PREDICTION_STARTED":
            # Start a new component execution for prediction
            component = ComponentExecution(
                component=WorkflowComponentType.PREDICTION,
                entry_time=workflow_event.timestamp
            )
            self.current_workflow.add_component(component)
            self.active_components["prediction"] = component
            
        elif event.event_type == "PREDICTION_COMPLETED":
            # Complete the prediction component
            if "prediction" in self.active_components:
                component = self.active_components["prediction"]
                component.complete(
                    success=workflow_event.success,
                    result=event.data
                )
                del self.active_components["prediction"]
                
        # Handle checkpoint events
        elif event.event_type == "CHECKPOINT_COMPLETED" or event.event_type == "CHECKPOINT_RESTORE_COMPLETED":
            # Complete the entire workflow if this is a final checkpoint
            if event.data.get("final", False):
                self.complete_workflow(
                    success=workflow_event.success,
                    results={"checkpoint_path": event.data.get("path")}
                )
    
    def start_workflow(self, workflow_id: Optional[str] = None) -> WorkflowExecution:
        """
        Start tracking a new workflow execution.
        
        Args:
            workflow_id: Optional workflow ID (defaults to a UUID)
            
        Returns:
            The new workflow execution
        """
        # Generate a workflow ID if not provided
        if not workflow_id:
            workflow_id = f"workflow_{uuid4()}"
            
        # Create a new workflow execution
        self.current_workflow = WorkflowExecution(workflow_id=workflow_id)
        
        if self.verbose:
            logger.info(f"Started tracking workflow: {workflow_id}")
            
        return self.current_workflow
    
    def complete_workflow(self, success: bool = True, results: Dict[str, Any] = None):
        """
        Complete the current workflow execution.
        
        Args:
            success: Whether the workflow completed successfully
            results: Optional results from the workflow
        """
        if not self.current_workflow:
            logger.warning("No active workflow to complete")
            return
            
        # Complete any active components
        for component_id, component in list(self.active_components.items()):
            component.complete(success=success)
            del self.active_components[component_id]
        
        # Complete the workflow
        self.current_workflow.complete(success=success, results=results)
        
        # Save the workflow execution to a file
        workflow_id = self.current_workflow.workflow_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        file_name = f"{workflow_id}.json"
        file_path = os.path.join(self.output_dir, file_name)
        
        self.current_workflow.save(file_path)
        
        if self.verbose:
            logger.info(f"Completed workflow: {workflow_id}")
            logger.info(f"Saved workflow tracking data to: {file_path}")
            
        # Return and reset the completed workflow
        completed_workflow = self.current_workflow
        self.current_workflow = None
        
        return completed_workflow
    
    def track_expert_contribution(self, 
                                expert_id: str, 
                                expert_type: Union[str, WorkflowComponentType],
                                confidence: float,
                                weight: float,
                                prediction: Any,
                                features_used: List[str] = None,
                                feature_importance: Dict[str, float] = None):
        """
        Track an expert model's contribution to a prediction.
        
        Args:
            expert_id: Identifier for the expert model
            expert_type: Type of expert model
            confidence: Confidence level of the expert's prediction
            weight: Weight assigned to the expert's prediction
            prediction: The expert's prediction
            features_used: List of features used by the expert
            feature_importance: Feature importance values
        """
        if not self.current_workflow:
            self.start_workflow()
            
        # Ensure expert_type is a WorkflowComponentType
        if isinstance(expert_type, str):
            try:
                expert_type = WorkflowComponentType(expert_type)
            except ValueError:
                if expert_type.upper() == "PHYSIOLOGICAL":
                    expert_type = WorkflowComponentType.PHYSIOLOGICAL_EXPERT
                elif expert_type.upper() == "ENVIRONMENTAL":
                    expert_type = WorkflowComponentType.ENVIRONMENTAL_EXPERT
                elif expert_type.upper() == "BEHAVIORAL":
                    expert_type = WorkflowComponentType.BEHAVIORAL_EXPERT
                elif expert_type.upper() == "MEDICATION":
                    expert_type = WorkflowComponentType.MEDICATION_EXPERT
                else:
                    logger.warning(f"Unknown expert type: {expert_type}, using 'other'")
                    expert_type = WorkflowComponentType.OTHER
                    
        # Create expert contribution
        contribution = ExpertContribution(
            expert_id=expert_id,
            expert_type=expert_type,
            confidence=confidence,
            weight=weight,
            prediction=prediction,
            features_used=features_used or [],
            feature_importance=feature_importance or {}
        )
        
        # Add contribution to workflow
        self.current_workflow.add_expert_contribution(contribution)
        
        if self.verbose:
            logger.debug(f"Tracked expert contribution: {expert_id}")
            
        return contribution
    
    def track_optimizer_performance(self,
                                  optimizer_id: str,
                                  optimizer_type: Union[str, WorkflowComponentType],
                                  best_fitness: float,
                                  convergence_curve: List[float] = None,
                                  parameters: Dict[str, List[Any]] = None,
                                  diversity_history: List[float] = None,
                                  exploration_exploitation_ratio: List[float] = None,
                                  iterations: int = 0,
                                  evaluations: int = 0,
                                  duration: float = 0.0):
        """
        Track the performance of an evolutionary optimizer.
        
        Args:
            optimizer_id: Identifier for the optimizer
            optimizer_type: Type of optimizer
            best_fitness: Best fitness value found
            convergence_curve: List of fitness values over iterations
            parameters: Dictionary of parameter values over iterations
            diversity_history: List of diversity values over iterations
            exploration_exploitation_ratio: List of exploration/exploitation values
            iterations: Number of iterations performed
            evaluations: Number of function evaluations
            duration: Duration of optimization in seconds
        """
        if not self.current_workflow:
            self.start_workflow()
            
        # Ensure optimizer_type is a WorkflowComponentType
        if isinstance(optimizer_type, str):
            try:
                optimizer_type = WorkflowComponentType(optimizer_type)
            except ValueError:
                if optimizer_type.upper() == "DE":
                    optimizer_type = WorkflowComponentType.DIFFERENTIAL_EVOLUTION
                elif optimizer_type.upper() == "ES":
                    optimizer_type = WorkflowComponentType.EVOLUTION_STRATEGY
                elif optimizer_type.upper() == "ACO":
                    optimizer_type = WorkflowComponentType.ANT_COLONY_OPTIMIZATION
                elif optimizer_type.upper() == "GWO":
                    optimizer_type = WorkflowComponentType.GREY_WOLF_OPTIMIZER
                else:
                    logger.warning(f"Unknown optimizer type: {optimizer_type}, using 'other'")
                    optimizer_type = WorkflowComponentType.OTHER
                    
        # Create optimizer performance record
        performance = OptimizerPerformance(
            optimizer_id=optimizer_id,
            optimizer_type=optimizer_type,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve or [],
            parameters=parameters or {},
            diversity_history=diversity_history or [],
            exploration_exploitation_ratio=exploration_exploitation_ratio or [],
            iterations=iterations,
            evaluations=evaluations,
            duration=duration
        )
        
        # Add performance record to workflow
        self.current_workflow.add_optimizer_performance(performance)
        
        if self.verbose:
            logger.debug(f"Tracked optimizer performance: {optimizer_id}")
            
        return performance
    
    def track_meta_learner_decision(self,
                                  selection_id: str,
                                  selected_algorithm: str,
                                  problem_features: Dict[str, Any] = None,
                                  confidence: float = 0.0,
                                  alternatives: Dict[str, float] = None):
        """
        Track a decision made by the meta-learner.
        
        Args:
            selection_id: Identifier for the selection decision
            selected_algorithm: Name of the selected algorithm
            problem_features: Features used to make the selection
            confidence: Confidence in the selection
            alternatives: Alternative algorithms and their scores
        """
        if not self.current_workflow:
            self.start_workflow()
            
        # Create meta-learner decision record
        decision = MetaLearnerDecision(
            selection_id=selection_id,
            selected_algorithm=selected_algorithm,
            problem_features=problem_features or {},
            confidence=confidence,
            alternatives=alternatives or {}
        )
        
        # Add decision to workflow
        self.current_workflow.add_meta_learner_decision(decision)
        
        if self.verbose:
            logger.debug(f"Tracked meta-learner decision: {selection_id} -> {selected_algorithm}")
            
        return decision
    
    def track_component(self, component_type: Union[str, WorkflowComponentType]):
        """
        Decorator to track component execution.
        
        Args:
            component_type: The type of component to track
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Ensure component_type is a WorkflowComponentType
                comp_type = (
                    WorkflowComponentType(component_type) 
                    if isinstance(component_type, str) 
                    else component_type
                )
                
                # Create a component execution
                component = ComponentExecution(
                    component=comp_type,
                    entry_time=datetime.now().isoformat()
                )
                
                # Add to active components
                comp_id = f"{comp_type.value}_{id(func)}"
                self.active_components[comp_id] = component
                
                # Ensure we have a workflow
                if not self.current_workflow:
                    self.start_workflow()
                
                self.current_workflow.add_component(component)
                
                if self.verbose:
                    logger.debug(f"Started tracking component: {comp_type.value}")
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Complete the component
                    component.complete(success=True, result=result)
                    
                    if self.verbose:
                        logger.debug(f"Completed tracking component: {comp_type.value}")
                    
                    return result
                except Exception as e:
                    # Complete the component with failure
                    component.complete(success=False, result={"error": str(e)})
                    
                    if self.verbose:
                        logger.error(f"Error in component {comp_type.value}: {str(e)}")
                    
                    # Re-raise the exception
                    raise
                finally:
                    # Remove from active components
                    if comp_id in self.active_components:
                        del self.active_components[comp_id]
            
            # Register the wrapped method
            self.tracked_methods[func.__qualname__] = wrapper
            return wrapper
        
        return decorator
    
    def track_moe_pipeline(self, pipeline):
        """
        Track an MoE pipeline instance by wrapping its methods.
        
        Args:
            pipeline: The MoE pipeline instance to track
            
        Returns:
            The modified pipeline instance
        """
        # Map pipeline methods to component types
        method_to_component = {
            "load_data": WorkflowComponentType.DATA_LOADING,
            "train": WorkflowComponentType.EXPERT_TRAINING,
            "predict": WorkflowComponentType.PREDICTION,
            "evaluate": WorkflowComponentType.EVALUATION,
            "set_patient": WorkflowComponentType.SYSTEM,
            "_create_checkpoint": WorkflowComponentType.CHECKPOINT,
            "load_checkpoint": WorkflowComponentType.CHECKPOINT
        }
        
        # Wrap each method
        for method_name, component_type in method_to_component.items():
            if hasattr(pipeline, method_name):
                original_method = getattr(pipeline, method_name)
                wrapped_method = self.track_component(component_type)(original_method)
                setattr(pipeline, method_name, wrapped_method)
                
                if self.verbose:
                    logger.debug(f"Tracked pipeline method: {method_name}")
        
        # Set the event manager if not already set
        if not self.event_manager and hasattr(pipeline, "event_manager"):
            self.event_manager = pipeline.event_manager
            self._register_event_listener()
            
        if self.verbose:
            logger.info(f"Tracking MoE pipeline: {pipeline.__class__.__name__}")
        
        return pipeline
    
    def get_tracked_executions(self) -> List[WorkflowExecution]:
        """
        Get all tracked workflow executions.
        
        Returns:
            List of workflow executions
        """
        executions = []
        
        # Load all workflow executions from the output directory
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join(self.output_dir, filename)
                    execution = WorkflowExecution.load(file_path)
                    executions.append(execution)
                except Exception as e:
                    logger.error(f"Error loading workflow execution from {filename}: {str(e)}")
        
        return executions 
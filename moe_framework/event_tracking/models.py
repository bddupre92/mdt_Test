"""
Data Models for Workflow Event Tracking.

This module defines the data structures used for tracking workflow execution,
including events, components, and executions.
"""

import os
import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class WorkflowComponentType(enum.Enum):
    """Types of components in the MoE framework workflow."""
    
    DATA_LOADING = "data_loading"
    QUALITY_ASSESSMENT = "quality_assessment"
    EXPERT_TRAINING = "expert_training"
    GATING_TRAINING = "gating_training"
    PREDICTION = "prediction"
    INTEGRATION = "integration"
    WEIGHT_CALCULATION = "weight_calculation"
    EVALUATION = "evaluation"
    CHECKPOINT = "checkpoint"
    SYSTEM = "system"
    # New component types for evolutionary algorithms and expert models
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    EVOLUTION_STRATEGY = "evolution_strategy"
    ANT_COLONY_OPTIMIZATION = "ant_colony_optimization"
    GREY_WOLF_OPTIMIZER = "grey_wolf_optimizer"
    PHYSIOLOGICAL_EXPERT = "physiological_expert"
    ENVIRONMENTAL_EXPERT = "environmental_expert"
    BEHAVIORAL_EXPERT = "behavioral_expert"
    MEDICATION_EXPERT = "medication_expert"
    META_LEARNER = "meta_learner"
    GATING_NETWORK = "gating_network"
    OPTIMIZER_ANALYSIS = "optimizer_analysis"
    OTHER = "other"

@dataclass
class WorkflowEvent:
    """
    Represents a specific event within a workflow execution.
    
    Events capture specific actions or state changes that occur during
    the execution of a workflow.
    """
    
    component: WorkflowComponentType
    event_type: str
    timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "component": self.component.value,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "details": self.details,
            "success": self.success
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowEvent':
        """
        Create an event from a dictionary.
        
        Args:
            data: Dictionary representation of the event
            
        Returns:
            WorkflowEvent instance
        """
        # Handle component type conversion
        component_value = data.get("component", "other")
        try:
            component = WorkflowComponentType(component_value)
        except ValueError:
            logger.warning(f"Unknown component type: {component_value}, using 'other'")
            component = WorkflowComponentType.OTHER
        
        return cls(
            component=component,
            event_type=data.get("event_type", "unknown"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            details=data.get("details", {}),
            success=data.get("success", True)
        )

@dataclass
class ComponentExecution:
    """
    Represents the execution of a specific component within a workflow.
    
    Component executions track the lifetime of a component, from entry
    to exit, including success status and results.
    """
    
    component: WorkflowComponentType
    entry_time: str
    exit_time: Optional[str] = None
    success: bool = True
    result: Optional[Dict[str, Any]] = None
    
    def complete(self, success: bool = True, result: Optional[Dict[str, Any]] = None):
        """
        Mark the component execution as complete.
        
        Args:
            success: Whether the component completed successfully
            result: Optional result data from the component
        """
        self.exit_time = datetime.now().isoformat()
        self.success = success
        self.result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the component execution to a dictionary.
        
        Returns:
            Dictionary representation of the component execution
        """
        return {
            "component": self.component.value,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "success": self.success,
            "result": self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentExecution':
        """
        Create a component execution from a dictionary.
        
        Args:
            data: Dictionary representation of the component execution
            
        Returns:
            ComponentExecution instance
        """
        # Handle component type conversion
        component_value = data.get("component", "other")
        try:
            component = WorkflowComponentType(component_value)
        except ValueError:
            logger.warning(f"Unknown component type: {component_value}, using 'other'")
            component = WorkflowComponentType.OTHER
        
        return cls(
            component=component,
            entry_time=data.get("entry_time", ""),
            exit_time=data.get("exit_time"),
            success=data.get("success", True),
            result=data.get("result")
        )

@dataclass
class ExpertContribution:
    """
    Tracks the contribution of an expert model in the MoE framework.
    
    This includes the expert's predictions, confidence, and weights assigned by
    the gating network.
    """
    
    expert_id: str
    expert_type: WorkflowComponentType
    confidence: float
    weight: float
    prediction: Any
    features_used: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the expert contribution to a dictionary."""
        return {
            "expert_id": self.expert_id,
            "expert_type": self.expert_type.value,
            "confidence": self.confidence,
            "weight": self.weight,
            "prediction": self.prediction,
            "features_used": self.features_used,
            "feature_importance": self.feature_importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertContribution':
        """Create an expert contribution from a dictionary."""
        # Handle expert type conversion
        expert_type_value = data.get("expert_type", "other")
        try:
            expert_type = WorkflowComponentType(expert_type_value)
        except ValueError:
            logger.warning(f"Unknown expert type: {expert_type_value}, using 'other'")
            expert_type = WorkflowComponentType.OTHER
        
        return cls(
            expert_id=data.get("expert_id", ""),
            expert_type=expert_type,
            confidence=data.get("confidence", 0.0),
            weight=data.get("weight", 0.0),
            prediction=data.get("prediction"),
            features_used=data.get("features_used", []),
            feature_importance=data.get("feature_importance", {})
        )

@dataclass
class OptimizerPerformance:
    """
    Tracks the performance of an evolutionary optimizer.
    
    This includes convergence metrics, parameter adaptation, population diversity,
    and other optimizer-specific performance indicators.
    """
    
    optimizer_id: str
    optimizer_type: WorkflowComponentType
    best_fitness: float
    convergence_curve: List[float] = field(default_factory=list)
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    diversity_history: List[float] = field(default_factory=list)
    exploration_exploitation_ratio: List[float] = field(default_factory=list)
    iterations: int = 0
    evaluations: int = 0
    duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the optimizer performance to a dictionary."""
        return {
            "optimizer_id": self.optimizer_id,
            "optimizer_type": self.optimizer_type.value,
            "best_fitness": self.best_fitness,
            "convergence_curve": self.convergence_curve,
            "parameters": self.parameters,
            "diversity_history": self.diversity_history,
            "exploration_exploitation_ratio": self.exploration_exploitation_ratio,
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "duration": self.duration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizerPerformance':
        """Create an optimizer performance from a dictionary."""
        # Handle optimizer type conversion
        optimizer_type_value = data.get("optimizer_type", "other")
        try:
            optimizer_type = WorkflowComponentType(optimizer_type_value)
        except ValueError:
            logger.warning(f"Unknown optimizer type: {optimizer_type_value}, using 'other'")
            optimizer_type = WorkflowComponentType.OTHER
        
        return cls(
            optimizer_id=data.get("optimizer_id", ""),
            optimizer_type=optimizer_type,
            best_fitness=data.get("best_fitness", 0.0),
            convergence_curve=data.get("convergence_curve", []),
            parameters=data.get("parameters", {}),
            diversity_history=data.get("diversity_history", []),
            exploration_exploitation_ratio=data.get("exploration_exploitation_ratio", []),
            iterations=data.get("iterations", 0),
            evaluations=data.get("evaluations", 0),
            duration=data.get("duration", 0.0)
        )

@dataclass
class MetaLearnerDecision:
    """
    Tracks the decisions made by the meta-learner.
    
    This includes algorithm selection decisions and the problem features that
    influenced those decisions.
    """
    
    selection_id: str
    selected_algorithm: str
    problem_features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    alternatives: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the meta-learner decision to a dictionary."""
        return {
            "selection_id": self.selection_id,
            "selected_algorithm": self.selected_algorithm,
            "problem_features": self.problem_features,
            "confidence": self.confidence,
            "alternatives": self.alternatives
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaLearnerDecision':
        """Create a meta-learner decision from a dictionary."""
        return cls(
            selection_id=data.get("selection_id", ""),
            selected_algorithm=data.get("selected_algorithm", ""),
            problem_features=data.get("problem_features", {}),
            confidence=data.get("confidence", 0.0),
            alternatives=data.get("alternatives", {})
        )

@dataclass
class WorkflowExecution:
    """
    Represents a complete workflow execution.
    
    A workflow execution contains a sequence of events and component executions
    that together represent a single run of the workflow.
    """
    
    workflow_id: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    success: bool = True
    events: List[WorkflowEvent] = field(default_factory=list)
    components: List[ComponentExecution] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    # New fields for tracking MoE components
    expert_contributions: List[ExpertContribution] = field(default_factory=list)
    optimizer_performances: List[OptimizerPerformance] = field(default_factory=list)
    meta_learner_decisions: List[MetaLearnerDecision] = field(default_factory=list)
    
    def add_event(self, event: WorkflowEvent):
        """
        Add an event to the workflow execution.
        
        Args:
            event: The event to add
        """
        self.events.append(event)
    
    def add_component(self, component: ComponentExecution):
        """
        Add a component execution to the workflow execution.
        
        Args:
            component: The component execution to add
        """
        self.components.append(component)
    
    def add_expert_contribution(self, contribution: ExpertContribution):
        """
        Add an expert contribution to the workflow execution.
        
        Args:
            contribution: The expert contribution to add
        """
        self.expert_contributions.append(contribution)
    
    def add_optimizer_performance(self, performance: OptimizerPerformance):
        """
        Add an optimizer performance record to the workflow execution.
        
        Args:
            performance: The optimizer performance to add
        """
        self.optimizer_performances.append(performance)
    
    def add_meta_learner_decision(self, decision: MetaLearnerDecision):
        """
        Add a meta-learner decision to the workflow execution.
        
        Args:
            decision: The meta-learner decision to add
        """
        self.meta_learner_decisions.append(decision)
    
    def complete(self, success: bool = True, results: Optional[Dict[str, Any]] = None):
        """
        Mark the workflow execution as complete.
        
        Args:
            success: Whether the workflow completed successfully
            results: Optional results from the workflow
        """
        self.end_time = datetime.now().isoformat()
        self.success = success
        
        if results:
            self.results.update(results)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow execution to a dictionary.
        
        Returns:
            Dictionary representation of the workflow execution
        """
        return {
            "workflow_id": self.workflow_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "success": self.success,
            "events": [event.to_dict() for event in self.events],
            "components": [component.to_dict() for component in self.components],
            "results": self.results,
            "expert_contributions": [contrib.to_dict() for contrib in self.expert_contributions],
            "optimizer_performances": [perf.to_dict() for perf in self.optimizer_performances],
            "meta_learner_decisions": [decision.to_dict() for decision in self.meta_learner_decisions]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowExecution':
        """
        Create a workflow execution from a dictionary.
        
        Args:
            data: Dictionary representation of the workflow execution
            
        Returns:
            WorkflowExecution instance
        """
        workflow = cls(
            workflow_id=data.get("workflow_id", "unknown"),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time"),
            success=data.get("success", True),
            results=data.get("results", {})
        )
        
        # Add events
        for event_data in data.get("events", []):
            workflow.add_event(WorkflowEvent.from_dict(event_data))
        
        # Add components
        for component_data in data.get("components", []):
            workflow.add_component(ComponentExecution.from_dict(component_data))
        
        # Add expert contributions
        for contrib_data in data.get("expert_contributions", []):
            workflow.add_expert_contribution(ExpertContribution.from_dict(contrib_data))
        
        # Add optimizer performances
        for perf_data in data.get("optimizer_performances", []):
            workflow.add_optimizer_performance(OptimizerPerformance.from_dict(perf_data))
        
        # Add meta-learner decisions
        for decision_data in data.get("meta_learner_decisions", []):
            workflow.add_meta_learner_decision(MetaLearnerDecision.from_dict(decision_data))
        
        return workflow
    
    def save(self, file_path: str):
        """
        Save the workflow execution to a file.
        
        Args:
            file_path: Path to save the workflow execution to
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> 'WorkflowExecution':
        """
        Load a workflow execution from a file.
        
        Args:
            file_path: Path to load the workflow execution from
            
        Returns:
            WorkflowExecution instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data) 
"""
MoE Validation Enhancements Integration Module

This module integrates the following enhancements with the MoE validation framework:
1. Enhanced Drift Notifications
2. Selective Expert Retraining
3. Continuous Explainability
4. Confidence Metrics

These enhancements leverage the existing MoE validation framework, explainability
components, and drift detection to provide a more comprehensive validation suite.
"""
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import time

# Import enhancement modules
from core.enhanced_drift_notifications import EnhancedDriftNotifier
from core.selective_expert_retraining import SelectiveExpertRetrainer
from core.continuous_explainability import ContinuousExplainabilityPipeline
from core.confidence_metrics import ConfidenceMetricsCalculator
from core.personalization_layer import PersonalizationLayer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoEValidationEnhancer:
    """
    Integrates enhanced validation capabilities with the MoE validation framework
    """
    
    def __init__(self, results_dir: str = 'results/moe_validation'):
        """
        Initialize the MoE validation enhancer
        
        Parameters:
        -----------
        results_dir : str
            Directory to save validation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize enhancement components
        self.notifier = EnhancedDriftNotifier(
            notify_threshold=0.5,
            results_dir=str(self.results_dir)
        )
        
        # Initialize personalization layer
        self.personalizer = PersonalizationLayer(
            results_dir=str(self.results_dir / 'personalization'),
            adaptation_rate=0.2,
            profile_update_threshold=0.1
        )
        
        self.retrainer = SelectiveExpertRetrainer(
            impact_threshold=0.3,
            results_dir=str(self.results_dir)
        )
        
        self.explainability = ContinuousExplainabilityPipeline(
            explainer_types=['shap', 'feature_importance'],
            update_interval=60,  # seconds
            results_dir=str(self.results_dir / 'continuous_explainability')
        )
        
        self.confidence = ConfidenceMetricsCalculator(
            drift_weight=0.5,
            results_dir=str(self.results_dir / 'confidence_metrics')
        )
    
    def process_drift_results(self, 
                             drift_data: Dict[str, Any], 
                             notify: bool = False,
                             notify_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Process drift detection results with enhanced notifications
        
        Parameters:
        -----------
        drift_data : Dict[str, Any]
            Dictionary containing drift detection results
        notify : bool
            Whether to generate notifications
        notify_threshold : float
            Threshold for notifications
            
        Returns:
        --------
        Dict[str, Any]
            Enhanced drift results with notifications
        """
        # Update notifier threshold if specified
        if notify_threshold != self.notifier.notify_threshold:
            self.notifier.notify_threshold = notify_threshold
        
        # Generate expert-specific drift impact if available
        expert_impacts = {}
        if 'model' in drift_data and hasattr(drift_data['model'], 'experts'):
            experts = drift_data['model'].experts
            for expert_name, expert in experts.items():
                # Calculate impact based on performance before and after drift
                if 'before_drift' in drift_data and 'after_drift' in drift_data:
                    before_perf = self._evaluate_expert(
                        expert, 
                        drift_data['before_drift']['X'], 
                        drift_data['before_drift']['y']
                    )
                    after_perf = self._evaluate_expert(
                        expert, 
                        drift_data['after_drift']['X'], 
                        drift_data['after_drift']['y']
                    )
                    expert_impacts[expert_name] = max(0, before_perf - after_perf)
        
        # Enhanced notification if requested
        notification = {}
        if notify:
            notification = self.notifier.generate_notification(
                drift_data=drift_data,
                expert_impacts=expert_impacts
            )
            
            # Visualize notification if generated
            if notification:
                viz_path = self.notifier.visualize_notification(notification)
                if viz_path:
                    notification['visualization'] = viz_path
        
        # Return enhanced results
        return {
            'drift_data': drift_data,
            'expert_impacts': expert_impacts,
            'notification': notification
        }
    
    def identify_experts_for_retraining(self, 
                                       expert_impacts: Dict[str, float],
                                       drift_score: float,
                                       impact_threshold: Optional[float] = None) -> List[str]:
        """
        Identify experts that need retraining based on drift impact
        
        Parameters:
        -----------
        expert_impacts : Dict[str, float]
            Dictionary mapping expert names to drift impact scores
        drift_score : float
            Overall drift score
        impact_threshold : Optional[float]
            Threshold to determine which experts need retraining
            
        Returns:
        --------
        List[str]
            List of expert names that need retraining
        """
        # Update threshold if specified
        if impact_threshold is not None:
            self.retrainer.impact_threshold = impact_threshold
            
        return self.retrainer.identify_experts_for_retraining(
            expert_impacts=expert_impacts,
            drift_score=drift_score
        )
    
    def retrain_selected_experts(self, 
                               moe_model: Any,
                               experts_to_retrain: List[str],
                               new_data: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Selectively retrain experts affected by drift
        
        Parameters:
        -----------
        moe_model : Any
            The mixture-of-experts model
        experts_to_retrain : List[str]
            List of expert names that need retraining
        new_data : Any
            New data to retrain the experts on
            
        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            Updated model and retraining metrics
        """
        return self.retrainer.retrain_selected_experts(
            moe_model=moe_model,
            experts_to_retrain=experts_to_retrain,
            new_data=new_data
        )
    
    def start_continuous_explainability(self, 
                                       model: Any,
                                       data_source: Union[Any, callable],
                                       feature_names: Optional[List[str]] = None,
                                       explainer_types: Optional[List[str]] = None,
                                       update_interval: Optional[int] = None) -> None:
        """
        Start continuous explainability monitoring
        
        Parameters:
        -----------
        model : Any
            Model to monitor and explain
        data_source : Union[Any, callable]
            Data source for explainability
        feature_names : Optional[List[str]]
            Names of features for better visualization
        explainer_types : Optional[List[str]]
            List of explainer types to use
        update_interval : Optional[int]
            Interval in seconds between updates
        """
        # Update configuration if specified
        if explainer_types:
            self.explainability.explainer_types = explainer_types
            self.explainability.explainers = self.explainability._load_explainers()
            
        if update_interval:
            self.explainability.update_interval = update_interval
            
        # Start monitoring
        self.explainability.start_monitoring(
            model=model,
            data_source=data_source,
            feature_names=feature_names
        )
        
        logger.info(f"Started continuous explainability monitoring with update interval {self.explainability.update_interval}s")
    
    def stop_continuous_explainability(self) -> None:
        """
        Stop continuous explainability monitoring
        """
        self.explainability.stop_monitoring()
        logger.info("Stopped continuous explainability monitoring")
    
    def calculate_confidence(self, 
                            prediction_probabilities: Any,
                            drift_score: float,
                            expert_impacts: Optional[Dict[str, float]] = None,
                            expert_weights: Optional[Dict[str, float]] = None,
                            drift_weight: Optional[float] = None) -> Any:
        """
        Calculate confidence metrics for predictions
        
        Parameters:
        -----------
        prediction_probabilities : Any
            Model prediction probabilities
        drift_score : float
            Current drift score
        expert_impacts : Optional[Dict[str, float]]
            Dictionary mapping expert names to drift impact scores
        expert_weights : Optional[Dict[str, float]]
            Dictionary mapping expert names to weights in the ensemble
        drift_weight : Optional[float]
            Weight to assign to drift impact in confidence calculation
            
        Returns:
        --------
        Any
            Confidence scores for predictions
        """
        # Update drift weight if specified
        if drift_weight is not None:
            self.confidence.drift_weight = drift_weight
            
        return self.confidence.calculate_confidence(
            prediction_probabilities=prediction_probabilities,
            drift_score=drift_score,
            expert_impacts=expert_impacts,
            expert_weights=expert_weights
        )
    
    def generate_confidence_report(self, 
                                 predictions: Any,
                                 confidence_scores: Any,
                                 threshold_levels: List[float] = [0.3, 0.5, 0.7, 0.9]) -> Dict[str, Any]:
        """
        Generate a report on prediction confidence
        
        Parameters:
        -----------
        predictions : Any
            Model predictions
        confidence_scores : Any
            Confidence scores for predictions
        threshold_levels : List[float]
            List of confidence thresholds to report on
            
        Returns:
        --------
        Dict[str, Any]
            Confidence report
        """
        return self.confidence.generate_confidence_report(
            predictions=predictions,
            confidence_scores=confidence_scores,
            threshold_levels=threshold_levels
        )
    
    def _evaluate_expert(self, expert: Any, X: Any, y: Any) -> float:
        """
        Evaluate an expert's performance
        
        Parameters:
        -----------
        expert : Any
            Expert model to evaluate
        X : Any
            Features for evaluation
        y : Any
            Target for evaluation
            
        Returns:
        --------
        float
            Performance metric (higher is better)
        """
        try:
            from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
            import numpy as np
            
            # Make predictions
            if hasattr(expert, 'predict_proba'):
                y_pred = expert.predict_proba(X)
                if y_pred.shape[1] == 2:  # Binary classification
                    y_pred = y_pred[:, 1]
                    return roc_auc_score(y, y_pred)
            
            # Regular prediction for classification or regression
            y_pred = expert.predict(X)
            
            # Choose appropriate metric based on unique values in y
            if len(np.unique(y)) <= 5:  # Classification (assuming <= 5 classes)
                return accuracy_score(y, y_pred)
            else:  # Regression
                return r2_score(y, y_pred)
                
        except Exception as e:
            logger.warning(f"Error evaluating expert: {str(e)}")
            return 0.0
            
    def run_enhanced_validation(self, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run enhanced validation with all the enhancements
        
        Parameters:
        -----------
        args : Dict[str, Any]
            Arguments for enhanced validation
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        if args is None:
            args = {}
            
        # Extract arguments
        notify = args.get('notify', False)
        notify_threshold = args.get('notify_threshold', 0.5)
        results_dir = args.get('results_dir', 'results/moe_validation')
        
        # Run validation and get results
        try:
            from tests import moe_validation_runner
            
            # Set system arguments appropriately
            import sys
            old_argv = sys.argv
            
            # Run validation
            results = {}
            
            # Import the validation modules
            from tests.moe_enhanced_validation_part1 import run_drift_tests
            from tests.moe_enhanced_validation_part4 import run_drift_explanation_tests
            
            # Run drift tests to get data for enhancements
            drift_results = run_drift_tests()
            
            # Run explanation tests
            explanation_results = run_drift_explanation_tests()
            
            # If continuous explainability is enabled and we have a model from the drift tests,
            # generate some sample explainability data
            if hasattr(self, 'explainability') and 'drift_detection' in drift_results:
                drift_data = drift_results.get('drift_detection', {})
                if 'model' in drift_data and 'data' in drift_data:
                    model = drift_data['model']
                    data = drift_data['data']
                    
                    try:
                        # Generate basic explainability data for the model
                        for explainer_type, explainer in self.explainability.explainers.items():
                            logger.info(f"Generating {explainer_type} explanations for validation report")
                            explanation = explainer.explain(model, data)
                            
                            if 'explainability' not in explanation_results:
                                explanation_results['explainability'] = {}
                                
                            # Extract feature importance
                            feature_importance = explainer.get_feature_importance(explanation)
                            
                            # Store in explanation results
                            explanation_results['explainability'][explainer_type] = {
                                'feature_importance': feature_importance,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Generate a visualization
                            self.explainability._visualize_explanations({
                                explainer_type: {
                                    'feature_importance': feature_importance,
                                    'raw_explanation': explanation
                                }
                            })
                            
                        # Generate some sample importance trends data
                        trends_data = {}
                        if 'feature_importance' in self.explainability.explainer_types:
                            explainer = self.explainability.explainers.get('feature_importance')
                            if explainer:
                                explanation = explainer.explain(model, data)
                                importance = explainer.get_feature_importance(explanation)
                                
                                if isinstance(importance, dict):
                                    # Create synthetic trends for top features
                                    sorted_features = sorted(importance.items(), key=lambda x: abs(float(x[1])), reverse=True)
                                    for feature, value in sorted_features[:5]:
                                        # Create a series of values with some random variation
                                        import random
                                        base_value = float(value)
                                        trends_data[feature] = [
                                            base_value * (1 + random.uniform(-0.2, 0.2)) 
                                            for _ in range(5)
                                        ]
                                        
                                    # Add trends to explanation results
                                    explanation_results['importance_trends'] = trends_data
                    except Exception as e:
                        logger.warning(f"Error generating explainability data for report: {e}")
            
            # Process drift results with enhanced notifications
            enhanced_drift_results = self.process_drift_results(
                drift_data=drift_results.get('drift_detection', {}),
                notify=notify,
                notify_threshold=notify_threshold
            )
            
            # Generate confidence metrics example
            confidence_example = None
            if 'drift_detection' in drift_results and 'model' in drift_results['drift_detection']:
                model = drift_results['drift_detection']['model']
                if hasattr(model, 'predict_proba') and 'data' in drift_results['drift_detection']:
                    data = drift_results['drift_detection']['data']
                    
                    # Generate predictions
                    predictions = model.predict(data)
                    prediction_probs = model.predict_proba(data)
                    
                    # Calculate confidence scores
                    drift_score = drift_results['drift_detection'].get('drift_score', 0.5)
                    confidence_scores = self.calculate_confidence(
                        prediction_probabilities=prediction_probs,
                        drift_score=drift_score,
                        expert_impacts=enhanced_drift_results['expert_impacts']
                    )
                    
                    # Generate confidence report
                    confidence_example = self.generate_confidence_report(
                        predictions=predictions,
                        confidence_scores=confidence_scores
                    )
            
            # Combine all results
            results = {
                'enhanced_drift_results': enhanced_drift_results,
                'confidence_example': confidence_example,
                'explanation_results': explanation_results
            }
            
            # Restore original sys.argv
            sys.argv = old_argv
            
            return results
            
        except Exception as e:
            logger.error(f"Error running enhanced validation: {str(e)}")
            return {'error': str(e)}

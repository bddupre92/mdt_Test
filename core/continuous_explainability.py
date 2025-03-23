"""
Continuous Explainability Pipeline Module

This module provides a continuous explainability pipeline that monitors models
and provides ongoing interpretations as new data arrives.
"""
import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContinuousExplainabilityPipeline:
    """
    Provides continuous model explanations as new data arrives.
    """
    
    def __init__(self, 
                explainer_types: List[str] = ['shap'], 
                update_interval: int = 60,  # seconds
                max_history: int = 100,
                results_dir: str = 'results/continuous_explainability'):
        """
        Initialize the continuous explainability pipeline
        
        Parameters:
        -----------
        explainer_types : List[str]
            List of explainer types to use
        update_interval : int
            Interval in seconds between updates
        max_history : int
            Maximum number of historical explanations to keep
        results_dir : str
            Directory to save explanation results
        """
        self.explainer_types = explainer_types
        self.update_interval = update_interval
        self.max_history = max_history
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Explanation history
        self.explanation_history = []
        
        # Monitoring thread
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Load explainers
        self.explainers = self._load_explainers()
        
        # Initialize log file
        self.log_file = self.results_dir / "continuous_explanations.json"
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def _load_explainers(self) -> Dict[str, Any]:
        """
        Load explainers for continuous monitoring
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary mapping explainer types to explainer instances
        """
        explainers = {}
        
        try:
            from explainability.explainer_factory import ExplainerFactory
            
            factory = ExplainerFactory()
            
            for explainer_type in self.explainer_types:
                try:
                    explainers[explainer_type] = factory.create_explainer(explainer_type)
                    logger.info(f"Loaded {explainer_type} explainer for continuous monitoring")
                except Exception as e:
                    logger.warning(f"Could not load {explainer_type} explainer: {str(e)}")
        except Exception as e:
            logger.warning(f"Could not load explainer factory: {str(e)}")
            
        return explainers
    
    def start_monitoring(self, 
                       model: Any, 
                       data_source: Union[Callable, Any],
                       feature_names: Optional[List[str]] = None):
        """
        Start continuous monitoring and explanation of model with new data
        
        Parameters:
        -----------
        model : Any
            Model to monitor and explain
        data_source : Union[Callable, Any]
            Either a callable that returns new data batches or the initial data
        feature_names : Optional[List[str]]
            Names of features for better visualization
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already active. Stop first before starting a new one.")
            return
            
        # Reset stop event
        self._stop_event.clear()
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(model, data_source, feature_names),
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info(f"Started continuous explainability monitoring with update interval {self.update_interval}s")
    
    def stop_monitoring(self):
        """
        Stop continuous monitoring
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_event.set()
            self._monitoring_thread.join(timeout=5.0)
            logger.info("Stopped continuous explainability monitoring")
        else:
            logger.warning("No active monitoring to stop")
    
    def _monitor_loop(self, 
                    model: Any, 
                    data_source: Union[Callable, Any],
                    feature_names: Optional[List[str]] = None):
        """
        Main monitoring loop
        
        Parameters:
        -----------
        model : Any
            Model to monitor and explain
        data_source : Union[Callable, Any]
            Either a callable that returns new data batches or the initial data
        feature_names : Optional[List[str]]
            Names of features for better visualization
        """
        # Initialize with provided data if not callable
        current_data = data_source() if callable(data_source) else data_source
        
        while not self._stop_event.is_set():
            try:
                # Generate explanations for current data
                explanations = self._generate_explanations(model, current_data, feature_names)
                
                # Add explanations to history
                if explanations:
                    self.explanation_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'explanations': explanations
                    })
                    
                    # Prune history if needed
                    if len(self.explanation_history) > self.max_history:
                        self.explanation_history = self.explanation_history[-self.max_history:]
                    
                    # Log explanations
                    self._log_explanation(explanations)
                    
                    # Generate visualizations
                    self._visualize_explanations(explanations)
                
                # Get new data if data_source is callable
                if callable(data_source):
                    new_data = data_source()
                    if new_data is not None:
                        current_data = new_data
                
                # Wait for next update
                time_to_wait = self.update_interval
                while time_to_wait > 0 and not self._stop_event.is_set():
                    time.sleep(min(1, time_to_wait))
                    time_to_wait -= 1
                    
            except Exception as e:
                logger.error(f"Error in continuous explanation loop: {str(e)}")
                time.sleep(self.update_interval)
    
    def _generate_explanations(self, 
                             model: Any, 
                             data: Any,
                             feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate explanations for current data using all available explainers
        
        Parameters:
        -----------
        model : Any
            Model to explain
        data : Any
            Data to explain
        feature_names : Optional[List[str]]
            Names of features for better visualization
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of explanations from different explainers
        """
        explanations = {}
        
        for explainer_type, explainer in self.explainers.items():
            try:
                # Generate explanation
                explanation = explainer.explain(model, data)
                
                # Get feature importance
                feature_importance = explainer.get_feature_importance(explanation)
                
                # Convert to dictionary with feature names if available
                if feature_names and isinstance(feature_importance, (list, np.ndarray)):
                    if len(feature_names) == len(feature_importance):
                        feature_importance = {
                            feature_names[i]: float(importance) 
                            for i, importance in enumerate(feature_importance)
                        }
                
                # Store explanation
                explanations[explainer_type] = {
                    'feature_importance': feature_importance,
                    'raw_explanation': explanation if not isinstance(explanation, np.ndarray) else explanation.tolist()
                }
                
                logger.debug(f"Generated {explainer_type} explanation")
                
            except Exception as e:
                logger.warning(f"Could not generate {explainer_type} explanation: {str(e)}")
        
        return explanations
    
    def _log_explanation(self, explanations: Dict[str, Any]) -> None:
        """
        Log explanations to the log file
        
        Parameters:
        -----------
        explanations : Dict[str, Any]
            Explanations to log
        """
        try:
            # Read existing logs
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
                
            # Add new log
            logs.append({
                'timestamp': datetime.now().isoformat(),
                'explanations': {
                    explainer_type: {
                        'feature_importance': explanation['feature_importance']
                    }
                    for explainer_type, explanation in explanations.items()
                }
            })
            
            # Keep only the latest entries
            logs = logs[-self.max_history:]
            
            # Write updated logs
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not log explanation: {str(e)}")
    
    def _visualize_explanations(self, explanations: Dict[str, Any]) -> None:
        """
        Visualize current explanations
        
        Parameters:
        -----------
        explanations : Dict[str, Any]
            Explanations to visualize
        """
        try:
            for explainer_type, explanation in explanations.items():
                # Skip if no feature importance
                if 'feature_importance' not in explanation:
                    continue
                    
                feature_importance = explanation['feature_importance']
                
                # Skip if feature importance is empty
                if not feature_importance:
                    continue
                
                # Create visualization
                plt.figure(figsize=(10, 6))
                
                if isinstance(feature_importance, dict):
                    # Sort features by absolute importance
                    features = [(k, v) for k, v in feature_importance.items()]
                    features.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Plot top 10 features
                    top_features = features[:10]
                    names = [f[0] for f in top_features]
                    values = [f[1] for f in top_features]
                    
                    # Create horizontal bar chart
                    plt.barh(names, values)
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top Features - {explainer_type} ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
                    plt.grid(axis='x', linestyle='--', alpha=0.6)
                    
                elif isinstance(feature_importance, (list, np.ndarray)):
                    # Plot feature importance as bar chart
                    plt.bar(range(len(feature_importance)), feature_importance)
                    plt.xlabel('Feature Index')
                    plt.ylabel('Feature Importance')
                    plt.title(f'Feature Importance - {explainer_type} ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
                    plt.grid(axis='y', linestyle='--', alpha=0.6)
                
                # Save visualization
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                viz_path = self.results_dir / f"explanation_{explainer_type}_{timestamp}.png"
                plt.savefig(viz_path)
                plt.close()
                
                logger.info(f"Saved {explainer_type} explanation visualization to {viz_path}")
                
        except Exception as e:
            logger.error(f"Could not visualize explanations: {str(e)}")
    
    def get_explanation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of explanations
        
        Returns:
        --------
        List[Dict[str, Any]]
            History of explanations
        """
        return self.explanation_history
    
    def get_feature_importance_trends(self) -> Dict[str, List[float]]:
        """
        Get trends in feature importance over time
        
        Returns:
        --------
        Dict[str, List[float]]
            Dictionary mapping feature names to lists of importance values over time
        """
        # Check if we have history
        if not self.explanation_history:
            return {}
            
        # Initialize result
        trends = {}
        
        # Get the first explainer type
        if not self.explanation_history[0]['explanations']:
            return {}
            
        explainer_type = list(self.explanation_history[0]['explanations'].keys())[0]
        
        # Extract feature names from the first entry
        first_entry = self.explanation_history[0]['explanations'][explainer_type]
        if 'feature_importance' not in first_entry:
            return {}
            
        feature_importance = first_entry['feature_importance']
        
        if isinstance(feature_importance, dict):
            # Initialize trends for each feature
            for feature in feature_importance:
                trends[feature] = []
                
            # Collect trends
            for entry in self.explanation_history:
                if explainer_type not in entry['explanations']:
                    continue
                    
                importance = entry['explanations'][explainer_type].get('feature_importance', {})
                
                for feature in trends:
                    trends[feature].append(importance.get(feature, 0))
        
        return trends
    
    def visualize_importance_trends(self) -> str:
        """
        Visualize trends in feature importance over time
        
        Returns:
        --------
        str
            Path to the saved visualization
        """
        trends = self.get_feature_importance_trends()
        
        if not trends:
            logger.warning("No feature importance trends to visualize")
            return ""
            
        try:
            # Get top features by average absolute importance
            top_features = sorted(
                trends.keys(),
                key=lambda f: np.mean([abs(v) for v in trends[f]]),
                reverse=True
            )[:5]  # Top 5 features
            
            # Create visualization
            plt.figure(figsize=(12, 7))
            
            for feature in top_features:
                plt.plot(trends[feature], label=feature)
                
            plt.xlabel('Time (samples)')
            plt.ylabel('Feature Importance')
            plt.title('Feature Importance Trends Over Time')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.results_dir / f"importance_trends_{timestamp}.png"
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Saved feature importance trends visualization to {viz_path}")
            return str(viz_path)
            
        except Exception as e:
            logger.error(f"Could not visualize importance trends: {str(e)}")
            return ""

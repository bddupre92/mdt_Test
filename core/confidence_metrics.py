"""
Confidence Metrics Module

This module provides confidence metrics for predictions that account for
both model uncertainty and drift severity.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfidenceMetricsCalculator:
    """
    Calculates confidence metrics for predictions that account for
    both model uncertainty and drift severity.
    """
    
    def __init__(self, 
                 drift_weight: float = 0.5,
                 results_dir: str = 'results/confidence_metrics'):
        """
        Initialize the confidence metrics calculator
        
        Parameters:
        -----------
        drift_weight : float
            Weight to assign to drift impact in confidence calculation (0-1)
        results_dir : str
            Directory to save confidence metrics results
        """
        self.drift_weight = drift_weight
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize confidence log
        self.confidence_log = self.results_dir / "confidence_log.json"
        if not self.confidence_log.exists():
            with open(self.confidence_log, 'w') as f:
                json.dump([], f)
    
    def calculate_confidence(self, 
                             prediction_probabilities: np.ndarray,
                             drift_score: float,
                             expert_impacts: Optional[Dict[str, float]] = None,
                             expert_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Calculate confidence metrics for predictions
        
        Parameters:
        -----------
        prediction_probabilities : np.ndarray
            Model prediction probabilities
        drift_score : float
            Current drift score
        expert_impacts : Optional[Dict[str, float]]
            Dictionary mapping expert names to drift impact scores
        expert_weights : Optional[Dict[str, float]]
            Dictionary mapping expert names to weights in the ensemble
            
        Returns:
        --------
        np.ndarray
            Confidence scores for each prediction
        """
        # Normalize drift score to [0, 1] where 0 means high drift, 1 means no drift
        drift_confidence = 1.0 - min(1.0, max(0.0, drift_score))
        
        # Calculate expert-weighted drift confidence if both impacts and weights are provided
        if expert_impacts and expert_weights:
            drift_confidence = self._calculate_expert_weighted_drift_confidence(
                expert_impacts, expert_weights
            )
        
        # Calculate model confidence from prediction probabilities
        model_confidence = self._calculate_model_confidence(prediction_probabilities)
        
        # Combine model confidence and drift confidence
        # drift_weight determines how much drift affects overall confidence
        combined_confidence = (
            (1 - self.drift_weight) * model_confidence + 
            self.drift_weight * drift_confidence
        )
        
        # Log confidence metrics
        self._log_confidence_metrics({
            'timestamp': datetime.now().isoformat(),
            'model_confidence_mean': float(np.mean(model_confidence)),
            'drift_confidence': float(drift_confidence),
            'combined_confidence_mean': float(np.mean(combined_confidence))
        })
        
        return combined_confidence
    
    def _calculate_model_confidence(self, 
                                   prediction_probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate model confidence from prediction probabilities
        
        Parameters:
        -----------
        prediction_probabilities : np.ndarray
            Model prediction probabilities
            
        Returns:
        --------
        np.ndarray
            Model confidence scores
        """
        # Handle different shapes of prediction probabilities
        if prediction_probabilities.ndim == 1:
            # Binary classification with single probability
            return np.maximum(prediction_probabilities, 1 - prediction_probabilities)
        
        elif prediction_probabilities.ndim == 2:
            # Multi-class classification
            # Confidence is the max probability
            return np.max(prediction_probabilities, axis=1)
        
        else:
            # Unexpected shape
            logger.warning(f"Unexpected shape for prediction probabilities: {prediction_probabilities.shape}")
            return np.ones(prediction_probabilities.shape[0])
    
    def _calculate_expert_weighted_drift_confidence(self,
                                                  expert_impacts: Dict[str, float],
                                                  expert_weights: Dict[str, float]) -> float:
        """
        Calculate drift confidence weighted by expert contributions
        
        Parameters:
        -----------
        expert_impacts : Dict[str, float]
            Dictionary mapping expert names to drift impact scores
        expert_weights : Dict[str, float]
            Dictionary mapping expert names to weights in the ensemble
            
        Returns:
        --------
        float
            Expert-weighted drift confidence
        """
        weighted_impact = 0.0
        total_weight = 0.0
        
        for expert, weight in expert_weights.items():
            if expert in expert_impacts:
                impact = expert_impacts[expert]
                weighted_impact += weight * impact
                total_weight += weight
        
        if total_weight > 0:
            # Normalize to [0, 1] where 0 means high drift, 1 means no drift
            return 1.0 - min(1.0, max(0.0, weighted_impact / total_weight))
        else:
            return 1.0  # Default to high confidence if no weights
    
    def _log_confidence_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log confidence metrics
        
        Parameters:
        -----------
        metrics : Dict[str, Any]
            Confidence metrics to log
        """
        try:
            with open(self.confidence_log, 'r') as f:
                logs = json.load(f)
                
            logs.append(metrics)
            
            with open(self.confidence_log, 'w') as f:
                json.dump(logs, f, indent=2)
                
            logger.debug(f"Logged confidence metrics to {self.confidence_log}")
        except Exception as e:
            logger.error(f"Could not log confidence metrics: {str(e)}")
    
    def get_confidence_history(self) -> List[Dict[str, Any]]:
        """
        Get confidence metrics history
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of confidence metrics logs
        """
        try:
            with open(self.confidence_log, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read confidence history: {str(e)}")
            return []
    
    def visualize_confidence_trends(self) -> str:
        """
        Visualize confidence metrics trends over time
        
        Returns:
        --------
        str
            Path to the saved visualization
        """
        history = self.get_confidence_history()
        
        if not history:
            logger.warning("No confidence history to visualize")
            return ""
            
        try:
            # Extract metrics
            timestamps = [entry['timestamp'] for entry in history]
            model_confidence = [entry['model_confidence_mean'] for entry in history]
            drift_confidence = [entry['drift_confidence'] for entry in history]
            combined_confidence = [entry['combined_confidence_mean'] for entry in history]
            
            # Convert timestamps to numeric indices for plotting
            indices = list(range(len(timestamps)))
            
            # Create visualization
            plt.figure(figsize=(12, 7))
            
            plt.plot(indices, model_confidence, 'b-', label='Model Confidence')
            plt.plot(indices, drift_confidence, 'r-', label='Drift-based Confidence')
            plt.plot(indices, combined_confidence, 'g-', label='Combined Confidence')
            
            plt.xlabel('Time (samples)')
            plt.ylabel('Confidence')
            plt.title('Confidence Metrics Trends Over Time')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Label with timestamps at regular intervals
            if len(timestamps) > 5:
                step = len(timestamps) // 5
                plt.xticks(
                    indices[::step],
                    [datetime.fromisoformat(t).strftime('%m-%d %H:%M') for t in timestamps[::step]],
                    rotation=45
                )
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.results_dir / f"confidence_trends_{timestamp}.png"
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Saved confidence trends visualization to {viz_path}")
            return str(viz_path)
            
        except Exception as e:
            logger.error(f"Could not visualize confidence trends: {str(e)}")
            return ""

    def apply_confidence_thresholds(self, 
                                   predictions: np.ndarray,
                                   confidence_scores: np.ndarray,
                                   threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply confidence thresholds to predictions
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        confidence_scores : np.ndarray
            Confidence scores for predictions
        threshold : float
            Confidence threshold
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Filtered predictions and their confidence scores
        """
        # Get indices where confidence exceeds threshold
        high_confidence_indices = confidence_scores >= threshold
        
        # Filter predictions and confidence scores
        filtered_predictions = predictions[high_confidence_indices]
        filtered_confidence = confidence_scores[high_confidence_indices]
        
        return filtered_predictions, filtered_confidence
    
    def generate_confidence_report(self, 
                                  predictions: np.ndarray,
                                  confidence_scores: np.ndarray,
                                  threshold_levels: List[float] = [0.3, 0.5, 0.7, 0.9]) -> Dict[str, Any]:
        """
        Generate a report on prediction confidence
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model predictions
        confidence_scores : np.ndarray
            Confidence scores for predictions
        threshold_levels : List[float]
            List of confidence thresholds to report on
            
        Returns:
        --------
        Dict[str, Any]
            Confidence report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'average_confidence': float(np.mean(confidence_scores)),
            'median_confidence': float(np.median(confidence_scores)),
            'std_confidence': float(np.std(confidence_scores)),
            'thresholds': {}
        }
        
        for threshold in threshold_levels:
            high_confidence_count = np.sum(confidence_scores >= threshold)
            
            report['thresholds'][str(threshold)] = {
                'count': int(high_confidence_count),
                'percentage': float(high_confidence_count / len(predictions) * 100)
            }
        
        # Visualize confidence distribution
        viz_path = self._visualize_confidence_distribution(confidence_scores)
        if viz_path:
            report['visualization'] = viz_path
        
        # Save report
        report_path = self.results_dir / f"confidence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved confidence report to {report_path}")
        except Exception as e:
            logger.error(f"Could not save confidence report: {str(e)}")
        
        return report
    
    def _visualize_confidence_distribution(self, confidence_scores: np.ndarray) -> str:
        """
        Visualize the distribution of confidence scores
        
        Parameters:
        -----------
        confidence_scores : np.ndarray
            Confidence scores to visualize
            
        Returns:
        --------
        str
            Path to the saved visualization
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Create histogram
            plt.hist(confidence_scores, bins=20, alpha=0.7, color='blue')
            
            # Add kernel density estimate
            if len(confidence_scores) > 10:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(confidence_scores)
                x = np.linspace(0, 1, 100)
                plt.plot(x, kde(x) * len(confidence_scores) / 20, 'r-', linewidth=2)
            
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.title('Distribution of Confidence Scores')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.results_dir / f"confidence_distribution_{timestamp}.png"
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Saved confidence distribution visualization to {viz_path}")
            return str(viz_path)
            
        except Exception as e:
            logger.error(f"Could not visualize confidence distribution: {str(e)}")
            return ""

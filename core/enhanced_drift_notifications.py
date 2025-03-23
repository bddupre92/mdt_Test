"""
Enhanced Drift Notifications Module

This module provides explanatory drift notifications that leverage the
explainability framework to make notifications more actionable for clinical staff.
"""
import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDriftNotifier:
    """
    Provides enhanced drift notifications with explanatory insights
    from the explainability framework.
    """
    
    def __init__(self, notify_threshold: float = 0.5, results_dir: str = 'results/moe_validation'):
        """
        Initialize the enhanced drift notifier
        
        Parameters:
        -----------
        notify_threshold : float
            Threshold for drift notifications (0-1, higher means only notify on severe drift)
        results_dir : str
            Directory to save notification results
        """
        self.notify_threshold = notify_threshold
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.notification_log = self.results_dir / "drift_notifications.json"
        
        # Initialize notification log if it doesn't exist
        if not self.notification_log.exists():
            with open(self.notification_log, 'w') as f:
                json.dump([], f)
    
    def _get_feature_importance(self, 
                               drift_data: Dict[str, Any], 
                               explainer_name: str = 'shap') -> Dict[str, float]:
        """
        Extract feature importance from drift data using the specified explainer
        
        Parameters:
        -----------
        drift_data : Dict[str, Any]
            Dictionary containing drift detection results
        explainer_name : str
            Name of the explainer to use (default: 'shap')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping feature names to importance scores
        """
        # If feature importance is already in the drift data, use it
        if 'feature_importance' in drift_data:
            return drift_data['feature_importance']
            
        # Otherwise, try to compute it using the explainability framework
        try:
            from explainability.explainer_factory import ExplainerFactory
            
            # Create appropriate explainer
            factory = ExplainerFactory()
            explainer = factory.create_explainer(explainer_name)
            
            # Get feature importance using the explainer
            if 'model' in drift_data and 'data' in drift_data:
                model = drift_data['model']
                data = drift_data['data']
                explanation = explainer.explain(model, data)
                return explainer.get_feature_importance(explanation)
            
            return {}
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {str(e)}")
            return {}
    
    def generate_notification(self, 
                             drift_data: Dict[str, Any],
                             expert_impacts: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate an enhanced drift notification with explanatory insights
        
        Parameters:
        -----------
        drift_data : Dict[str, Any]
            Dictionary containing drift detection results
        expert_impacts : Optional[Dict[str, float]]
            Dictionary mapping expert names to drift impact scores
            
        Returns:
        --------
        Dict[str, Any]
            Enhanced notification with explanatory insights
        """
        # Extract drift score and check threshold
        drift_score = drift_data.get('drift_score', 0.0)
        if drift_score < self.notify_threshold:
            return {}
            
        # Get feature importance to explain the drift
        feature_importance = self._get_feature_importance(drift_data)
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Get top contributing features (up to 5)
        top_features = sorted_features[:5]
        
        # Generate notification
        notification = {
            'timestamp': datetime.now().isoformat(),
            'drift_score': drift_score,
            'threshold': self.notify_threshold,
            'message': f"Significant drift detected (score: {drift_score:.4f})",
            'top_contributing_features': {f: i for f, i in top_features},
            'action_recommendations': []
        }
        
        # Add expert-specific impact if available
        if expert_impacts:
            # Sort experts by impact
            sorted_experts = sorted(
                expert_impacts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            notification['expert_impacts'] = {e: i for e, i in sorted_experts}
            
            # Generate expert-specific recommendations
            for expert, impact in sorted_experts:
                if impact > self.notify_threshold:
                    notification['action_recommendations'].append(
                        f"Consider retraining expert '{expert}' (impact: {impact:.4f})"
                    )
        
        # Generate feature-specific recommendations
        for feature, importance in top_features:
            abs_importance = abs(importance)
            if abs_importance > self.notify_threshold / 2:
                direction = "increase" if importance > 0 else "decrease"
                notification['action_recommendations'].append(
                    f"Monitor feature '{feature}' which shows significant {direction} in importance"
                )
        
        # Log notification
        self._log_notification(notification)
        
        return notification
    
    def _log_notification(self, notification: Dict[str, Any]) -> None:
        """
        Log the notification to the notification log file
        
        Parameters:
        -----------
        notification : Dict[str, Any]
            Notification to log
        """
        try:
            # Read existing notifications
            with open(self.notification_log, 'r') as f:
                notifications = json.load(f)
                
            # Add new notification
            notifications.append(notification)
            
            # Write updated notifications
            with open(self.notification_log, 'w') as f:
                json.dump(notifications, f, indent=2)
                
            logger.info(f"Drift notification logged to {self.notification_log}")
        except Exception as e:
            logger.error(f"Could not log notification: {str(e)}")
    
    def visualize_notification(self, notification: Dict[str, Any]) -> str:
        """
        Create a visualization of the notification
        
        Parameters:
        -----------
        notification : Dict[str, Any]
            Notification to visualize
            
        Returns:
        --------
        str
            Path to the saved visualization
        """
        if not notification:
            logger.warning("No notification to visualize")
            return ""
            
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Plot drift score
            ax1.bar(['Drift Score', 'Threshold'], 
                  [notification['drift_score'], notification['threshold']],
                  color=['red', 'blue'])
            ax1.set_title('Drift Detection')
            ax1.set_ylabel('Score')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Plot top contributing features
            if 'top_contributing_features' in notification and notification['top_contributing_features']:
                features = list(notification['top_contributing_features'].keys())
                importances = list(notification['top_contributing_features'].values())
                
                # Sort by absolute importance
                sorted_indices = np.argsort([abs(i) for i in importances])[::-1]
                features = [features[i] for i in sorted_indices]
                importances = [importances[i] for i in sorted_indices]
                
                colors = ['green' if i > 0 else 'red' for i in importances]
                ax2.barh(features, importances, color=colors)
                ax2.set_title('Feature Contribution to Drift')
                ax2.set_xlabel('Importance')
                ax2.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = self.results_dir / f"drift_notification_{timestamp}.png"
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Drift notification visualization saved to {viz_path}")
            return str(viz_path)
        except Exception as e:
            logger.error(f"Could not visualize notification: {str(e)}")
            return ""

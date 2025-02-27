"""
framework_evaluator.py
-------------------
Comprehensive framework evaluation tools
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FrameworkEvaluator:
    def __init__(self):
        self.metrics_history = []
        self.feature_importance_history = []
        self.prediction_history = []
        self.drift_history = []
        
    def evaluate_prediction_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: Optional[np.ndarray] = None) -> Dict:
        """Evaluate prediction performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            
        self.metrics_history.append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        return metrics
    
    def track_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray):
        """Track feature importance over time"""
        self.feature_importance_history.append({
            'timestamp': datetime.now(),
            'importance': dict(zip(feature_names, importance_scores))
        })
    
    def track_prediction(self, features: Dict, prediction: float, actual: Optional[float] = None):
        """Track individual predictions"""
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        })
    
    def track_drift_event(self, drift_info: Dict):
        """Track drift detection events"""
        self.drift_history.append({
            'timestamp': datetime.now(),
            **drift_info
        })
    
    def plot_framework_performance(self, save_path: Optional[str] = None):
        """Generate comprehensive performance visualization"""
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: Metrics Over Time
        plt.subplot(3, 1, 1)
        metrics_df = pd.DataFrame(self.metrics_history)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in metrics_df:
                plt.plot(metrics_df['timestamp'], metrics_df[metric], label=metric)
        plt.title('Prediction Performance Metrics Over Time')
        plt.ylabel('Score')
        plt.legend()
        
        # Plot 2: Feature Importance Heatmap
        plt.subplot(3, 1, 2)
        if self.feature_importance_history:
            importance_df = pd.DataFrame([h['importance'] for h in self.feature_importance_history])
            sns.heatmap(importance_df.T, cmap='YlOrRd', cbar_kws={'label': 'Importance'})
            plt.title('Feature Importance Evolution')
            plt.ylabel('Features')
        
        # Plot 3: Drift Events and Model Updates
        plt.subplot(3, 1, 3)
        if self.drift_history:
            drift_df = pd.DataFrame(self.drift_history)
            plt.scatter(drift_df['timestamp'], drift_df['severity'], 
                       c=drift_df['mean_shift'], cmap='viridis', 
                       label='Drift Events')
            plt.colorbar(label='Mean Shift')
        plt.title('Drift Events and Severity')
        plt.ylabel('Severity')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {}
            
        latest_metrics = self.metrics_history[-1]
        metrics_df = pd.DataFrame(self.metrics_history)
        
        report = {
            'current_performance': {
                metric: value for metric, value in latest_metrics.items()
                if metric != 'timestamp'
            },
            'metric_trends': {
                metric: {
                    'mean': metrics_df[metric].mean(),
                    'std': metrics_df[metric].std(),
                    'trend': 'improving' if len(metrics_df) > 1 and 
                            metrics_df[metric].iloc[-1] > metrics_df[metric].iloc[-2]
                            else 'declining'
                }
                for metric in metrics_df.columns if metric != 'timestamp'
            },
            'drift_summary': {
                'total_drifts': len(self.drift_history),
                'avg_severity': np.mean([d['severity'] for d in self.drift_history]) 
                               if self.drift_history else 0,
                'most_drifted_features': self._get_most_drifted_features()
            }
        }
        
        return report
    
    def _get_most_drifted_features(self) -> List[Tuple[str, int]]:
        """Get features that drifted most frequently"""
        if not self.drift_history:
            return []
            
        feature_drift_counts = {}
        for drift in self.drift_history:
            for feature in drift.get('drifting_features', []):
                feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1
                
        return sorted(feature_drift_counts.items(), key=lambda x: x[1], reverse=True)

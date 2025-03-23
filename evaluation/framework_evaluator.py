"""Framework evaluation and visualization module."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

class FrameworkEvaluator:
    def __init__(self):
        self.metrics_history = []
        self.feature_importance_history = []
        self.prediction_history = []
        self.drift_history = []
        
    def evaluate_prediction_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: Optional[np.ndarray] = None) -> Dict:
        """Evaluate prediction performance metrics"""
        # Convert predictions to binary if needed
        if y_prob is not None:
            y_pred = (y_prob > 0.5).astype(int)
            
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
    
    def track_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray) -> None:
        """Track feature importance over time."""
        self.feature_importance_history.append({
            'timestamp': datetime.now(),
            'features': feature_names,
            'importance': dict(zip(feature_names, importance_scores))
        })
    
    def track_drift_event(self, drift_info: Dict) -> None:
        """Track drift detection events."""
        self.drift_history.append({
            'timestamp': datetime.now(),
            **drift_info
        })
    
    def plot_framework_performance(self, save_path: Optional[str] = None) -> None:
        """Generate comprehensive performance visualization."""
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
        plt.grid(True)
        
        # Plot 2: Feature Importance Evolution
        plt.subplot(3, 1, 2)
        if self.feature_importance_history:
            importance_df = pd.DataFrame([
                h['importance'] for h in self.feature_importance_history
            ])
            sns.heatmap(importance_df.T, cmap='YlOrRd', 
                       cbar_kws={'label': 'Importance'})
            plt.title('Feature Importance Evolution')
            plt.ylabel('Features')
        
        # Plot 3: Drift Events Timeline
        plt.subplot(3, 1, 3)
        if self.drift_history:
            drift_df = pd.DataFrame(self.drift_history)
            plt.scatter(drift_df['timestamp'], drift_df['severity'],
                       c=drift_df['severity'], cmap='RdYlBu_r',
                       s=100, alpha=0.6)
            plt.title('Drift Events Timeline')
            plt.ylabel('Drift Severity')
            plt.colorbar(label='Severity')
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_drift_analysis(self, save_path: Optional[str] = None) -> None:
        """Generate detailed drift analysis visualization."""
        if not self.drift_history:
            return
            
        drift_df = pd.DataFrame(self.drift_history)
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Drift Severity Distribution
        plt.subplot(2, 2, 1)
        sns.histplot(drift_df['severity'], bins=20)
        plt.title('Drift Severity Distribution')
        plt.xlabel('Severity')
        
        # Plot 2: Drift Frequency Over Time
        plt.subplot(2, 2, 2)
        drift_df['hour'] = drift_df['timestamp'].dt.hour
        hourly_drifts = drift_df.groupby('hour').size()
        plt.bar(hourly_drifts.index, hourly_drifts.values)
        plt.title('Drift Frequency by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Drifts')
        
        # Plot 3: Feature Drift Impact
        plt.subplot(2, 2, 3)
        if 'drifted_features' in drift_df.columns:
            feature_counts = pd.Series(
                [f for features in drift_df['drifted_features'] 
                 for f in features]
            ).value_counts()
            feature_counts.plot(kind='bar')
            plt.title('Most Frequently Drifted Features')
            plt.xticks(rotation=45)
        
        # Plot 4: Severity vs Performance Impact
        plt.subplot(2, 2, 4)
        metrics_df = pd.DataFrame(self.metrics_history)
        if not metrics_df.empty and not drift_df.empty:
            merged_df = pd.merge_asof(
                drift_df.sort_values('timestamp'),
                metrics_df.sort_values('timestamp'),
                on='timestamp'
            )
            plt.scatter(merged_df['severity'], merged_df['accuracy'])
            plt.xlabel('Drift Severity')
            plt.ylabel('Accuracy')
            plt.title('Drift Impact on Performance')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
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
                'recent_drifts': len([
                    d for d in self.drift_history 
                    if (datetime.now() - d['timestamp']).total_seconds() < 3600
                ]),
                'avg_severity': np.mean([d['severity'] for d in self.drift_history])
                if self.drift_history else 0
            }
        }
        
        if self.feature_importance_history:
            latest_importance = self.feature_importance_history[-1]['importance']
            report['feature_importance'] = {
                'current': latest_importance,
                'trending_features': self._get_trending_features()
            }
            
        return report
    
    def _get_trending_features(self) -> List[str]:
        """Identify features with significant importance changes."""
        if len(self.feature_importance_history) < 2:
            return []
            
        current = pd.Series(self.feature_importance_history[-1]['importance'])
        previous = pd.Series(self.feature_importance_history[-2]['importance'])
        changes = (current - previous) / previous
        
        return list(changes[abs(changes) > 0.1].index)
        
    def _get_most_drifted_features(self) -> List[Tuple[str, int]]:
        """Get features that drifted most frequently."""
        if not self.drift_history or 'drifted_features' not in self.drift_history[0]:
            return []
            
        feature_counts = {}
        for event in self.drift_history:
            for feature in event['drifted_features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
                
        return sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

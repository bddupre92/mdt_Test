"""
drift_analysis.py
---------------
Visualization tools for drift detection analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os

class DriftAnalyzer:
    """Analyzer for drift detection results."""
    
    def __init__(self):
        self.drift_history = []
        self.severity_history = []
        self.error_history = []
        self.prediction_history = []
        
    def add_drift_event(self, 
                       timestamp: datetime,
                       severity: float,
                       error: float,
                       predictions: np.ndarray,
                       true_values: Optional[np.ndarray] = None) -> None:
        """Add a drift detection event."""
        self.drift_history.append({
            'timestamp': timestamp,
            'severity': severity,
            'error': error
        })
        self.severity_history.append(severity)
        self.error_history.append(error)
        self.prediction_history.append({
            'predictions': predictions,
            'true_values': true_values
        })
    
    def plot_drift_errors(self, save_path: Optional[str] = None) -> None:
        """Plot drift detection errors over time."""
        if not self.drift_history:
            return
            
        plt.figure(figsize=(12, 6))
        df = pd.DataFrame(self.drift_history)
        
        plt.plot(df['timestamp'], df['error'], 'r-', label='Detection Error')
        plt.fill_between(df['timestamp'], 
                        df['error'] - np.std(self.error_history),
                        df['error'] + np.std(self.error_history),
                        alpha=0.2, color='r')
        
        plt.title('Drift Detection Errors Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_drift_severity(self, save_path: Optional[str] = None) -> None:
        """Plot drift severity over time."""
        if not self.drift_history:
            return
            
        plt.figure(figsize=(12, 6))
        df = pd.DataFrame(self.drift_history)
        
        # Plot severity
        plt.plot(df['timestamp'], df['severity'], 'b-', label='Drift Severity')
        
        # Add threshold line if available
        if hasattr(self, 'drift_threshold'):
            plt.axhline(y=self.drift_threshold, color='r', linestyle='--',
                       label='Drift Threshold')
        
        # Add drift points
        drift_points = df[df['severity'] > (self.drift_threshold if hasattr(self, 'drift_threshold') else 0.5)]
        plt.scatter(drift_points['timestamp'], drift_points['severity'],
                   color='r', s=100, alpha=0.6, label='Drift Detected')
        
        plt.title('Drift Severity Over Time')
        plt.xlabel('Time')
        plt.ylabel('Severity')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_drift_true_vs_pred(self, save_path: Optional[str] = None) -> None:
        """Plot true vs predicted values with drift points."""
        true_values_available = any(
            ph['true_values'] is not None for ph in self.prediction_history
        )
        
        if not true_values_available:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Collect all predictions and true values
        all_preds = []
        all_true = []
        for ph in self.prediction_history:
            if ph['true_values'] is not None:
                all_preds.extend(ph['predictions'])
                all_true.extend(ph['true_values'])
        
        # Create scatter plot
        plt.scatter(all_true, all_preds, alpha=0.5, label='Predictions')
        
        # Add perfect prediction line
        min_val = min(min(all_true), min(all_preds))
        max_val = max(max(all_true), max(all_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                 label='Perfect Prediction')
        
        plt.title('True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_drift_analysis(self, save_path: Optional[str] = None) -> None:
        """Generate comprehensive drift analysis visualization."""
        if not self.drift_history:
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Severity Distribution
        plt.subplot(2, 2, 1)
        sns.histplot(self.severity_history, bins=20)
        plt.title('Drift Severity Distribution')
        plt.xlabel('Severity')
        
        # Plot 2: Error Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(self.error_history, bins=20)
        plt.title('Detection Error Distribution')
        plt.xlabel('Error')
        
        # Plot 3: Severity vs Error
        plt.subplot(2, 2, 3)
        plt.scatter(self.severity_history, self.error_history, alpha=0.5)
        plt.title('Severity vs Error')
        plt.xlabel('Severity')
        plt.ylabel('Error')
        
        # Plot 4: Time Series Analysis
        plt.subplot(2, 2, 4)
        df = pd.DataFrame(self.drift_history)
        plt.plot(df['timestamp'], df['severity'], 'b-', label='Severity')
        plt.plot(df['timestamp'], df['error'], 'r-', label='Error')
        plt.title('Severity and Error Over Time')
        plt.xlabel('Time')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_analysis(self, save_dir: str) -> None:
        """Save all drift analysis plots."""
        os.makedirs(save_dir, exist_ok=True)
        
        self.plot_drift_errors(os.path.join(save_dir, 'drift_errors.png'))
        self.plot_drift_severity(os.path.join(save_dir, 'drift_severity.png'))
        self.plot_drift_true_vs_pred(os.path.join(save_dir, 'drift_true_vs_pred.png'))
        self.plot_drift_analysis(os.path.join(save_dir, 'drift_analysis.png'))

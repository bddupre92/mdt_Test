import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import argparse
from pathlib import Path
import time
import warnings
from scipy import stats

from utils.json_utils import save_json
from utils.plotting import save_plot, setup_plot_style

class DriftExplainer:
    """
    Explainer for concept drift in data streams
    """
    def __init__(self, detector=None, meta_learner=None):
        """
        Initialize the drift explainer
        
        Parameters:
        -----------
        detector : object, optional
            Drift detector to use
        meta_learner : object, optional
            Meta-learner to explain
        """
        self.detector = detector
        self.meta_learner = meta_learner
        self.explanation = None
    
    def explain(self, X_before, y_before, X_after, y_after, drift_point=None, **kwargs):
        """
        Generate explanation for detected drift
        
        Parameters:
        -----------
        X_before : np.ndarray or pd.DataFrame
            Features before drift
        y_before : np.ndarray or pd.Series
            Targets before drift
        X_after : np.ndarray or pd.DataFrame
            Features after drift
        y_after : np.ndarray or pd.Series
            Targets after drift
        drift_point : int, optional
            Index of the drift point
        **kwargs : dict
            Additional parameters for the explanation
            
        Returns:
        --------
        Dict[str, Any]
            Explanation results
        """
        # Extract drift information
        explanation = {
            'data_info': self._extract_data_info(X_before, y_before, X_after, y_after, drift_point),
            'distribution_changes': self._analyze_distribution_changes(X_before, X_after),
            'performance_impact': self._analyze_performance_impact(X_before, y_before, X_after, y_after),
            'feature_importance': self._analyze_feature_importance(X_before, y_before, X_after, y_after)
        }
        
        # Store explanation
        self.explanation = explanation
        
        return explanation
    
    def _extract_data_info(self, X_before, y_before, X_after, y_after, drift_point):
        """
        Extract basic information about the data before and after drift
        
        Returns:
        --------
        Dict[str, Any]
            Basic information about the data
        """
        info = {
            'before_size': len(X_before),
            'after_size': len(X_after),
            'n_features': X_before.shape[1],
            'drift_point': drift_point
        }
        
        # Calculate basic statistics for each dataset
        info['before_stats'] = {
            'y_mean': float(np.mean(y_before)),
            'y_std': float(np.std(y_before)),
            'y_min': float(np.min(y_before)),
            'y_max': float(np.max(y_before))
        }
        
        info['after_stats'] = {
            'y_mean': float(np.mean(y_after)),
            'y_std': float(np.std(y_after)),
            'y_min': float(np.min(y_after)),
            'y_max': float(np.max(y_after))
        }
        
        # Calculate changes in statistics
        info['target_changes'] = {
            'mean_change': info['after_stats']['y_mean'] - info['before_stats']['y_mean'],
            'std_change': info['after_stats']['y_std'] - info['before_stats']['y_std'],
            'range_change': (info['after_stats']['y_max'] - info['after_stats']['y_min']) - 
                          (info['before_stats']['y_max'] - info['before_stats']['y_min'])
        }
        
        return info
    
    def _analyze_distribution_changes(self, X_before, X_after):
        """
        Analyze changes in feature distributions
        
        Returns:
        --------
        Dict[str, Any]
            Distribution change analysis results
        """
        results = {
            'feature_distribution_changes': [],
            'most_changed_features': []
        }
        
        # Get feature names or generate them
        if hasattr(X_before, 'columns'):
            feature_names = X_before.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X_before.shape[1])]
        
        # Convert to numpy arrays if needed
        X_before_np = X_before.values if hasattr(X_before, 'values') else X_before
        X_after_np = X_after.values if hasattr(X_after, 'values') else X_after
        
        # Calculate distribution changes for each feature
        feature_changes = []
        
        for i in range(X_before.shape[1]):
            # Extract feature values
            before_values = X_before_np[:, i]
            after_values = X_after_np[:, i]
            
            # Basic statistics changes
            mean_before = np.mean(before_values)
            mean_after = np.mean(after_values)
            std_before = np.std(before_values)
            std_after = np.std(after_values)
            
            # Statistical tests
            try:
                # Kolmogorov-Smirnov test for distribution change
                ks_statistic, ks_pvalue = stats.ks_2samp(before_values, after_values)
                
                # T-test for mean change
                t_statistic, t_pvalue = stats.ttest_ind(before_values, after_values, equal_var=False)
                
                # F-test for variance change
                f_statistic = np.var(before_values, ddof=1) / np.var(after_values, ddof=1)
                f_pvalue = 1 - stats.f.cdf(f_statistic, len(before_values) - 1, len(after_values) - 1)
                
                # Ensure consistent p-value direction (lower = more significant change)
                if f_statistic < 1:
                    f_statistic = 1 / f_statistic
                    f_pvalue = 1 - f_pvalue
            except Exception as e:
                # Handle cases where tests cannot be performed
                ks_statistic, ks_pvalue = 0, 1
                t_statistic, t_pvalue = 0, 1
                f_statistic, f_pvalue = 1, 1
                logging.warning(f"Could not perform statistical tests on feature {i}: {str(e)}")
            
            # Calculate overall change score (lower p-value = more significant change)
            change_score = (1 - ks_pvalue) + (1 - t_pvalue) + (1 - f_pvalue)
            
            # Record changes
            feature_change = {
                'feature': feature_names[i],
                'mean_before': float(mean_before),
                'mean_after': float(mean_after),
                'mean_change': float(mean_after - mean_before),
                'mean_change_pct': float((mean_after - mean_before) / (abs(mean_before) + 1e-10)),
                'std_before': float(std_before),
                'std_after': float(std_after),
                'std_change': float(std_after - std_before),
                'ks_test': {
                    'statistic': float(ks_statistic),
                    'pvalue': float(ks_pvalue)
                },
                't_test': {
                    'statistic': float(t_statistic),
                    'pvalue': float(t_pvalue)
                },
                'f_test': {
                    'statistic': float(f_statistic),
                    'pvalue': float(f_pvalue)
                },
                'change_score': float(change_score)
            }
            
            feature_changes.append(feature_change)
        
        # Sort features by change score
        feature_changes.sort(key=lambda x: x['change_score'], reverse=True)
        
        # Store all changes
        results['feature_distribution_changes'] = feature_changes
        
        # Get top changed features
        top_n = min(5, len(feature_changes))
        results['most_changed_features'] = [
            {'feature': fc['feature'], 'change_score': fc['change_score']}
            for fc in feature_changes[:top_n]
        ]
        
        # Calculate overall distribution change
        results['overall_distribution_change'] = {
            'mean_ks_statistic': float(np.mean([fc['ks_test']['statistic'] for fc in feature_changes])),
            'max_ks_statistic': float(np.max([fc['ks_test']['statistic'] for fc in feature_changes])),
            'mean_change_score': float(np.mean([fc['change_score'] for fc in feature_changes])),
            'max_change_score': float(np.max([fc['change_score'] for fc in feature_changes]))
        }
        
        return results
    
    def _analyze_performance_impact(self, X_before, y_before, X_after, y_after):
        """
        Analyze impact of drift on model performance
        
        Returns:
        --------
        Dict[str, Any]
            Performance impact analysis results
        """
        results = {
            'model_performance_before': {},
            'model_performance_after': {},
            'performance_change': {}
        }
        
        # Use meta_learner if provided, otherwise create a simple model
        model = self.meta_learner
        
        if model is None:
            # Create a simple random forest model
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            
            # Train on data before drift
            model.fit(X_before, y_before)
        
        # Evaluate model on both datasets
        try:
            # Predict on before data
            y_pred_before = model.predict(X_before)
            
            # Predict on after data
            y_pred_after = model.predict(X_after)
            
            # Calculate metrics for before data
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse_before = mean_squared_error(y_before, y_pred_before)
            mae_before = mean_absolute_error(y_before, y_pred_before)
            r2_before = r2_score(y_before, y_pred_before)
            
            results['model_performance_before'] = {
                'mse': float(mse_before),
                'mae': float(mae_before),
                'r2': float(r2_before)
            }
            
            # Calculate metrics for after data
            mse_after = mean_squared_error(y_after, y_pred_after)
            mae_after = mean_absolute_error(y_after, y_pred_after)
            
            # R2 can be negative if model performs poorly
            try:
                r2_after = r2_score(y_after, y_pred_after)
            except:
                r2_after = -1.0
            
            results['model_performance_after'] = {
                'mse': float(mse_after),
                'mae': float(mae_after),
                'r2': float(r2_after)
            }
            
            # Calculate performance changes
            results['performance_change'] = {
                'mse_change': float(mse_after - mse_before),
                'mse_change_pct': float((mse_after - mse_before) / (mse_before + 1e-10)),
                'mae_change': float(mae_after - mae_before),
                'mae_change_pct': float((mae_after - mae_before) / (mae_before + 1e-10)),
                'r2_change': float(r2_after - r2_before)
            }
            
            # Determine if drift is harmful
            results['harmful_drift'] = results['performance_change']['mse_change'] > 0
            
            # Determine severity of impact
            if results['performance_change']['mse_change_pct'] > 1.0:
                severity = 'high'
            elif results['performance_change']['mse_change_pct'] > 0.5:
                severity = 'medium'
            elif results['performance_change']['mse_change_pct'] > 0.1:
                severity = 'low'
            else:
                severity = 'negligible'
            
            results['impact_severity'] = severity
            
        except Exception as e:
            logging.error(f"Error analyzing performance impact: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _analyze_feature_importance(self, X_before, y_before, X_after, y_after):
        """
        Analyze changes in feature importance
        
        Returns:
        --------
        Dict[str, Any]
            Feature importance change analysis
        """
        results = {
            'feature_importance_before': {},
            'feature_importance_after': {},
            'importance_change': {}
        }
        
        # Get feature names or generate them
        if hasattr(X_before, 'columns'):
            feature_names = X_before.columns.tolist()
        else:
            feature_names = [f'Feature {i}' for i in range(X_before.shape[1])]
        
        try:
            # Create models to analyze feature importance
            from sklearn.ensemble import RandomForestRegressor
            
            model_before = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            model_after = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            
            # Train models
            model_before.fit(X_before, y_before)
            model_after.fit(X_after, y_after)
            
            # Get feature importances
            importance_before = model_before.feature_importances_
            importance_after = model_after.feature_importances_
            
            # Store feature importances
            for i, feature in enumerate(feature_names):
                results['feature_importance_before'][feature] = float(importance_before[i])
                results['feature_importance_after'][feature] = float(importance_after[i])
                results['importance_change'][feature] = float(importance_after[i] - importance_before[i])
            
            # Calculate overall change in feature importance
            importance_change_values = list(results['importance_change'].values())
            results['overall_importance_change'] = {
                'mean_absolute_change': float(np.mean(np.abs(importance_change_values))),
                'max_absolute_change': float(np.max(np.abs(importance_change_values)))
            }
            
            # Find top features with importance change
            sorted_features = sorted(
                results['importance_change'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            top_n = min(5, len(sorted_features))
            results['top_importance_changes'] = [
                {'feature': f, 'change': c} for f, c in sorted_features[:top_n]
            ]
            
        except Exception as e:
            logging.error(f"Error analyzing feature importance: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def plot(self, plot_type='distribution_changes', **kwargs):
        """
        Generate plots from the drift explanation
        
        Parameters:
        -----------
        plot_type : str, optional
            Type of plot ('distribution_changes', 'performance_impact', 'feature_importance')
        **kwargs : dict
            Additional parameters for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot
        """
        if self.explanation is None:
            raise ValueError("No explanation generated. Call explain() first.")
        
        # Set up matplotlib
        setup_plot_style()
        
        try:
            if plot_type == 'distribution_changes':
                return self._plot_distribution_changes(**kwargs)
            elif plot_type == 'performance_impact':
                return self._plot_performance_impact(**kwargs)
            elif plot_type == 'feature_importance':
                return self._plot_feature_importance(**kwargs)
            elif plot_type == 'target_distribution':
                return self._plot_target_distribution(**kwargs)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
                
        except Exception as e:
            logging.error(f"Error generating plot: {str(e)}")
            
            # Create a simple error figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            ax.set_title(f"Error: {type(e).__name__}")
            ax.axis('off')
            
            return fig
    
    def _plot_distribution_changes(self, top_n=5, **kwargs):
        """
        Plot distribution changes for top changed features
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Distribution changes plot
        """
        # Get distribution changes
        dist_changes = self.explanation['distribution_changes']['feature_distribution_changes']
        
        # Sort by change score and get top N
        sorted_changes = sorted(dist_changes, key=lambda x: x['change_score'], reverse=True)
        top_changes = sorted_changes[:min(top_n, len(sorted_changes))]
        
        # Create grid of plots
        n_features = len(top_changes)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        
        # Handle single plot case
        if n_features == 1:
            axes = np.array([axes])
        
        # Ensure axes is always a 2D array
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Create plots for each feature
        for i, feature_change in enumerate(top_changes):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Extract feature name and statistics
            feature = feature_change['feature']
            ks_stat = feature_change['ks_test']['statistic']
            ks_pval = feature_change['ks_test']['pvalue']
            mean_before = feature_change['mean_before']
            mean_after = feature_change['mean_after']
            
            # Add title with feature name and KS test results
            ax.set_title(f"{feature}\nKS statistic: {ks_stat:.3f}, p-value: {ks_pval:.3f}")
            
            # Add mean lines
            ax.axvline(mean_before, color='blue', linestyle='--', 
                      label=f'Before mean: {mean_before:.3f}')
            ax.axvline(mean_after, color='red', linestyle='--', 
                      label=f'After mean: {mean_after:.3f}')
            
            # Add legend
            ax.legend()
            
            # Set labels
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        # Hide unused subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        # Add suptitle
        fig.suptitle('Feature Distribution Changes Before and After Drift', 
                    fontsize=16, y=1.02)
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_impact(self, **kwargs):
        """
        Plot performance impact of drift
        
        Returns:
        --------
        matplotlib.figure.Figure
            Performance impact plot
        """
        # Get performance impact data
        perf_before = self.explanation['performance_impact']['model_performance_before']
        perf_after = self.explanation['performance_impact']['model_performance_after']
        
        # Create figure with 3 subplots (MSE, MAE, R2)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot MSE
        axes[0].bar(['Before Drift', 'After Drift'], 
                   [perf_before.get('mse', 0), perf_after.get('mse', 0)])
        axes[0].set_title('Mean Squared Error')
        axes[0].set_ylabel('MSE')
        
        # Plot MAE
        axes[1].bar(['Before Drift', 'After Drift'], 
                   [perf_before.get('mae', 0), perf_after.get('mae', 0)])
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_ylabel('MAE')
        
        # Plot R2
        axes[2].bar(['Before Drift', 'After Drift'], 
                   [perf_before.get('r2', 0), perf_after.get('r2', 0)])
        axes[2].set_title('R² Score')
        axes[2].set_ylabel('R²')
        
        # Add value labels
        for i, ax in enumerate(axes):
            metric = ['mse', 'mae', 'r2'][i]
            for j, rect in enumerate(ax.patches):
                height = rect.get_height()
                value = perf_before.get(metric, 0) if j == 0 else perf_after.get(metric, 0)
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Add suptitle
        severity = self.explanation['performance_impact'].get('impact_severity', 'unknown')
        is_harmful = self.explanation['performance_impact'].get('harmful_drift', False)
        
        title = f'Performance Impact of Drift (Severity: {severity.capitalize()})'
        if is_harmful:
            title += ' - HARMFUL DRIFT'
        
        fig.suptitle(title, fontsize=16, y=1.05)
        
        plt.tight_layout()
        return fig
    
    def _plot_feature_importance(self, top_n=10, **kwargs):
        """
        Plot feature importance changes
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Feature importance plot
        """
        # Get feature importance data
        importance_before = self.explanation['feature_importance']['feature_importance_before']
        importance_after = self.explanation['feature_importance']['feature_importance_after']
        importance_change = self.explanation['feature_importance']['importance_change']
        
        # Sort features by absolute change
        sorted_features = sorted(
            importance_change.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top N features
        top_features = [f for f, _ in sorted_features[:min(top_n, len(sorted_features))]]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot feature importances before and after
        x = np.arange(len(top_features))
        width = 0.35
        
        before_values = [importance_before.get(f, 0) for f in top_features]
        after_values = [importance_after.get(f, 0) for f in top_features]
        
        ax1.bar(x - width/2, before_values, width, label='Before Drift')
        ax1.bar(x + width/2, after_values, width, label='After Drift')
        
        ax1.set_title('Feature Importance Before and After Drift')
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_features, rotation=45, ha='right')
        ax1.set_ylabel('Importance')
        ax1.legend()
        
        # Plot importance changes
        change_values = [importance_change.get(f, 0) for f in top_features]
        colors = ['green' if v >= 0 else 'red' for v in change_values]
        
        ax2.bar(top_features, change_values, color=colors)
        ax2.set_title('Feature Importance Changes')
        ax2.set_xticklabels(top_features, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Importance Change')
        
        # Add value labels for change plot
        for i, v in enumerate(change_values):
            ax2.text(i, v + (0.01 if v >= 0 else -0.03),
                   f'{v:.3f}', ha='center')
        
        # Add suptitle
        fig.suptitle('Feature Importance Analysis', fontsize=16, y=0.98)
        
        plt.tight_layout()
        return fig
    
    def _plot_target_distribution(self, **kwargs):
        """
        Plot target distribution before and after drift
        
        Returns:
        --------
        matplotlib.figure.Figure
            Target distribution plot
        """
        # Get data info
        data_info = self.explanation['data_info']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot target statistics
        metrics = ['mean', 'std', 'min', 'max']
        before_values = [data_info['before_stats'][f'y_{m}'] for m in metrics]
        after_values = [data_info['after_stats'][f'y_{m}'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, before_values, width, label='Before Drift')
        ax1.bar(x + width/2, after_values, width, label='After Drift')
        
        ax1.set_title('Target Statistics Before and After Drift')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.capitalize() for m in metrics])
        ax1.set_ylabel('Value')
        ax1.legend()
        
        # Plot target changes
        changes = [
            data_info['target_changes']['mean_change'],
            data_info['target_changes']['std_change'],
            data_info['target_changes']['range_change']
        ]
        
        change_labels = ['Mean Change', 'Std Dev Change', 'Range Change']
        colors = ['green' if c >= 0 else 'red' for c in changes]
        
        ax2.bar(change_labels, changes, color=colors)
        ax2.set_title('Target Distribution Changes')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_ylabel('Change')
        
        # Add value labels
        for i, v in enumerate(changes):
            ax2.text(i, v + (0.01 if v >= 0 else -0.03),
                   f'{v:.3f}', ha='center')
        
        # Add suptitle
        fig.suptitle('Target Distribution Analysis', fontsize=16, y=0.98)
        
        plt.tight_layout()
        return fig

def explain_drift(args):
    """
    Explain drift when detected
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
        
    Returns:
    --------
    Dict[str, Any]
        Drift explanation results
    """
    logging.info("Explaining drift in the data")
    
    # Get parameters from args
    window_size = args.drift_window if hasattr(args, 'drift_window') else 50
    drift_threshold = args.drift_threshold if hasattr(args, 'drift_threshold') else 0.5
    significance_level = args.significance_level if hasattr(args, 'significance_level') else 0.05
    visualize = args.visualize if hasattr(args, 'visualize') else False
    data_path = args.data_path if hasattr(args, 'data_path') else None
    
    # Create results directory
    results_dir = Path('results/drift_explanation')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate or load data with drift
    if data_path:
        try:
            # Load data from file
            data = pd.read_csv(data_path)
            
            # Assume the first column is timestamp and the last column is target
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
            
            logging.info(f"Loaded {len(y)} samples from {data_path}")
            
            # Use midpoint as drift point for explanation
            drift_point = len(X) // 2
            is_synthetic = False
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.info("Falling back to synthetic data")
            is_synthetic = True
            data_path = None
    
    if not data_path:
        # Generate synthetic data with drift
        from core.drift_detection import generate_synthetic_data_with_drift
        
        n_samples = 1000
        n_features = 10
        drift_points = [n_samples // 2]  # Single drift point at the middle
        
        X, y, _ = generate_synthetic_data_with_drift(
            n_samples=n_samples, 
            n_features=n_features,
            drift_points=drift_points,
            noise_level=0.1
        )
        
        logging.info(f"Generated {n_samples} synthetic samples with drift point at {drift_points[0]}")
        
        # Set drift point for explanation
        drift_point = drift_points[0]
        is_synthetic = True
    
    # Split data before and after drift
    X_before = X[:drift_point]
    y_before = y[:drift_point]
    X_after = X[drift_point:]
    y_after = y[drift_point:]
    
    logging.info(f"Data split: {len(X_before)} samples before drift, {len(X_after)} samples after drift")
    
    # Create explainer
    explainer = DriftExplainer()
    
    # Generate explanation
    explanation = explainer.explain(X_before, y_before, X_after, y_after, drift_point)
    
    # Generate plots if requested
    plot_paths = {}
    
    if visualize:
        plot_types = ['distribution_changes', 'performance_impact', 'feature_importance', 'target_distribution']
        
        for plot_type in plot_types:
            try:
                # Generate plot
                fig = explainer.plot(plot_type)
                
                # Save plot
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                
                plot_path = save_plot(
                    fig, 
                    f"drift_explanation_{plot_type}_{timestamp}", 
                    plot_type='drift'
                )
                
                plot_paths[plot_type] = str(plot_path)
                
                logging.info(f"Generated {plot_type} plot for drift explanation")
                
            except Exception as e:
                logging.error(f"Error generating {plot_type} plot: {str(e)}")
    
    # Create comprehensive results
    results = {
        'explanation': explanation,
        'plot_paths': plot_paths,
        'metadata': {
            'data_source': data_path if data_path else 'synthetic',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'drift_point': drift_point,
            'window_size': window_size,
            'drift_threshold': drift_threshold,
            'significance_level': significance_level
        }
    }
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'drift_explanation_results_{timestamp}.json'
    
    save_json(results, results_file)
    logging.info(f"Drift explanation results saved to {results_file}")
    
    # Print summary
    print("\nDrift Explanation Summary:")
    print("=========================")
    print(f"Data source: {'Synthetic' if is_synthetic else data_path}")
    print(f"Drift point: {drift_point} (out of {len(X)} samples)")
    
    if 'target_changes' in explanation['data_info']:
        target_changes = explanation['data_info']['target_changes']
        print(f"\nTarget Changes:")
        print(f"  Mean change: {target_changes['mean_change']:.4f}")
        print(f"  Std dev change: {target_changes['std_change']:.4f}")
        print(f"  Range change: {target_changes['range_change']:.4f}")
    
    if 'most_changed_features' in explanation['distribution_changes']:
        top_features = explanation['distribution_changes']['most_changed_features']
        print(f"\nMost Changed Features:")
        for i, feature in enumerate(top_features):
            print(f"  {i+1}. {feature['feature']} (score: {feature['change_score']:.4f})")
    
    if 'performance_impact' in explanation and 'harmful_drift' in explanation['performance_impact']:
        harmful = explanation['performance_impact']['harmful_drift']
        severity = explanation['performance_impact'].get('impact_severity', 'unknown')
        print(f"\nPerformance Impact:")
        print(f"  Harmful drift: {'Yes' if harmful else 'No'}")
        print(f"  Impact severity: {severity.capitalize()}")
        
        if 'performance_change' in explanation['performance_impact']:
            perf_change = explanation['performance_impact']['performance_change']
            print(f"  MSE change: {perf_change.get('mse_change', 0):.4f} ({perf_change.get('mse_change_pct', 0)*100:.1f}%)")
            print(f"  R² change: {perf_change.get('r2_change', 0):.4f}")
    
    if plot_paths:
        print("\nGenerated Plots:")
        for plot_type, path in plot_paths.items():
            print(f"  {plot_type}: {path}")
    
    print(f"\nResults saved to {results_file}")
    
    return results

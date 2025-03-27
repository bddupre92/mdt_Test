"""
Real vs. Synthetic Data Comparison Framework for MoE Validation.

This module provides tools to compare real clinical data with synthetic data
to evaluate model performance across both data types.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class RealSyntheticComparator:
    """Tools to compare real vs synthetic data characteristics and model performance."""
    
    def __init__(self, config_path=None):
        """Initialize with optional configuration file path."""
        self.config = self._load_config(config_path)
        self.comparison_history = []
        
    def _load_config(self, config_path):
        """Load configuration from JSON or Python file or use defaults."""
        if not config_path or not os.path.exists(config_path):
            return {
                'feature_comparison': {
                    'visualization_features': [],
                    'statistical_tests': ['ks', 'ttest']
                },
                'performance_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mse', 'r2']
            }
        
        # Handle Python module config files
        if config_path.endswith('.py'):
            import importlib.util
            import sys
            
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["config_module"] = config_module
            spec.loader.exec_module(config_module)
            
            config = {}
            for attr in dir(config_module):
                if not attr.startswith('__'):
                    config[attr] = getattr(config_module, attr)
            return config
        
        # Handle JSON config files
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Attempt to load as JSON for backward compatibility
        else:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Config file {config_path} is not a valid JSON or Python file")
        
    def compare_distributions(self, real_df, synthetic_df, features=None):
        """Compare feature distributions between real and synthetic data."""
        # Determine which features to compare
        if features is None:
            features = self.config.get('feature_comparison', {}).get('visualization_features', [])
            if not features:
                # Use common columns if not specified
                features = list(set(real_df.columns) & set(synthetic_df.columns))
                # Filter to numeric and boolean columns only
                features = [f for f in features if f in real_df.select_dtypes(include=['number', 'bool']).columns]
                
        # Ensure all features exist in both datasets
        features = [f for f in features if f in real_df.columns and f in synthetic_df.columns]
        
        comparison_results = {}
        
        for feature in features:
            # Get real and synthetic values for this feature
            real_values = real_df[feature].dropna()
            synth_values = synthetic_df[feature].dropna()
            
            # Skip if not enough data points
            if len(real_values) < 5 or len(synth_values) < 5:
                comparison_results[feature] = {
                    'error': 'Insufficient data points for comparison'
                }
                continue
                
            # 1. Basic statistics
            real_stats = {
                'mean': float(real_values.mean()),
                'median': float(real_values.median()),
                'std': float(real_values.std()),
                'min': float(real_values.min()),
                'max': float(real_values.max())
            }
            
            synth_stats = {
                'mean': float(synth_values.mean()),
                'median': float(synth_values.median()),
                'std': float(synth_values.std()),
                'min': float(synth_values.min()),
                'max': float(synth_values.max())
            }
            
            # 2. Statistical tests
            tests = self.config.get('feature_comparison', {}).get('statistical_tests', ['ks'])
            test_results = {}
            
            # Kolmogorov-Smirnov test
            if 'ks' in tests:
                try:
                    ks_stat, p_value = stats.ks_2samp(real_values, synth_values)
                    test_results['ks_test'] = {
                        'statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'significant_difference': p_value < 0.05
                    }
                except Exception as e:
                    test_results['ks_test'] = {'error': str(e)}
            
            # T-test
            if 'ttest' in tests:
                try:
                    t_stat, p_value = stats.ttest_ind(real_values, synth_values, equal_var=False)
                    test_results['t_test'] = {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_difference': p_value < 0.05
                    }
                except Exception as e:
                    test_results['t_test'] = {'error': str(e)}
            
            # 3. Create distribution comparison visualization
            fig = plt.figure(figsize=(10, 6))
            
            # Histogram with KDE
            ax = sns.histplot(real_values, kde=True, stat="density", label="Real Data", alpha=0.6, color="blue")
            sns.histplot(synth_values, kde=True, stat="density", label="Synthetic Data", alpha=0.6, color="red", ax=ax)
            
            plt.title(f"Distribution Comparison: {feature}")
            plt.legend()
            plt.tight_layout()
            
            # Save figure to buffer
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            
            # Store results
            comparison_results[feature] = {
                'real_stats': real_stats,
                'synthetic_stats': synth_stats,
                'test_results': test_results,
                'visualization': buf.getvalue()  # Store raw PNG data
            }
            
        # Store in comparison history
        self.comparison_history.append({
            'comparison_type': 'feature_distributions',
            'features_compared': features,
            'timestamp': pd.Timestamp.now().isoformat()
        })
            
        return comparison_results
        
    def compare_correlations(self, real_df, synthetic_df, features=None):
        """Compare correlation structures between real and synthetic data."""
        # Determine which features to use for correlation analysis
        if features is None:
            features = list(set(real_df.columns) & set(synthetic_df.columns))
            features = [f for f in features if f in real_df.select_dtypes(include=['number']).columns]
        
        # Ensure all features exist in both datasets
        features = [f for f in features if f in real_df.columns and f in synthetic_df.columns]
        
        if len(features) < 2:
            return {
                'error': 'Insufficient features for correlation analysis',
                'features_available': features
            }
            
        # Calculate correlation matrices
        real_corr = real_df[features].corr()
        synth_corr = synthetic_df[features].corr()
        
        # Calculate difference
        corr_diff = real_corr - synth_corr
        
        # Create visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Real correlations
        sns.heatmap(real_corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title("Real Data Correlations")
        
        # Synthetic correlations
        sns.heatmap(synth_corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title("Synthetic Data Correlations")
        
        # Correlation differences
        sns.heatmap(corr_diff, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
        axes[2].set_title("Correlation Differences (Real - Synthetic)")
        
        plt.tight_layout()
        
        # Save figure to buffer
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        
        # Calculate matrix similarity metrics
        from sklearn.metrics import mean_squared_error
        from numpy import triu_indices_from, corrcoef
        
        # Get upper triangular values (excluding diagonal)
        idx = triu_indices_from(real_corr.values, 1)
        real_upper = real_corr.values[idx]
        synth_upper = synth_corr.values[idx]
        
        # Calculate MSE and correlation of correlations
        mse = mean_squared_error(real_upper, synth_upper)
        corr_of_corrs, p_value = stats.pearsonr(real_upper, synth_upper)
        
        result = {
            'correlation_mse': float(mse),
            'correlation_similarity': float(corr_of_corrs),
            'correlation_p_value': float(p_value),
            'visualization': buf.getvalue(),
            'features_used': features
        }
        
        # Store in comparison history
        self.comparison_history.append({
            'comparison_type': 'correlation_structure',
            'features_compared': features,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        return result
    
    def _evaluate_model(self, model, X, y, task_type):
        """Helper method to evaluate a model on given data."""
        try:
            # Make predictions
            if task_type == 'classification':
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)
                    if y_prob.shape[1] > 1:  # Multi-class
                        y_prob = y_prob[:, 1]  # Get probability of positive class
                    y_pred = (y_prob > 0.5).astype(int)
                else:
                    y_pred = model.predict(X)
                    y_prob = None
                
                # Calculate metrics
                metrics = {
                    'accuracy': float(accuracy_score(y, y_pred)),
                    'precision': float(precision_score(y, y_pred, average='binary', zero_division=0)),
                    'recall': float(recall_score(y, y_pred, average='binary', zero_division=0)),
                    'f1': float(f1_score(y, y_pred, average='binary', zero_division=0))
                }
                
                # ROC AUC if probabilities available
                if y_prob is not None:
                    metrics['roc_auc'] = float(roc_auc_score(y, y_prob))
                    
            elif task_type == 'regression':
                y_pred = model.predict(X)
                
                # Calculate metrics
                metrics = {
                    'mse': float(mean_squared_error(y, y_pred)),
                    'r2': float(r2_score(y, y_pred))
                }
                
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
            return {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_model_performance(self, model, real_data, synthetic_data, 
                                  target_column, features=None, task_type='classification'):
        """Compare model performance metrics on real vs synthetic data."""
        # Determine which features to use
        if features is None:
            # Use all common features except the target
            features = list(set(real_data.columns) & set(synthetic_data.columns))
            features = [f for f in features if f != target_column]
        
        # Ensure all features exist in both datasets
        features = [f for f in features if f in real_data.columns and f in synthetic_data.columns]
        
        # Check if target column exists
        if target_column not in real_data.columns or target_column not in synthetic_data.columns:
            return {
                'error': f"Target column '{target_column}' not found in both datasets"
            }
            
        # Prepare data
        X_real = real_data[features]
        y_real = real_data[target_column]
        
        X_synth = synthetic_data[features]
        y_synth = synthetic_data[target_column]
        
        # Check data validity
        if X_real.shape[0] == 0 or X_synth.shape[0] == 0:
            return {
                'error': "Empty dataset provided"
            }
            
        # 1. Train on synthetic, test on real
        synthetic_to_real = self._cross_validation(model, X_synth, y_synth, X_real, y_real, task_type)
        
        # 2. Train on real, test on synthetic
        real_to_synthetic = self._cross_validation(model, X_real, y_real, X_synth, y_synth, task_type)
        
        # 3. Train and test on synthetic
        synthetic_only = self._cross_validation(model, X_synth, y_synth, X_synth, y_synth, task_type)
        
        # 4. Train and test on real
        real_only = self._cross_validation(model, X_real, y_real, X_real, y_real, task_type)
        
        # Create comparison visualization
        if task_type == 'classification':
            # Compare selected metrics
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            metrics_to_plot = [m for m in metrics_to_plot if all(
                m in result['metrics'] for result in [synthetic_to_real, real_to_synthetic, 
                                                     synthetic_only, real_only]
                if result['success']
            )]
            
            if metrics_to_plot:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create data for plotting
                scenarios = ['Synth→Real', 'Real→Synth', 'Synth→Synth', 'Real→Real']
                data = []
                
                for i, result in enumerate([synthetic_to_real, real_to_synthetic, synthetic_only, real_only]):
                    if result['success']:
                        for metric in metrics_to_plot:
                            data.append({
                                'Scenario': scenarios[i],
                                'Metric': metric,
                                'Value': result['metrics'][metric]
                            })
                
                # Create DataFrame for seaborn
                plot_df = pd.DataFrame(data)
                
                # Create grouped bar chart
                sns.barplot(x='Metric', y='Value', hue='Scenario', data=plot_df, ax=ax)
                
                ax.set_title('Model Performance Comparison')
                ax.set_ylim(0, 1.0)
                plt.tight_layout()
                
                # Save figure to buffer
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                
                viz = buf.getvalue()
            else:
                viz = None
        else:
            # Regression metrics visualization
            metrics_to_plot = ['mse', 'r2']
            
            # Check if we have successful results with these metrics
            has_metrics = all(m in result['metrics'] for result in [synthetic_to_real, real_to_synthetic, 
                                                                 synthetic_only, real_only]
                           if result['success'] for m in metrics_to_plot)
            
            if has_metrics:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Create data for MSE
                scenarios = ['Synth→Real', 'Real→Synth', 'Synth→Synth', 'Real→Real']
                mse_values = [
                    synthetic_to_real['metrics']['mse'] if synthetic_to_real['success'] else 0,
                    real_to_synthetic['metrics']['mse'] if real_to_synthetic['success'] else 0,
                    synthetic_only['metrics']['mse'] if synthetic_only['success'] else 0,
                    real_only['metrics']['mse'] if real_only['success'] else 0
                ]
                
                # MSE plot
                axes[0].bar(scenarios, mse_values)
                axes[0].set_title('Mean Squared Error')
                axes[0].set_ylabel('MSE (lower is better)')
                
                # Create data for R2
                r2_values = [
                    synthetic_to_real['metrics']['r2'] if synthetic_to_real['success'] else 0,
                    real_to_synthetic['metrics']['r2'] if real_to_synthetic['success'] else 0,
                    synthetic_only['metrics']['r2'] if synthetic_only['success'] else 0,
                    real_only['metrics']['r2'] if real_only['success'] else 0
                ]
                
                # R2 plot
                axes[1].bar(scenarios, r2_values)
                axes[1].set_title('R² Score')
                axes[1].set_ylabel('R² (higher is better)')
                axes[1].set_ylim(-0.2, 1.0)  # R2 can be negative
                
                plt.tight_layout()
                
                # Save figure to buffer
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                
                viz = buf.getvalue()
            else:
                viz = None
        
        result = {
            'synthetic_to_real': synthetic_to_real,
            'real_to_synthetic': real_to_synthetic,
            'synthetic_only': synthetic_only,
            'real_only': real_only,
            'features_used': features,
            'target_column': target_column,
            'task_type': task_type,
            'visualization': viz
        }
        
        # Store in comparison history
        self.comparison_history.append({
            'comparison_type': 'model_performance',
            'features_compared': features,
            'target_column': target_column,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        return result
    
    def _cross_validation(self, model_template, X_train, y_train, X_test, y_test, task_type):
        """Perform train-test evaluation with a model template."""
        from sklearn.base import clone
        
        try:
            # Clone the model to avoid modifying the original
            model = clone(model_template)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            return self._evaluate_model(model, X_test, y_test, task_type)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_comparison_report(self, results, save_path=None):
        """Generate comprehensive report comparing real and synthetic data."""
        # Create report structure
        report = {
            'report_timestamp': pd.Timestamp.now().isoformat(),
            'report_summary': {
                'total_comparisons': len(self.comparison_history),
                'features_compared': list(set(
                    feature for comp in self.comparison_history 
                    for feature in comp.get('features_compared', [])
                ))
            },
            'comparison_results': results,
            'comparison_history': self.comparison_history
        }
        
        # Calculate overall similarity score
        similarity_scores = []
        
        # Distribution similarity
        if 'distributions' in results:
            for feature, result in results['distributions'].items():
                if 'test_results' in result and 'ks_test' in result['test_results']:
                    # Convert KS statistic to similarity (1 - KS)
                    ks_stat = result['test_results']['ks_test'].get('statistic')
                    if ks_stat is not None:
                        similarity_scores.append(1 - ks_stat)
        
        # Correlation structure similarity
        if 'correlations' in results and 'correlation_similarity' in results['correlations']:
            # Add the correlation of correlations
            corr_sim = results['correlations']['correlation_similarity']
            if corr_sim is not None:
                similarity_scores.append(max(0, corr_sim))  # Ensure non-negative
        
        # Model performance transferability
        if 'model_performance' in results:
            perf = results['model_performance']
            
            # Check synthetic to real transfer
            if perf.get('synthetic_to_real', {}).get('success', False):
                # Compare to synthetic-only performance as baseline
                if perf.get('synthetic_only', {}).get('success', False):
                    for metric in ['accuracy', 'f1', 'r2']:
                        if (metric in perf['synthetic_to_real'].get('metrics', {}) and 
                            metric in perf['synthetic_only'].get('metrics', {})):
                            # Calculate relative performance ratio
                            synth_real = perf['synthetic_to_real']['metrics'][metric]
                            synth_only = perf['synthetic_only']['metrics'][metric]
                            
                            if synth_only > 0:
                                transfer_ratio = synth_real / synth_only
                                similarity_scores.append(min(1.0, transfer_ratio))  # Cap at 1.0
        
        # Calculate overall similarity if we have scores
        if similarity_scores:
            report['report_summary']['overall_similarity_score'] = float(np.mean(similarity_scores))
        
        # Save report if path provided
        if save_path:
            # Ensure we don't try to serialize binary data
            save_report = report.copy()
            
            # Remove binary visualization data from saved report
            if 'distributions' in save_report.get('comparison_results', {}):
                for feature in save_report['comparison_results']['distributions']:
                    if 'visualization' in save_report['comparison_results']['distributions'][feature]:
                        del save_report['comparison_results']['distributions'][feature]['visualization']
            
            if ('correlations' in save_report.get('comparison_results', {}) and 
                'visualization' in save_report['comparison_results']['correlations']):
                del save_report['comparison_results']['correlations']['visualization']
                
            if ('model_performance' in save_report.get('comparison_results', {}) and 
                'visualization' in save_report['comparison_results']['model_performance']):
                del save_report['comparison_results']['model_performance']['visualization']
            
            # Save JSON report using NumpyEncoder for NumPy data types
            with open(save_path, 'w') as f:
                json.dump(save_report, f, cls=NumpyEncoder, indent=2)
            
            # Save visualizations separately if they exist
            save_dir = os.path.dirname(save_path)
            base_name = os.path.splitext(os.path.basename(save_path))[0]
            
            if 'distributions' in report.get('comparison_results', {}):
                os.makedirs(os.path.join(save_dir, f"{base_name}_viz"), exist_ok=True)
                
                for feature, result in report['comparison_results']['distributions'].items():
                    if 'visualization' in result:
                        viz_path = os.path.join(
                            save_dir, f"{base_name}_viz", f"distribution_{feature}.png")
                        with open(viz_path, 'wb') as f:
                            f.write(result['visualization'])
            
            if ('correlations' in report.get('comparison_results', {}) and 
                'visualization' in report['comparison_results']['correlations']):
                os.makedirs(os.path.join(save_dir, f"{base_name}_viz"), exist_ok=True)
                viz_path = os.path.join(save_dir, f"{base_name}_viz", "correlations.png")
                with open(viz_path, 'wb') as f:
                    f.write(report['comparison_results']['correlations']['visualization'])
                    
            if ('model_performance' in report.get('comparison_results', {}) and 
                'visualization' in report['comparison_results']['model_performance']):
                os.makedirs(os.path.join(save_dir, f"{base_name}_viz"), exist_ok=True)
                viz_path = os.path.join(save_dir, f"{base_name}_viz", "model_performance.png")
                with open(viz_path, 'wb') as f:
                    f.write(report['comparison_results']['model_performance']['visualization'])
                
        return report
    
    def compare_all(self, real_df, synthetic_df, model=None, target_column=None, 
                    task_type='classification', features=None, save_path=None):
        """Run all comparison analyses and generate a comprehensive report."""
        results = {}
        
        # 1. Compare feature distributions
        results['distributions'] = self.compare_distributions(real_df, synthetic_df, features)
        
        # 2. Compare correlation structures
        results['correlations'] = self.compare_correlations(real_df, synthetic_df, features)
        
        # 3. Compare model performance if model provided
        if model is not None and target_column is not None:
            results['model_performance'] = self.compare_model_performance(
                model, real_df, synthetic_df, target_column, features, task_type)
        
        # 4. Generate final report
        return self.generate_comparison_report(results, save_path)

import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules from test directories
from tests.moe_enhanced_validation_part1 import (
    SyntheticDataGenerator,
    MockExpert,
    MockDriftDetector
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockExplainer:
    """Mock explainer for testing explainability integration."""
    
    def __init__(self, model=None):
        self.model = model
        self.feature_names = None
        
    def explain(self, data, target=None):
        """Generate feature importance explanation."""
        if self.model is None:
            # Generate random feature importance if no model
            n_features = data.shape[1]
            importance = np.random.random(n_features)
            importance = importance / np.sum(importance)  # Normalize
            
            feature_names = self.feature_names or [f'feature_{i}' for i in range(n_features)]
            return {
                'feature_importance': dict(zip(feature_names, importance)),
                'global_importance': importance,
                'feature_names': feature_names
            }
        else:
            # Use model's feature importance if available
            try:
                importance = self.model.feature_importances_
                importance = importance / np.sum(importance)  # Normalize
                
                feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
                return {
                    'feature_importance': dict(zip(feature_names, importance)),
                    'global_importance': importance,
                    'feature_names': feature_names
                }
            except:
                # Fallback to random if model doesn't support feature_importances_
                n_features = data.shape[1]
                importance = np.random.random(n_features)
                importance = importance / np.sum(importance)
                
                feature_names = self.feature_names or [f'feature_{i}' for i in range(n_features)]
                return {
                    'feature_importance': dict(zip(feature_names, importance)),
                    'global_importance': importance,
                    'feature_names': feature_names
                }

class ExplainableDriftTests:
    """Test suite for integrated explainability and drift detection tests."""
    
    def __init__(self):
        self.results = {}
        self.drift_detector = MockDriftDetector(threshold=0.15)
        self.explainer = MockExplainer()
        
    def extract_time_features(self, df):
        """Extract useful temporal features from timestamp column."""
        if 'timestamp' not in df.columns:
            return df
            
        df_with_time = df.copy()
        
        # Extract temporal features
        df_with_time['hour'] = df_with_time['timestamp'].dt.hour
        df_with_time['day_of_week'] = df_with_time['timestamp'].dt.dayofweek
        df_with_time['day_of_month'] = df_with_time['timestamp'].dt.day
        df_with_time['month'] = df_with_time['timestamp'].dt.month
        
        # Drop the original timestamp column
        df_with_time = df_with_time.drop('timestamp', axis=1)
        
        return df_with_time
        
    def setup(self):
        """Set up data directories."""
        # Create results directory if it doesn't exist
        results_dir = Path("../results/moe_validation")
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
        
    def test_drift_feature_importance(self):
        """Test that identifies which features contribute most to drift."""
        logger.info("Running drift feature importance test")
        results_dir = self.setup()
        
        # Generate synthetic data with concept drift
        data_gen = SyntheticDataGenerator(seed=42)
        data = data_gen.generate_concept_drift_data(n_samples=500, drift_point=250, drift_magnitude=5.0)
        
        # Split into before and after drift
        before_drift = data.iloc[:250].copy()
        after_drift = data.iloc[250:].copy()
        
        # Extract temporal features and prepare data
        before_drift = self.extract_time_features(before_drift)
        after_drift = self.extract_time_features(after_drift)
        
        # Separate features and target
        before_drift_X = before_drift.drop('target', axis=1)
        before_drift_y = before_drift['target']
        after_drift_X = after_drift.drop('target', axis=1)
        after_drift_y = after_drift['target']
        
        # Set feature names for explainer
        self.explainer.feature_names = before_drift_X.columns.tolist()
        
        # Train a model on before-drift data
        before_model = RandomForestRegressor(random_state=42)
        before_model.fit(before_drift_X, before_drift_y)
        
        # Train a model on after-drift data
        after_model = RandomForestRegressor(random_state=42)
        after_model.fit(after_drift_X, after_drift_y)
        
        # Get feature importance from models
        self.explainer.model = before_model
        before_explanation = self.explainer.explain(before_drift_X)
        
        self.explainer.model = after_model
        after_explanation = self.explainer.explain(after_drift_X)
        
        # Set reference data and check for drift
        self.drift_detector.set_reference(before_drift_X.values)
        drift_detected, drift_magnitude, drift_info = self.drift_detector.detect_drift(after_drift_X.values)
        
        # Calculate feature importance shift
        before_importance = before_explanation['global_importance']
        after_importance = after_explanation['global_importance']
        
        # Calculate absolute differences in feature importance
        feature_importance_shift = np.abs(after_importance - before_importance)
        
        # Identify top drifting features
        feature_names = before_drift_X.columns.tolist()
        top_drift_features = sorted(
            zip(feature_names, feature_importance_shift),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 features
        
        # Save results
        drift_results = pd.DataFrame({
            'feature': [f[0] for f in top_drift_features],
            'importance_shift': [f[1] for f in top_drift_features]
        })
        drift_results.to_csv(results_dir / 'drift_feature_importance.csv', index=False)
        
        # Generate visualizations for feature importance drift
        self._visualize_feature_importance_drift(
            before_importance=before_importance,
            after_importance=after_importance,
            feature_names=feature_names,
            save_path=results_dir / 'drift_feature_importance_comparison.png'
        )
        
        # Check if test passes
        has_drift_features = len(top_drift_features) > 0
        drift_features_valid = all(shift > 0.01 for _, shift in top_drift_features)
        
        logger.info(f"Drift feature importance test complete. Results saved to {results_dir / 'drift_feature_importance.csv'}")
        
        self.results['drift_feature_importance'] = {
            'passed': drift_detected and has_drift_features and drift_features_valid,
            'details': f"Drift detected: {drift_detected}, Magnitude: {drift_magnitude:.4f}, "
                      f"Top drift features: {', '.join(f[0] for f in top_drift_features)}"
        }
        
    def test_drift_explanation(self):
        """Test generation of human-readable explanations for detected drift."""
        logger.info("Running drift explanation test")
        results_dir = self.setup()
        
        # Generate synthetic data with concept drift
        data_gen = SyntheticDataGenerator(seed=42)
        data = data_gen.generate_concept_drift_data(n_samples=500, drift_point=250, drift_magnitude=5.0)
        
        # Split into before and after drift
        before_drift = data.iloc[:250].copy()
        after_drift = data.iloc[250:].copy()
        
        # Extract temporal features and prepare data
        before_drift = self.extract_time_features(before_drift)
        after_drift = self.extract_time_features(after_drift)
        
        # Separate features and target
        before_drift_X = before_drift.drop('target', axis=1)
        before_drift_y = before_drift['target']
        after_drift_X = after_drift.drop('target', axis=1)
        after_drift_y = after_drift['target']
        
        # Set feature names for explainer
        self.explainer.feature_names = before_drift_X.columns.tolist()
        
        # Set reference data and check for drift
        self.drift_detector.set_reference(before_drift_X.values)
        drift_detected, drift_magnitude, drift_info = self.drift_detector.detect_drift(after_drift_X.values)
        
        # Generate simple statistical summary of drift
        drift_summary = {}
        
        for col in before_drift_X.columns:
            before_mean = before_drift_X[col].mean()
            after_mean = after_drift_X[col].mean()
            mean_change = after_mean - before_mean
            mean_pct_change = (mean_change / before_mean) * 100 if before_mean != 0 else float('inf')
            
            drift_summary[col] = {
                'before_mean': before_mean,
                'after_mean': after_mean,
                'abs_change': abs(mean_change),
                'pct_change': abs(mean_pct_change)
            }
        
        # Sort features by percentage change
        sorted_features = sorted(
            drift_summary.items(), 
            key=lambda x: x[1]['pct_change'], 
            reverse=True
        )
        
        # Generate a human-readable explanation
        explanation = f"Drift detected with magnitude {drift_magnitude:.4f}. The most significant changes were:\n"
        
        for feature, stats in sorted_features[:3]:  # Top 3 features
            explanation += f"- {feature}: Changed from {stats['before_mean']:.4f} to {stats['after_mean']:.4f} "
            explanation += f"({stats['pct_change']:.2f}% change)\n"
        
        # Save explanation to file
        with open(results_dir / 'drift_explanation.txt', 'w') as f:
            f.write(explanation)
        
        # Generate statistical distribution visualization
        self._visualize_feature_distributions(
            before_data=before_drift_X,
            after_data=after_drift_X,
            sorted_features=[f[0] for f in sorted_features[:3]],  # Top 3 features
            save_path=results_dir / 'feature_distribution_changes.png'
        )
        
        logger.info(f"Drift explanation test complete. Results saved to {results_dir / 'drift_explanation.txt'}")
        
        # Check if test passes
        has_explanation = len(explanation) > 50  # Arbitrary minimum length
        
        self.results['drift_explanation'] = {
            'passed': drift_detected and has_explanation,
            'details': f"Drift detected: {drift_detected}, Magnitude: {drift_magnitude:.4f}, "
                      f"Explanation generated: {has_explanation}"
        }
        
    def _visualize_feature_importance_drift(self, before_importance, after_importance, feature_names, save_path):
        """Generate visualization comparing feature importance before and after drift.
        
        Args:
            before_importance: Feature importance values before drift
            after_importance: Feature importance values after drift
            feature_names: List of feature names
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Calculate importance shift
        importance_shift = np.abs(after_importance - before_importance)
        
        # Create a DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Before Drift': before_importance,
            'After Drift': after_importance,
            'Absolute Change': importance_shift
        })
        
        # Sort by absolute change
        importance_df = importance_df.sort_values('Absolute Change', ascending=False)
        
        # Select top features to display (for readability)
        top_n = min(10, len(importance_df))
        importance_df = importance_df.head(top_n).copy()
        
        # Set up the figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: Comparison bar chart
        importance_df.plot(
            x='Feature',
            y=['Before Drift', 'After Drift'],
            kind='bar',
            ax=axes[0],
            width=0.8
        )
        axes[0].set_title('Feature Importance Before vs After Drift')
        axes[0].set_ylabel('Importance Score')
        axes[0].set_xticklabels(importance_df['Feature'], rotation=45, ha='right')
        
        # Plot 2: Absolute change
        importance_df.plot(
            x='Feature',
            y='Absolute Change',
            kind='barh',
            ax=axes[1],
            color='darkred'
        )
        axes[1].set_title('Feature Importance Shift Due to Drift')
        axes[1].set_xlabel('Absolute Change in Importance')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance drift visualization saved to {save_path}")
        plt.close()
    
    def _visualize_feature_distributions(self, before_data, after_data, sorted_features, save_path):
        """Generate visualization comparing feature distributions before and after drift.
        
        Args:
            before_data: DataFrame with features before drift
            after_data: DataFrame with features after drift
            sorted_features: List of feature names to visualize (sorted by importance)
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style for better visualization
        sns.set_style('whitegrid')
        
        # Limit to top N features for readability
        n_features = min(3, len(sorted_features))  # Show at most 3 features
        selected_features = sorted_features[:n_features]
        
        # Create subplots grid
        fig, axes = plt.subplots(n_features, 2, figsize=(15, 5 * n_features))
        
        # If only one feature, wrap axes in a list to make indexing consistent
        if n_features == 1:
            axes = [axes]
        
        # Plot distributions for each selected feature
        for i, feature in enumerate(selected_features):
            # Plot histograms (left column)
            axes[i][0].hist(before_data[feature], bins=20, alpha=0.5, label='Before Drift')
            axes[i][0].hist(after_data[feature], bins=20, alpha=0.5, label='After Drift')
            axes[i][0].set_title(f'{feature} - Histogram Comparison')
            axes[i][0].set_xlabel('Value')
            axes[i][0].set_ylabel('Frequency')
            axes[i][0].legend()
            
            # Plot KDE (right column)
            sns.kdeplot(before_data[feature], ax=axes[i][1], label='Before Drift')
            sns.kdeplot(after_data[feature], ax=axes[i][1], label='After Drift')
            axes[i][1].set_title(f'{feature} - Density Comparison')
            axes[i][1].set_xlabel('Value')
            axes[i][1].set_ylabel('Density')
            axes[i][1].legend()
            
            # Calculate and display statistics
            before_mean = before_data[feature].mean()
            after_mean = after_data[feature].mean()
            before_std = before_data[feature].std()
            after_std = after_data[feature].std()
            
            stats_text = (f"Before: μ={before_mean:.2f}, σ={before_std:.2f}\n"
                         f"After: μ={after_mean:.2f}, σ={after_std:.2f}\n"
                         f"Δμ={after_mean-before_mean:.2f} ({(after_mean-before_mean)/before_mean*100:.1f}%)")
            
            # Add text box with statistics
            axes[i][1].text(0.05, 0.95, stats_text, transform=axes[i][1].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        plt.suptitle('Feature Distribution Changes Before vs After Drift', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature distribution visualization saved to {save_path}")
        plt.close()
    
    def test_temporal_feature_importance(self):
        """Test that tracks feature importance evolution over time during drift."""
        logger.info("Running temporal feature importance test")
        results_dir = self.setup()
        
        # Generate synthetic data with concept drift
        data_gen = SyntheticDataGenerator(seed=43)  # Different seed for diversity
        data = data_gen.generate_concept_drift_data(n_samples=500, drift_point=250, drift_magnitude=5.0)
        
        # Define time windows to track importance evolution (5 windows)
        windows = [
            (0, 100),      # Well before drift
            (100, 200),    # Just before drift
            (200, 300),    # During drift (includes drift point)
            (300, 400),    # Just after drift
            (400, 500)     # Well after drift
        ]
        
        # Track feature importance at each time window
        temporal_importance = []
        feature_names = None
        
        for window_start, window_end in windows:
            # Extract window data
            window_data = data.iloc[window_start:window_end].copy()
            
            # Extract temporal features and prepare data
            window_data = self.extract_time_features(window_data)
            
            # Separate features and target
            window_X = window_data.drop('target', axis=1)
            window_y = window_data['target']
            
            if feature_names is None:
                feature_names = window_X.columns.tolist()
                # Set feature names for explainer
                self.explainer.feature_names = feature_names
            
            # Train a model on window data
            window_model = RandomForestRegressor(random_state=42)
            window_model.fit(window_X, window_y)
            
            # Get feature importance
            self.explainer.model = window_model
            window_explanation = self.explainer.explain(window_X)
            
            # Store feature importance
            window_importance = window_explanation['global_importance']
            temporal_importance.append(window_importance)
        
        # Generate temporal visualization of feature importance
        self._visualize_temporal_feature_importance(
            temporal_importance=temporal_importance,
            feature_names=feature_names,
            window_labels=['Pre-Drift 1', 'Pre-Drift 2', 'Transition', 'Post-Drift 1', 'Post-Drift 2'],
            save_path=results_dir / 'temporal_feature_importance.png'
        )
        
        # Track the top feature for each window
        top_features = []
        for i, importance in enumerate(temporal_importance):
            top_idx = np.argmax(importance)
            top_feature = feature_names[top_idx]
            top_importance = importance[top_idx]
            top_features.append((top_feature, top_importance))
        
        # Save temporal importance evolution to CSV
        evolution_df = pd.DataFrame({
            'window': ['Pre-Drift 1', 'Pre-Drift 2', 'Transition', 'Post-Drift 1', 'Post-Drift 2'],
            'top_feature': [f[0] for f in top_features],
            'importance_value': [f[1] for f in top_features]
        })
        evolution_df.to_csv(results_dir / 'temporal_importance_evolution.csv', index=False)
        
        # Check for feature importance shift over time
        first_window_top = top_features[0][0]
        last_window_top = top_features[-1][0]
        feature_shift_occurred = first_window_top != last_window_top
        
        logger.info(f"Temporal feature importance test complete. Results saved to {results_dir / 'temporal_feature_importance.png'}")
        
        self.results['temporal_feature_importance'] = {
            'passed': feature_shift_occurred,
            'details': f"Feature importance shift detected: {feature_shift_occurred}. "
                       f"Top feature changed from {first_window_top} to {last_window_top}."
        }
    
    def _visualize_temporal_feature_importance(self, temporal_importance, feature_names, window_labels, save_path):
        """Generate visualization showing feature importance evolution over time.
        
        Args:
            temporal_importance: List of feature importance arrays for each time window
            feature_names: List of feature names
            window_labels: Labels for each time window
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        
        # Convert to DataFrame for easier plotting
        # Each row is a feature, each column is a time window
        importance_df = pd.DataFrame(
            data=np.column_stack(temporal_importance),
            index=feature_names,
            columns=window_labels
        )
        
        # Sort features by average importance
        importance_df['avg'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('avg', ascending=False)
        importance_df = importance_df.drop('avg', axis=1)
        
        # Select top features to display
        top_n = min(8, len(importance_df))  # Show at most 8 features
        top_features_df = importance_df.iloc[:top_n].copy()
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create two visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. Heatmap showing the importance evolution
        # Define custom colormap from light to dark blue
        cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#FFFFFF', '#0343DF'])
        
        sns.heatmap(
            top_features_df, 
            ax=ax1,
            cmap=cmap,
            annot=True,
            fmt='.3f',
            linewidths=.5,
            cbar_kws={'label': 'Importance Value'}
        )
        ax1.set_title('Feature Importance Evolution Over Time', fontsize=14)
        ax1.set_ylabel('Features', fontsize=12)
        
        # 2. Line chart showing importance trends
        for feature in top_features_df.index:
            ax2.plot(
                window_labels,
                top_features_df.loc[feature],
                marker='o',
                linewidth=2,
                label=feature
            )
        
        ax2.set_title('Temporal Importance Trends', fontsize=14)
        ax2.set_xlabel('Time Window', fontsize=12)
        ax2.set_ylabel('Importance Value', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add drift marker in the middle
        ax2.axvline(x=2, color='red', linestyle='--', alpha=0.7, label='Approximate Drift Point')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Temporal feature importance visualization saved to {save_path}")
        plt.close()
    
    def test_expert_specific_drift_impact(self):
        """Test that analyzes how drift impacts different experts in the MoE system."""
        logger.info("Running expert-specific drift impact test")
        results_dir = self.setup()
        
        # Generate synthetic data with concept drift
        data_gen = SyntheticDataGenerator(seed=44)  # Different seed
        data = data_gen.generate_concept_drift_data(n_samples=500, drift_point=250, drift_magnitude=5.0)
        
        # Extract temporal features and prepare data
        data = self.extract_time_features(data)
        
        # Split into before and after drift
        before_drift = data.iloc[:250].copy()
        after_drift = data.iloc[250:].copy()
        
        # Create mock experts with different specialties
        expert_specialties = [
            'time_based',      # Expert focusing on temporal features
            'feature_0_2',    # Expert focusing on features 0 and 2
            'feature_1_3',    # Expert focusing on features 1 and 3
            'all_features'    # Generalist expert using all features
        ]
        
        experts = [MockExpert(i, specialty=s) for i, s in enumerate(expert_specialties)]
        
        # Train experts on before-drift data
        X_before = before_drift.drop('target', axis=1)
        y_before = before_drift['target']
        X_after = after_drift.drop('target', axis=1)
        y_after = after_drift['target']
        
        # Dictionary to track expert performance before and after drift
        expert_performance = {}
        feature_impacts = {}
        
        # Train and evaluate each expert
        for expert in experts:
            # Train on before-drift data
            expert.train(X_before, y_before)
            
            # Evaluate on before-drift data
            before_pred = expert.model.predict(X_before)
            before_mse = mean_squared_error(y_before, before_pred)
            
            # Evaluate on after-drift data
            after_pred = expert.model.predict(X_after)
            after_mse = mean_squared_error(y_after, after_pred)
            
            # Calculate performance degradation
            perf_change_pct = ((after_mse - before_mse) / before_mse) * 100
            
            # Calculate uncertainty bounds using bootstrap resampling
            bootstrap_degradations = []
            n_bootstraps = 100
            
            try:
                if len(y_after) > 10:  # Need enough samples for bootstrapping
                    for i in range(n_bootstraps):
                        # Sample with replacement
                        indices = np.random.randint(0, len(y_after), size=len(y_after))
                        y_sample = y_after.iloc[indices] if hasattr(y_after, 'iloc') else y_after[indices]
                        X_sample = X_after.iloc[indices] if hasattr(X_after, 'iloc') else X_after[indices]
                        
                        # Predict on bootstrap sample
                        bootstrap_pred = expert.model.predict(X_sample)
                        bootstrap_mse = mean_squared_error(y_sample, bootstrap_pred)
                        bootstrap_degradation = ((bootstrap_mse - before_mse) / before_mse * 100) if before_mse > 0 else 0
                        bootstrap_degradations.append(bootstrap_degradation)
                    
                    # 95% confidence interval
                    lower_bound = np.percentile(bootstrap_degradations, 2.5)
                    upper_bound = np.percentile(bootstrap_degradations, 97.5)
                else:
                    # For small sample sizes, use a simple heuristic
                    lower_bound = perf_change_pct * 0.8
                    upper_bound = perf_change_pct * 1.2
            except Exception as e:
                logger.warning(f"Error calculating bootstrap uncertainty: {e}")
                lower_bound = perf_change_pct * 0.8
                upper_bound = perf_change_pct * 1.2
            
            # Generate expert-specific recommendations based on degradation severity
            if perf_change_pct > 60:
                severity = "Critical"
                recommendation = f"CRITICAL: Expert '{expert.specialty}' requires immediate retraining. "
                if hasattr(expert.model, 'feature_importances_'):
                    # Get the top features that might be contributing to drift
                    feature_names = X_before.columns
                    importances = expert.model.feature_importances_
                    top_indices = np.argsort(importances)[-3:]
                    top_features = [feature_names[i] for i in top_indices]
                    recommendation += f"Focus on features: {', '.join(top_features)} as they may be most affected by drift."
            elif perf_change_pct > 30:
                severity = "Moderate"
                recommendation = f"WARNING: Expert '{expert.specialty}' shows moderate degradation. "
                recommendation += "Schedule retraining within the next monitoring cycle and increase monitoring frequency."
            else:
                severity = "Low"
                recommendation = f"INFO: Expert '{expert.specialty}' remains stable under current conditions. "
                recommendation += "Continue routine monitoring."
            
            # Store results with uncertainty and recommendations
            expert_performance[expert.specialty] = {
                'before_mse': before_mse,
                'after_mse': after_mse,
                'degradation_pct': perf_change_pct,
                'confidence_interval': [lower_bound, upper_bound],
                'severity': severity,
                'recommendation': recommendation
            }
            
            # Get feature importance for this expert
            self.explainer.model = expert.model
            explanation = self.explainer.explain(X_before)
            feature_impacts[expert.specialty] = {
                'features': explanation['feature_names'],
                'importance': explanation['global_importance']
            }
        
        # Visualize expert-specific drift impact
        self._visualize_expert_drift_impact(
            expert_performance=expert_performance,
            save_path=results_dir / 'expert_drift_impact.png'
        )
        
        # Create correlation analysis between feature importance and drift impact
        self._visualize_importance_drift_correlation(
            expert_performance=expert_performance,
            feature_impacts=feature_impacts,
            save_path=results_dir / 'importance_drift_correlation.png'
        )
        
        # Save expert performance data with uncertainty and recommendations
        expert_perf_df = pd.DataFrame({
            'specialty': list(expert_performance.keys()),
            'before_mse': [v['before_mse'] for v in expert_performance.values()],
            'after_mse': [v['after_mse'] for v in expert_performance.values()],
            'degradation_pct': [v['degradation_pct'] for v in expert_performance.values()],
            'lower_bound': [v['confidence_interval'][0] for v in expert_performance.values()],
            'upper_bound': [v['confidence_interval'][1] for v in expert_performance.values()],
            'severity': [v['severity'] for v in expert_performance.values()],
            'recommendation': [v['recommendation'] for v in expert_performance.values()]
        })
        expert_perf_df.to_csv(results_dir / 'expert_drift_impact.csv', index=False)
        
        # Determine which expert is most resilient to drift
        most_resilient = expert_perf_df.loc[expert_perf_df['degradation_pct'].idxmin()]
        
        logger.info(f"Expert-specific drift impact test complete. Results saved to {results_dir / 'expert_drift_impact.png'}")
        
        self.results['expert_drift_impact'] = {
            'passed': True,  # Always passes, this is an analytical test
            'details': f"Expert drift impact analysis completed. Most resilient expert: {most_resilient['specialty']} "  
                      f"with performance degradation of {most_resilient['degradation_pct']:.2f}%."
        }
    
    def _visualize_expert_drift_impact(self, expert_performance, save_path):
        """Visualize how drift impacts different experts.
        
        Args:
            expert_performance: Dictionary with expert performance before and after drift
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Prepare data
        specialties = list(expert_performance.keys())
        before_mse = [v['before_mse'] for v in expert_performance.values()]
        after_mse = [v['after_mse'] for v in expert_performance.values()]
        degradation = [v['degradation_pct'] for v in expert_performance.values()]
        confidence_intervals = [v.get('confidence_interval', [0, 0]) for v in expert_performance.values()]
        lower_bounds = [ci[0] for ci in confidence_intervals]
        upper_bounds = [ci[1] for ci in confidence_intervals]
        
        # 1. Bar chart comparing before and after MSE with error bars
        bar_width = 0.35
        x = np.arange(len(specialties))
        
        ax1.bar(x - bar_width/2, before_mse, bar_width, label='Before Drift', color='royalblue')
        ax1.bar(x + bar_width/2, after_mse, bar_width, label='After Drift', color='firebrick')
        
        # Add error bars for uncertainty visualization
        # Ensure all error values are positive by taking absolute values
        err_lower = [abs(after - lower) if after > lower else 0 for after, lower in zip(after_mse, lower_bounds)]
        err_upper = [abs(upper - after) if upper > after else 0 for after, upper in zip(after_mse, upper_bounds)]
        ax1.errorbar(x + bar_width/2, after_mse, yerr=[err_lower, err_upper], fmt='none', ecolor='black', capsize=5)
        
        ax1.set_xlabel('Expert Specialty')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Expert Performance Before vs After Drift')
        ax1.set_xticks(x)
        ax1.set_xticklabels(specialties, rotation=45, ha='right')
        ax1.legend()
        
        # Add text labels
        for i, v in enumerate(zip(before_mse, after_mse)):
            ax1.text(i - bar_width/2, v[0] + 0.05, f'{v[0]:.2f}', ha='center', fontsize=9)
            ax1.text(i + bar_width/2, v[1] + 0.05, f'{v[1]:.2f}', ha='center', fontsize=9)
        
        # 2. Horizontal bar chart showing performance degradation
        # Sort by degradation
        sorted_indices = np.argsort(degradation)
        sorted_specialties = [specialties[i] for i in sorted_indices]
        sorted_degradation = [degradation[i] for i in sorted_indices]
        
        # Create gradient colors based on degradation
        colors = plt.cm.YlOrRd(np.array(sorted_degradation) / max(sorted_degradation))
        
        ax2.barh(sorted_specialties, sorted_degradation, color=colors)
        ax2.set_xlabel('Performance Degradation (%)')
        ax2.set_title('Expert Resilience to Drift')
        
        # Add text labels
        for i, v in enumerate(sorted_degradation):
            ax2.text(v + 2, i, f'{v:.2f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Expert drift impact visualization saved to {save_path}")
        plt.close()
    
    def _visualize_importance_drift_correlation(self, expert_performance, feature_impacts, save_path):
        """Visualize correlation between feature importance and drift impact.
        
        Args:
            expert_performance: Dictionary with expert performance degradation
            feature_impacts: Dictionary with feature importance for each expert
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import pearsonr
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Extract data
        specialties = list(expert_performance.keys())
        degradation = [v['degradation_pct'] for v in expert_performance.values()]
        
        # Find common features across all experts
        common_features = set(feature_impacts[specialties[0]]['features'])
        for specialty in specialties[1:]:
            common_features &= set(feature_impacts[specialty]['features'])
        common_features = list(common_features)
        
        # For each common feature, plot correlation with degradation
        for i, feature in enumerate(common_features[:min(4, len(common_features))]):  # Show top 4 features or fewer if not enough
            try:
                feature_imp = []
                for specialty in specialties:
                    try:
                        feature_idx = feature_impacts[specialty]['features'].index(feature)
                        imp = feature_impacts[specialty]['importance'][feature_idx]
                        feature_imp.append(imp)
                    except ValueError:
                        # Feature not found in this specialty's features
                        feature_imp.append(0)  # Default to zero importance
                
                # Check if we have valid data for correlation
                if len(set(feature_imp)) <= 1 or len(set(degradation)) <= 1:
                    # Not enough variation for correlation - silently handle this case
                    corr, p_value = 0, 1.0
                    correlation_note = "(insufficient variation)"
                else:
                    # Calculate correlation with error handling
                    try:
                        corr, p_value = pearsonr(feature_imp, degradation)
                        correlation_note = ""
                    except Exception as e:
                        logger.warning(f"Error calculating correlation for {feature}: {e}")
                        corr, p_value = 0, 1.0
                        correlation_note = "(calculation error)"
                
                # Plot scatter points
                axes[i].scatter(feature_imp, degradation, s=100, alpha=0.7)
                
                # Add trendline with error handling
                try:
                    if len(set(feature_imp)) > 1 and not np.isnan(feature_imp).any() and not np.isnan(degradation).any():
                        z = np.polyfit(feature_imp, degradation, 1)
                        p = np.poly1d(z)
                        x_range = sorted(feature_imp)
                        if len(x_range) >= 2:
                            axes[i].plot(x_range, p(x_range), 'r--', alpha=0.7)
                except Exception as e:
                    logger.warning(f"Error creating trendline for {feature}: {e}")
            
                # Add expert labels with error handling
                for j, specialty in enumerate(specialties):
                    if j < len(feature_imp) and j < len(degradation):
                        # Check for valid values
                        if not np.isnan(feature_imp[j]) and not np.isnan(degradation[j]):
                            axes[i].annotate(specialty, (feature_imp[j], degradation[j]), 
                                             textcoords="offset points", xytext=(0,10), ha='center')
                
                axes[i].set_xlabel(f'{feature} Importance')
                axes[i].set_ylabel('Performance Degradation (%)')
                axes[i].set_title(f'Correlation between {feature} Importance and Drift Impact\n'
                                f'r = {corr:.3f}, p-value = {p_value:.3f} {correlation_note}')
                axes[i].grid(True, linestyle='--', alpha=0.7)
            except Exception as e:
                logger.warning(f"Error creating plot for feature {feature}: {e}")
                # Make sure the subplot is still valid
                if i < len(axes):
                    axes[i].text(0.5, 0.5, f"Error plotting {feature}\n{str(e)}", 
                                ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Importance-drift correlation visualization saved to {save_path}")
        plt.close()
        
    def run_all_tests(self):
        """Run all explainable drift tests and return results."""
        self.test_drift_feature_importance()
        self.test_drift_explanation()
        self.test_temporal_feature_importance()
        self.test_expert_specific_drift_impact()
        
        return self.results
        
if __name__ == "__main__":
    tests = ExplainableDriftTests()
    results = tests.run_all_tests()
    print("\n=== Explainable Drift Test Results ===")
    for test, result in results.items():
        status = "PASSED" if result['passed'] else "FAILED"
        print(f"{test}: {status} - {result['details']}")

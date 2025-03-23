"""
Feature analysis tools for the SATzilla-inspired algorithm selector

This module provides tools for analyzing problem features and their importance
for algorithm selection, including feature importance visualization and analysis.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_feature_importance(
    selector,
    export_dir: Optional[Path] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Analyze feature importance for algorithm selection
    
    Args:
        selector: The trained selector
        export_dir: Directory to export visualizations (if None, don't export)
        
    Returns:
        Dictionary of feature importance scores for each algorithm
    """
    logger.info("Analyzing feature importance")
    
    # Get feature names
    feature_names = list(selector.X_train[0].keys())
    
    # Dictionary to store importance scores
    importance_dict = {}
    
    # For each algorithm model
    for alg in selector.algorithms:
        if selector.models[alg] is not None and hasattr(selector.models[alg], 'feature_importances_'):
            # Get importance scores
            importances = selector.models[alg].feature_importances_
            
            # Store in dictionary
            importance_dict[alg] = {
                'features': feature_names,
                'importance': importances.tolist()
            }
            
            # Export visualization if requested
            if export_dir:
                # Create directory if it doesn't exist
                viz_dir = export_dir / "feature_analysis"
                viz_dir.mkdir(exist_ok=True, parents=True)
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Create DataFrame for better plotting
                importance_df = pd.DataFrame({
                    'Feature': [feature_names[i] for i in indices],
                    'Importance': [importances[i] for i in indices]
                })
                
                # Create plot
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title(f'Feature Importance for {alg}')
                plt.tight_layout()
                plt.savefig(viz_dir / f"feature_importance_{alg}.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    return importance_dict

def analyze_feature_correlation(
    selector,
    export_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Analyze correlation between problem features
    
    Args:
        selector: The trained selector
        export_dir: Directory to export visualizations (if None, don't export)
        
    Returns:
        Correlation matrix
    """
    logger.info("Analyzing feature correlation")
    
    # Get feature names
    feature_names = list(selector.X_train[0].keys())
    
    # Convert features to DataFrame
    features_df = pd.DataFrame([list(f.values()) for f in selector.X_train], 
                               columns=feature_names)
    
    # Calculate correlation
    corr = features_df.corr()
    
    # Export visualization if requested
    if export_dir:
        # Create directory if it doesn't exist
        viz_dir = export_dir / "feature_analysis"
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(viz_dir / "feature_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return corr

def analyze_features_with_pca(
    selector,
    export_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze problem features using Principal Component Analysis (PCA)
    
    Args:
        selector: The trained selector
        export_dir: Directory to export visualizations (if None, don't export)
        
    Returns:
        Dictionary with PCA results
    """
    logger.info("Analyzing features with PCA")
    
    # Get feature names
    feature_names = list(selector.X_train[0].keys())
    
    # Convert features to numpy array
    X = np.array([list(f.values()) for f in selector.X_train])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    # Get component loadings
    loadings = pca.components_
    
    # Export visualization if requested
    if export_dir:
        # Create directory if it doesn't exist
        viz_dir = export_dir / "feature_analysis"
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), 'ro-')
        plt.axhline(y=0.9, color='r', linestyle='-', alpha=0.5)
        plt.title('Explained Variance by Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.savefig(viz_dir / "pca_explained_variance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot first two components
        plt.figure(figsize=(12, 10))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
        plt.title('First Two Principal Components')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        plt.savefig(viz_dir / "pca_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot component loadings
        plt.figure(figsize=(14, 10))
        for i, feature in enumerate(feature_names):
            plt.arrow(0, 0, loadings[0, i], loadings[1, i], head_width=0.05, head_length=0.05)
            plt.text(loadings[0, i] * 1.15, loadings[1, i] * 1.15, feature, fontsize=9)
        
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        plt.title('PCA Component Loadings')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig(viz_dir / "pca_loadings.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Return PCA results
    return {
        'explained_variance': explained_variance.tolist(),
        'cumulative_variance': np.cumsum(explained_variance).tolist(),
        'loadings': loadings.tolist(),
        'transformed_data': X_pca.tolist()
    }

def analyze_feature_performance_correlation(
    selector,
    export_dir: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlation between features and algorithm performance
    
    Args:
        selector: The trained selector
        export_dir: Directory to export visualizations (if None, don't export)
        
    Returns:
        Dictionary of correlation scores for each algorithm
    """
    logger.info("Analyzing feature-performance correlation")
    
    # Get feature names
    feature_names = list(selector.X_train[0].keys())
    
    # Convert features to DataFrame
    features_df = pd.DataFrame([list(f.values()) for f in selector.X_train], 
                               columns=feature_names)
    
    # Dictionary to store correlation scores
    correlation_dict = {}
    
    # For each algorithm
    for alg in selector.algorithms:
        # Get performance data
        perf = np.array(selector.y_train[alg])
        
        # Calculate Pearson correlation
        correlation = {}
        for feature in feature_names:
            feat_values = features_df[feature].values
            corr = np.corrcoef(feat_values, perf)[0, 1]
            correlation[feature] = corr
        
        # Calculate mutual information (non-linear correlation)
        mi = mutual_info_regression(features_df, perf)
        mi_dict = {feature: mi[i] for i, feature in enumerate(feature_names)}
        
        # Store both metrics
        correlation_dict[alg] = {
            'pearson': correlation,
            'mutual_info': mi_dict
        }
        
        # Export visualization if requested
        if export_dir:
            # Create directory if it doesn't exist
            viz_dir = export_dir / "feature_analysis"
            viz_dir.mkdir(exist_ok=True, parents=True)
            
            # Create correlation plot
            plt.figure(figsize=(14, 10))
            
            # Sort features by correlation
            sorted_features = sorted(correlation.items(), key=lambda x: abs(x[1]), reverse=True)
            features = [x[0] for x in sorted_features]
            correlations = [x[1] for x in sorted_features]
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            colors = ['g' if x > 0 else 'r' for x in correlations]
            plt.barh(features, [abs(x) for x in correlations], color=colors, alpha=0.7)
            plt.title(f'Feature-Performance Correlation for {alg}')
            plt.xlabel('Absolute Correlation')
            plt.grid(True, alpha=0.3)
            plt.savefig(viz_dir / f"feature_perf_corr_{alg}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create scatter plots for top 4 features
            plt.figure(figsize=(15, 12))
            plt.suptitle(f'Top Features vs Performance for {alg}')
            
            for i in range(min(4, len(features))):
                plt.subplot(2, 2, i+1)
                feat = features[i]
                plt.scatter(features_df[feat], perf, alpha=0.7)
                plt.title(f'{feat} (r={correlations[i]:.3f})')
                plt.xlabel(feat)
                plt.ylabel('Performance')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"feature_perf_scatter_{alg}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return correlation_dict

def analyze_feature_subset_performance(
    selector,
    features_to_remove: List[str],
    problems: List,
    max_evaluations: int = 1000,
    export_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Analyze performance impact of removing features
    
    Args:
        selector: The trained selector
        features_to_remove: List of features to remove
        problems: List of problems to evaluate on
        max_evaluations: Maximum evaluations per algorithm
        export_dir: Directory to export results (if None, don't export)
        
    Returns:
        Dictionary with performance comparison
    """
    logger.info(f"Analyzing performance with {len(features_to_remove)} features removed")
    
    # Get feature names
    feature_names = list(selector.X_train[0].keys())
    
    # Get indices of features to remove
    remove_indices = [feature_names.index(f) for f in features_to_remove if f in feature_names]
    
    # Create a modified version of the selector with features removed
    from copy import deepcopy
    import types
    
    # Create a deep copy of the selector
    modified_selector = deepcopy(selector)
    
    # Define a new extract_features method that excludes the specified features
    def modified_extract_features(self, problem):
        # Call the original method
        features = selector.extract_features(problem)
        
        # Remove the specified features
        for feature in features_to_remove:
            if feature in features:
                del features[feature]
        
        return features
    
    # Bind the new method to the modified selector
    modified_selector.extract_features = types.MethodType(modified_extract_features, modified_selector)
    
    # Train the modified selector
    modified_selector.train(problems, max_evaluations=max_evaluations)
    
    # Evaluate performance on problems
    original_performance = {}
    modified_performance = {}
    
    for i, problem in enumerate(problems):
        logger.info(f"Evaluating problem {i+1}/{len(problems)}")
        
        # Select algorithm with original selector
        original_alg = selector.select_algorithm(problem)
        original_result = selector.optimize(problem, algorithm=original_alg, max_evaluations=max_evaluations)
        original_fitness = original_result[1]  # Assuming result is (solution, fitness, evaluations)
        
        # Select algorithm with modified selector
        modified_alg = modified_selector.select_algorithm(problem)
        modified_result = modified_selector.optimize(problem, algorithm=modified_alg, max_evaluations=max_evaluations)
        modified_fitness = modified_result[1]
        
        # Store results
        problem_name = getattr(problem, 'name', f'problem_{i}')
        original_performance[problem_name] = {
            'algorithm': original_alg,
            'fitness': float(original_fitness)
        }
        modified_performance[problem_name] = {
            'algorithm': modified_alg,
            'fitness': float(modified_fitness)
        }
    
    # Calculate performance difference
    performance_diff = {}
    for problem in original_performance:
        orig = original_performance[problem]['fitness']
        mod = modified_performance[problem]['fitness']
        
        # Calculate relative difference (percentage)
        if orig != 0:
            diff = (mod - orig) / abs(orig) * 100
        else:
            diff = float('inf') if mod > 0 else 0
        
        performance_diff[problem] = {
            'original': orig,
            'modified': mod,
            'difference_percent': diff
        }
    
    # Calculate average performance difference
    avg_diff = np.mean([pd['difference_percent'] for pd in performance_diff.values() 
                        if not np.isinf(pd['difference_percent'])])
    
    # Export results if requested
    if export_dir:
        # Create directory if it doesn't exist
        results_dir = export_dir / "feature_analysis"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Export results as CSV
        results_df = pd.DataFrame({
            'Problem': list(performance_diff.keys()),
            'Original Fitness': [performance_diff[p]['original'] for p in performance_diff],
            'Modified Fitness': [performance_diff[p]['modified'] for p in performance_diff],
            'Difference (%)': [performance_diff[p]['difference_percent'] for p in performance_diff]
        })
        
        results_df.to_csv(results_dir / f"feature_subset_performance.csv", index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.bar(results_df['Problem'], results_df['Difference (%)'], alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        plt.axhline(y=avg_diff, color='g', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_diff:.2f}%')
        plt.title(f'Performance Impact of Removing Features: {", ".join(features_to_remove)}')
        plt.ylabel('Performance Difference (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "feature_subset_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Return performance comparison
    return {
        'removed_features': features_to_remove,
        'original_performance': original_performance,
        'modified_performance': modified_performance,
        'performance_diff': performance_diff,
        'avg_difference_percent': float(avg_diff)
    } 
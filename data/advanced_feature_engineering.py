"""
Advanced Feature Engineering Module

This module provides advanced feature engineering capabilities for the MoE framework,
extending the preprocessing pipeline with sophisticated feature creation methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats as stats

from data.preprocessing_pipeline import PreprocessingOperation


class PolynomialFeatureGenerator(PreprocessingOperation):
    """Generate polynomial and interaction features."""
    
    def __init__(self, degree: int = 2, include_bias: bool = False, 
                 interaction_only: bool = False, exclude_cols: List[str] = None):
        """Initialize the polynomial feature generator.
        
        Args:
            degree: The degree of the polynomial features
            include_bias: Whether to include a bias column
            interaction_only: Whether to only include interaction features
            exclude_cols: Columns to exclude from feature generation
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.exclude_cols = exclude_cols or []
        self.poly = None
        self.feature_names = []
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the polynomial feature generator to the data."""
        # Filter numeric columns and exclude specified columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        if not feature_cols:
            return
            
        # Initialize and fit the polynomial feature generator
        self.poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        self.poly.fit(data[feature_cols])
        
        # Store feature names for later use
        input_features = feature_cols
        self.feature_names = self.poly.get_feature_names_out(input_features)
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate polynomial features from the data."""
        result = data.copy()
        
        if self.poly is None:
            return result
            
        # Filter numeric columns and exclude specified columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        if not feature_cols:
            return result
            
        # Generate polynomial features
        poly_features = self.poly.transform(data[feature_cols])
        
        # Convert to DataFrame with proper column names
        poly_df = pd.DataFrame(
            poly_features, 
            columns=self.feature_names,
            index=data.index
        )
        
        # Remove original features if they exist in the polynomial features
        if self.include_bias:
            # Skip the first column which is the bias term (constant 1)
            poly_df = poly_df.iloc[:, 1:]
        
        # Add polynomial features to the result
        for col in poly_df.columns:
            if col not in feature_cols:  # Skip original features
                result[f"poly_{col}"] = poly_df[col]
                
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'degree': self.degree,
            'include_bias': self.include_bias,
            'interaction_only': self.interaction_only,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'degree' in params:
            self.degree = params['degree']
        if 'include_bias' in params:
            self.include_bias = params['include_bias']
        if 'interaction_only' in params:
            self.interaction_only = params['interaction_only']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        if self.poly is None:
            return {}
            
        # Calculate the number of new features created
        new_features_count = len(transformed_data.columns) - len(data.columns)
        
        # Calculate the variance explained by the new features
        if new_features_count > 0:
            original_variance = data.select_dtypes(include=['int64', 'float64']).var().sum()
            transformed_variance = transformed_data.select_dtypes(include=['int64', 'float64']).var().sum()
            variance_ratio = transformed_variance / original_variance if original_variance > 0 else 1.0
        else:
            variance_ratio = 1.0
            
        return {
            'new_features_count': new_features_count,
            'variance_ratio': variance_ratio
        }


class DimensionalityReducer(PreprocessingOperation):
    """Reduce dimensionality of the data using various techniques."""
    
    def __init__(self, method: str = 'pca', n_components: int = 2, 
                 kernel: str = 'rbf', exclude_cols: List[str] = None,
                 prefix: str = 'reduced'):
        """Initialize the dimensionality reducer.
        
        Args:
            method: Method for dimensionality reduction. Options: 'pca', 'kernel_pca', 'tsne'
            n_components: Number of components to keep
            kernel: Kernel to use for kernel PCA
            exclude_cols: Columns to exclude from dimensionality reduction
            prefix: Prefix for the new feature names
        """
        self.method = method
        self.n_components = n_components
        self.kernel = kernel
        self.exclude_cols = exclude_cols or []
        self.prefix = prefix
        self.reducer = None
        self.feature_names = []
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the dimensionality reducer to the data."""
        # Filter numeric columns and exclude specified columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        if not feature_cols or len(feature_cols) <= self.n_components:
            return
            
        # Initialize the appropriate reducer
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
        elif self.method == 'kernel_pca':
            self.reducer = KernelPCA(n_components=self.n_components, kernel=self.kernel)
        elif self.method == 'tsne':
            self.reducer = TSNE(n_components=self.n_components)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.method}")
            
        # Fit the reducer
        self.reducer.fit(data[feature_cols])
        
        # Generate feature names
        self.feature_names = [f"{self.prefix}_{self.method}_{i}" for i in range(self.n_components)]
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Reduce dimensionality of the data."""
        result = data.copy()
        
        if self.reducer is None:
            return result
            
        # Filter numeric columns and exclude specified columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        if not feature_cols or len(feature_cols) <= self.n_components:
            return result
            
        # Transform the data
        reduced_data = self.reducer.transform(data[feature_cols])
        
        # Add reduced features to the result
        for i in range(self.n_components):
            result[self.feature_names[i]] = reduced_data[:, i]
            
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'method': self.method,
            'n_components': self.n_components,
            'kernel': self.kernel,
            'exclude_cols': self.exclude_cols,
            'prefix': self.prefix
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'method' in params:
            self.method = params['method']
        if 'n_components' in params:
            self.n_components = params['n_components']
        if 'kernel' in params:
            self.kernel = params['kernel']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        if 'prefix' in params:
            self.prefix = params['prefix']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        if self.reducer is None or self.method not in ['pca', 'kernel_pca']:
            return {}
            
        metrics = {}
        
        # For PCA, we can get explained variance ratio
        if self.method == 'pca' and hasattr(self.reducer, 'explained_variance_ratio_'):
            metrics['explained_variance_ratio'] = sum(self.reducer.explained_variance_ratio_)
            
        return metrics


class StatisticalFeatureGenerator(PreprocessingOperation):
    """Generate statistical features from the data."""
    
    def __init__(self, window_sizes: List[int] = None, exclude_cols: List[str] = None,
                 group_by: str = None, stats: List[str] = None):
        """Initialize the statistical feature generator.
        
        Args:
            window_sizes: List of window sizes for rolling statistics
            exclude_cols: Columns to exclude from feature generation
            group_by: Column to group by for grouped statistics
            stats: List of statistics to compute. Options: 'mean', 'std', 'min', 'max', 'skew', 'kurt'
        """
        self.window_sizes = window_sizes or [5, 10, 20]
        self.exclude_cols = exclude_cols or []
        self.group_by = group_by
        self.stats = stats or ['mean', 'std', 'min', 'max']
        self.stats_funcs = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'skew': stats.skew,
            'kurt': stats.kurtosis
        }
        self.group_stats = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the statistical feature generator to the data."""
        # If group_by is specified, compute group statistics
        if self.group_by and self.group_by in data.columns:
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            feature_cols = [col for col in numeric_cols if col not in self.exclude_cols and col != self.group_by]
            
            if not feature_cols:
                return
                
            # Compute group statistics
            for stat in self.stats:
                if stat in self.stats_funcs:
                    self.group_stats[stat] = data.groupby(self.group_by)[feature_cols].agg(self.stats_funcs[stat])
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate statistical features from the data."""
        result = data.copy()
        
        # Filter numeric columns and exclude specified columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        if not feature_cols:
            return result
            
        # Generate rolling window statistics
        for window in self.window_sizes:
            for col in feature_cols:
                for stat in self.stats:
                    if stat in self.stats_funcs:
                        result[f"{col}_roll_{window}_{stat}"] = data[col].rolling(window=window, min_periods=1).agg(self.stats_funcs[stat])
        
        # Add group statistics if available
        if self.group_by and self.group_by in data.columns and self.group_stats:
            for stat in self.stats:
                if stat in self.group_stats:
                    for col in self.group_stats[stat].columns:
                        result[f"{col}_group_{stat}"] = data[self.group_by].map(self.group_stats[stat][col])
                        
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'window_sizes': self.window_sizes,
            'exclude_cols': self.exclude_cols,
            'group_by': self.group_by,
            'stats': self.stats
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'window_sizes' in params:
            self.window_sizes = params['window_sizes']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
        if 'group_by' in params:
            self.group_by = params['group_by']
        if 'stats' in params:
            self.stats = params['stats']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        # Calculate the number of new features created
        new_features_count = len(transformed_data.columns) - len(data.columns)
        
        return {
            'new_features_count': new_features_count
        }


class ClusterFeatureGenerator(PreprocessingOperation):
    """Generate cluster-based features."""
    
    def __init__(self, n_clusters: int = 3, method: str = 'kmeans', 
                 exclude_cols: List[str] = None):
        """Initialize the cluster feature generator.
        
        Args:
            n_clusters: Number of clusters to create
            method: Clustering method to use. Options: 'kmeans'
            exclude_cols: Columns to exclude from clustering
        """
        self.n_clusters = n_clusters
        self.method = method
        self.exclude_cols = exclude_cols or []
        self.clusterer = None
        self.cluster_centers = None
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the cluster feature generator to the data."""
        # Filter numeric columns and exclude specified columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        if not feature_cols:
            return
            
        # Initialize and fit the clusterer
        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.clusterer.fit(data[feature_cols])
            self.cluster_centers = self.clusterer.cluster_centers_
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate cluster-based features."""
        result = data.copy()
        
        if self.clusterer is None:
            return result
            
        # Filter numeric columns and exclude specified columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        feature_cols = [col for col in numeric_cols if col not in self.exclude_cols]
        
        if not feature_cols:
            return result
            
        # Predict clusters
        clusters = self.clusterer.predict(data[feature_cols])
        result['cluster_id'] = clusters
        
        # Calculate distance to each cluster center
        for i in range(self.n_clusters):
            distances = np.sqrt(((data[feature_cols].values - self.cluster_centers[i]) ** 2).sum(axis=1))
            result[f'distance_to_cluster_{i}'] = distances
            
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'n_clusters': self.n_clusters,
            'method': self.method,
            'exclude_cols': self.exclude_cols
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'n_clusters' in params:
            self.n_clusters = params['n_clusters']
        if 'method' in params:
            self.method = params['method']
        if 'exclude_cols' in params:
            self.exclude_cols = params['exclude_cols']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        if self.clusterer is None:
            return {}
            
        # For KMeans, we can get inertia (sum of squared distances to nearest centroid)
        if self.method == 'kmeans' and hasattr(self.clusterer, 'inertia_'):
            return {
                'inertia': self.clusterer.inertia_,
                'new_features_count': self.n_clusters + 1  # cluster_id + distances to centers
            }
            
        return {}

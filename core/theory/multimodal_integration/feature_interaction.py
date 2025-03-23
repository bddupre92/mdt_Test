"""Feature Interaction Analysis for Multimodal Integration.

This module provides tools for analyzing interactions between features from different modalities,
particularly for physiological signals and contextual information relevant to migraine prediction.

Key features:
1. Cross-modality correlation analysis
2. Multivariate feature importance assessment
3. Interaction detection algorithms
4. Dimensionality reduction for multimodal data
5. Visualization of feature relationships
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

from core.theory.multimodal_integration import FeatureInteractionAnalyzer, ModalityData
from .. import base

class CrossModalInteractionAnalyzer(FeatureInteractionAnalyzer):
    """Analyzer for interactions between features from different modalities."""
    
    def __init__(self,
                 interaction_method: str = 'correlation',
                 significance_level: float = 0.05,
                 min_correlation: float = 0.1,
                 random_state: Optional[int] = None):
        """Initialize cross-modal interaction analyzer.
        
        Args:
            interaction_method: Method for analyzing interactions
                - 'correlation': Pearson/Spearman correlation
                - 'mutual_info': Mutual information
                - 'granger': Granger causality
                - 'transfer_entropy': Transfer entropy
            significance_level: P-value threshold for significant interactions
            min_correlation: Minimum correlation coefficient to consider
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If an invalid interaction method is specified
        """
        # Validate interaction method
        valid_methods = ['correlation', 'mutual_info', 'granger', 'transfer_entropy']
        if interaction_method not in valid_methods:
            raise ValueError(f"Invalid interaction method: {interaction_method}. "
                           f"Must be one of: {', '.join(valid_methods)}")
            
        self.interaction_method = interaction_method
        self.significance_level = significance_level
        self.min_correlation = min_correlation
        
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize storage for interaction results
        self.interaction_matrix = None
        self.p_values = None
        self.feature_names = None
        self.modality_mapping = None
    
    def analyze_interactions(self, 
                           *data_sources: Union[np.ndarray, ModalityData],
                           feature_names: Optional[List[str]] = None,
                           **kwargs) -> Dict[str, Any]:
        """Analyze interactions between features from multiple data sources.
        
        Args:
            *data_sources: Data sources to analyze
            feature_names: Names of features (optional)
            **kwargs: Additional parameters
                - method_params: Parameters specific to interaction method
                - temporal_window: Window size for temporal analysis
                - lag_order: Maximum lag order for Granger/TE analysis
        
        Returns:
            Dictionary containing:
                - interaction_matrix: Matrix of interaction strengths
                - p_values: Matrix of statistical significance
                - significant_pairs: List of significant feature pairs
                - graph: NetworkX graph of interactions
                - clusters: Hierarchical clustering results
        """
        # Extract data arrays and validate
        data_arrays = []
        modality_labels = []
        
        for i, source in enumerate(data_sources):
            if isinstance(source, ModalityData):
                data_arrays.append(source.data)
                modality_labels.extend([source.modality_type] * source.data.shape[1])
            else:
                data_arrays.append(source)
                modality_labels.extend([f"modality_{i}"] * source.shape[1])
        
        # Combine all features into a single matrix
        X = np.hstack(data_arrays)
        n_features = X.shape[1]
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        self.feature_names = feature_names
        self.modality_mapping = dict(zip(feature_names, modality_labels))
        
        # Compute interactions based on selected method
        if self.interaction_method == 'correlation':
            interaction_matrix, p_values = self._compute_correlations(X)
        elif self.interaction_method == 'mutual_info':
            interaction_matrix = self._compute_mutual_information(X)
            p_values = self._compute_mi_significance(X)
        elif self.interaction_method == 'granger':
            interaction_matrix, p_values = self._compute_granger_causality(
                X, lag_order=kwargs.get('lag_order', 1)
            )
        elif self.interaction_method == 'transfer_entropy':
            interaction_matrix = self._compute_transfer_entropy(
                X, temporal_window=kwargs.get('temporal_window', 10)
            )
            p_values = np.zeros_like(interaction_matrix)  # TE doesn't provide p-values
        else:
            raise ValueError(f"Unknown interaction method: {self.interaction_method}")
        
        # Store results
        self.interaction_matrix = interaction_matrix
        self.p_values = p_values
        
        # Find significant interactions
        significant_pairs = self._find_significant_pairs(
            interaction_matrix, p_values, feature_names
        )
        
        # Create interaction graph
        graph = self._create_interaction_graph(
            interaction_matrix, p_values, feature_names, modality_labels
        )
        
        # Perform hierarchical clustering
        clusters = self._cluster_features(interaction_matrix)
        
        return {
            'interaction_matrix': interaction_matrix,
            'p_values': p_values,
            'significant_pairs': significant_pairs,
            'graph': graph,
            'clusters': clusters,
            'feature_names': feature_names,
            'modality_mapping': self.modality_mapping
        }
    
    def _compute_correlations(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute correlation matrix and p-values.
        
        Args:
            X: Data matrix (samples × features)
            
        Returns:
            Tuple containing:
                - Correlation matrix
                - Matrix of p-values
        """
        n_features = X.shape[1]
        corr_matrix = np.zeros((n_features, n_features))
        p_values = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr, p_value = stats.pearsonr(X[:, i], X[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                p_values[i, j] = p_value
                p_values[j, i] = p_value
                
        np.fill_diagonal(corr_matrix, 1.0)
        np.fill_diagonal(p_values, 0.0)
        
        return corr_matrix, p_values
    
    def _compute_mutual_information(self, X: np.ndarray) -> np.ndarray:
        """Compute mutual information between features.
        
        Args:
            X: Data matrix (samples × features)
            
        Returns:
            Matrix of mutual information values
        """
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = self._estimate_mutual_information(X[:, i], X[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
                
        np.fill_diagonal(mi_matrix, 1.0)
        return mi_matrix
    
    def _compute_mi_significance(self, X: np.ndarray, n_permutations: int = 1000) -> np.ndarray:
        """Compute significance of mutual information values using permutation tests.
        
        Args:
            X: Data matrix (samples × features)
            n_permutations: Number of permutations for significance testing
            
        Returns:
            Matrix of p-values
        """
        n_features = X.shape[1]
        p_values = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                p_value = self._permutation_test(
                    X[:, i], X[:, j], n_permutations
                )
                p_values[i, j] = p_value
                p_values[j, i] = p_value
                
        np.fill_diagonal(p_values, 0.0)
        return p_values
    
    def _compute_granger_causality(self, 
                                 X: np.ndarray,
                                 lag_order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Granger causality between features.
        
        Args:
            X: Data matrix (samples × features)
            lag_order: Number of lags to consider
            
        Returns:
            Tuple containing:
                - Matrix of Granger causality F-statistics
                - Matrix of p-values
        """
        n_features = X.shape[1]
        gc_matrix = np.zeros((n_features, n_features))
        p_values = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    f_stat, p_value = self._granger_test(
                        X[:, i], X[:, j], lag_order
                    )
                    gc_matrix[i, j] = f_stat
                    p_values[i, j] = p_value
                    
        return gc_matrix, p_values
    
    def _compute_transfer_entropy(self, 
                                X: np.ndarray,
                                temporal_window: int = 10) -> np.ndarray:
        """Compute transfer entropy between features.
        
        Args:
            X: Data matrix (samples × features)
            temporal_window: Window size for temporal analysis
            
        Returns:
            Matrix of transfer entropy values
        """
        n_features = X.shape[1]
        te_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    te = self._estimate_transfer_entropy(
                        X[:, i], X[:, j], temporal_window
                    )
                    te_matrix[i, j] = te
                    
        return te_matrix
    
    def _find_significant_pairs(self,
                              interaction_matrix: np.ndarray,
                              p_values: np.ndarray,
                              feature_names: List[str]) -> List[Dict[str, Any]]:
        """Find significant feature interactions.
        
        Args:
            interaction_matrix: Matrix of interaction strengths
            p_values: Matrix of p-values
            feature_names: List of feature names
            
        Returns:
            List of dictionaries containing significant pairs and their properties
        """
        significant_pairs = []
        n_features = len(feature_names)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if (p_values[i, j] < self.significance_level and 
                    abs(interaction_matrix[i, j]) > self.min_correlation):
                    
                    pair_info = {
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'modality1': self.modality_mapping[feature_names[i]],
                        'modality2': self.modality_mapping[feature_names[j]],
                        'interaction_strength': interaction_matrix[i, j],
                        'p_value': p_values[i, j]
                    }
                    significant_pairs.append(pair_info)
        
        return significant_pairs
    
    def _create_interaction_graph(self,
                                interaction_matrix: np.ndarray,
                                p_values: np.ndarray,
                                feature_names: List[str],
                                modality_labels: List[str]) -> nx.Graph:
        """Create a NetworkX graph of feature interactions.
        
        Args:
            interaction_matrix: Matrix of interaction strengths
            p_values: Matrix of p-values
            feature_names: List of feature names
            modality_labels: List of modality labels
            
        Returns:
            NetworkX graph representing feature interactions
        """
        G = nx.Graph()
        
        # Add nodes with attributes
        for i, (name, modality) in enumerate(zip(feature_names, modality_labels)):
            G.add_node(name, modality=modality)
        
        # Add edges for significant interactions
        n_features = len(feature_names)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if (p_values[i, j] < self.significance_level and 
                    abs(interaction_matrix[i, j]) > self.min_correlation):
                    
                    G.add_edge(
                        feature_names[i],
                        feature_names[j],
                        weight=abs(interaction_matrix[i, j]),
                        p_value=p_values[i, j]
                    )
        
        return G
    
    def _cluster_features(self, interaction_matrix: np.ndarray) -> Dict[str, Any]:
        """Perform hierarchical clustering of features based on interactions.
        
        Args:
            interaction_matrix: Matrix of interaction strengths
            
        Returns:
            Dictionary containing clustering results
        """
        # Convert interaction matrix to distance matrix
        distances = 1 - np.abs(interaction_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(
            squareform(distances), method='complete'
        )
        
        # Get cluster assignments at different levels
        n_clusters_range = range(2, min(6, len(self.feature_names)))
        cluster_assignments = {}
        
        for n_clusters in n_clusters_range:
            labels = hierarchy.fcluster(
                linkage_matrix, n_clusters, criterion='maxclust'
            )
            cluster_assignments[n_clusters] = labels
        
        return {
            'linkage_matrix': linkage_matrix,
            'cluster_assignments': cluster_assignments,
            'feature_names': self.feature_names
        }
    
    def visualize_interactions(self, 
                             interaction_results: Dict[str, Any],
                             **kwargs) -> Dict[str, Any]:
        """Visualize feature interactions.
        
        Args:
            interaction_results: Results from analyze_interactions
            **kwargs: Additional visualization parameters
                - plot_type: Type of visualization
                - fig_size: Figure size
                - color_map: Color map for plotting
                
        Returns:
            Dictionary containing visualization objects
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plot_type = kwargs.get('plot_type', 'heatmap')
        fig_size = kwargs.get('fig_size', (10, 8))
        cmap = kwargs.get('color_map', 'coolwarm')
        
        visualizations = {}
        
        if plot_type == 'heatmap' or plot_type == 'all':
            # Create heatmap of interactions
            plt.figure(figsize=fig_size)
            sns.heatmap(
                interaction_results['interaction_matrix'],
                xticklabels=interaction_results['feature_names'],
                yticklabels=interaction_results['feature_names'],
                cmap=cmap,
                center=0,
                annot=True
            )
            plt.title('Feature Interaction Heatmap')
            visualizations['heatmap'] = plt.gcf()
            
        if plot_type == 'network' or plot_type == 'all':
            # Create network visualization
            plt.figure(figsize=fig_size)
            G = interaction_results['graph']
            pos = nx.spring_layout(G)
            
            # Draw nodes colored by modality
            modalities = nx.get_node_attributes(G, 'modality')
            unique_modalities = set(modalities.values())
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_modalities)))
            color_map = dict(zip(unique_modalities, colors))
            
            nx.draw(G, pos,
                   node_color=[color_map[modalities[node]] for node in G.nodes()],
                   with_labels=True,
                   node_size=1000,
                   font_size=8)
            
            plt.title('Feature Interaction Network')
            visualizations['network'] = plt.gcf()
            
        if plot_type == 'dendrogram' or plot_type == 'all':
            # Create dendrogram
            plt.figure(figsize=fig_size)
            hierarchy.dendrogram(
                interaction_results['clusters']['linkage_matrix'],
                labels=interaction_results['feature_names'],
                leaf_rotation=90
            )
            plt.title('Feature Clustering Dendrogram')
            visualizations['dendrogram'] = plt.gcf()
            
        return visualizations
    
    def _estimate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate mutual information between two variables using KDE.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Estimated mutual information
        """
        from sklearn.neighbors import KernelDensity
        
        # Standardize variables
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        
        # Estimate densities
        xy = np.vstack([x, y]).T
        kde_joint = KernelDensity(kernel='gaussian').fit(xy)
        kde_x = KernelDensity(kernel='gaussian').fit(x.reshape(-1, 1))
        kde_y = KernelDensity(kernel='gaussian').fit(y.reshape(-1, 1))
        
        # Compute MI using samples
        n_samples = len(x)
        mi_samples = []
        
        for i in range(n_samples):
            log_px = kde_x.score_samples(x[i].reshape(1, -1))
            log_py = kde_y.score_samples(y[i].reshape(1, -1))
            log_pxy = kde_joint.score_samples(xy[i].reshape(1, -1))
            mi_samples.append(log_pxy - log_px - log_py)
            
        return np.mean(mi_samples)
    
    def _permutation_test(self, 
                         x: np.ndarray,
                         y: np.ndarray,
                         n_permutations: int) -> float:
        """Perform permutation test for mutual information significance.
        
        Args:
            x: First variable
            y: Second variable
            n_permutations: Number of permutations
            
        Returns:
            P-value from permutation test
        """
        observed_mi = self._estimate_mutual_information(x, y)
        permuted_mis = []
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            mi_perm = self._estimate_mutual_information(x, y_perm)
            permuted_mis.append(mi_perm)
            
        p_value = np.mean(np.array(permuted_mis) >= observed_mi)
        return p_value
    
    def _granger_test(self,
                      x: np.ndarray,
                      y: np.ndarray,
                      lag_order: int) -> Tuple[float, float]:
        """Perform Granger causality test between two variables.
        
        Args:
            x: First variable (potential cause)
            y: Second variable (potential effect)
            lag_order: Number of lags to consider
            
        Returns:
            Tuple containing:
                - F-statistic
                - P-value
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Prepare data
        data = np.column_stack([y, x])  # y first as it's the dependent variable
        
        # Perform Granger test
        test_results = grangercausalitytests(data, maxlag=lag_order, verbose=False)
        
        # Extract results for the specified lag order
        f_stat = test_results[lag_order][0]['ssr_ftest'][0]
        p_value = test_results[lag_order][0]['ssr_ftest'][1]
        
        return f_stat, p_value
    
    def _estimate_transfer_entropy(self,
                                 x: np.ndarray,
                                 y: np.ndarray,
                                 temporal_window: int) -> float:
        """Estimate transfer entropy from x to y.
        
        Args:
            x: Source variable
            y: Target variable
            temporal_window: Window size for temporal analysis
            
        Returns:
            Estimated transfer entropy
        """
        # Prepare time-lagged variables
        x_past = x[:-1]
        y_past = y[:-1]
        y_present = y[1:]
        
        # Estimate probabilities using KDE
        from sklearn.neighbors import KernelDensity
        
        # Joint state space
        xyz = np.vstack([x_past, y_past, y_present]).T
        xy = np.vstack([x_past, y_past]).T
        yz = np.vstack([y_past, y_present]).T
        
        # Fit KDE models
        kde_xyz = KernelDensity(kernel='gaussian').fit(xyz)
        kde_xy = KernelDensity(kernel='gaussian').fit(xy)
        kde_yz = KernelDensity(kernel='gaussian').fit(yz)
        kde_y = KernelDensity(kernel='gaussian').fit(y_past.reshape(-1, 1))
        
        # Compute TE using samples
        te_samples = []
        n_samples = len(x_past)
        
        for i in range(n_samples):
            log_pxyz = kde_xyz.score_samples(xyz[i].reshape(1, -1))
            log_pxy = kde_xy.score_samples(xy[i].reshape(1, -1))
            log_pyz = kde_yz.score_samples(yz[i].reshape(1, -1))
            log_py = kde_y.score_samples(y_past[i].reshape(1, -1))
            
            te_samples.append(log_pxyz + log_py - log_pxy - log_pyz)
            
        return np.mean(te_samples)

class MultimodalFeatureSelector:
    """Feature selection for multimodal data based on interaction analysis."""
    
    def __init__(self,
                 n_features: Optional[int] = None,
                 selection_method: str = 'interaction_strength',
                 random_state: Optional[int] = None):
        """Initialize multimodal feature selector.
        
        Args:
            n_features: Number of features to select
            selection_method: Method for feature selection
                - 'interaction_strength': Based on interaction strength
                - 'clustering': Based on cluster representatives
                - 'graph_centrality': Based on network centrality
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.selection_method = selection_method
        
        if random_state is not None:
            np.random.seed(random_state)
            
        self.selected_features_ = None
        self.feature_scores_ = None
    
    def fit(self,
            interaction_results: Dict[str, Any],
            X: Optional[np.ndarray] = None) -> 'MultimodalFeatureSelector':
        """Fit the feature selector.
        
        Args:
            interaction_results: Results from CrossModalInteractionAnalyzer
            X: Original feature matrix (optional)
            
        Returns:
            self
        """
        if self.selection_method == 'interaction_strength':
            self._select_by_interaction_strength(interaction_results)
        elif self.selection_method == 'clustering':
            self._select_by_clustering(interaction_results)
        elif self.selection_method == 'graph_centrality':
            self._select_by_graph_centrality(interaction_results)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
            
        return self
    
    def _select_by_interaction_strength(self, interaction_results: Dict[str, Any]):
        """Select features based on overall interaction strength."""
        interaction_matrix = np.abs(interaction_results['interaction_matrix'])
        feature_importance = np.sum(interaction_matrix, axis=1)
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        feature_names = interaction_results['feature_names']
        
        if self.n_features is None:
            self.n_features = len(feature_names)
            
        self.selected_features_ = [feature_names[i] for i in sorted_idx[:self.n_features]]
        self.feature_scores_ = {
            feature_names[i]: feature_importance[i] for i in sorted_idx[:self.n_features]
        }
    
    def _select_by_clustering(self, interaction_results: Dict[str, Any]):
        """Select features based on cluster representatives."""
        clusters = interaction_results['clusters']
        linkage = clusters['linkage_matrix']
        feature_names = clusters['feature_names']
        
        if self.n_features is None:
            self.n_features = len(feature_names) // 2
            
        # Get cluster assignments
        labels = hierarchy.fcluster(linkage, self.n_features, criterion='maxclust')
        
        # Select representative from each cluster
        selected_features = []
        feature_scores = {}
        
        for cluster_id in range(1, self.n_features + 1):
            cluster_members = [i for i, label in enumerate(labels) if label == cluster_id]
            
            if cluster_members:
                # Select member with highest average interaction strength
                interaction_strengths = np.sum(
                    np.abs(interaction_results['interaction_matrix'][cluster_members, :]),
                    axis=1
                )
                representative_idx = cluster_members[np.argmax(interaction_strengths)]
                
                selected_features.append(feature_names[representative_idx])
                feature_scores[feature_names[representative_idx]] = np.max(interaction_strengths)
        
        self.selected_features_ = selected_features
        self.feature_scores_ = feature_scores
    
    def _select_by_graph_centrality(self, interaction_results: Dict[str, Any]):
        """Select features based on graph centrality measures."""
        G = interaction_results['graph']
        
        # Compute different centrality measures
        degree_cent = nx.degree_centrality(G)
        between_cent = nx.betweenness_centrality(G)
        close_cent = nx.closeness_centrality(G)
        
        # Combine centrality measures
        feature_scores = {}
        for node in G.nodes():
            feature_scores[node] = (
                degree_cent[node] +
                between_cent[node] +
                close_cent[node]
            ) / 3.0
            
        # Sort features by combined centrality
        sorted_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if self.n_features is None:
            self.n_features = len(G.nodes()) // 2
            
        self.selected_features_ = [f[0] for f in sorted_features[:self.n_features]]
        self.feature_scores_ = dict(sorted_features[:self.n_features]) 
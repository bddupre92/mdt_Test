"""Missing Data Handling for Multimodal Integration.

This module provides tools for handling missing data in multimodal time series,
particularly for physiological signals and contextual information relevant to migraine prediction.

Key features:
1. Multiple imputation techniques
2. Pattern analysis for missingness
3. Uncertainty quantification for imputed values
4. Missing data simulation and validation
5. Adaptive imputation strategies
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
# Required before importing IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import softmax

from core.theory.multimodal_integration import MissingDataHandler, ModalityData
from .. import base

class MultimodalMissingDataHandler(MissingDataHandler):
    """Handler for missing data in multimodal time series."""
    
    def __init__(self,
                 imputation_method: str = 'auto',
                 max_missing_ratio: float = 0.3,
                 min_samples_present: int = 10,
                 random_state: Optional[int] = None):
        """Initialize missing data handler.
        
        Args:
            imputation_method: Method for imputing missing values
                - 'auto': Automatically select best method
                - 'interpolation': Time-based interpolation
                - 'knn': K-nearest neighbors imputation
                - 'mice': Multiple imputation by chained equations
                - 'pattern': Pattern-based imputation
            max_missing_ratio: Maximum ratio of missing values allowed
            min_samples_present: Minimum number of present samples required
            random_state: Random seed for reproducibility
        """
        # Validate imputation method
        valid_methods = ['auto', 'interpolation', 'knn', 'mice', 'pattern']
        if imputation_method not in valid_methods:
            raise ValueError(f"Invalid imputation method: {imputation_method}. "
                           f"Must be one of: {', '.join(valid_methods)}")
        
        self.imputation_method = imputation_method
        self.max_missing_ratio = max_missing_ratio
        self.min_samples_present = min_samples_present
        
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize storage for missing patterns
        self.missing_patterns = None
        self.imputation_stats = None
        self.uncertainty_estimates = None
    
    def detect_missing_patterns(self, 
                              *data_sources: Union[np.ndarray, ModalityData]) -> Dict[str, Any]:
        """Detect patterns of missing data across multiple sources.
        
        Args:
            *data_sources: Data sources to analyze
            
        Returns:
            Dictionary containing:
                - missing_patterns: Binary matrix of missing patterns
                - pattern_frequencies: Frequency of each pattern
                - modality_stats: Missing data statistics per modality
                - temporal_stats: Temporal statistics of missingness
        """
        # Extract data arrays and create missing indicators
        data_arrays = []
        modality_labels = []
        missing_indicators = []
        
        for i, source in enumerate(data_sources):
            if isinstance(source, ModalityData):
                data = source.data
                modality = source.modality_type
            else:
                data = source
                modality = f"modality_{i}"
                
            data_arrays.append(data)
            modality_labels.extend([modality] * data.shape[1])
            missing_indicators.append(np.isnan(data))
        
        # Combine all indicators
        X_missing = np.hstack(missing_indicators)
        n_samples, n_features = X_missing.shape
        
        # Analyze missing patterns
        unique_patterns, pattern_counts = np.unique(X_missing, axis=0, return_counts=True)
        pattern_frequencies = pattern_counts / n_samples
        
        # Compute modality-specific statistics
        modality_stats = {}
        current_idx = 0
        
        for i, source in enumerate(data_arrays):
            n_features_source = source.shape[1]
            missing_ratio = np.mean(missing_indicators[i])
            missing_per_feature = np.mean(missing_indicators[i], axis=0)
            
            modality_stats[modality_labels[current_idx]] = {
                'missing_ratio': missing_ratio,
                'missing_per_feature': missing_per_feature,
                'n_features': n_features_source
            }
            current_idx += n_features_source
        
        # Analyze temporal patterns
        temporal_stats = self._analyze_temporal_patterns(X_missing)
        
        # Store results
        self.missing_patterns = {
            'patterns': unique_patterns,
            'frequencies': pattern_frequencies,
            'modality_stats': modality_stats,
            'temporal_stats': temporal_stats
        }
        
        return self.missing_patterns
    
    def impute(self,
               *data_sources: Union[np.ndarray, ModalityData],
               **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
        """Impute missing values in multiple data sources.
        
        Args:
            *data_sources: Data sources with missing values
            **kwargs: Additional parameters
                - temporal_info: Temporal information for time-based methods
                - modality_weights: Importance weights for each modality
                - n_imputations: Number of imputations for multiple imputation
                
        Returns:
            Tuple containing:
                - List of imputed data arrays
                - Uncertainty estimates for imputed values
                
        Raises:
            ValueError: If an invalid imputation method is specified
        """
        # Extract parameters
        temporal_info = kwargs.get('temporal_info', None)
        modality_weights = kwargs.get('modality_weights', None)
        n_imputations = kwargs.get('n_imputations', 5)
        
        # Extract data arrays
        data_arrays = []
        modality_labels = []
        
        for i, source in enumerate(data_sources):
            if isinstance(source, ModalityData):
                data_arrays.append(source.data)
                modality_labels.extend([source.modality_type] * source.data.shape[1])
            else:
                data_arrays.append(source)
                modality_labels.extend([f"modality_{i}"] * source.shape[1])
        
        # Combine all data
        X = np.hstack(data_arrays)
        n_samples, n_features = X.shape
        
        # Validate input data
        if n_samples == 0 or n_features == 0:
            raise ValueError("Empty data array provided")
        
        # Select imputation method
        if self.imputation_method == 'auto':
            method = self._select_imputation_method(X, temporal_info)
        else:
            method = self.imputation_method
            
        # Validate method
        valid_methods = ['auto', 'interpolation', 'knn', 'mice', 'pattern']
        if method not in valid_methods:
            raise ValueError(f"Invalid imputation method: {method}. "
                           f"Must be one of: {', '.join(valid_methods)}")
        
        # Perform imputation based on selected method
        if method == 'interpolation':
            if temporal_info is None:
                # If no temporal info provided, generate a sequence
                temporal_info = np.arange(X.shape[0])
                
            X_imputed, uncertainties = self._interpolation_imputation(
                X, temporal_info
            )
        elif method == 'knn':
            X_imputed, uncertainties = self._knn_imputation(
                X, n_neighbors=kwargs.get('n_neighbors', 5)
            )
        elif method == 'mice':
            X_imputed, uncertainties = self._mice_imputation(
                X, n_imputations=n_imputations
            )
        elif method == 'pattern':
            X_imputed, uncertainties = self._pattern_based_imputation(
                X, modality_weights
            )
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        # Split imputed data back into original arrays
        imputed_arrays = []
        current_idx = 0
        
        for source in data_arrays:
            n_features_source = source.shape[1]
            imputed_source = X_imputed[:, current_idx:current_idx + n_features_source]
            imputed_arrays.append(imputed_source)
            current_idx += n_features_source
        
        # Store imputation statistics
        self.imputation_stats = {
            'method_used': method,
            'n_imputed_values': np.sum(np.isnan(X)),
            'imputation_quality': self._assess_imputation_quality(X, X_imputed)
        }
        
        self.uncertainty_estimates = uncertainties
        
        return imputed_arrays, uncertainties
    
    def _analyze_temporal_patterns(self, X_missing: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal patterns in missing data.
        
        Args:
            X_missing: Binary matrix indicating missing values
            
        Returns:
            Dictionary containing temporal statistics
        """
        n_samples, n_features = X_missing.shape
        
        # Compute run lengths of missing values
        run_lengths = []
        for j in range(n_features):
            runs = self._find_runs(X_missing[:, j])
            run_lengths.extend(runs)
        
        # Analyze periodicity
        periodicity = self._detect_periodicity(X_missing)
        
        # Compute temporal correlation of missingness
        temporal_correlation = np.corrcoef(X_missing.T)
        
        return {
            'run_lengths': {
                'mean': np.mean(run_lengths),
                'std': np.std(run_lengths),
                'max': np.max(run_lengths)
            },
            'periodicity': periodicity,
            'temporal_correlation': temporal_correlation
        }
    
    def _find_runs(self, x: np.ndarray) -> List[int]:
        """Find lengths of consecutive True values in boolean array.
        
        Args:
            x: Boolean array
            
        Returns:
            List of run lengths
        """
        # Pad array to handle runs at the beginning and end
        padded = np.hstack([[False], x, [False]])
        
        # Find run boundaries
        run_starts = np.where(padded[1:] & ~padded[:-1])[0]
        run_ends = np.where(~padded[1:] & padded[:-1])[0]
        
        # Calculate run lengths
        run_lengths = run_ends - run_starts
        
        return run_lengths.tolist()
    
    def _detect_periodicity(self, X_missing: np.ndarray) -> Dict[str, Any]:
        """Detect periodic patterns in missing data.
        
        Args:
            X_missing: Binary matrix indicating missing values
            
        Returns:
            Dictionary containing periodicity information
        """
        from scipy.signal import periodogram
        
        # Compute periodogram for each feature
        periodicities = []
        dominant_frequencies = []
        
        for j in range(X_missing.shape[1]):
            f, pxx = periodogram(X_missing[:, j].astype(float))
            
            # Find dominant frequency
            dominant_idx = np.argmax(pxx[1:]) + 1  # Skip zero frequency
            dominant_freq = f[dominant_idx]
            
            periodicities.append({
                'frequencies': f,
                'power': pxx,
                'dominant_frequency': dominant_freq
            })
            dominant_frequencies.append(dominant_freq)
        
        return {
            'feature_periodicities': periodicities,
            'dominant_frequencies': dominant_frequencies,
            'mean_frequency': np.mean(dominant_frequencies)
        }
    
    def _select_imputation_method(self,
                                X: np.ndarray,
                                temporal_info: Optional[np.ndarray] = None) -> str:
        """Automatically select best imputation method.
        
        Args:
            X: Data matrix with missing values
            temporal_info: Temporal information if available
            
        Returns:
            Selected imputation method
        """
        missing_ratio = np.mean(np.isnan(X))
        n_samples, n_features = X.shape
        
        # Check if temporal information is available and useful
        if temporal_info is not None and self._is_regular_temporal(temporal_info):
            return 'interpolation'
        
        # For high-dimensional data with moderate missing ratio, use MICE
        if n_features > 10 and missing_ratio < 0.3:
            return 'mice'
        
        # For data with clear patterns, use pattern-based imputation
        if self.missing_patterns is not None and len(self.missing_patterns['patterns']) < 10:
            return 'pattern'
        
        # Default to KNN for other cases
        return 'knn'
    
    def _is_regular_temporal(self, temporal_info: np.ndarray) -> bool:
        """Check if temporal information is regularly spaced.
        
        Args:
            temporal_info: Array of timestamps
            
        Returns:
            True if temporal spacing is regular
        """
        if len(temporal_info) < 2:
            return False
        
        # Compute time differences
        time_diffs = np.diff(temporal_info)
        
        # Check if differences are approximately constant
        return np.std(time_diffs) / np.mean(time_diffs) < 0.1
    
    def _interpolation_imputation(self,
                                X: np.ndarray,
                                temporal_info: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform time-based interpolation.
        
        Args:
            X: Data matrix with missing values
            temporal_info: Temporal information
            
        Returns:
            Tuple containing:
                - Imputed data matrix
                - Uncertainty estimates
        """
        n_samples, n_features = X.shape
        X_imputed = X.copy()
        uncertainties = np.zeros_like(X)
        
        for j in range(n_features):
            # Find missing indices
            missing_idx = np.isnan(X[:, j])
            if not np.any(missing_idx):
                continue
                
            # Find valid data points
            valid_idx = ~missing_idx
            if np.sum(valid_idx) < 2:
                continue
                
            # Perform interpolation
            f = interp1d(
                temporal_info[valid_idx],
                X[valid_idx, j],
                kind='cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Impute missing values
            X_imputed[missing_idx, j] = f(temporal_info[missing_idx])
            
            # Estimate uncertainty based on distance to nearest valid points
            for idx in np.where(missing_idx)[0]:
                nearest_valid = np.abs(temporal_info[valid_idx] - temporal_info[idx])
                uncertainties[idx, j] = np.min(nearest_valid)
        
        return X_imputed, uncertainties
    
    def _knn_imputation(self,
                       X: np.ndarray,
                       n_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Perform K-nearest neighbors imputation.
        
        Args:
            X: Data matrix with missing values
            n_neighbors: Number of neighbors to use
            
        Returns:
            Tuple containing:
                - Imputed data matrix
                - Uncertainty estimates
        """
        # Initialize KNN imputer
        imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights='distance'
        )
        
        # Perform imputation
        X_imputed = imputer.fit_transform(X)
        
        # Estimate uncertainties using standard deviation of nearby points
        uncertainties = np.zeros_like(X)
        missing_mask = np.isnan(X)
        
        # For each feature
        for j in range(X.shape[1]):
            # Find non-missing indices for this feature
            valid_idx = ~np.isnan(X[:, j])
            if np.sum(valid_idx) < n_neighbors:
                continue
                
            # For each sample with missing value
            for i in np.where(missing_mask[:, j])[0]:
                # Find nearest neighbors using Euclidean distance
                dists = np.sqrt(np.sum((X_imputed[valid_idx] - X_imputed[i])**2, axis=1))
                neighbor_idx = np.argsort(dists)[:n_neighbors]
                
                # Compute weighted standard deviation
                weights = 1 / (dists[neighbor_idx] + 1e-8)
                neighbor_values = X_imputed[valid_idx][neighbor_idx, j]
                uncertainties[i, j] = np.sqrt(
                    np.average((neighbor_values - X_imputed[i, j])**2, weights=weights)
                )
        
        return X_imputed, uncertainties
    
    def _mice_imputation(self,
                        X: np.ndarray,
                        n_imputations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Perform multiple imputation by chained equations (MICE).
        
        Args:
            X: Data matrix with missing values
            n_imputations: Number of imputations
            
        Returns:
            Tuple containing:
                - Imputed data matrix
                - Uncertainty estimates
        """
        # Initialize imputer
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100),
            max_iter=10,
            random_state=42
        )
        
        # Perform multiple imputations
        imputations = []
        for _ in range(n_imputations):
            X_imputed = imputer.fit_transform(X)
            imputations.append(X_imputed)
        
        # Combine imputations
        X_combined = np.mean(imputations, axis=0)
        uncertainties = np.std(imputations, axis=0)
        
        return X_combined, uncertainties
    
    def _pattern_based_imputation(self,
                                X: np.ndarray,
                                modality_weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Impute missing values using pattern-based approaches.
        
        Args:
            X: Data matrix with missing values
            modality_weights: Importance weights for each modality
            
        Returns:
            Tuple containing:
                - Imputed data matrix
                - Uncertainty estimates
        """
        # Create a copy of the input data
        X_imputed = X.copy()
        uncertainties = np.zeros_like(X)
        
        # Get missing mask
        missing_mask = np.isnan(X)
        
        # If no missing values, return original data
        if not np.any(missing_mask):
            return X, uncertainties
        
        # Get complete samples (rows without missing values)
        complete_rows = ~np.any(missing_mask, axis=1)
        
        # If no complete rows, use mean imputation as fallback
        if not np.any(complete_rows):
            # Mean imputation as fallback
            col_means = np.nanmean(X, axis=0)
            col_stds = np.nanstd(X, axis=0)
            
            # Impute missing values with column means
            for j in range(X.shape[1]):
                missing_in_col = missing_mask[:, j]
                X_imputed[missing_in_col, j] = col_means[j]
                uncertainties[missing_in_col, j] = col_stds[j]
                
            return X_imputed, uncertainties
        
        # Get complete rows
        X_complete = X[complete_rows]
        
        # For each row with missing values
        for i in range(X.shape[0]):
            if np.any(missing_mask[i]):
                # Get missing pattern for this row
                pattern = missing_mask[i]
                
                # Find similar samples based on observed values
                similarities = self._compute_similarity(
                    X_imputed[i], X_complete, pattern, modality_weights
                )
                
                # Get top-k similar samples
                k = min(5, len(similarities))
                top_indices = np.argsort(similarities)[:k]
                top_similarities = similarities[top_indices]
                
                # Normalize similarities to sum to 1
                weights = softmax(-top_similarities)
                
                # For each missing feature
                for j in np.where(pattern)[0]:
                    # Get values from similar samples
                    values = X_complete[top_indices, j]
                    
                    # Weighted average for imputation
                    X_imputed[i, j] = np.sum(values * weights)
                    
                    # Weighted standard deviation for uncertainty
                    if k > 1:
                        uncertainties[i, j] = np.sqrt(
                            np.sum(weights * (values - X_imputed[i, j])**2)
                        )
                    else:
                        # If only one similar sample, use column std
                        uncertainties[i, j] = np.nanstd(X[:, j])
        
        # Final check for any remaining NaN values
        if np.any(np.isnan(X_imputed)):
            # Fill any remaining NaNs with column means
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape[1]):
                missing_in_col = np.isnan(X_imputed[:, j])
                if np.any(missing_in_col):
                    X_imputed[missing_in_col, j] = col_means[j]
                    uncertainties[missing_in_col, j] = np.nanstd(X[:, j])
        
        return X_imputed, uncertainties
    
    def _compute_similarity(self,
                          x: np.ndarray,
                          X_complete: np.ndarray,
                          pattern: np.ndarray,
                          modality_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Compute similarity between a sample and complete cases.
        
        Args:
            x: Sample with missing values
            X_complete: Matrix of complete cases
            pattern: Missing pattern for the sample
            modality_weights: Importance weights for each modality
            
        Returns:
            Array of similarity scores
        """
        # Use only observed features for similarity computation
        observed = ~pattern
        
        if modality_weights is not None:
            # Apply modality weights
            weights = np.ones(len(observed))
            current_idx = 0
            
            for modality, weight in modality_weights.items():
                n_features = self.missing_patterns['modality_stats'][modality]['n_features']
                weights[current_idx:current_idx + n_features] = weight
                current_idx += n_features
                
            observed_weights = weights[observed]
        else:
            observed_weights = None
        
        # Compute weighted Euclidean distance
        diff = x[observed].reshape(1, -1) - X_complete[:, observed]
        if observed_weights is not None:
            diff = diff * np.sqrt(observed_weights)
            
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        # Convert distances to similarities
        similarities = 1 / (distances + 1e-8)
        
        return similarities
    
    def _assess_imputation_quality(self,
                                 X_original: np.ndarray,
                                 X_imputed: np.ndarray) -> Dict[str, float]:
        """Assess quality of imputation.
        
        Args:
            X_original: Original data matrix with missing values
            X_imputed: Imputed data matrix
            
        Returns:
            Dictionary containing quality metrics
        """
        # Find originally observed values
        observed = ~np.isnan(X_original)
        
        # Compute metrics only on observed values
        mse = np.mean((X_original[observed] - X_imputed[observed])**2)
        mae = np.mean(np.abs(X_original[observed] - X_imputed[observed]))
        
        # Compute correlation between observed values
        corr = np.corrcoef(X_original[observed], X_imputed[observed])[0, 1]
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': corr
        }
    
    def simulate_missing_data(self,
                            X: np.ndarray,
                            missing_ratio: float = 0.2,
                            pattern_type: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """Simulate missing data for testing imputation methods.
        
        Args:
            X: Complete data matrix
            missing_ratio: Ratio of values to make missing
            pattern_type: Type of missing pattern
                - 'random': Missing completely at random
                - 'temporal': Time-dependent missing
                - 'structured': Structured missing patterns
            
        Returns:
            Tuple containing:
                - Data matrix with simulated missing values
                - Binary matrix indicating which values were made missing
        """
        n_samples, n_features = X.shape
        X_missing = X.copy()
        missing_mask = np.zeros_like(X, dtype=bool)
        
        if pattern_type == 'random':
            # Randomly select values to make missing
            n_missing = int(missing_ratio * X.size)
            idx = np.random.choice(X.size, n_missing, replace=False)
            row_idx, col_idx = np.unravel_index(idx, X.shape)
            X_missing[row_idx, col_idx] = np.nan
            missing_mask[row_idx, col_idx] = True
            
        elif pattern_type == 'temporal':
            # Create time-dependent missing patterns
            for j in range(n_features):
                # Generate random segments to make missing
                segment_length = np.random.randint(5, 20)
                n_segments = int(missing_ratio * n_samples / segment_length)
                
                for _ in range(n_segments):
                    start = np.random.randint(0, n_samples - segment_length)
                    X_missing[start:start + segment_length, j] = np.nan
                    missing_mask[start:start + segment_length, j] = True
                    
        elif pattern_type == 'structured':
            # Create structured missing patterns
            n_patterns = 3
            pattern_length = n_features // n_patterns
            
            for i in range(0, n_samples, n_patterns):
                pattern = np.random.randint(n_patterns)
                start_feature = pattern * pattern_length
                end_feature = start_feature + pattern_length
                
                if np.random.random() < missing_ratio:
                    X_missing[i, start_feature:end_feature] = np.nan
                    missing_mask[i, start_feature:end_feature] = True
                    
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
        return X_missing, missing_mask 
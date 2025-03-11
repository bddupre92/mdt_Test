"""Reliability Modeling for Multimodal Integration.

This module provides tools for assessing and modeling the reliability of different data sources,
particularly for physiological signals and contextual information relevant to migraine prediction.

Key features:
1. Source-specific confidence scoring
2. Temporal reliability assessment
3. Conflict resolution between sources
4. Adaptive weighting mechanisms
5. Quality metrics for data sources
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from core.theory.multimodal_integration import ReliabilityModel, ModalityData
from .. import base

class MultimodalReliabilityModel(ReliabilityModel):
    """Model for assessing and tracking reliability of multimodal data sources."""
    
    def __init__(self,
                 reliability_method: str = 'auto',
                 noise_threshold: float = 0.3,
                 outlier_sensitivity: float = 0.05,
                 temporal_decay: bool = True,
                 temporal_window: int = 100,
                 random_state: Optional[int] = None):
        """Initialize reliability model.
        
        Args:
            reliability_method: Method for assessing reliability
                - 'auto': Automatically select best method
                - 'signal_quality': Based on signal quality metrics
                - 'statistical': Statistical outlier detection
                - 'historical': Based on historical reliability
                - 'ensemble': Combine multiple methods
            noise_threshold: Threshold for noise level
            outlier_sensitivity: Sensitivity for outlier detection
            temporal_decay: Whether to apply temporal decay to reliability
            temporal_window: Window size for temporal analysis
            random_state: Random seed for reproducibility
        """
        # Validate reliability method
        valid_methods = [
            'auto', 'signal_quality', 'statistical', 'historical', 'ensemble',
            'adaptive', 'temporal', 'conflict'
        ]
        if reliability_method not in valid_methods:
            raise ValueError(f"Invalid reliability method: {reliability_method}. "
                           f"Must be one of: {', '.join(valid_methods)}")
            
        self.reliability_method = reliability_method
        self.noise_threshold = noise_threshold
        self.outlier_sensitivity = outlier_sensitivity
        self.temporal_decay = temporal_decay
        self.temporal_window = temporal_window
        self.min_confidence = 0.1  # Minimum confidence threshold
        
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize storage for reliability tracking
        self.reliability_scores = {}
        self.temporal_trends = {}
        self.conflict_history = {}
        self.quality_metrics = {}
        
        # To store historical reliability
        self.modality_reliability_history = {}
    
    def assess_reliability(self,
                         *data_sources: Union[np.ndarray, ModalityData],
                         **kwargs) -> Dict[str, float]:
        """Assess reliability of multiple data sources.
        
        Args:
            *data_sources: Data sources to assess
            **kwargs: Additional parameters
                - temporal_info: Temporal information for time-based methods
                - reference_data: Reference data for validation
                - quality_thresholds: Thresholds for quality metrics
                
        Returns:
            Dictionary mapping source identifiers to reliability scores (0-1)
            
        Raises:
            ValueError: If no data sources are provided or temporal_info length doesn't match data
        """
        # Extract parameters
        temporal_info = kwargs.get('temporal_info', None)
        reference_data = kwargs.get('reference_data', None)
        quality_thresholds = kwargs.get('quality_thresholds', None)
        
        # Extract data arrays and modality information
        data_arrays = []
        modality_labels = []
        
        for i, source in enumerate(data_sources):
            if isinstance(source, ModalityData):
                data_arrays.append(source.data)
                modality_labels.append(source.modality_type)
            else:
                data_arrays.append(source)
                modality_labels.append(f"modality_{i}")
        
        # Validate input data
        if not data_arrays:
            raise ValueError("No data sources provided")
            
        # Validate temporal info
        if temporal_info is not None:
            # Check if any data arrays have incorrect length
            for i, data in enumerate(data_arrays):
                if len(data) != len(temporal_info):
                    raise ValueError(
                        f"Temporal info length ({len(temporal_info)}) does not match "
                        f"data array length ({len(data)}) for modality {modality_labels[i]}"
                    )
        
        # Compute reliability based on selected method
        if self.reliability_method == 'auto':
            reliability_scores = self._auto_select_method(
                data_arrays, modality_labels, temporal_info
            )
        elif self.reliability_method == 'signal_quality':
            reliability_scores = self._signal_quality_reliability(
                data_arrays, modality_labels, temporal_info
            )
        elif self.reliability_method == 'statistical':
            reliability_scores = self._statistical_outlier_detection(
                data_arrays, modality_labels, temporal_info
            )
        elif self.reliability_method == 'historical':
            reliability_scores = self._historical_reliability(
                data_arrays, modality_labels, temporal_info
            )
        elif self.reliability_method == 'ensemble':
            reliability_scores = self._ensemble_reliability(
                data_arrays, modality_labels, **kwargs
            )
        elif self.reliability_method == 'adaptive':
            reliability_scores = self._adaptive_reliability(
                data_arrays, modality_labels, temporal_info
            )
        elif self.reliability_method == 'temporal':
            reliability_scores = self._temporal_reliability(
                data_arrays, modality_labels, temporal_info
            )
        elif self.reliability_method == 'conflict':
            reliability_scores = self._conflict_based_reliability(
                data_arrays, modality_labels
            )
        else:
            raise ValueError(f"Unknown reliability method: {self.reliability_method}")
        
        # Update reliability tracking
        self._update_reliability_tracking(reliability_scores, temporal_info)
        
        # Compute quality metrics
        self._compute_quality_metrics(
            data_arrays, modality_labels, reference_data, quality_thresholds
        )
        
        return reliability_scores
    
    def update_reliability(self,
                         reliability_scores: Dict[str, float],
                         new_evidence: Dict[str, Any]) -> Dict[str, float]:
        """Update reliability scores based on new evidence.
        
        Args:
            reliability_scores: Current reliability scores
            new_evidence: New evidence to consider
                - prediction_errors: Prediction errors per source
                - conflict_indicators: Indicators of source conflicts
                - quality_updates: Updated quality metrics
                
        Returns:
            Updated reliability scores
        """
        # Extract new evidence
        prediction_errors = new_evidence.get('prediction_errors', {})
        conflict_indicators = new_evidence.get('conflict_indicators', {})
        quality_updates = new_evidence.get('quality_updates', {})
        
        # Update scores based on prediction errors
        if prediction_errors:
            reliability_scores = self._update_from_errors(
                reliability_scores, prediction_errors
            )
        
        # Update scores based on conflicts
        if conflict_indicators:
            reliability_scores = self._update_from_conflicts(
                reliability_scores, conflict_indicators
            )
        
        # Update scores based on quality metrics
        if quality_updates:
            reliability_scores = self._update_from_quality(
                reliability_scores, quality_updates
            )
        
        # Ensure minimum confidence
        for source in reliability_scores:
            reliability_scores[source] = max(
                self.noise_threshold,
                reliability_scores[source]
            )
        
        return reliability_scores
    
    def _auto_select_method(self,
                           data_arrays: List[np.ndarray],
                           modality_labels: List[str],
                           temporal_info: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute reliability scores using an ensemble of methods.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            temporal_info: Temporal information if available
            
        Returns:
            Dictionary of reliability scores
        """
        # Compute reliability using different methods
        signal_quality_scores = self._signal_quality_reliability(
            data_arrays, modality_labels, temporal_info
        )
        
        statistical_scores = self._statistical_outlier_detection(
            data_arrays, modality_labels, temporal_info
        )
        
        historical_scores = self._historical_reliability(
            data_arrays, modality_labels, temporal_info
        )
        
        # Combine scores using weighted average
        reliability_scores = {}
        for modality in modality_labels:
            # Create weighted scores list
            scores = [
                (0.3, signal_quality_scores[modality]),
                (0.3, statistical_scores[modality]),
                (0.4, historical_scores[modality])
            ]
            
            # Calculate weighted sum
            reliability_scores[modality] = sum(weight * score for weight, score in scores)
        
        return reliability_scores
    
    def _temporal_reliability(self,
                            data_arrays: List[np.ndarray],
                            modality_labels: List[str],
                            temporal_info: np.ndarray) -> Dict[str, float]:
        """Compute time-based reliability scores.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            temporal_info: Temporal information
            
        Returns:
            Dictionary of reliability scores
        """
        reliability_scores = {}
        
        for i, (data, modality) in enumerate(zip(data_arrays, modality_labels)):
            # Compute temporal metrics
            temporal_consistency = self._compute_temporal_consistency(
                data, temporal_info
            )
            
            # Analyze temporal patterns
            if data.shape[0] > self.temporal_window:
                # Compute rolling statistics
                rolling_mean = pd.DataFrame(data).rolling(
                    window=self.temporal_window, min_periods=1
                ).mean().values
                rolling_std = pd.DataFrame(data).rolling(
                    window=self.temporal_window, min_periods=1
                ).std().values
                
                # Detect anomalous temporal patterns
                temporal_anomalies = np.abs(
                    (data - rolling_mean) / (rolling_std + 1e-8)
                ) > 3
                temporal_quality = 1 - np.mean(temporal_anomalies)
            else:
                temporal_quality = 1.0
            
            # Combine temporal metrics
            reliability_scores[modality] = 0.6 * temporal_consistency + 0.4 * temporal_quality
        
        return reliability_scores
    
    def _conflict_based_reliability(self,
                                  data_arrays: List[np.ndarray],
                                  modality_labels: List[str]) -> Dict[str, float]:
        """Compute reliability scores based on conflicts between sources.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            
        Returns:
            Dictionary of reliability scores
        """
        n_sources = len(data_arrays)
        conflict_matrix = np.zeros((n_sources, n_sources))
        
        # Compute pairwise conflicts
        for i in range(n_sources):
            for j in range(i + 1, n_sources):
                conflict_score = self._compute_source_conflict(
                    data_arrays[i], data_arrays[j]
                )
                conflict_matrix[i, j] = conflict_score
                conflict_matrix[j, i] = conflict_score
        
        # Convert conflicts to reliability scores
        reliability_scores = {}
        for i, modality in enumerate(modality_labels):
            # Use average inverse conflict as reliability
            conflicts = conflict_matrix[i, :]
            reliability = 1 / (np.mean(conflicts) + 1e-8)
            
            # Normalize to [0, 1]
            reliability = (reliability - np.min(reliability)) / (
                np.max(reliability) - np.min(reliability) + 1e-8
            )
            
            reliability_scores[modality] = reliability
        
        return reliability_scores
    
    def _ensemble_reliability(self,
                            data_arrays: List[np.ndarray],
                            modality_labels: List[str],
                            **kwargs) -> Dict[str, float]:
        """Compute reliability scores using an ensemble of methods.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of reliability scores
        """
        # Compute reliability using different methods
        adaptive_scores = self._adaptive_reliability(
            data_arrays, modality_labels, kwargs.get('temporal_info')
        )
        
        temporal_scores = self._temporal_reliability(
            data_arrays, modality_labels, kwargs.get('temporal_info', np.arange(len(data_arrays[0])))
        )
        
        conflict_scores = self._conflict_based_reliability(
            data_arrays, modality_labels
        )
        
        # Combine scores using weighted average
        reliability_scores = {}
        for modality in modality_labels:
            scores = [
                (0.4, adaptive_scores[modality]),
                (0.3, temporal_scores[modality]),
                (0.3, conflict_scores[modality])
            ]
            
            reliability_scores[modality] = sum(w * s for w, s in scores)
        
        return reliability_scores
    
    def _compute_temporal_consistency(self,
                                   data: np.ndarray,
                                   temporal_info: np.ndarray) -> float:
        """Compute temporal consistency score.
        
        Args:
            data: Data array
            temporal_info: Temporal information
            
        Returns:
            Temporal consistency score
        """
        if len(data) < 2:
            return 1.0
        
        # Compute first-order differences
        diffs = np.diff(data, axis=0)
        
        # Compute time intervals
        time_intervals = np.diff(temporal_info)
        
        # Normalize differences by time intervals
        normalized_diffs = diffs / time_intervals[:, np.newaxis]
        
        # Compute consistency score based on variation
        consistency = 1 / (1 + np.std(normalized_diffs))
        
        return consistency
    
    def _compute_source_conflict(self,
                               data1: np.ndarray,
                               data2: np.ndarray) -> float:
        """Compute conflict score between two data sources.
        
        Args:
            data1: First data array
            data2: Second data array
            
        Returns:
            Conflict score
        """
        # Standardize data
        scaler = StandardScaler()
        data1_std = scaler.fit_transform(np.nan_to_num(data1))
        data2_std = scaler.fit_transform(np.nan_to_num(data2))
        
        # Compute correlation-based conflict
        corr_matrix = np.corrcoef(data1_std.T, data2_std.T)
        n_features1 = data1.shape[1]
        cross_corr = corr_matrix[:n_features1, n_features1:]
        
        # Convert correlations to conflicts
        conflicts = 1 - np.abs(cross_corr)
        
        return np.mean(conflicts)
    
    def _update_reliability_tracking(self,
                                   reliability_scores: Dict[str, float],
                                   temporal_info: Optional[np.ndarray] = None):
        """Update reliability tracking information.
        
        Args:
            reliability_scores: Current reliability scores
            temporal_info: Temporal information if available
        """
        current_time = len(self.temporal_trends.get(next(iter(reliability_scores)), []))
        
        # Update temporal trends
        for source, score in reliability_scores.items():
            if source not in self.temporal_trends:
                self.temporal_trends[source] = []
            self.temporal_trends[source].append(score)
            
            # Maintain fixed window size
            if len(self.temporal_trends[source]) > self.temporal_window:
                self.temporal_trends[source] = self.temporal_trends[source][-self.temporal_window:]
        
        # Store current scores
        self.reliability_scores = reliability_scores
    
    def _compute_quality_metrics(self,
                               data_arrays: List[np.ndarray],
                               modality_labels: List[str],
                               reference_data: Optional[Dict[str, np.ndarray]] = None,
                               quality_thresholds: Optional[Dict[str, float]] = None):
        """Compute quality metrics for each data source.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            reference_data: Reference data for validation
            quality_thresholds: Thresholds for quality metrics
        """
        for i, (data, modality) in enumerate(zip(data_arrays, modality_labels)):
            metrics = {
                'completeness': 1 - np.mean(np.isnan(data)),
                'variance_explained': np.var(np.nan_to_num(data)) / (np.var(data) + 1e-8),
                'temporal_stability': np.mean(np.abs(np.diff(data, axis=0)))
            }
            
            # Compare with reference if available
            if reference_data and modality in reference_data:
                ref_data = reference_data[modality]
                metrics['reference_correlation'] = np.corrcoef(
                    np.nan_to_num(data).ravel(),
                    ref_data.ravel()
                )[0, 1]
            
            # Check against thresholds
            if quality_thresholds:
                metrics['threshold_compliance'] = {}
                for metric, threshold in quality_thresholds.items():
                    if metric in metrics:
                        metrics['threshold_compliance'][metric] = metrics[metric] >= threshold
            
            self.quality_metrics[modality] = metrics
    
    def _update_from_errors(self,
                          reliability_scores: Dict[str, float],
                          prediction_errors: Dict[str, float]) -> Dict[str, float]:
        """Update reliability scores based on prediction errors.
        
        Args:
            reliability_scores: Current reliability scores
            prediction_errors: Dictionary of prediction errors
            
        Returns:
            Updated reliability scores
        """
        # Convert errors to reliability updates
        error_scale = 0.1  # Scale factor for error impact
        
        for source in reliability_scores:
            if source in prediction_errors:
                error = prediction_errors[source]
                # Update score inversely proportional to error
                reliability_scores[source] *= (1 - error_scale * error)
                
        return reliability_scores
    
    def _update_from_conflicts(self,
                             reliability_scores: Dict[str, float],
                             conflict_indicators: Dict[str, List[str]]) -> Dict[str, float]:
        """Update reliability scores based on conflicts.
        
        Args:
            reliability_scores: Current reliability scores
            conflict_indicators: Dictionary mapping sources to conflicting sources
            
        Returns:
            Updated reliability scores
        """
        conflict_penalty = 0.1  # Penalty for each conflict
        
        for source, conflicts in conflict_indicators.items():
            if source in reliability_scores:
                # Apply penalty based on number of conflicts
                n_conflicts = len(conflicts)
                reliability_scores[source] *= (1 - conflict_penalty * n_conflicts)
                
        return reliability_scores
    
    def _update_from_quality(self,
                           reliability_scores: Dict[str, float],
                           quality_updates: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Update reliability scores based on quality metrics.
        
        Args:
            reliability_scores: Current reliability scores
            quality_updates: Dictionary of updated quality metrics
            
        Returns:
            Updated reliability scores
        """
        quality_weight = 0.2  # Weight for quality metrics
        
        for source, metrics in quality_updates.items():
            if source in reliability_scores:
                # Compute average quality score
                quality_score = np.mean(list(metrics.values()))
                
                # Update reliability score
                reliability_scores[source] = (
                    (1 - quality_weight) * reliability_scores[source] +
                    quality_weight * quality_score
                )
                
        return reliability_scores
    
    def _signal_quality_reliability(self,
                                  data_arrays: List[np.ndarray],
                                  modality_labels: List[str],
                                  temporal_info: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute reliability scores based on signal quality metrics.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            temporal_info: Temporal information if available
            
        Returns:
            Dictionary mapping modality labels to reliability scores
        """
        reliability_scores = {}
        
        # Check for empty arrays
        if not data_arrays or any(data.size == 0 for data in data_arrays):
            raise ValueError("Empty data array provided")
        
        for i, (data, modality) in enumerate(zip(data_arrays, modality_labels)):
            # Compute basic quality metrics
            completeness = 1 - np.mean(np.isnan(data))
            
            # Detect outliers using robust covariance estimation
            if data.ndim > 1 and data.shape[1] > 1:
                robust_cov = MinCovDet(random_state=42).fit(
                    np.nan_to_num(data)
                )
                mahal_dist = robust_cov.mahalanobis(np.nan_to_num(data))
                outlier_ratio = np.mean(
                    mahal_dist > stats.chi2.ppf(self.outlier_sensitivity, data.shape[1])
                )
            else:
                # For 1D data or single feature data
                flat_data = np.nan_to_num(data).ravel()
                z_scores = np.abs(stats.zscore(flat_data, nan_policy='omit'))
                outlier_ratio = np.mean(z_scores > 3)
            
            # Compute temporal consistency if temporal info available
            if temporal_info is not None:
                temporal_scores = self._compute_temporal_consistency(
                    data, temporal_info
                )
                temporal_weight = 0.3
            else:
                temporal_scores = 1.0
                temporal_weight = 0.0
            
            # Combine metrics into final reliability score
            reliability_scores[modality] = (
                0.4 * completeness +
                0.3 * (1 - outlier_ratio) +
                temporal_weight * temporal_scores
            )
        
        return reliability_scores
    
    def _statistical_outlier_detection(self,
                                     data_arrays: List[np.ndarray],
                                     modality_labels: List[str],
                                     temporal_info: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute reliability scores based on statistical outlier detection.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            temporal_info: Temporal information if available
            
        Returns:
            Dictionary mapping modality labels to reliability scores
        """
        reliability_scores = {}
        
        # Check for empty arrays
        if not data_arrays or any(data.size == 0 for data in data_arrays):
            raise ValueError("Empty data array provided")
        
        for i, (data, modality) in enumerate(zip(data_arrays, modality_labels)):
            # Handle missing values
            data_clean = np.nan_to_num(data)
            
            # Initialize outlier detectors
            detectors = [
                IsolationForest(contamination=self.outlier_sensitivity,
                              random_state=42),
                LocalOutlierFactor(contamination=self.outlier_sensitivity,
                                 novelty=True),
                EllipticEnvelope(contamination=self.outlier_sensitivity,
                               random_state=42)
            ]
            
            # Compute outlier scores using each detector
            outlier_scores = []
            for detector in detectors:
                try:
                    if isinstance(detector, LocalOutlierFactor):
                        detector.fit(data_clean)
                        scores = -detector.score_samples(data_clean)
                    else:
                        detector.fit(data_clean)
                        scores = -detector.score_samples(data_clean)
                    outlier_scores.append(scores)
                except Exception:
                    # If a detector fails, skip it
                    continue
            
            if outlier_scores:
                # Combine outlier scores using weighted average
                combined_scores = np.mean(outlier_scores, axis=0)
                
                # Convert to reliability score (inverse of outlier score)
                reliability = 1 / (1 + np.exp(combined_scores))  # Sigmoid transformation
                reliability_scores[modality] = np.mean(reliability)
            else:
                # If all detectors fail, assign default reliability
                reliability_scores[modality] = 0.5
        
        return reliability_scores
    
    def _historical_reliability(self,
                              data_arrays: List[np.ndarray],
                              modality_labels: List[str],
                              temporal_info: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute reliability scores based on historical data.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            temporal_info: Temporal information if available
            
        Returns:
            Dictionary mapping modality labels to reliability scores
        """
        reliability_scores = {}
        
        # Check for empty arrays
        if not data_arrays or any(data.size == 0 for data in data_arrays):
            raise ValueError("Empty data array provided")
        
        for i, (data, modality) in enumerate(zip(data_arrays, modality_labels)):
            # Compute historical reliability
            historical_reliability = self._compute_historical_reliability(
                data, temporal_info
            )
            
            reliability_scores[modality] = historical_reliability
        
        return reliability_scores
    
    def _compute_historical_reliability(self,
                                       data: np.ndarray,
                                       temporal_info: Optional[np.ndarray] = None) -> float:
        """Compute historical reliability score.
        
        Args:
            data: Data array
            temporal_info: Temporal information if available
            
        Returns:
            Historical reliability score
        """
        if len(data) < 2:
            return 1.0
        
        # Compute historical consistency
        historical_consistency = self._compute_historical_consistency(
            data, temporal_info
        )
        
        # Compute historical stability
        historical_stability = self._compute_historical_stability(
            data, temporal_info
        )
        
        # Combine historical metrics
        historical_reliability = 0.6 * historical_consistency + 0.4 * historical_stability
        
        return historical_reliability
    
    def _compute_historical_consistency(self,
                                       data: np.ndarray,
                                       temporal_info: Optional[np.ndarray] = None) -> float:
        """Compute historical consistency score.
        
        Args:
            data: Data array
            temporal_info: Temporal information if available
            
        Returns:
            Historical consistency score
        """
        if len(data) < 2:
            return 1.0
        
        # Compute first-order differences
        diffs = np.diff(data, axis=0)
        
        # Compute time intervals if temporal info is available
        if temporal_info is not None:
            time_intervals = np.diff(temporal_info)
            
            # Handle multidimensional data
            if data.ndim > 1:
                # Normalize differences by time intervals for each feature
                normalized_diffs = np.zeros_like(diffs)
                for j in range(diffs.shape[1]):
                    normalized_diffs[:, j] = diffs[:, j] / time_intervals
            else:
                # For 1D data
                normalized_diffs = diffs / time_intervals
        else:
            # If no temporal info, use unit time intervals
            normalized_diffs = diffs
        
        # Compute consistency score based on variation
        consistency = 1 / (1 + np.std(normalized_diffs))
        
        return consistency
    
    def _compute_historical_stability(self,
                                      data: np.ndarray,
                                      temporal_info: Optional[np.ndarray] = None) -> float:
        """Compute historical stability score.
        
        Args:
            data: Data array
            temporal_info: Temporal information if available
            
        Returns:
            Historical stability score
        """
        if len(data) < 2:
            return 1.0
        
        # Compute historical stability
        if temporal_info is not None:
            # Use temporal information to compute weighted stability
            time_diffs = np.diff(temporal_info)
            weights = 1 / (time_diffs + 1e-8)
            
            # Handle multidimensional data
            if data.ndim > 1:
                # Compute weighted stability for each feature and average
                stabilities = []
                for j in range(data.shape[1]):
                    feature_diffs = (data[1:, j] - data[:-1, j])**2
                    weighted_std = np.sqrt(np.average(feature_diffs, weights=weights))
                    stabilities.append(weighted_std)
                weighted_std = np.mean(stabilities)
            else:
                # For 1D data
                weighted_std = np.sqrt(np.average((data[1:] - data[:-1])**2, weights=weights))
                
            stability = 1 / (1 + weighted_std)
        else:
            # If no temporal info, use standard deviation
            stability = 1 / (1 + np.std(data))
        
        return stability
    
    def _adaptive_reliability(self,
                             data_arrays: List[np.ndarray],
                             modality_labels: List[str],
                             temporal_info: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute reliability scores using an adaptive approach.
        
        Args:
            data_arrays: List of data arrays
            modality_labels: List of modality labels
            temporal_info: Temporal information if available
            
        Returns:
            Dictionary mapping modality labels to reliability scores
        """
        reliability_scores = {}
        
        # Check for empty arrays
        if not data_arrays or any(data.size == 0 for data in data_arrays):
            raise ValueError("Empty data array provided")
        
        for i, (data, modality) in enumerate(zip(data_arrays, modality_labels)):
            # Compute adaptive reliability
            adaptive_reliability = self._compute_adaptive_reliability(
                data, temporal_info
            )
            
            reliability_scores[modality] = adaptive_reliability
        
        return reliability_scores
    
    def _compute_adaptive_reliability(self,
                                     data: np.ndarray,
                                     temporal_info: Optional[np.ndarray] = None) -> float:
        """Compute adaptive reliability score.
        
        Args:
            data: Data array
            temporal_info: Temporal information if available
            
        Returns:
            Adaptive reliability score
        """
        if len(data) < 2:
            return 1.0
        
        # Compute adaptive consistency
        adaptive_consistency = self._compute_adaptive_consistency(
            data, temporal_info
        )
        
        # Compute adaptive stability
        adaptive_stability = self._compute_adaptive_stability(
            data, temporal_info
        )
        
        # Combine adaptive metrics
        adaptive_reliability = 0.6 * adaptive_consistency + 0.4 * adaptive_stability
        
        return adaptive_reliability
    
    def _compute_adaptive_consistency(self,
                                      data: np.ndarray,
                                      temporal_info: Optional[np.ndarray] = None) -> float:
        """Compute adaptive consistency score.
        
        Args:
            data: Data array
            temporal_info: Temporal information if available
            
        Returns:
            Adaptive consistency score
        """
        if len(data) < 2:
            return 1.0
        
        # Compute first-order differences
        diffs = np.diff(data, axis=0)
        
        # Compute time intervals if temporal info is available
        if temporal_info is not None:
            time_intervals = np.diff(temporal_info)
            
            # Handle multidimensional data
            if data.ndim > 1:
                # Normalize differences by time intervals for each feature
                normalized_diffs = np.zeros_like(diffs)
                for j in range(diffs.shape[1]):
                    normalized_diffs[:, j] = diffs[:, j] / time_intervals
            else:
                # For 1D data
                normalized_diffs = diffs / time_intervals
        else:
            # If no temporal info, use unit time intervals
            normalized_diffs = diffs
        
        # Compute consistency score based on variation
        consistency = 1 / (1 + np.std(normalized_diffs))
        
        return consistency
    
    def _compute_adaptive_stability(self,
                                   data: np.ndarray,
                                   temporal_info: Optional[np.ndarray] = None) -> float:
        """Compute adaptive stability score.
        
        Args:
            data: Data array
            temporal_info: Temporal information if available
            
        Returns:
            Adaptive stability score
        """
        if len(data) < 2:
            return 1.0
        
        # Compute adaptive stability
        adaptive_stability = np.std(data) / (np.std(data) + 1e-8)
        
        return adaptive_stability 
"""
Feature Interactions for Migraine Analysis
========================================

This module provides implementations for analyzing interactions between different physiological
and contextual features in migraine prediction. It focuses on:

1. Prodrome phase indicator analysis
2. Trigger interaction detection
3. Feature importance ranking
4. Cross-modal correlation analysis
5. Temporal lead/lag relationships

The analysis takes into account the complex interplay between different physiological signals,
environmental factors, and behavioral patterns that may contribute to migraine onset.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats, signal
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime, timedelta
import pandas as pd

from core.theory.multimodal_integration import ModalityData
from core.theory.temporal_modeling.causal_inference import CausalInferenceAnalyzer
from . import MigraineFeatureInteractionAnalyzer

class FeatureInteractionAnalyzer(MigraineFeatureInteractionAnalyzer):
    """Implementation of migraine-specific feature interaction analysis."""
    
    def __init__(self, 
                 significance_threshold: float = 0.05,
                 min_correlation: float = 0.2,
                 max_time_lag: int = 48):  # 48 hours max lag
        """
        Initialize the feature interaction analyzer.
        
        Parameters
        ----------
        significance_threshold : float
            P-value threshold for significant interactions
        min_correlation : float
            Minimum correlation coefficient to consider
        max_time_lag : int
            Maximum time lag to consider in hours
        """
        self.significance_threshold = significance_threshold
        self.min_correlation = min_correlation
        self.max_time_lag = max_time_lag
        # Initialize with empty DataFrame
        self.causal_analyzer = CausalInferenceAnalyzer(pd.DataFrame())
        
    def analyze_prodrome_indicators(self, 
                                  data_sources: Dict[str, ModalityData], 
                                  time_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Analyze prodrome phase indicators across multiple data sources.
        
        Parameters
        ----------
        data_sources : Dict[str, ModalityData]
            Dictionary of data sources keyed by modality name
        time_window : Optional[Tuple[float, float]]
            Optional time window for analysis (start_time, end_time)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - indicators: List of identified prodrome indicators
            - temporal_patterns: Temporal patterns in indicators
            - significance: Statistical significance of indicators
            - feature_importance: Importance scores for each indicator
        """
        indicators = []
        temporal_patterns = {}
        significance_scores = {}
        feature_importance = {}
        
        # Extract time-aligned features from each modality
        aligned_features = self._align_time_series(data_sources, time_window)
        
        # Check if any data is available
        if not any(len(data) > 0 for data in aligned_features.values()):
            return {
                'indicators': [],
                'temporal_patterns': {},
                'significance': {},
                'feature_importance': {},
                'cross_modal_interactions': {
                    'correlation_matrix': np.array([]),
                    'modalities': [],
                    'connected_components': []
                }
            }
        
        # Analyze each modality for prodrome indicators
        for modality_name, modality_data in data_sources.items():
            # Skip if no data in time window
            if modality_data.data is None or len(modality_data.data) == 0:
                continue
                
            # Extract features for this modality
            features = self._extract_prodrome_features(modality_data)
            
            # Skip if no features extracted
            if not features:
                continue
                
            # Analyze temporal patterns
            patterns = self._analyze_temporal_patterns(features)
            temporal_patterns[modality_name] = patterns
            
            # Calculate statistical significance
            sig_scores = self._calculate_significance(features)
            significance_scores[modality_name] = sig_scores
            
            # Identify significant indicators
            for feature_name, score in sig_scores.items():
                if score < self.significance_threshold:
                    indicators.append({
                        'modality': modality_name,
                        'feature': feature_name,
                        'significance': score,
                        'temporal_pattern': patterns.get(feature_name, {})
                    })
            
            # Calculate feature importance
            importance = self._calculate_feature_importance(features)
            feature_importance[modality_name] = importance
        
        # Analyze cross-modal interactions
        cross_modal = self._analyze_cross_modal_interactions(aligned_features)
        
        return {
            'indicators': indicators,
            'temporal_patterns': temporal_patterns,
            'significance': significance_scores,
            'feature_importance': feature_importance,
            'cross_modal_interactions': cross_modal
        }
    
    def detect_trigger_interactions(self, 
                                  triggers: Dict[str, np.ndarray],
                                  physiological_responses: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Detect interactions between multiple triggers and physiological responses.
        
        Parameters
        ----------
        triggers : Dict[str, np.ndarray]
            Dictionary of trigger time series keyed by trigger name
        physiological_responses : Dict[str, np.ndarray]
            Dictionary of physiological response time series
            
        Returns
        -------
        Dict[str, float]
            Interaction strengths between triggers and responses
        """
        interactions = {}
        
        # Convert data to arrays for analysis
        trigger_names = list(triggers.keys())
        response_names = list(physiological_responses.keys())
        
        # Analyze pairwise interactions
        for trigger_name in trigger_names:
            trigger_data = triggers[trigger_name]
            
            for response_name in response_names:
                response_data = physiological_responses[response_name]
                
                # Ensure equal lengths
                min_len = min(len(trigger_data), len(response_data))
                t_data = trigger_data[:min_len]
                r_data = response_data[:min_len]
                
                # Calculate interaction metrics
                correlation = stats.pearsonr(t_data, r_data)[0]
                mutual_info = mutual_info_regression(t_data.reshape(-1, 1), r_data)[0]
                
                # Analyze temporal relationship
                time_lag = self._find_optimal_time_lag(t_data, r_data)
                
                # Store results
                key = f"{trigger_name}->{response_name}"
                interactions[key] = {
                    'correlation': correlation,
                    'mutual_information': mutual_info,
                    'time_lag': time_lag,
                    'significance': self._calculate_interaction_significance(t_data, r_data)
                }
        
        # Analyze multi-trigger interactions
        if len(trigger_names) > 1:
            multi_trigger = self._analyze_multi_trigger_interactions(triggers, physiological_responses)
            interactions['multi_trigger'] = multi_trigger
        
        return interactions
    
    def rank_feature_importance(self, 
                              features: Dict[str, np.ndarray], 
                              migraine_occurrences: np.ndarray) -> List[Tuple[str, float]]:
        """
        Rank features by importance for migraine prediction.
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Dictionary of feature time series
        migraine_occurrences : np.ndarray
            Binary array indicating migraine occurrences
            
        Returns
        -------
        List[Tuple[str, float]]
            Ranked list of (feature_name, importance_score) tuples
        """
        importance_scores = []
        
        # Prepare feature matrix
        feature_names = list(features.keys())
        X = np.column_stack([features[name] for name in feature_names])
        
        # Calculate importance using multiple methods
        for feature_idx, feature_name in enumerate(feature_names):
            # Mutual information
            mi_score = mutual_info_regression(X[:, [feature_idx]], migraine_occurrences)[0]
            
            # Correlation
            corr = abs(stats.pearsonr(X[:, feature_idx], migraine_occurrences)[0])
            
            # Granger causality
            granger_score = self._calculate_granger_causality(
                X[:, feature_idx], 
                migraine_occurrences
            )
            
            # Temporal lead analysis
            lead_score = self._analyze_temporal_lead(
                X[:, feature_idx],
                migraine_occurrences
            )
            
            # Combine scores
            combined_score = (
                0.3 * mi_score +
                0.2 * corr +
                0.3 * granger_score +
                0.2 * lead_score
            )
            
            importance_scores.append((feature_name, combined_score))
        
        # Sort by importance score
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return importance_scores
    
    def _align_time_series(self, 
                          data_sources: Dict[str, ModalityData],
                          time_window: Optional[Tuple[Union[float, datetime], Union[float, datetime]]] = None) -> Dict[str, np.ndarray]:
        """Align multiple time series to a common time grid."""
        aligned_data = {}
        
        # Find common time range if not specified
        if time_window is None:
            start_times = []
            end_times = []
            for data in data_sources.values():
                if data.timestamps is not None:
                    start_times.append(np.min(data.timestamps))
                    end_times.append(np.max(data.timestamps))
            
            if start_times and end_times:
                time_window = (max(start_times), min(end_times))
        
        # Align each data source
        for name, data in data_sources.items():
            if data.timestamps is not None and time_window is not None:
                # Convert timestamps to float if needed
                if isinstance(time_window[0], datetime):
                    start_time = time_window[0].timestamp()
                    end_time = time_window[1].timestamp()
                    if isinstance(data.timestamps[0], datetime):
                        timestamps = np.array([t.timestamp() for t in data.timestamps])
                    else:
                        timestamps = data.timestamps
                else:
                    start_time = float(time_window[0])
                    end_time = float(time_window[1])
                    if isinstance(data.timestamps[0], datetime):
                        timestamps = np.array([t.timestamp() for t in data.timestamps])
                    else:
                        timestamps = data.timestamps
                    
                # Extract data within time window
                mask = ((timestamps >= start_time) & (timestamps <= end_time))
                aligned_data[name] = data.data[mask]
            else:
                aligned_data[name] = data.data
                
        return aligned_data
    
    def _extract_prodrome_features(self, modality_data: ModalityData) -> Dict[str, np.ndarray]:
        """Extract features relevant to prodrome phase detection."""
        features = {}
        
        if modality_data.modality_type == 'ecg':
            # Extract HRV features
            if len(modality_data.data.shape) == 1:
                features['hrv_changes'] = self._calculate_hrv_changes(modality_data.data)
        
        elif modality_data.modality_type == 'eeg':
            # Extract EEG band power changes
            if len(modality_data.data.shape) == 1:
                features['alpha_power'] = self._calculate_band_power(modality_data.data, (8, 13))
                features['theta_power'] = self._calculate_band_power(modality_data.data, (4, 8))
        
        elif modality_data.modality_type == 'skin_conductance':
            # Extract EDA features
            if len(modality_data.data.shape) == 1:
                features['scr_rate'] = self._calculate_scr_rate(modality_data.data)
        
        return features
    
    def _analyze_temporal_patterns(self, features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Analyze temporal patterns in features."""
        patterns = {}
        
        for name, data in features.items():
            # Calculate basic statistics
            patterns[name] = {
                'trend': self._calculate_trend(data),
                'periodicity': self._detect_periodicity(data),
                'changepoints': self._detect_changepoints(data)
            }
            
        return patterns
    
    def _calculate_significance(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate statistical significance of features."""
        significance = {}
        
        for name, data in features.items():
            # Perform statistical tests
            if len(data) > 1:
                # Test for significant changes
                baseline = np.mean(data[:len(data)//3])  # Use first third as baseline
                test_data = data[len(data)//3:]
                t_stat, p_value = stats.ttest_1samp(test_data - baseline, 0)
                significance[name] = p_value
            else:
                significance[name] = 1.0
                
        return significance
    
    def _calculate_feature_importance(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate importance scores for features."""
        importance = {}
        
        # Handle empty features
        if not features:
            return importance
            
        # Create feature matrix
        try:
            feature_matrix = np.column_stack([features[name] for name in features])
        except ValueError:
            # Handle case where features have different lengths or are empty
            return {name: 0.0 for name in features}
            
        if feature_matrix.size == 0:
            return {name: 0.0 for name in features}
            
        # Normalize features
        try:
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_matrix)
        except ValueError:
            # Handle case where normalization fails
            return {name: 0.0 for name in features}
        
        # Calculate importance using variance and temporal characteristics
        for i, name in enumerate(features):
            variance_score = np.var(normalized_features[:, i])
            temporal_score = self._calculate_temporal_importance(normalized_features[:, i])
            importance[name] = 0.7 * variance_score + 0.3 * temporal_score
            
        return importance
    
    def _analyze_cross_modal_interactions(self, 
                                        aligned_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze interactions between different modalities."""
        interactions = {}
        
        # Create correlation matrix
        modalities = list(aligned_features.keys())
        n_modalities = len(modalities)
        correlation_matrix = np.zeros((n_modalities, n_modalities))
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    correlation_matrix[i, j] = self._calculate_modal_correlation(
                        aligned_features[mod1],
                        aligned_features[mod2]
                    )
        
        # Create interaction graph
        G = nx.Graph()
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i < j and abs(correlation_matrix[i, j]) > self.min_correlation:
                    G.add_edge(mod1, mod2, weight=abs(correlation_matrix[i, j]))
        
        # Find strongly connected components
        connected_components = list(nx.connected_components(G))
        
        interactions['correlation_matrix'] = correlation_matrix
        interactions['modalities'] = modalities
        interactions['connected_components'] = connected_components
        
        return interactions
    
    def _find_optimal_time_lag(self, trigger: np.ndarray, response: np.ndarray) -> int:
        """Find the optimal time lag between trigger and response."""
        max_lag = min(len(trigger), self.max_time_lag)
        correlations = []
        
        for lag in range(max_lag):
            if lag == 0:
                corr = stats.pearsonr(trigger, response)[0]
            else:
                corr = stats.pearsonr(trigger[:-lag], response[lag:])[0]
            correlations.append(abs(corr))
            
        return int(np.argmax(correlations))
    
    def _calculate_interaction_significance(self, 
                                         trigger: np.ndarray, 
                                         response: np.ndarray) -> float:
        """Calculate statistical significance of interaction."""
        # Perform permutation test
        n_permutations = 1000
        observed_corr = abs(stats.pearsonr(trigger, response)[0])
        
        # Generate null distribution
        null_dist = []
        for _ in range(n_permutations):
            shuffled_trigger = np.random.permutation(trigger)
            null_corr = abs(stats.pearsonr(shuffled_trigger, response)[0])
            null_dist.append(null_corr)
            
        # Calculate p-value
        p_value = np.mean(np.array(null_dist) >= observed_corr)
        
        return float(p_value)
    
    def _analyze_multi_trigger_interactions(self,
                                          triggers: Dict[str, np.ndarray],
                                          responses: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze interactions between multiple triggers."""
        results = {}
        
        # Create trigger combination matrix
        trigger_names = list(triggers.keys())
        n_triggers = len(trigger_names)
        
        if n_triggers < 2:
            return results
            
        # Analyze pairwise trigger interactions
        interaction_matrix = np.zeros((n_triggers, n_triggers))
        synergy_scores = {}
        
        for i, t1 in enumerate(trigger_names):
            for j, t2 in enumerate(trigger_names):
                if i < j:
                    # Calculate interaction score
                    score = self._calculate_trigger_synergy(
                        triggers[t1],
                        triggers[t2],
                        responses
                    )
                    interaction_matrix[i, j] = score
                    interaction_matrix[j, i] = score
                    
                    if abs(score) > self.min_correlation:
                        synergy_scores[f"{t1}-{t2}"] = score
        
        results['interaction_matrix'] = interaction_matrix
        results['trigger_names'] = trigger_names
        results['synergy_scores'] = synergy_scores
        
        return results
    
    def _calculate_trigger_synergy(self,
                                 trigger1: np.ndarray,
                                 trigger2: np.ndarray,
                                 responses: Dict[str, np.ndarray]) -> float:
        """Calculate synergy score between two triggers."""
        # Handle empty or invalid data
        if len(trigger1) == 0 or len(trigger2) == 0 or not responses:
            return 0.0

        try:
            # Ensure triggers are 1D arrays of the same length
            t1 = np.asarray(trigger1).flatten()
            t2 = np.asarray(trigger2).flatten()
            min_len = min(len(t1), len(t2))
            t1 = t1[:min_len]
            t2 = t2[:min_len]

            # Combine triggers
            combined_effect = (t1 + t2) / 2  # Simple average of the two triggers
            
            # Calculate individual and combined effects
            individual_effects = []
            for response_data in responses.values():
                if len(response_data) < min_len:
                    continue
                    
                response = np.asarray(response_data[:min_len]).flatten()
                try:
                    effect1 = abs(stats.pearsonr(t1, response)[0])
                    effect2 = abs(stats.pearsonr(t2, response)[0])
                    combined = abs(stats.pearsonr(combined_effect, response)[0])
                    
                    # Positive score indicates synergy, negative indicates redundancy
                    synergy = combined - max(effect1, effect2)
                    individual_effects.append(synergy)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                    
            return float(np.mean(individual_effects)) if individual_effects else 0.0
        except (ValueError, np.linalg.LinAlgError, TypeError):
            return 0.0
    
    def _calculate_temporal_importance(self, feature: np.ndarray) -> float:
        """Calculate importance score based on temporal characteristics."""
        if len(feature) < 2:
            return 0.0
            
        try:
            # Calculate trend strength
            trend = np.polyfit(np.arange(len(feature)), feature, 1)[0]
        except (ValueError, np.linalg.LinAlgError):
            trend = 0.0
            
        try:
            # Calculate periodicity
            freqs = np.fft.fftfreq(len(feature))
            fft = np.fft.fft(feature)
            periodicity = np.max(np.abs(fft)) / len(feature)
        except (ValueError, ZeroDivisionError):
            periodicity = 0.0
        
        # Combine scores
        return float(0.6 * abs(trend) + 0.4 * periodicity)
    
    def _calculate_modal_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate correlation between two modalities."""
        # Handle empty data
        if len(data1) < 2 or len(data2) < 2:
            return 0.0
            
        # Ensure equal lengths
        min_len = min(len(data1), len(data2))
        d1 = data1[:min_len]
        d2 = data2[:min_len]
        
        try:
            return float(stats.pearsonr(d1, d2)[0])
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_hrv_changes(self, ecg_data: np.ndarray) -> np.ndarray:
        """Calculate HRV changes over time."""
        # Simple implementation - use rolling window
        window_size = 100
        hrv = np.array([np.std(ecg_data[i:i+window_size]) 
                       for i in range(0, len(ecg_data)-window_size)])
        return hrv
    
    def _calculate_band_power(self, eeg_data: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
        """Calculate power in specific frequency band."""
        # Simple implementation using FFT
        fft = np.fft.fft(eeg_data)
        freqs = np.fft.fftfreq(len(eeg_data))
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.abs(fft[mask])
    
    def _calculate_scr_rate(self, eda_data: np.ndarray) -> np.ndarray:
        """Calculate skin conductance response rate."""
        # Simple implementation - count peaks
        peaks, _ = signal.find_peaks(eda_data, height=0.1)
        return np.array([len(peaks)])
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate linear trend in data."""
        if len(data) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(data)), data, 1)[0])
    
    def _detect_periodicity(self, data: np.ndarray) -> Dict[str, float]:
        """Detect periodic patterns in data."""
        if len(data) < 2:
            return {
                'frequency': 0.0,
                'amplitude': 0.0
            }
            
        freqs = np.fft.fftfreq(len(data))
        fft = np.fft.fft(data)
        main_freq_idx = np.argmax(np.abs(fft))
        
        return {
            'frequency': float(freqs[main_freq_idx]),
            'amplitude': float(np.abs(fft[main_freq_idx])) / len(data)
        }
    
    def _detect_changepoints(self, data: np.ndarray) -> List[int]:
        """Detect points of significant change in data."""
        # Simple implementation using rolling statistics
        window_size = max(10, len(data) // 10)
        means = np.array([np.mean(data[i:i+window_size])
                         for i in range(0, len(data)-window_size)])
        
        # Find points where mean changes significantly
        threshold = np.std(means) * 2
        changes = np.where(np.abs(np.diff(means)) > threshold)[0]
        
        return changes.tolist()
    
    def _calculate_granger_causality(self, feature: np.ndarray, target: np.ndarray) -> float:
        """Calculate Granger causality score."""
        # Create a DataFrame with the feature and target
        df = pd.DataFrame({
            'feature': feature,
            'target': target
        })
        
        # Update causal analyzer with new data
        self.causal_analyzer.data = df
        
        # Analyze Granger causality
        result = self.causal_analyzer.analyze_granger_causality(
            'feature',
            'target'
        )
        return float(1 - result.get('p_value', 1.0))  # Convert p-value to score
    
    def _analyze_temporal_lead(self, feature: np.ndarray, target: np.ndarray) -> float:
        """Analyze temporal lead/lag relationship."""
        # Calculate cross-correlation
        cross_corr = signal.correlate(target - np.mean(target),
                                    feature - np.mean(feature),
                                    mode='full')
        
        # Find the lag with maximum correlation
        lags = np.arange(-(len(target)-1), len(target))
        max_lag = lags[np.argmax(np.abs(cross_corr))]
        
        # Convert lag to score (higher score for positive lag, indicating feature leads target)
        if max_lag > 0:
            return float(np.exp(-max_lag / len(target)))  # Decay with lag
        else:
            return 0.0  # No lead relationship 
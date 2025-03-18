"""
Trigger Identification for Migraine Analysis
=========================================

This module provides implementations for identifying and analyzing migraine triggers
through causal inference, sensitivity analysis, and temporal pattern recognition.

Key Features:
1. Causal inference for trigger-symptom relationships
2. Sensitivity analysis for trigger thresholds
3. Multi-trigger interaction modeling
4. Personalized trigger profile generation
5. Temporal pattern recognition for triggers

The analysis takes into account both individual triggers and their interactions,
while considering temporal relationships and personalized sensitivity thresholds.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import pandas as pd
import networkx as nx
from dataclasses import dataclass

from core.theory.temporal_modeling.causal_inference import CausalInferenceAnalyzer
from core.theory.multimodal_integration import ModalityData
from . import MigraineTriggerAnalyzer

@dataclass
class TriggerEvent:
    """Represents a trigger event with its characteristics."""
    trigger_type: str
    timestamp: datetime
    intensity: float
    duration: Optional[timedelta] = None
    confidence: float = 1.0
    associated_symptoms: List[str] = None
    context: Dict[str, Any] = None

@dataclass
class TriggerProfile:
    """Represents a personalized trigger profile."""
    trigger_sensitivities: Dict[str, float]
    interaction_effects: Dict[str, float]
    temporal_patterns: Dict[str, Dict[str, Any]]
    threshold_ranges: Dict[str, Tuple[float, float]]
    confidence_scores: Dict[str, float]

class TriggerIdentificationAnalyzer(MigraineTriggerAnalyzer):
    """Implementation of migraine trigger identification and analysis."""
    
    def __init__(self, 
                 causal_threshold: float = 0.05,
                 sensitivity_window: int = 48,  # hours
                 min_confidence: float = 0.7,
                 max_interaction_order: int = 3):
        """
        Initialize the trigger identification analyzer.
        
        Parameters
        ----------
        causal_threshold : float
            P-value threshold for causal relationships
        sensitivity_window : int
            Time window (hours) for sensitivity analysis
        min_confidence : float
            Minimum confidence score for trigger identification
        max_interaction_order : int
            Maximum order of trigger interactions to consider
        """
        self.causal_threshold = causal_threshold
        self.sensitivity_window = sensitivity_window
        self.min_confidence = min_confidence
        self.max_interaction_order = max_interaction_order
        self.causal_analyzer = CausalInferenceAnalyzer(pd.DataFrame())
        
    def identify_triggers(self,
                         symptom_data: Dict[str, np.ndarray],
                         potential_triggers: Dict[str, np.ndarray],
                         timestamps: np.ndarray,
                         context_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Identify potential migraine triggers from time series data.
        
        Parameters
        ----------
        symptom_data : Dict[str, np.ndarray]
            Time series of migraine symptoms
        potential_triggers : Dict[str, np.ndarray]
            Time series of potential trigger factors
        timestamps : np.ndarray
            Timestamps for the data points
        context_data : Optional[Dict[str, np.ndarray]]
            Additional contextual data
            
        Returns
        -------
        Dict[str, Any]
            Identified triggers with confidence scores and relationships
        """
        # Initialize empty results
        results = {
            'triggers': [],
            'causal_scores': {},
            'sensitivity_thresholds': {},
            'temporal_patterns': {},
            'interaction_effects': {},
            'confidence_scores': {}
        }
        
        # Handle empty or invalid data
        if len(timestamps) == 0 or len(potential_triggers) == 0 or len(symptom_data) == 0:
            return results
            
        # Validate timestamps
        if isinstance(timestamps[0], datetime):
            # For datetime timestamps, check they're in chronological order
            if not all(t1 <= t2 for t1, t2 in zip(timestamps[:-1], timestamps[1:])):
                return results
        else:
            # For numeric timestamps, check they're non-negative and monotonic
            if not all(t >= 0 for t in timestamps) or not all(t1 <= t2 for t1, t2 in zip(timestamps[:-1], timestamps[1:])):
                return results
        
        # Analyze each potential trigger
        for trigger_name, trigger_data in potential_triggers.items():
            # Skip if data lengths don't match
            if len(trigger_data) != len(timestamps):
                continue
                
            # Perform causal analysis
            causal_score = self._analyze_causal_relationship(
                trigger_data,
                symptom_data,
                timestamps
            )
            
            # Calculate sensitivity threshold
            sensitivity = self._analyze_trigger_sensitivity(
                trigger_data,
                symptom_data,
                timestamps
            )
            
            # Detect temporal patterns
            temporal_pattern = self._detect_temporal_patterns(
                trigger_data,
                symptom_data,
                timestamps
            )
            
            # Calculate confidence score
            confidence = self._calculate_trigger_confidence(
                causal_score,
                sensitivity,
                temporal_pattern
            )
            
            if confidence >= self.min_confidence:
                results['triggers'].append(trigger_name)
                results['causal_scores'][trigger_name] = causal_score
                results['sensitivity_thresholds'][trigger_name] = sensitivity
                results['temporal_patterns'][trigger_name] = temporal_pattern
                results['confidence_scores'][trigger_name] = confidence
        
        # Analyze trigger interactions
        if len(results['triggers']) > 1:
            results['interaction_effects'] = self._analyze_trigger_interactions(
                {name: potential_triggers[name] for name in results['triggers']},
                symptom_data,
                timestamps
            )
        
        return results
    
    def generate_trigger_profile(self,
                               trigger_history: List[TriggerEvent],
                               migraine_events: List[datetime],
                               context_history: Optional[Dict[str, List[Any]]] = None) -> TriggerProfile:
        """
        Generate a personalized trigger profile based on historical data.
        
        Parameters
        ----------
        trigger_history : List[TriggerEvent]
            Historical trigger events
        migraine_events : List[datetime]
            Timestamps of migraine occurrences
        context_history : Optional[Dict[str, List[Any]]]
            Historical context data
            
        Returns
        -------
        TriggerProfile
            Personalized trigger profile
        """
        # Calculate trigger sensitivities
        sensitivities = self._calculate_trigger_sensitivities(
            trigger_history,
            migraine_events
        )
        
        # Analyze interaction effects
        interactions = self._analyze_historical_interactions(
            trigger_history,
            migraine_events
        )
        
        # Detect temporal patterns
        patterns = self._analyze_temporal_trigger_patterns(
            trigger_history,
            migraine_events
        )
        
        # Calculate threshold ranges
        thresholds = self._calculate_threshold_ranges(
            trigger_history,
            migraine_events
        )
        
        # Calculate confidence scores
        confidence = self._calculate_profile_confidence(
            trigger_history,
            migraine_events,
            patterns
        )
        
        return TriggerProfile(
            trigger_sensitivities=sensitivities,
            interaction_effects=interactions,
            temporal_patterns=patterns,
            threshold_ranges=thresholds,
            confidence_scores=confidence
        )
    
    def analyze_trigger_sensitivity(self,
                                  trigger_data: np.ndarray,
                                  symptom_data: Dict[str, np.ndarray],
                                  baseline_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Analyze sensitivity thresholds for triggers.
        
        Parameters
        ----------
        trigger_data : np.ndarray
            Time series of trigger measurements
        symptom_data : Dict[str, np.ndarray]
            Time series of symptoms
        baseline_period : Optional[Tuple[datetime, datetime]]
            Period for baseline comparison
            
        Returns
        -------
        Dict[str, Any]
            Sensitivity analysis results
        """
        results = {}
        
        # Calculate baseline statistics if period provided
        if baseline_period is not None:
            baseline_stats = self._calculate_baseline_stats(
                trigger_data,
                baseline_period
            )
            results['baseline'] = baseline_stats
        
        # Perform threshold analysis
        thresholds = self._analyze_threshold_levels(
            trigger_data,
            symptom_data
        )
        results['thresholds'] = thresholds
        
        # Analyze temporal sensitivity
        temporal_sensitivity = self._analyze_temporal_sensitivity(
            trigger_data,
            symptom_data
        )
        results['temporal_sensitivity'] = temporal_sensitivity
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_sensitivity_confidence(
            trigger_data,
            symptom_data
        )
        results['confidence_intervals'] = confidence_intervals
        
        return results
    
    def _analyze_causal_relationship(self,
                                   trigger_data: np.ndarray,
                                   symptom_data: Dict[str, np.ndarray],
                                   timestamps: np.ndarray) -> float:
        """Analyze causal relationship between trigger and symptoms."""
        # Handle insufficient data
        if len(trigger_data) < 3:  # Need at least 3 points for meaningful analysis
            return 0.0
            
        # Convert data to pandas DataFrame
        df = pd.DataFrame({
            'trigger': trigger_data,
            **{f'symptom_{i}': data for i, data in enumerate(symptom_data.values())}
        })
        
        # Update causal analyzer
        self.causal_analyzer.data = df
        
        # Calculate Granger causality with error handling
        max_causal_score = 0.0
        for symptom_col in df.columns[1:]:  # Skip trigger column
            try:
                result = self.causal_analyzer.analyze_granger_causality(
                    'trigger',
                    symptom_col,
                    max_lag=min(12, len(trigger_data) // 4)  # Adjust max lag based on data length
                )
                causal_score = 1 - result.get('p_value', 1.0)
                max_causal_score = max(max_causal_score, causal_score)
            except (ValueError, np.linalg.LinAlgError):
                # If Granger test fails, fall back to correlation analysis
                correlation = abs(stats.pearsonr(trigger_data, df[symptom_col])[0])
                max_causal_score = max(max_causal_score, correlation)
        
        return float(max_causal_score)
    
    def _analyze_trigger_sensitivity(self,
                                   trigger_data: np.ndarray,
                                   symptom_data: Dict[str, np.ndarray],
                                   timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze sensitivity thresholds for the trigger."""
        sensitivity = {}
        
        # Calculate basic statistics
        trigger_mean = np.mean(trigger_data)
        trigger_std = np.std(trigger_data)
        
        # Find trigger events (peaks in trigger data)
        peaks, _ = self._find_trigger_events(trigger_data)
        
        # Calculate symptom response for different trigger levels
        levels = np.linspace(
            trigger_mean - 2*trigger_std,
            trigger_mean + 2*trigger_std,
            10
        )
        
        for level in levels:
            # Find episodes above threshold
            episodes = trigger_data > level
            
            # Calculate average symptom response
            response = np.mean([
                np.mean([symptom[i:i+self.sensitivity_window] 
                        for symptom in symptom_data.values()])
                for i in np.where(episodes)[0]
            ])
            
            sensitivity[f'level_{level:.2f}'] = float(response)
        
        return sensitivity
    
    def _detect_temporal_patterns(self,
                                trigger_data: np.ndarray,
                                symptom_data: Dict[str, np.ndarray],
                                timestamps: np.ndarray) -> Dict[str, Any]:
        """Detect temporal patterns in trigger-symptom relationships."""
        patterns = {}
        
        # Handle empty or invalid data
        if len(trigger_data) == 0 or len(timestamps) == 0:
            return {
                'daily': {'hourly_means': [0.0] * 24, 'peak_hours': [], 'strength': 0.0},
                'weekly': {'daily_means': [0.0] * 7, 'peak_days': [], 'strength': 0.0},
                'temporal_relationships': {'strength': 0.0}
            }
        
        # Ensure timestamps and data have the same length
        min_len = min(len(trigger_data), len(timestamps))
        trigger_data = trigger_data[:min_len]
        timestamps = timestamps[:min_len]
        
        # Convert timestamps to hours if datetime
        if isinstance(timestamps[0], datetime):
            hours = np.array([t.hour for t in timestamps])
        else:
            hours = np.array([int(t % 24) for t in timestamps])
        
        # Analyze daily patterns
        daily_pattern = self._analyze_daily_pattern(
            trigger_data,
            hours
        )
        patterns['daily'] = daily_pattern
        
        # Analyze weekly patterns if enough data
        if len(timestamps) >= 7*24:
            weekly_pattern = self._analyze_weekly_pattern(
                trigger_data,
                timestamps
            )
            patterns['weekly'] = weekly_pattern
        else:
            patterns['weekly'] = {
                'daily_means': [0.0] * 7,
                'peak_days': [],
                'strength': 0.0
            }
        
        # Analyze lead/lag relationships
        temporal_relationships = self._analyze_temporal_relationships(
            trigger_data,
            {name: data[:min_len] for name, data in symptom_data.items()}
        )
        patterns['temporal_relationships'] = temporal_relationships
        
        return patterns
    
    def _calculate_trigger_confidence(self,
                                    causal_score: float,
                                    sensitivity: Dict[str, float],
                                    temporal_pattern: Dict[str, Any]) -> float:
        """Calculate confidence score for trigger identification."""
        # Weight different factors
        causal_weight = 0.4
        sensitivity_weight = 0.3
        pattern_weight = 0.3
        
        # Calculate sensitivity score
        sensitivity_score = np.mean(list(sensitivity.values()))
        
        # Calculate pattern score
        pattern_strength = np.mean([
            temporal_pattern['daily'].get('strength', 0),
            temporal_pattern.get('weekly', {}).get('strength', 0),
            temporal_pattern['temporal_relationships'].get('strength', 0)
        ])
        
        # Combine scores
        confidence = (
            causal_weight * causal_score +
            sensitivity_weight * sensitivity_score +
            pattern_weight * pattern_strength
        )
        
        return float(confidence)
    
    def _analyze_trigger_interactions(self,
                                    triggers: Dict[str, np.ndarray],
                                    symptom_data: Dict[str, np.ndarray],
                                    timestamps: np.ndarray) -> Dict[str, float]:
        """Analyze interactions between multiple triggers."""
        interactions = {}
        trigger_names = list(triggers.keys())
        
        # Analyze pairwise interactions
        for i, name1 in enumerate(trigger_names):
            for j, name2 in enumerate(trigger_names):
                if i < j:
                    interaction_score = self._calculate_interaction_score(
                        triggers[name1],
                        triggers[name2],
                        symptom_data
                    )
                    interactions[f"{name1}-{name2}"] = float(interaction_score)
        
        # Analyze higher-order interactions up to max_interaction_order
        if len(trigger_names) > 2:
            for order in range(3, min(len(trigger_names) + 1, self.max_interaction_order + 1)):
                higher_order = self._analyze_higher_order_interactions(
                    triggers,
                    symptom_data,
                    order
                )
                interactions.update(higher_order)
        
        return interactions
    
    def _find_trigger_events(self,
                           trigger_data: np.ndarray,
                           threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find significant trigger events in the data."""
        if threshold is None:
            threshold = np.mean(trigger_data) + np.std(trigger_data)
        
        # Find peaks above threshold
        peaks, properties = signal.find_peaks(
            trigger_data,
            height=threshold,
            distance=20  # Minimum samples between peaks
        )
        
        return peaks, properties
    
    def _analyze_daily_pattern(self,
                             trigger_data: np.ndarray,
                             hours: np.ndarray) -> Dict[str, Any]:
        """Analyze daily patterns in trigger data."""
        # Handle empty data
        if len(trigger_data) == 0:
            return {
                'hourly_means': [0.0] * 24,
                'peak_hours': [],
                'strength': 0.0
            }
        
        # Calculate average trigger levels by hour
        hourly_means = []
        for h in range(24):
            hour_mask = hours == h
            if np.any(hour_mask):
                hourly_means.append(float(np.mean(trigger_data[hour_mask])))
            else:
                hourly_means.append(0.0)
        
        # Convert to numpy array for calculations
        hourly_means = np.array(hourly_means)
        
        # Find peak times (avoid division by zero)
        mean_level = np.mean(hourly_means)
        if mean_level > 0:
            peak_hours = np.where(hourly_means > mean_level + np.std(hourly_means))[0]
            pattern_strength = np.std(hourly_means) / mean_level
        else:
            peak_hours = np.array([])
            pattern_strength = 0.0
        
        return {
            'hourly_means': hourly_means.tolist(),
            'peak_hours': peak_hours.tolist(),
            'strength': float(pattern_strength)
        }
    
    def _analyze_weekly_pattern(self,
                              trigger_data: np.ndarray,
                              timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyze weekly patterns in trigger data."""
        if isinstance(timestamps[0], datetime):
            days = np.array([t.weekday() for t in timestamps])
        else:
            days = (timestamps // 24) % 7
        
        # Calculate average trigger levels by day
        daily_means = [np.mean(trigger_data[days == d]) for d in range(7)]
        
        # Find peak days
        peak_days = np.where(daily_means > np.mean(daily_means) + np.std(daily_means))[0]
        
        # Calculate pattern strength
        pattern_strength = np.std(daily_means) / np.mean(daily_means)
        
        return {
            'daily_means': daily_means,
            'peak_days': peak_days.tolist(),
            'strength': float(pattern_strength)
        }
    
    def _analyze_temporal_relationships(self,
                                     trigger_data: np.ndarray,
                                     symptom_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze temporal relationships between triggers and symptoms."""
        relationships = {}
        max_lag = self.sensitivity_window
        
        for symptom_name, symptom in symptom_data.items():
            try:
                # Normalize data to avoid numerical issues
                t_norm = (trigger_data - np.mean(trigger_data)) / (np.std(trigger_data) + 1e-10)
                s_norm = (symptom - np.mean(symptom)) / (np.std(symptom) + 1e-10)
                
                # Calculate cross-correlation
                cross_corr = signal.correlate(s_norm, t_norm, mode='full')
                
                # Find the lag with maximum correlation
                lags = np.arange(-(len(symptom)-1), len(symptom))
                max_lag_idx = np.argmax(np.abs(cross_corr))
                max_lag_time = lags[max_lag_idx]
                
                # Calculate normalized correlation
                max_corr = cross_corr[max_lag_idx] / len(trigger_data)
                
                relationships[symptom_name] = {
                    'lag': int(max_lag_time),
                    'correlation': float(max_corr)
                }
            except (ValueError, ZeroDivisionError):
                relationships[symptom_name] = {
                    'lag': 0,
                    'correlation': 0.0
                }
        
        # Calculate overall relationship strength
        correlations = [abs(rel['correlation']) for rel in relationships.values()]
        strength = float(np.mean(correlations)) if correlations else 0.0
        relationships['strength'] = strength
        
        return relationships
    
    def _calculate_interaction_score(self,
                                   trigger1: np.ndarray,
                                   trigger2: np.ndarray,
                                   symptom_data: Dict[str, np.ndarray]) -> float:
        """Calculate interaction score between two triggers."""
        # Normalize triggers
        t1_norm = (trigger1 - np.mean(trigger1)) / np.std(trigger1)
        t2_norm = (trigger2 - np.mean(trigger2)) / np.std(trigger2)
        
        # Calculate combined effect
        combined = t1_norm * t2_norm
        
        # Calculate correlation with symptoms
        max_correlation = 0.0
        for symptom in symptom_data.values():
            symptom_norm = (symptom - np.mean(symptom)) / np.std(symptom)
            correlation = abs(stats.pearsonr(combined, symptom_norm)[0])
            max_correlation = max(max_correlation, correlation)
        
        return float(max_correlation)
    
    def _analyze_higher_order_interactions(self,
                                         triggers: Dict[str, np.ndarray],
                                         symptom_data: Dict[str, np.ndarray],
                                         order: int) -> Dict[str, float]:
        """Analyze higher-order interactions between triggers."""
        interactions = {}
        trigger_names = list(triggers.keys())
        
        # Generate combinations of triggers
        from itertools import combinations
        for combo in combinations(trigger_names, order):
            # Calculate combined effect
            combined = np.ones_like(triggers[combo[0]])
            for name in combo:
                trigger_norm = (triggers[name] - np.mean(triggers[name])) / np.std(triggers[name])
                combined *= trigger_norm
            
            # Calculate correlation with symptoms
            max_correlation = 0.0
            for symptom in symptom_data.values():
                symptom_norm = (symptom - np.mean(symptom)) / np.std(symptom)
                correlation = abs(stats.pearsonr(combined, symptom_norm)[0])
                max_correlation = max(max_correlation, correlation)
            
            interactions['-'.join(combo)] = float(max_correlation)
        
        return interactions
    
    def _calculate_trigger_sensitivities(self,
                                       trigger_history: List[TriggerEvent],
                                       migraine_events: List[datetime]) -> Dict[str, float]:
        """Calculate trigger sensitivities from historical data."""
        sensitivities = {}
        trigger_types = set(event.trigger_type for event in trigger_history)
        
        for trigger_type in trigger_types:
            # Get trigger events of this type
            type_events = [e for e in trigger_history if e.trigger_type == trigger_type]
            
            # Calculate sensitivity
            sensitivity = self._calculate_type_sensitivity(
                type_events,
                migraine_events
            )
            sensitivities[trigger_type] = float(sensitivity)
        
        return sensitivities
    
    def _calculate_type_sensitivity(self,
                                  trigger_events: List[TriggerEvent],
                                  migraine_events: List[datetime]) -> float:
        """Calculate sensitivity for a specific trigger type."""
        if not trigger_events or not migraine_events:
            return 0.0
        
        # Count trigger events followed by migraines
        followed_by_migraine = 0
        window = timedelta(hours=self.sensitivity_window)
        
        for event in trigger_events:
            # Check if any migraine occurs within the window
            for migraine in migraine_events:
                if event.timestamp < migraine <= event.timestamp + window:
                    followed_by_migraine += 1
                    break
        
        return followed_by_migraine / len(trigger_events)
    
    def _analyze_historical_interactions(self,
                                      trigger_history: List[TriggerEvent],
                                      migraine_events: List[datetime]) -> Dict[str, float]:
        """Analyze trigger interactions from historical data."""
        interactions = {}
        trigger_types = set(event.trigger_type for event in trigger_history)
        
        # Analyze pairwise interactions
        for t1 in trigger_types:
            for t2 in trigger_types:
                if t1 < t2:
                    score = self._calculate_historical_interaction(
                        trigger_history,
                        migraine_events,
                        t1,
                        t2
                    )
                    interactions[f"{t1}-{t2}"] = float(score)
        
        return interactions
    
    def _calculate_historical_interaction(self,
                                        trigger_history: List[TriggerEvent],
                                        migraine_events: List[datetime],
                                        type1: str,
                                        type2: str) -> float:
        """Calculate historical interaction score between two trigger types."""
        window = timedelta(hours=self.sensitivity_window)
        
        # Find co-occurring triggers
        cooccurrences = 0
        followed_by_migraine = 0
        
        for event1 in (e for e in trigger_history if e.trigger_type == type1):
            # Find type2 events within window of event1
            for event2 in (e for e in trigger_history if e.trigger_type == type2):
                if abs(event2.timestamp - event1.timestamp) <= window:
                    cooccurrences += 1
                    # Check if followed by migraine
                    for migraine in migraine_events:
                        if event1.timestamp < migraine <= event1.timestamp + window:
                            followed_by_migraine += 1
                            break
        
        return followed_by_migraine / cooccurrences if cooccurrences > 0 else 0.0
    
    def _analyze_temporal_trigger_patterns(self,
                                         trigger_history: List[TriggerEvent],
                                         migraine_events: List[datetime]) -> Dict[str, Dict[str, Any]]:
        """Analyze temporal patterns in historical trigger data."""
        patterns = {}
        trigger_types = set(event.trigger_type for event in trigger_history)
        
        for trigger_type in trigger_types:
            # Get events of this type
            type_events = [e for e in trigger_history if e.trigger_type == trigger_type]
            
            # Analyze daily pattern
            daily = self._analyze_historical_daily_pattern(type_events)
            
            # Analyze weekly pattern
            weekly = self._analyze_historical_weekly_pattern(type_events)
            
            # Analyze seasonal pattern if enough data
            seasonal = self._analyze_historical_seasonal_pattern(type_events)
            
            patterns[trigger_type] = {
                'daily': daily,
                'weekly': weekly,
                'seasonal': seasonal
            }
        
        return patterns
    
    def _analyze_historical_daily_pattern(self,
                                        trigger_events: List[TriggerEvent]) -> Dict[str, Any]:
        """Analyze daily patterns in historical trigger events."""
        if not trigger_events:
            return {'strength': 0.0, 'peak_hours': []}
        
        # Count events by hour
        hours = [0] * 24
        for event in trigger_events:
            hours[event.timestamp.hour] += 1
        
        # Find peak hours
        mean_events = np.mean(hours)
        std_events = np.std(hours)
        peak_hours = [h for h, count in enumerate(hours) if count > mean_events + std_events]
        
        # Calculate pattern strength
        strength = std_events / mean_events if mean_events > 0 else 0.0
        
        return {
            'strength': float(strength),
            'peak_hours': peak_hours,
            'hourly_counts': hours
        }
    
    def _analyze_historical_weekly_pattern(self,
                                         trigger_events: List[TriggerEvent]) -> Dict[str, Any]:
        """Analyze weekly patterns in historical trigger events."""
        if not trigger_events:
            return {'strength': 0.0, 'peak_days': []}
        
        # Count events by day
        days = [0] * 7
        for event in trigger_events:
            days[event.timestamp.weekday()] += 1
        
        # Find peak days
        mean_events = np.mean(days)
        std_events = np.std(days)
        peak_days = [d for d, count in enumerate(days) if count > mean_events + std_events]
        
        # Calculate pattern strength
        strength = std_events / mean_events if mean_events > 0 else 0.0
        
        return {
            'strength': float(strength),
            'peak_days': peak_days,
            'daily_counts': days
        }
    
    def _analyze_historical_seasonal_pattern(self,
                                           trigger_events: List[TriggerEvent]) -> Dict[str, Any]:
        """Analyze seasonal patterns in historical trigger events."""
        if not trigger_events:
            return {'strength': 0.0, 'peak_months': []}
        
        # Count events by month
        months = [0] * 12
        for event in trigger_events:
            months[event.timestamp.month - 1] += 1
        
        # Find peak months
        mean_events = np.mean(months)
        std_events = np.std(months)
        peak_months = [m for m, count in enumerate(months) if count > mean_events + std_events]
        
        # Calculate pattern strength
        strength = std_events / mean_events if mean_events > 0 else 0.0
        
        return {
            'strength': float(strength),
            'peak_months': peak_months,
            'monthly_counts': months
        }
    
    def _calculate_threshold_ranges(self,
                                  trigger_history: List[TriggerEvent],
                                  migraine_events: List[datetime]) -> Dict[str, Tuple[float, float]]:
        """Calculate threshold ranges for each trigger type."""
        thresholds = {}
        trigger_types = set(event.trigger_type for event in trigger_history)
        
        for trigger_type in trigger_types:
            # Get events of this type
            type_events = [e for e in trigger_history if e.trigger_type == trigger_type]
            
            # Find events that preceded migraines
            migraine_triggers = []
            window = timedelta(hours=self.sensitivity_window)
            
            for event in type_events:
                for migraine in migraine_events:
                    if event.timestamp < migraine <= event.timestamp + window:
                        migraine_triggers.append(event.intensity)
                        break
            
            if migraine_triggers:
                # Calculate threshold range
                lower = np.percentile(migraine_triggers, 25)
                upper = np.percentile(migraine_triggers, 75)
                thresholds[trigger_type] = (float(lower), float(upper))
            else:
                thresholds[trigger_type] = (0.0, 0.0)
        
        return thresholds
    
    def _calculate_profile_confidence(self,
                                    trigger_history: List[TriggerEvent],
                                    migraine_events: List[datetime],
                                    patterns: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for trigger profile components."""
        confidence = {}
        trigger_types = set(event.trigger_type for event in trigger_history)
        
        for trigger_type in trigger_types:
            # Get events of this type
            type_events = [e for e in trigger_history if e.trigger_type == trigger_type]
            
            # Calculate temporal consistency (ensure between 0 and 1)
            temporal_scores = [
                min(1.0, patterns[trigger_type]['daily']['strength']),
                min(1.0, patterns[trigger_type]['weekly']['strength']),
                min(1.0, patterns[trigger_type]['seasonal']['strength'])
            ]
            temporal_score = np.mean(temporal_scores)
            
            # Calculate prediction accuracy (already between 0 and 1)
            accuracy = self._calculate_type_sensitivity(type_events, migraine_events)
            
            # Calculate data sufficiency (already between 0 and 1)
            data_score = min(1.0, len(type_events) / 100)  # Saturate at 100 events
            
            # Combine scores (weighted average ensures result between 0 and 1)
            confidence[trigger_type] = float(
                0.4 * temporal_score +
                0.4 * accuracy +
                0.2 * data_score
            )
        
        return confidence
    
    def _analyze_threshold_levels(self,
                                trigger_data: np.ndarray,
                                symptom_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze different threshold levels for trigger detection."""
        thresholds = {}
        
        # Calculate basic statistics
        trigger_mean = np.mean(trigger_data)
        trigger_std = np.std(trigger_data)
        
        # Define threshold levels
        levels = np.linspace(
            trigger_mean - 2*trigger_std,
            trigger_mean + 2*trigger_std,
            10
        )
        
        for level in levels:
            # Find episodes above threshold
            episodes = trigger_data > level
            
            # Calculate symptom response
            responses = []
            for symptom in symptom_data.values():
                # Calculate average symptom level after trigger episodes
                response = np.mean([
                    np.mean(symptom[i:i+self.sensitivity_window])
                    for i in np.where(episodes)[0]
                    if i + self.sensitivity_window <= len(symptom)
                ])
                responses.append(response)
            
            # Store average response for this threshold
            thresholds[f'level_{level:.2f}'] = float(np.mean(responses))
        
        return thresholds
    
    def _analyze_temporal_sensitivity(self,
                                    trigger_data: np.ndarray,
                                    symptom_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze temporal sensitivity patterns."""
        sensitivity = {}
        
        # Find trigger events
        peaks, _ = self._find_trigger_events(trigger_data)
        
        # Analyze response times
        response_times = []
        response_magnitudes = []
        
        for peak in peaks:
            if peak + self.sensitivity_window >= len(trigger_data):
                continue
                
            # Calculate average symptom response in window
            for symptom in symptom_data.values():
                window = symptom[peak:peak+self.sensitivity_window]
                max_response = np.max(window)
                response_time = np.argmax(window)
                
                response_times.append(response_time)
                response_magnitudes.append(max_response)
        
        if response_times:
            sensitivity['mean_response_time'] = float(np.mean(response_times))
            sensitivity['std_response_time'] = float(np.std(response_times))
            sensitivity['mean_magnitude'] = float(np.mean(response_magnitudes))
            sensitivity['std_magnitude'] = float(np.std(response_magnitudes))
        else:
            sensitivity['mean_response_time'] = 0.0
            sensitivity['std_response_time'] = 0.0
            sensitivity['mean_magnitude'] = 0.0
            sensitivity['std_magnitude'] = 0.0
        
        return sensitivity
    
    def _calculate_sensitivity_confidence(self,
                                       trigger_data: np.ndarray,
                                       symptom_data: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for sensitivity analysis."""
        confidence = {}
        
        # Bootstrap parameters
        n_bootstrap = 1000
        confidence_level = 0.95
        
        # Prepare data
        n_samples = len(trigger_data)
        
        for symptom_name, symptom in symptom_data.items():
            bootstrap_responses = []
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(n_samples, n_samples)
                t_sample = trigger_data[indices]
                s_sample = symptom[indices]
                
                # Calculate response
                peaks, _ = self._find_trigger_events(t_sample)
                if len(peaks) > 0:
                    responses = []
                    for peak in peaks:
                        if peak + self.sensitivity_window < len(s_sample):
                            response = np.mean(s_sample[peak:peak+self.sensitivity_window])
                            responses.append(response)
                    
                    if responses:
                        bootstrap_responses.append(np.mean(responses))
            
            if bootstrap_responses:
                # Calculate confidence intervals
                lower = np.percentile(bootstrap_responses, (1 - confidence_level) * 100 / 2)
                upper = np.percentile(bootstrap_responses, (1 + confidence_level) * 100 / 2)
                confidence[symptom_name] = (float(lower), float(upper))
            else:
                confidence[symptom_name] = (0.0, 0.0)
        
        return confidence
    
    def _calculate_baseline_stats(self,
                                  trigger_data: np.ndarray,
                                  baseline_period: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate baseline statistics for trigger data."""
        # Convert timestamps to indices if needed
        if isinstance(baseline_period[0], datetime):
            start_idx = 0
            end_idx = len(trigger_data)
        else:
            start_idx = int(baseline_period[0])
            end_idx = int(baseline_period[1])
        
        # Extract baseline data
        baseline = trigger_data[start_idx:end_idx]
        
        # Calculate statistics
        stats = {
            'mean': float(np.mean(baseline)),
            'std': float(np.std(baseline)),
            'median': float(np.median(baseline)),
            'min': float(np.min(baseline)),
            'max': float(np.max(baseline)),
            'q25': float(np.percentile(baseline, 25)),
            'q75': float(np.percentile(baseline, 75))
        }
        
        return stats 
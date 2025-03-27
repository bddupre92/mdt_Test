"""
Domain-Specific Preprocessing Module

This module provides specialized preprocessing operations for medical and clinical data,
particularly focusing on migraine-related features and patterns. It extends the
preprocessing pipeline with domain-specific knowledge and transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import re
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
import logging

from data.preprocessing_pipeline import PreprocessingOperation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicationNormalizer(PreprocessingOperation):
    """Normalize medication names and dosages in clinical data."""
    
    def __init__(self, medication_cols: List[str] = None, 
                 dosage_cols: List[str] = None,
                 medication_mapping: Dict[str, str] = None,
                 patient_id_col: str = 'patient_id',
                 timestamp_col: str = 'date'):
        """Initialize the medication normalizer.
        
        Args:
            medication_cols: Columns containing medication names
            dosage_cols: Columns containing dosage information
            medication_mapping: Dictionary mapping raw medication names to standardized names
            patient_id_col: Name of the patient ID column
            timestamp_col: Name of the timestamp column
        """
        self.medication_cols = medication_cols or []
        self.dosage_cols = dosage_cols or []
        self.medication_mapping = medication_mapping or {}
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.common_units = {
            'mg': 'mg',
            'milligram': 'mg',
            'milligrams': 'mg',
            'g': 'g',
            'gram': 'g',
            'grams': 'g',
            'mcg': 'mcg',
            'microgram': 'mcg',
            'micrograms': 'mcg',
            'ml': 'ml',
            'milliliter': 'ml',
            'milliliters': 'ml'
        }
        self.dosage_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)')
        self.detected_medications = set()
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the medication normalizer to the data."""
        # Detect medications if no mapping provided
        if not self.medication_mapping and self.medication_cols:
            for col in self.medication_cols:
                if col in data.columns:
                    unique_meds = data[col].dropna().unique()
                    self.detected_medications.update(unique_meds)
                    
            # Create a basic mapping for detected medications
            for med in self.detected_medications:
                # Convert to lowercase and remove special characters
                normalized = re.sub(r'[^\w\s]', '', str(med).lower())
                # Remove dosage information if present
                normalized = re.sub(r'\d+\s*(?:mg|g|mcg|ml)', '', normalized).strip()
                # Add to mapping if not already present
                if med not in self.medication_mapping:
                    self.medication_mapping[med] = normalized
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Normalize medication names and dosages in the data."""
        result = data.copy()
        
        # Normalize medication names
        for col in self.medication_cols:
            if col in result.columns:
                result[f"{col}_normalized"] = result[col].map(
                    lambda x: self.medication_mapping.get(x, x) if pd.notna(x) else x
                )
                
        # Normalize dosage information
        for col in self.dosage_cols:
            if col in result.columns:
                result[f"{col}_value"] = result[col].apply(self._extract_dosage_value)
                result[f"{col}_unit"] = result[col].apply(self._extract_dosage_unit)
                
        return result
        
    def _extract_dosage_value(self, dosage_str: str) -> Optional[float]:
        """Extract the numeric value from a dosage string."""
        if not dosage_str or not isinstance(dosage_str, str):
            return None
            
        match = self.dosage_pattern.search(dosage_str)
        if match:
            return float(match.group(1))
        return None
        
    def _extract_dosage_unit(self, dosage_str: str) -> Optional[str]:
        """Extract the unit from a dosage string."""
        if not dosage_str or not isinstance(dosage_str, str):
            return None
            
        match = self.dosage_pattern.search(dosage_str)
        if match:
            raw_unit = match.group(2).lower()
            return self.common_units.get(raw_unit, raw_unit)
        return None
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'medication_cols': self.medication_cols,
            'dosage_cols': self.dosage_cols,
            'medication_mapping': self.medication_mapping,
            'patient_id_col': self.patient_id_col,
            'timestamp_col': self.timestamp_col
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'medication_cols' in params:
            self.medication_cols = params['medication_cols']
        if 'dosage_cols' in params:
            self.dosage_cols = params['dosage_cols']
        if 'medication_mapping' in params:
            self.medication_mapping = params['medication_mapping']
        if 'patient_id_col' in params:
            self.patient_id_col = params['patient_id_col']
        if 'timestamp_col' in params:
            self.timestamp_col = params['timestamp_col']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Calculate the percentage of medications that were normalized
        for col in self.medication_cols:
            if col in data.columns and f"{col}_normalized" in transformed_data.columns:
                orig_values = data[col].dropna()
                norm_values = transformed_data[f"{col}_normalized"].dropna()
                
                if len(orig_values) > 0:
                    # Count how many values changed during normalization
                    changed = sum(orig_values != norm_values)
                    metrics[f"{col}_normalization_rate"] = changed / len(orig_values)
                    
        return metrics


class SymptomExtractor(PreprocessingOperation):
    """Extract and normalize symptoms from clinical text data."""
    
    def __init__(self, text_cols: List[str] = None, 
                 symptom_dictionary: Dict[str, List[str]] = None,
                 extract_severity: bool = True):
        """Initialize the symptom extractor.
        
        Args:
            text_cols: Columns containing clinical text
            symptom_dictionary: Dictionary mapping symptom categories to related terms
            extract_severity: Whether to extract severity information
        """
        self.text_cols = text_cols or []
        self.symptom_dictionary = symptom_dictionary or self._default_symptom_dictionary()
        self.extract_severity = extract_severity
        self.severity_terms = {
            'mild': 1,
            'slight': 1,
            'minimal': 1,
            'moderate': 2,
            'medium': 2,
            'average': 2,
            'severe': 3,
            'intense': 3,
            'extreme': 3,
            'debilitating': 3
        }
        
    def _default_symptom_dictionary(self) -> Dict[str, List[str]]:
        """Create a default dictionary of migraine-related symptoms."""
        return {
            'headache': ['headache', 'head pain', 'head ache', 'cephalgia'],
            'nausea': ['nausea', 'nauseated', 'queasy', 'sick to stomach'],
            'vomiting': ['vomiting', 'vomit', 'throwing up', 'emesis'],
            'photophobia': ['photophobia', 'light sensitivity', 'sensitive to light'],
            'phonophobia': ['phonophobia', 'sound sensitivity', 'sensitive to sound', 'noise sensitivity'],
            'aura': ['aura', 'visual disturbance', 'visual symptoms', 'visual aura'],
            'dizziness': ['dizziness', 'dizzy', 'vertigo', 'lightheaded', 'light headed'],
            'fatigue': ['fatigue', 'tired', 'exhausted', 'lethargy'],
            'neck_pain': ['neck pain', 'neck stiffness', 'stiff neck']
        }
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the symptom extractor to the data."""
        # Nothing to fit for this operation
        pass
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract symptoms from clinical text data."""
        result = data.copy()
        
        for col in self.text_cols:
            if col in result.columns:
                # Create binary features for each symptom category
                for symptom, terms in self.symptom_dictionary.items():
                    result[f"has_{symptom}"] = result[col].apply(
                        lambda x: self._check_for_symptom(x, terms) if pd.notna(x) else 0
                    )
                    
                    # Extract severity if requested
                    if self.extract_severity:
                        result[f"{symptom}_severity"] = result[col].apply(
                            lambda x: self._extract_symptom_severity(x, terms) if pd.notna(x) else 0
                        )
                        
        return result
        
    def _check_for_symptom(self, text: str, terms: List[str]) -> int:
        """Check if any of the symptom terms appear in the text."""
        if not isinstance(text, str):
            return 0
            
        text = text.lower()
        for term in terms:
            if term in text:
                return 1
        return 0
        
    def _extract_symptom_severity(self, text: str, symptom_terms: List[str]) -> int:
        """Extract the severity of a symptom from text."""
        if not isinstance(text, str):
            return 0
            
        text = text.lower()
        
        # Check if any symptom term is present
        symptom_present = False
        for term in symptom_terms:
            if term in text:
                symptom_present = True
                break
                
        if not symptom_present:
            return 0
            
        # Look for severity terms near symptom terms
        max_severity = 0
        
        for severity_term, severity_value in self.severity_terms.items():
            if severity_term in text:
                max_severity = max(max_severity, severity_value)
                
        # Default to moderate (2) if symptom is present but no severity is specified
        return max_severity if max_severity > 0 else 2
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'text_cols': self.text_cols,
            'symptom_dictionary': self.symptom_dictionary,
            'extract_severity': self.extract_severity
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'text_cols' in params:
            self.text_cols = params['text_cols']
        if 'symptom_dictionary' in params:
            self.symptom_dictionary = params['symptom_dictionary']
        if 'extract_severity' in params:
            self.extract_severity = params['extract_severity']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Calculate the percentage of records with each symptom
        for symptom in self.symptom_dictionary.keys():
            col_name = f"has_{symptom}"
            if col_name in transformed_data.columns:
                symptom_rate = transformed_data[col_name].mean()
                metrics[f"{symptom}_rate"] = symptom_rate
                
        return metrics


class TemporalPatternExtractor(PreprocessingOperation):
    """Extract temporal patterns from time series clinical data."""
    
    def __init__(self, timestamp_col: str, 
                 event_cols: List[str] = None,
                 window_days: int = 30,
                 min_events: int = 3,
                 extract_frequency: bool = True,
                 extract_periodicity: bool = True,
                 extract_clustering: bool = True):
        """Initialize the temporal pattern extractor.
        
        Args:
            timestamp_col: Column containing timestamps
            event_cols: Columns indicating events (e.g., migraine attacks)
            window_days: Window size in days for pattern analysis
            min_events: Minimum number of events required for pattern analysis
            extract_frequency: Whether to extract frequency features
            extract_periodicity: Whether to extract periodicity features
            extract_clustering: Whether to extract temporal clustering features
        """
        self.timestamp_col = timestamp_col
        self.event_cols = event_cols or []
        self.window_days = window_days
        self.min_events = min_events
        self.extract_frequency = extract_frequency
        self.extract_periodicity = extract_periodicity
        self.extract_clustering = extract_clustering
        self.frequency_stats = {}
        self.periodicity_stats = {}
        self.clustering_stats = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the temporal pattern extractor to the data."""
        if self.timestamp_col not in data.columns:
            logger.warning(f"Timestamp column '{self.timestamp_col}' not found in data")
            return
            
        # Ensure timestamp column is datetime
        timestamps = pd.to_datetime(data[self.timestamp_col], errors='coerce')
        
        # Process each event column
        for event_col in self.event_cols:
            if event_col in data.columns:
                # Get events (where event column is True/1)
                events = data[data[event_col] == 1]
                event_timestamps = pd.to_datetime(events[self.timestamp_col], errors='coerce').dropna()
                
                if len(event_timestamps) >= self.min_events:
                    # Calculate frequency statistics
                    if self.extract_frequency:
                        self._calculate_frequency_stats(event_timestamps, event_col)
                        
                    # Calculate periodicity statistics
                    if self.extract_periodicity:
                        self._calculate_periodicity_stats(event_timestamps, event_col)
                        
                    # Calculate clustering statistics
                    if self.extract_clustering:
                        self._calculate_clustering_stats(event_timestamps, event_col)
        
    def _calculate_frequency_stats(self, timestamps: pd.Series, event_col: str) -> None:
        """Calculate frequency statistics for events."""
        # Sort timestamps
        timestamps = timestamps.sort_values()
        
        # Calculate total duration in days
        duration_days = (timestamps.max() - timestamps.min()).total_seconds() / (24 * 3600)
        
        if duration_days > 0:
            # Calculate events per month (30 days)
            events_per_month = len(timestamps) * 30 / duration_days
            
            # Store statistics
            self.frequency_stats[event_col] = {
                'count': len(timestamps),
                'duration_days': duration_days,
                'events_per_month': events_per_month
            }
        
    def _calculate_periodicity_stats(self, timestamps: pd.Series, event_col: str) -> None:
        """Calculate periodicity statistics for events."""
        # Sort timestamps
        timestamps = timestamps.sort_values()
        
        if len(timestamps) >= 3:
            # Calculate intervals between events in days
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds() / (24 * 3600)
                intervals.append(interval)
                
            # Calculate statistics of intervals
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            cv_interval = std_interval / mean_interval if mean_interval > 0 else 0
            
            # Store statistics
            self.periodicity_stats[event_col] = {
                'mean_interval_days': mean_interval,
                'std_interval_days': std_interval,
                'cv_interval': cv_interval
            }
        
    def _calculate_clustering_stats(self, timestamps: pd.Series, event_col: str) -> None:
        """Calculate temporal clustering statistics for events."""
        # Sort timestamps
        timestamps = timestamps.sort_values()
        
        if len(timestamps) >= self.min_events:
            # Calculate clusters (events within 3 days of each other)
            clusters = []
            current_cluster = [timestamps.iloc[0]]
            
            for i in range(1, len(timestamps)):
                if (timestamps.iloc[i] - current_cluster[-1]).total_seconds() <= 3 * 24 * 3600:
                    # Add to current cluster
                    current_cluster.append(timestamps.iloc[i])
                else:
                    # Start a new cluster
                    clusters.append(current_cluster)
                    current_cluster = [timestamps.iloc[i]]
                    
            # Add the last cluster
            if current_cluster:
                clusters.append(current_cluster)
                
            # Calculate statistics
            cluster_sizes = [len(cluster) for cluster in clusters]
            mean_cluster_size = np.mean(cluster_sizes)
            max_cluster_size = np.max(cluster_sizes)
            cluster_count = len(clusters)
            
            # Store statistics
            self.clustering_stats[event_col] = {
                'cluster_count': cluster_count,
                'mean_cluster_size': mean_cluster_size,
                'max_cluster_size': max_cluster_size
            }
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract temporal patterns from the data."""
        result = data.copy()
        
        if self.timestamp_col not in result.columns:
            return result
            
        # Ensure timestamp column is datetime
        result[self.timestamp_col] = pd.to_datetime(result[self.timestamp_col], errors='coerce')
        
        # Add patient-level frequency features
        if self.extract_frequency and self.frequency_stats:
            for event_col, stats in self.frequency_stats.items():
                result[f"{event_col}_monthly_frequency"] = stats['events_per_month']
                
        # Add patient-level periodicity features
        if self.extract_periodicity and self.periodicity_stats:
            for event_col, stats in self.periodicity_stats.items():
                result[f"{event_col}_mean_interval_days"] = stats['mean_interval_days']
                result[f"{event_col}_cv_interval"] = stats['cv_interval']
                
        # Add patient-level clustering features
        if self.extract_clustering and self.clustering_stats:
            for event_col, stats in self.clustering_stats.items():
                result[f"{event_col}_cluster_count"] = stats['cluster_count']
                result[f"{event_col}_mean_cluster_size"] = stats['mean_cluster_size']
                
        # Calculate time-based features for each record
        for event_col in self.event_cols:
            if event_col in result.columns:
                # Calculate days since last event
                result[f"days_since_last_{event_col}"] = result.apply(
                    lambda row: self._calculate_days_since_last_event(
                        row[self.timestamp_col], result, event_col
                    ),
                    axis=1
                )
                
                # Calculate days until next event
                result[f"days_until_next_{event_col}"] = result.apply(
                    lambda row: self._calculate_days_until_next_event(
                        row[self.timestamp_col], result, event_col
                    ),
                    axis=1
                )
                
        return result
        
    def _calculate_days_since_last_event(self, timestamp, data: pd.DataFrame, event_col: str) -> float:
        """Calculate days since the last event."""
        if pd.isna(timestamp):
            return np.nan
            
        # Get events before this timestamp
        events = data[(data[event_col] == 1) & (data[self.timestamp_col] < timestamp)]
        
        if len(events) == 0:
            return np.nan
            
        # Get the most recent event
        last_event = events[self.timestamp_col].max()
        
        # Calculate days difference
        days_diff = (timestamp - last_event).total_seconds() / (24 * 3600)
        
        return days_diff
        
    def _calculate_days_until_next_event(self, timestamp, data: pd.DataFrame, event_col: str) -> float:
        """Calculate days until the next event."""
        if pd.isna(timestamp):
            return np.nan
            
        # Get events after this timestamp
        events = data[(data[event_col] == 1) & (data[self.timestamp_col] > timestamp)]
        
        if len(events) == 0:
            return np.nan
            
        # Get the next event
        next_event = events[self.timestamp_col].min()
        
        # Calculate days difference
        days_diff = (next_event - timestamp).total_seconds() / (24 * 3600)
        
        return days_diff
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'timestamp_col': self.timestamp_col,
            'event_cols': self.event_cols,
            'window_days': self.window_days,
            'min_events': self.min_events,
            'extract_frequency': self.extract_frequency,
            'extract_periodicity': self.extract_periodicity,
            'extract_clustering': self.extract_clustering
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'timestamp_col' in params:
            self.timestamp_col = params['timestamp_col']
        if 'event_cols' in params:
            self.event_cols = params['event_cols']
        if 'window_days' in params:
            self.window_days = params['window_days']
        if 'min_events' in params:
            self.min_events = params['min_events']
        if 'extract_frequency' in params:
            self.extract_frequency = params['extract_frequency']
        if 'extract_periodicity' in params:
            self.extract_periodicity = params['extract_periodicity']
        if 'extract_clustering' in params:
            self.extract_clustering = params['extract_clustering']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Calculate the percentage of records with temporal features
        for event_col in self.event_cols:
            days_since_col = f"days_since_last_{event_col}"
            if days_since_col in transformed_data.columns:
                coverage = transformed_data[days_since_col].notna().mean()
                metrics[f"{days_since_col}_coverage"] = coverage
                
        return metrics


class ComorbidityAnalyzer(PreprocessingOperation):
    """Analyze and extract comorbidity features from clinical data."""
    
    def __init__(self, condition_cols: List[str] = None, 
                 comorbidity_groups: Dict[str, List[str]] = None,
                 calculate_indices: bool = True):
        """Initialize the comorbidity analyzer.
        
        Args:
            condition_cols: Columns containing condition/diagnosis information
            comorbidity_groups: Dictionary mapping comorbidity groups to related conditions
            calculate_indices: Whether to calculate comorbidity indices
        """
        self.condition_cols = condition_cols or []
        self.comorbidity_groups = comorbidity_groups or self._default_comorbidity_groups()
        self.calculate_indices = calculate_indices
        self.condition_weights = {
            'hypertension': 1,
            'diabetes': 1,
            'coronary_artery_disease': 1,
            'heart_failure': 2,
            'arrhythmia': 1,
            'stroke': 2,
            'asthma': 1,
            'copd': 2,
            'chronic_kidney_disease': 2,
            'liver_disease': 2,
            'cancer': 2,
            'depression': 1,
            'anxiety': 1,
            'bipolar_disorder': 1,
            'schizophrenia': 2,
            'epilepsy': 1,
            'parkinsons': 2,
            'multiple_sclerosis': 2,
            'fibromyalgia': 1,
            'chronic_fatigue': 1,
            'irritable_bowel_syndrome': 1,
            'inflammatory_bowel_disease': 1,
            'rheumatoid_arthritis': 1,
            'osteoarthritis': 1,
            'osteoporosis': 1,
            'sleep_apnea': 1,
            'insomnia': 1,
            'obesity': 1,
            'hypothyroidism': 1,
            'hyperthyroidism': 1
        }
        
    def _default_comorbidity_groups(self) -> Dict[str, List[str]]:
        """Create a default dictionary of comorbidity groups."""
        return {
            'cardiovascular': [
                'hypertension', 'coronary artery disease', 'heart failure', 
                'arrhythmia', 'stroke', 'tia', 'atherosclerosis'
            ],
            'respiratory': [
                'asthma', 'copd', 'sleep apnea', 'pulmonary fibrosis', 
                'chronic bronchitis', 'emphysema'
            ],
            'metabolic': [
                'diabetes', 'obesity', 'metabolic syndrome', 'hyperlipidemia',
                'hypothyroidism', 'hyperthyroidism'
            ],
            'neurological': [
                'epilepsy', 'parkinsons', 'multiple sclerosis', 'alzheimers',
                'dementia', 'neuropathy'
            ],
            'psychiatric': [
                'depression', 'anxiety', 'bipolar disorder', 'schizophrenia',
                'ptsd', 'ocd'
            ],
            'pain': [
                'fibromyalgia', 'chronic pain', 'arthritis', 'back pain',
                'neck pain', 'osteoarthritis', 'rheumatoid arthritis'
            ],
            'gastrointestinal': [
                'irritable bowel syndrome', 'inflammatory bowel disease', 'crohns',
                'ulcerative colitis', 'gerd', 'peptic ulcer'
            ]
        }
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the comorbidity analyzer to the data."""
        # Nothing to fit for this operation
        pass
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract comorbidity features from the data."""
        result = data.copy()
        
        # Create binary features for each comorbidity group
        for group, conditions in self.comorbidity_groups.items():
            result[f"has_{group}_comorbidity"] = 0
            
            for col in self.condition_cols:
                if col in result.columns:
                    # Check if any condition in the group is present
                    result[f"has_{group}_comorbidity"] = result.apply(
                        lambda row: 1 if self._check_for_conditions(row[col], conditions) else row[f"has_{group}_comorbidity"],
                        axis=1
                    )
                    
        # Calculate comorbidity count and index
        if self.calculate_indices:
            result['comorbidity_count'] = 0
            result['comorbidity_index'] = 0
            
            for condition, weight in self.condition_weights.items():
                has_condition = 0
                
                for col in self.condition_cols:
                    if col in result.columns:
                        # Check if the condition is present
                        result[f"has_{condition}"] = result.apply(
                            lambda row: 1 if self._check_for_conditions(row[col], [condition]) else 0,
                            axis=1
                        )
                        
                        # Update comorbidity count and index
                        result['comorbidity_count'] += result[f"has_{condition}"]
                        result['comorbidity_index'] += result[f"has_{condition}"] * weight
                        
        return result
        
    def _check_for_conditions(self, text: Any, conditions: List[str]) -> bool:
        """Check if any of the conditions appear in the text."""
        if not isinstance(text, str):
            return False
            
        text = text.lower()
        for condition in conditions:
            if condition.lower() in text:
                return True
        return False
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'condition_cols': self.condition_cols,
            'comorbidity_groups': self.comorbidity_groups,
            'calculate_indices': self.calculate_indices
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'condition_cols' in params:
            self.condition_cols = params['condition_cols']
        if 'comorbidity_groups' in params:
            self.comorbidity_groups = params['comorbidity_groups']
        if 'calculate_indices' in params:
            self.calculate_indices = params['calculate_indices']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Calculate the percentage of records with each comorbidity group
        for group in self.comorbidity_groups.keys():
            col_name = f"has_{group}_comorbidity"
            if col_name in transformed_data.columns:
                comorbidity_rate = transformed_data[col_name].mean()
                metrics[f"{group}_comorbidity_rate"] = comorbidity_rate
                
        # Calculate average comorbidity count and index if available
        if 'comorbidity_count' in transformed_data.columns:
            metrics['avg_comorbidity_count'] = transformed_data['comorbidity_count'].mean()
            
        if 'comorbidity_index' in transformed_data.columns:
            metrics['avg_comorbidity_index'] = transformed_data['comorbidity_index'].mean()
            
        return metrics


class PhysiologicalSignalProcessor(PreprocessingOperation):
    """Process physiological signals and extract relevant features for migraine prediction."""
    
    def __init__(self, 
                 vital_cols: List[str] = None,
                 patient_id_col: str = 'patient_id',
                 timestamp_col: str = 'date',
                 calculate_variability: bool = True,
                 calculate_trends: bool = True,
                 window_size: int = 7):
        """Initialize the physiological signal processor.
        
        Args:
            vital_cols: Columns containing vital signs data (heart rate, blood pressure, etc.)
            patient_id_col: Column containing patient identifiers
            timestamp_col: Column containing timestamps
            calculate_variability: Whether to calculate variability metrics
            calculate_trends: Whether to calculate trend features
            window_size: Window size in days for calculating trends and variability
        """
        self.vital_cols = vital_cols or []
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.calculate_variability = calculate_variability
        self.calculate_trends = calculate_trends
        self.window_size = window_size
        self.baseline_stats = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the physiological signal processor to the data.
        
        Args:
            data: The data to fit the processor to
            **kwargs: Additional arguments
        """
        if self.patient_id_col in data.columns and self.timestamp_col in data.columns:
            # Calculate baseline statistics for each patient and vital sign
            for col in self.vital_cols:
                if col in data.columns:
                    # Group by patient and calculate statistics
                    grouped = data.groupby(self.patient_id_col)[col]
                    self.baseline_stats[col] = {
                        'mean': grouped.mean().to_dict(),
                        'std': grouped.std().to_dict(),
                        'min': grouped.min().to_dict(),
                        'max': grouped.max().to_dict()
                    }
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process physiological signals and extract features.
        
        Args:
            data: The data to transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed data with physiological features
        """
        result = data.copy()
        
        if self.patient_id_col not in result.columns or self.timestamp_col not in result.columns:
            logger.warning("Patient ID or timestamp column missing. Cannot process physiological signals.")
            return result
            
        # Ensure timestamp column is datetime type
        if result[self.timestamp_col].dtype != 'datetime64[ns]':
            try:
                result[self.timestamp_col] = pd.to_datetime(result[self.timestamp_col])
            except Exception as e:
                logger.error(f"Failed to convert timestamp column to datetime: {e}")
                return result
                
        # Process each vital sign column
        for col in self.vital_cols:
            if col in result.columns:
                # Calculate normalized values
                if col in self.baseline_stats and self.patient_id_col in result.columns:
                    # Normalize based on patient's baseline
                    result[f"{col}_normalized"] = result.apply(
                        lambda row: self._normalize_value(row[col], col, row[self.patient_id_col]), 
                        axis=1
                    )
                    
                # Calculate variability metrics
                if self.calculate_variability:
                    # Group by patient and calculate rolling statistics
                    result = self._add_variability_features(result, col)
                    
                # Calculate trend features
                if self.calculate_trends:
                    result = self._add_trend_features(result, col)
                    
        return result
        
    def _normalize_value(self, value, col, patient_id):
        """Normalize a value based on the patient's baseline statistics."""
        if pd.isna(value) or patient_id not in self.baseline_stats[col]['mean']:
            return np.nan
            
        mean = self.baseline_stats[col]['mean'][patient_id]
        std = self.baseline_stats[col]['std'][patient_id]
        
        if std == 0 or pd.isna(std):
            return 0  # Avoid division by zero
            
        return (value - mean) / std
        
    def _add_variability_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add variability features for a vital sign column."""
        result = data.copy()
        
        # Sort by patient and timestamp
        result = result.sort_values([self.patient_id_col, self.timestamp_col])
        
        # Group by patient
        grouped = result.groupby(self.patient_id_col)
        
        # Calculate rolling statistics
        window = self.window_size
        
        # Apply rolling calculations for each patient
        rolling_std = grouped[col].rolling(window=window, min_periods=2).std().reset_index(level=0, drop=True)
        rolling_range = grouped[col].rolling(window=window, min_periods=2).apply(
            lambda x: x.max() - x.min() if len(x) >= 2 else np.nan
        ).reset_index(level=0, drop=True)
        
        # Add to result dataframe
        result[f"{col}_rolling_std"] = rolling_std
        result[f"{col}_rolling_range"] = rolling_range
        
        return result
        
    def _add_trend_features(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add trend features for a vital sign column."""
        result = data.copy()
        
        # Sort by patient and timestamp
        result = result.sort_values([self.patient_id_col, self.timestamp_col])
        
        # Group by patient
        grouped = result.groupby(self.patient_id_col)
        
        # Calculate rolling mean and slope
        window = self.window_size
        
        # Apply rolling calculations for each patient
        rolling_mean = grouped[col].rolling(window=window, min_periods=2).mean().reset_index(level=0, drop=True)
        
        # Calculate slope (rate of change)
        # This is a simple approximation - for each window, we calculate (last_value - first_value) / time_diff
        def calculate_slope(series):
            if len(series) < 2:
                return np.nan
            # Get first and last values and their timestamps
            first_idx = series.index[0]
            last_idx = series.index[-1]
            first_val = series.iloc[0]
            last_val = series.iloc[-1]
            
            # Calculate time difference - handle different index types
            try:
                # For datetime indices
                time_diff = (last_idx - first_idx).total_seconds() / (24 * 3600)
            except (AttributeError, TypeError):
                # For numeric indices, assume each step is one day
                time_diff = float(last_idx - first_idx)
                
            if time_diff == 0:
                return 0  # Avoid division by zero
            return (last_val - first_val) / time_diff
        
        # Apply the slope calculation to each window
        rolling_slope = grouped.apply(
            lambda group: group[col].rolling(window=window, min_periods=2).apply(
                calculate_slope, raw=False
            )
        ).reset_index(level=0, drop=True)
        
        # Add to result dataframe
        result[f"{col}_rolling_mean"] = rolling_mean
        result[f"{col}_rolling_slope"] = rolling_slope
        
        return result
        
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation."""
        return {
            'vital_cols': self.vital_cols,
            'patient_id_col': self.patient_id_col,
            'timestamp_col': self.timestamp_col,
            'calculate_variability': self.calculate_variability,
            'calculate_trends': self.calculate_trends,
            'window_size': self.window_size
        }
        
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation."""
        if 'vital_cols' in params:
            self.vital_cols = params['vital_cols']
        if 'patient_id_col' in params:
            self.patient_id_col = params['patient_id_col']
        if 'timestamp_col' in params:
            self.timestamp_col = params['timestamp_col']
        if 'calculate_variability' in params:
            self.calculate_variability = params['calculate_variability']
        if 'calculate_trends' in params:
            self.calculate_trends = params['calculate_trends']
        if 'window_size' in params:
            self.window_size = params['window_size']
            
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation."""
        metrics = {}
        
        # Calculate the percentage of non-null values in the generated features
        for col in self.vital_cols:
            for feature_suffix in ['_normalized', '_rolling_std', '_rolling_range', '_rolling_mean', '_rolling_slope']:
                feature_name = f"{col}{feature_suffix}"
                if feature_name in transformed_data.columns:
                    non_null_rate = transformed_data[feature_name].notna().mean()
                    metrics[f"{feature_name}_non_null_rate"] = non_null_rate
        
        return metrics


class EnvironmentalTriggerAnalyzer(PreprocessingOperation):
    """Analyze environmental factors that may trigger migraines."""
    
    def __init__(self, 
                 weather_cols: List[str] = None,
                 pollution_cols: List[str] = None,
                 light_cols: List[str] = None,
                 noise_cols: List[str] = None,
                 timestamp_col: str = None,
                 location_col: str = None,
                 window_days: int = 3,
                 trigger_threshold: float = 1.5):
        """Initialize the environmental trigger analyzer.
        
        Args:
            weather_cols: Columns containing weather data (temperature, humidity, pressure, etc.)
            pollution_cols: Columns containing pollution data (PM2.5, PM10, AQI, etc.)
            light_cols: Columns containing light exposure data
            noise_cols: Columns containing noise exposure data
            timestamp_col: Column containing timestamps
            location_col: Column containing location information
            window_days: Window size in days for analyzing changes
            trigger_threshold: Threshold for identifying potential triggers (in standard deviations)
        """
        self.weather_cols = weather_cols or []
        self.pollution_cols = pollution_cols or []
        self.light_cols = light_cols or []
        self.noise_cols = noise_cols or []
        self.timestamp_col = timestamp_col
        self.location_col = location_col
        self.window_days = window_days
        self.trigger_threshold = trigger_threshold
        self.baseline_stats = {}
        self.seasonal_patterns = {}
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the environmental trigger analyzer to the data.
        
        Args:
            data: The data to fit the analyzer to
            **kwargs: Additional arguments
        """
        logger.info("Fitting EnvironmentalTriggerAnalyzer")
        
        # Calculate baseline statistics for environmental factors
        all_env_cols = self.weather_cols + self.pollution_cols + self.light_cols + self.noise_cols
        
        for col in all_env_cols:
            if col in data.columns:
                self.baseline_stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75)
                }
        
        # Analyze seasonal patterns if timestamp is available
        if self.timestamp_col and self.timestamp_col in data.columns:
            if data[self.timestamp_col].dtype == 'datetime64[ns]' or isinstance(data[self.timestamp_col].iloc[0], (datetime, pd.Timestamp)):
                # Extract month and season for seasonal analysis
                data_with_time = data.copy()
                data_with_time['month'] = pd.to_datetime(data_with_time[self.timestamp_col]).dt.month
                
                # Define seasons (Northern Hemisphere)
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 'winter'
                    elif month in [3, 4, 5]:
                        return 'spring'
                    elif month in [6, 7, 8]:
                        return 'summer'
                    else:  # 9, 10, 11
                        return 'fall'
                
                data_with_time['season'] = data_with_time['month'].apply(get_season)
                
                # Calculate seasonal statistics for environmental factors
                for col in all_env_cols:
                    if col in data.columns:
                        # Monthly patterns
                        monthly_means = data_with_time.groupby('month')[col].mean()
                        monthly_stds = data_with_time.groupby('month')[col].std()
                        
                        # Seasonal patterns
                        seasonal_means = data_with_time.groupby('season')[col].mean()
                        seasonal_stds = data_with_time.groupby('season')[col].std()
                        
                        self.seasonal_patterns[col] = {
                            'monthly_means': monthly_means.to_dict(),
                            'monthly_stds': monthly_stds.to_dict(),
                            'seasonal_means': seasonal_means.to_dict(),
                            'seasonal_stds': seasonal_stds.to_dict()
                        }
        
        # Analyze location-specific patterns if location is available
        if self.location_col and self.location_col in data.columns:
            for col in all_env_cols:
                if col in data.columns:
                    location_means = data.groupby(self.location_col)[col].mean()
                    location_stds = data.groupby(self.location_col)[col].std()
                    
                    self.baseline_stats[f"{col}_by_location"] = {
                        'location_means': location_means.to_dict(),
                        'location_stds': location_stds.to_dict()
                    }
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Analyze environmental factors and extract trigger features.
        
        Args:
            data: The data to transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed data with environmental trigger features
        """
        logger.info("Transforming data with EnvironmentalTriggerAnalyzer")
        result = data.copy()
        
        # Process all environmental columns
        all_env_cols = self.weather_cols + self.pollution_cols + self.light_cols + self.noise_cols
        
        # Add timestamp-based features if available
        if self.timestamp_col and self.timestamp_col in result.columns:
            if result[self.timestamp_col].dtype == 'datetime64[ns]' or isinstance(result[self.timestamp_col].iloc[0], (datetime, pd.Timestamp)):
                # Extract time components
                result['month'] = pd.to_datetime(result[self.timestamp_col]).dt.month
                result['day_of_year'] = pd.to_datetime(result[self.timestamp_col]).dt.dayofyear
                
                # Define seasons (Northern Hemisphere)
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 'winter'
                    elif month in [3, 4, 5]:
                        return 'spring'
                    elif month in [6, 7, 8]:
                        return 'summer'
                    else:  # 9, 10, 11
                        return 'fall'
                
                result['season'] = result['month'].apply(get_season)
        
        # Calculate deviations and rate of change for each environmental factor
        for col in all_env_cols:
            if col in result.columns and col in self.baseline_stats:
                # Calculate deviation from baseline
                baseline_mean = self.baseline_stats[col]['mean']
                baseline_std = self.baseline_stats[col]['std']
                
                result[f"{col}_deviation"] = (result[col] - baseline_mean) / baseline_std
                
                # Identify potential triggers (significant deviations)
                result[f"{col}_trigger"] = (abs(result[f"{col}_deviation"]) > self.trigger_threshold).astype(int)
                
                # Calculate rate of change if timestamp is available
                if self.timestamp_col and self.timestamp_col in result.columns:
                    if result[self.timestamp_col].dtype == 'datetime64[ns]' or isinstance(result[self.timestamp_col].iloc[0], (datetime, pd.Timestamp)):
                        # Sort by timestamp
                        if not result.index.equals(result.sort_values(by=self.timestamp_col).index):
                            result = result.sort_values(by=self.timestamp_col)
                        
                        # Calculate day-to-day changes
                        result[f"{col}_change"] = result[col].diff()
                        
                        # Identify rapid changes (potential triggers)
                        result[f"{col}_rapid_change"] = (abs(result[f"{col}_change"]) > (baseline_std * self.trigger_threshold)).astype(int)
                
                # Calculate seasonal deviations if seasonal patterns are available
                if 'season' in result.columns and col in self.seasonal_patterns:
                    # Create a function to get expected seasonal value
                    def get_seasonal_expected(season):
                        return self.seasonal_patterns[col]['seasonal_means'].get(season, baseline_mean)
                    
                    # Calculate deviation from seasonal norm
                    result[f"{col}_seasonal_deviation"] = result.apply(
                        lambda row: (row[col] - get_seasonal_expected(row['season'])) / baseline_std 
                        if row['season'] in self.seasonal_patterns[col]['seasonal_means'] else 0, 
                        axis=1
                    )
                    
                    # Identify seasonal anomalies
                    result[f"{col}_seasonal_anomaly"] = (abs(result[f"{col}_seasonal_deviation"]) > self.trigger_threshold).astype(int)
        
        # Process weather-specific columns
        for col in self.weather_cols:
            if col in result.columns:
                # Identify extreme weather conditions
                if 'temp' in col.lower():
                    # Extreme temperatures
                    if col in self.baseline_stats:
                        q25 = self.baseline_stats[col]['q25']
                        q75 = self.baseline_stats[col]['q75']
                        iqr = q75 - q25
                        
                        result[f"{col}_extreme_low"] = (result[col] < (q25 - 1.5 * iqr)).astype(int)
                        result[f"{col}_extreme_high"] = (result[col] > (q75 + 1.5 * iqr)).astype(int)
                
                elif 'pressure' in col.lower():
                    # Pressure changes (barometric pressure is a known migraine trigger)
                    if self.timestamp_col and self.timestamp_col in result.columns:
                        if result[self.timestamp_col].dtype == 'datetime64[ns]' or isinstance(result[self.timestamp_col].iloc[0], (datetime, pd.Timestamp)):
                            # Sort by timestamp
                            if not result.index.equals(result.sort_values(by=self.timestamp_col).index):
                                result = result.sort_values(by=self.timestamp_col)
                            
                            # Calculate pressure changes over 24-hour periods
                            result[f"{col}_24h_change"] = result[col].diff(24)
                            
                            # Identify significant pressure changes
                            if col in self.baseline_stats:
                                std = self.baseline_stats[col]['std']
                                result[f"{col}_significant_change"] = (abs(result[f"{col}_24h_change"]) > std).astype(int)
        
        # Process pollution-specific columns
        for col in self.pollution_cols:
            if col in result.columns:
                # Identify high pollution days
                if col in self.baseline_stats:
                    mean = self.baseline_stats[col]['mean']
                    std = self.baseline_stats[col]['std']
                    
                    result[f"{col}_high"] = (result[col] > (mean + std)).astype(int)
                    result[f"{col}_very_high"] = (result[col] > (mean + 2 * std)).astype(int)
        
        # Process light-specific columns
        for col in self.light_cols:
            if col in result.columns:
                # Identify bright light exposure (known migraine trigger)
                if col in self.baseline_stats:
                    q75 = self.baseline_stats[col]['q75']
                    max_val = self.baseline_stats[col]['max']
                    threshold = q75 + (max_val - q75) * 0.5
                    
                    result[f"{col}_bright_exposure"] = (result[col] > threshold).astype(int)
        
        # Process noise-specific columns
        for col in self.noise_cols:
            if col in result.columns:
                # Identify loud noise exposure (known migraine trigger)
                if col in self.baseline_stats:
                    q75 = self.baseline_stats[col]['q75']
                    max_val = self.baseline_stats[col]['max']
                    threshold = q75 + (max_val - q75) * 0.5
                    
                    result[f"{col}_loud_exposure"] = (result[col] > threshold).astype(int)
        
        # Create combined trigger indicators
        trigger_cols = [col for col in result.columns if col.endswith('_trigger') or 
                        col.endswith('_rapid_change') or 
                        col.endswith('_seasonal_anomaly') or
                        col.endswith('_extreme_low') or 
                        col.endswith('_extreme_high') or
                        col.endswith('_significant_change') or
                        col.endswith('_high') or 
                        col.endswith('_very_high') or
                        col.endswith('_bright_exposure') or
                        col.endswith('_loud_exposure')]
        
        if trigger_cols:
            # Count the number of potential environmental triggers
            result['env_trigger_count'] = result[trigger_cols].sum(axis=1)
            
            # Flag if any environmental triggers are present
            result['has_env_trigger'] = (result['env_trigger_count'] > 0).astype(int)
            
            # Categorize trigger severity
            result['env_trigger_severity'] = pd.cut(
                result['env_trigger_count'], 
                bins=[-1, 0, 1, 3, float('inf')],
                labels=['none', 'mild', 'moderate', 'severe']
            )
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation.
        
        Returns:
            A dictionary of parameters
        """
        return {
            'weather_cols': self.weather_cols,
            'pollution_cols': self.pollution_cols,
            'light_cols': self.light_cols,
            'noise_cols': self.noise_cols,
            'timestamp_col': self.timestamp_col,
            'location_col': self.location_col,
            'window_days': self.window_days,
            'trigger_threshold': self.trigger_threshold
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation.
        
        Args:
            params: A dictionary of parameters
        """
        if 'weather_cols' in params:
            self.weather_cols = params['weather_cols']
        if 'pollution_cols' in params:
            self.pollution_cols = params['pollution_cols']
        if 'light_cols' in params:
            self.light_cols = params['light_cols']
        if 'noise_cols' in params:
            self.noise_cols = params['noise_cols']
        if 'timestamp_col' in params:
            self.timestamp_col = params['timestamp_col']
        if 'location_col' in params:
            self.location_col = params['location_col']
        if 'window_days' in params:
            self.window_days = params['window_days']
        if 'trigger_threshold' in params:
            self.trigger_threshold = params['trigger_threshold']
    
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation.
        
        Args:
            data: The original data
            transformed_data: The transformed data
            
        Returns:
            A dictionary of quality metrics
        """
        metrics = {
            'num_environmental_features_added': len(transformed_data.columns) - len(data.columns),
        }
        
        # Calculate completeness of environmental data
        env_cols = self.weather_cols + self.pollution_cols + self.light_cols + self.noise_cols
        valid_cols = [col for col in env_cols if col in data.columns]
        
        if valid_cols:
            metrics['environmental_data_completeness'] = data[valid_cols].notna().mean().mean() * 100
        
        # Calculate trigger statistics if available
        if 'has_env_trigger' in transformed_data.columns:
            metrics['env_trigger_rate'] = transformed_data['has_env_trigger'].mean() * 100
        
        if 'env_trigger_count' in transformed_data.columns:
            metrics['avg_env_trigger_count'] = transformed_data['env_trigger_count'].mean()
            metrics['max_env_trigger_count'] = transformed_data['env_trigger_count'].max()
        
        return metrics


class AdvancedFeatureEngineer(PreprocessingOperation):
    """Advanced feature engineering for migraine data.
    
    This class implements advanced feature engineering techniques specifically
    for migraine data, including interaction features, temporal patterns,
    and domain-specific derived features.
    """
    
    def __init__(self, 
                 feature_groups: Dict[str, List[str]] = None,
                 interaction_pairs: List[Tuple[str, str]] = None,
                 temporal_cols: List[str] = None,
                 timestamp_col: str = None,
                 target_col: str = None,
                 patient_id_col: str = None,
                 polynomial_degree: int = 2,
                 create_clusters: bool = True,
                 n_clusters: int = 5):
        """Initialize the advanced feature engineer.
        
        Args:
            feature_groups: Dictionary mapping group names to lists of column names
            interaction_pairs: List of column pairs to create interaction features for
            temporal_cols: Columns to create temporal features for
            timestamp_col: Column containing timestamps
            target_col: Target column (e.g., migraine occurrence or severity)
            patient_id_col: Column containing patient identifiers
            polynomial_degree: Degree for polynomial feature generation
            create_clusters: Whether to create cluster features
            n_clusters: Number of clusters to create if clustering is enabled
        """
        self.feature_groups = feature_groups or {}
        self.interaction_pairs = interaction_pairs or []
        self.temporal_cols = temporal_cols or []
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.patient_id_col = patient_id_col
        self.polynomial_degree = polynomial_degree
        self.create_clusters = create_clusters
        self.n_clusters = n_clusters
        
        # Models and transformers for feature engineering
        self.poly_transformer = None
        self.cluster_models = {}
        self.feature_importances = {}
        self.temporal_patterns = {}
        self.lag_features = [1, 3, 7, 14, 30]  # Default lag periods in days
        
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Fit the feature engineer to the data.
        
        Args:
            data: The data to fit the feature engineer to
            **kwargs: Additional arguments
        """
        logger.info("Fitting AdvancedFeatureEngineer")
        
        # Prepare data for fitting
        numeric_data = data.select_dtypes(include=['number'])
        numeric_cols = numeric_data.columns.tolist()
        
        # Fit polynomial features transformer
        if self.polynomial_degree > 1 and len(numeric_cols) > 0:
            from sklearn.preprocessing import PolynomialFeatures
            self.poly_transformer = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False, interaction_only=True)
            # Fit on a sample of data to avoid memory issues with large datasets
            sample_size = min(1000, len(data))
            self.poly_transformer.fit(numeric_data.head(sample_size))
        
        # Fit clustering models for each feature group if enabled
        if self.create_clusters:
            from sklearn.cluster import KMeans
            
            for group_name, cols in self.feature_groups.items():
                valid_cols = [col for col in cols if col in numeric_cols]
                if len(valid_cols) >= 2:  # Need at least 2 features for meaningful clusters
                    try:
                        # Initialize and fit KMeans
                        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                        # Use only the valid columns for clustering
                        cluster_data = data[valid_cols].copy()
                        # Handle missing values for clustering
                        cluster_data = cluster_data.fillna(cluster_data.mean())
                        kmeans.fit(cluster_data)
                        self.cluster_models[group_name] = kmeans
                    except Exception as e:
                        logger.warning(f"Failed to fit clustering model for {group_name}: {e}")
        
        # Learn temporal patterns if timestamp column is available
        if self.timestamp_col and self.timestamp_col in data.columns and self.temporal_cols:
            if data[self.timestamp_col].dtype == 'datetime64[ns]' or isinstance(data[self.timestamp_col].iloc[0], (datetime, pd.Timestamp)):
                # Ensure data is sorted by timestamp
                sorted_data = data.sort_values(by=self.timestamp_col)
                
                # Extract temporal components
                sorted_data['hour'] = pd.to_datetime(sorted_data[self.timestamp_col]).dt.hour
                sorted_data['day_of_week'] = pd.to_datetime(sorted_data[self.timestamp_col]).dt.dayofweek
                sorted_data['day_of_month'] = pd.to_datetime(sorted_data[self.timestamp_col]).dt.day
                sorted_data['month'] = pd.to_datetime(sorted_data[self.timestamp_col]).dt.month
                
                # Analyze temporal patterns for each column
                for col in self.temporal_cols:
                    if col in sorted_data.columns:
                        # Calculate hourly, daily, and monthly patterns
                        hourly_means = sorted_data.groupby('hour')[col].mean()
                        daily_means = sorted_data.groupby('day_of_week')[col].mean()
                        monthly_means = sorted_data.groupby('month')[col].mean()
                        
                        self.temporal_patterns[col] = {
                            'hourly_means': hourly_means.to_dict(),
                            'daily_means': daily_means.to_dict(),
                            'monthly_means': monthly_means.to_dict()
                        }
                        
                        # If target column is available, analyze correlations with time
                        if self.target_col and self.target_col in sorted_data.columns:
                            # Group by temporal components and calculate target mean
                            hourly_target = sorted_data.groupby('hour')[self.target_col].mean()
                            daily_target = sorted_data.groupby('day_of_week')[self.target_col].mean()
                            monthly_target = sorted_data.groupby('month')[self.target_col].mean()
                            
                            self.temporal_patterns[f"{col}_target"] = {
                                'hourly_target': hourly_target.to_dict(),
                                'daily_target': daily_target.to_dict(),
                                'monthly_target': monthly_target.to_dict()
                            }
        
        # Calculate feature importances if target column is available
        if self.target_col and self.target_col in data.columns:
            try:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # Determine if classification or regression based on target column
                is_classification = data[self.target_col].dtype == 'bool' or data[self.target_col].nunique() <= 10
                
                # Prepare feature matrix
                X = data.select_dtypes(include=['number']).drop(columns=[self.target_col], errors='ignore')
                y = data[self.target_col]
                
                # Handle missing values for importance calculation
                X = X.fillna(X.mean())
                
                # Train a simple model to get feature importances
                if is_classification:
                    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                
                model.fit(X, y)
                
                # Store feature importances
                for i, col in enumerate(X.columns):
                    self.feature_importances[col] = model.feature_importances_[i]
            except Exception as e:
                logger.warning(f"Failed to calculate feature importances: {e}")
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the data with advanced feature engineering.
        
        Args:
            data: The data to transform
            **kwargs: Additional arguments
            
        Returns:
            Transformed data with engineered features
        """
        logger.info("Transforming data with AdvancedFeatureEngineer")
        result = data.copy()
        
        # Create polynomial and interaction features
        if self.poly_transformer is not None:
            numeric_data = result.select_dtypes(include=['number'])
            numeric_cols = numeric_data.columns.tolist()
            
            if len(numeric_cols) > 0:
                try:
                    # Generate polynomial features
                    poly_features = self.poly_transformer.transform(numeric_data)
                    poly_feature_names = self.poly_transformer.get_feature_names_out(numeric_cols)
                    
                    # Add only interaction features (degree 2 or higher)
                    interaction_features = pd.DataFrame(
                        poly_features[:, len(numeric_cols):],
                        index=result.index,
                        columns=poly_feature_names[len(numeric_cols):]
                    )
                    
                    # Rename columns to make them more interpretable
                    new_cols = {}
                    for col in interaction_features.columns:
                        new_col = col.replace(' ', '*')
                        new_cols[col] = f"interaction_{new_col}"
                    
                    interaction_features = interaction_features.rename(columns=new_cols)
                    
                    # Add to result dataframe
                    result = pd.concat([result, interaction_features], axis=1)
                except Exception as e:
                    logger.warning(f"Failed to generate polynomial features: {e}")
        
        # Create explicit interaction features for specified pairs
        for col1, col2 in self.interaction_pairs:
            if col1 in result.columns and col2 in result.columns:
                # Check if both columns are numeric
                if pd.api.types.is_numeric_dtype(result[col1]) and pd.api.types.is_numeric_dtype(result[col2]):
                    # Create interaction feature
                    result[f"interaction_{col1}_{col2}"] = result[col1] * result[col2]
                    
                    # Create ratio features if no zeros in denominator
                    if (result[col2] != 0).all():
                        result[f"ratio_{col1}_to_{col2}"] = result[col1] / result[col2]
                    
                    if (result[col1] != 0).all():
                        result[f"ratio_{col2}_to_{col1}"] = result[col2] / result[col1]
        
        # Add cluster assignments for each feature group
        for group_name, model in self.cluster_models.items():
            if group_name in self.feature_groups:
                cols = self.feature_groups[group_name]
                valid_cols = [col for col in cols if col in result.columns]
                
                if len(valid_cols) >= 2:
                    try:
                        # Prepare data for clustering
                        cluster_data = result[valid_cols].copy()
                        # Handle missing values
                        cluster_data = cluster_data.fillna(cluster_data.mean())
                        
                        # Predict clusters
                        clusters = model.predict(cluster_data)
                        result[f"cluster_{group_name}"] = clusters
                        
                        # Add distance to cluster centers as features
                        distances = model.transform(cluster_data)
                        for i in range(self.n_clusters):
                            result[f"distance_to_cluster_{group_name}_{i}"] = distances[:, i]
                    except Exception as e:
                        logger.warning(f"Failed to generate cluster features for {group_name}: {e}")
        
        # Create temporal features if timestamp column is available
        if self.timestamp_col and self.timestamp_col in result.columns:
            if result[self.timestamp_col].dtype == 'datetime64[ns]' or isinstance(result[self.timestamp_col].iloc[0], (datetime, pd.Timestamp)):
                # Extract time components
                result['hour_of_day'] = pd.to_datetime(result[self.timestamp_col]).dt.hour
                result['day_of_week'] = pd.to_datetime(result[self.timestamp_col]).dt.dayofweek
                result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
                result['day_of_month'] = pd.to_datetime(result[self.timestamp_col]).dt.day
                result['month'] = pd.to_datetime(result[self.timestamp_col]).dt.month
                result['quarter'] = pd.to_datetime(result[self.timestamp_col]).dt.quarter
                
                # Create cyclical features for time components
                result['hour_sin'] = np.sin(2 * np.pi * result['hour_of_day'] / 24)
                result['hour_cos'] = np.cos(2 * np.pi * result['hour_of_day'] / 24)
                result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
                result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
                result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
                result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
                
                # Create temporal pattern deviation features
                for col in self.temporal_patterns.keys():
                    if col in self.temporal_cols and col in result.columns:
                        # Calculate deviations from temporal patterns
                        if 'hourly_means' in self.temporal_patterns[col]:
                            result[f"{col}_hourly_deviation"] = result.apply(
                                lambda row: row[col] - self.temporal_patterns[col]['hourly_means'].get(row['hour_of_day'], 0),
                                axis=1
                            )
                        
                        if 'daily_means' in self.temporal_patterns[col]:
                            result[f"{col}_daily_deviation"] = result.apply(
                                lambda row: row[col] - self.temporal_patterns[col]['daily_means'].get(row['day_of_week'], 0),
                                axis=1
                            )
                        
                        if 'monthly_means' in self.temporal_patterns[col]:
                            result[f"{col}_monthly_deviation"] = result.apply(
                                lambda row: row[col] - self.temporal_patterns[col]['monthly_means'].get(row['month'], 0),
                                axis=1
                            )
                
                # Create lag features if patient_id is available for longitudinal data
                if self.patient_id_col and self.patient_id_col in result.columns:
                    # Ensure data is sorted by patient and timestamp
                    result = result.sort_values(by=[self.patient_id_col, self.timestamp_col])
                    
                    # Create lag features for each temporal column and each lag period
                    for col in self.temporal_cols:
                        if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
                            for lag in self.lag_features:
                                # Create lag feature within each patient group
                                result[f"{col}_lag_{lag}"] = result.groupby(self.patient_id_col)[col].shift(lag)
                                
                                # Calculate change from previous period
                                if lag == 1:  # Only for lag 1 to avoid too many features
                                    result[f"{col}_change"] = result[col] - result[f"{col}_lag_{lag}"]
                                    result[f"{col}_pct_change"] = result.groupby(self.patient_id_col)[col].pct_change(lag)
                            
                            # Calculate rolling statistics for each patient
                            result[f"{col}_rolling_mean_7d"] = result.groupby(self.patient_id_col)[col].transform(
                                lambda x: x.rolling(window=7, min_periods=1).mean()
                            )
                            result[f"{col}_rolling_std_7d"] = result.groupby(self.patient_id_col)[col].transform(
                                lambda x: x.rolling(window=7, min_periods=1).std()
                            )
                            result[f"{col}_rolling_max_7d"] = result.groupby(self.patient_id_col)[col].transform(
                                lambda x: x.rolling(window=7, min_periods=1).max()
                            )
        
        # Create domain-specific migraine features
        if self.target_col and self.target_col in result.columns:
            # Calculate days since last migraine for each patient
            if self.patient_id_col and self.patient_id_col in result.columns and self.timestamp_col and self.timestamp_col in result.columns:
                # Convert timestamps to datetime if needed
                if result[self.timestamp_col].dtype != 'datetime64[ns]':
                    result['_timestamp'] = pd.to_datetime(result[self.timestamp_col])
                else:
                    result['_timestamp'] = result[self.timestamp_col]
                
                # Sort by patient and timestamp
                result = result.sort_values(by=[self.patient_id_col, '_timestamp'])
                
                # Create a feature for days since last migraine
                def calc_days_since_last_migraine(group):
                    # Find dates with migraines
                    migraine_dates = group.loc[group[self.target_col] > 0, '_timestamp']
                    if len(migraine_dates) > 0:
                        # Calculate days since most recent migraine for each row
                        days = []
                        for date in group['_timestamp']:
                            # Find dates of previous migraines
                            prev_migraines = migraine_dates[migraine_dates < date]
                            if len(prev_migraines) > 0:
                                # Calculate days since most recent migraine
                                days_since = (date - prev_migraines.max()).total_seconds() / (24 * 3600)
                                days.append(days_since)
                            else:
                                days.append(np.nan)  # No previous migraine
                        return days
                    else:
                        return [np.nan] * len(group)
                
                # Apply the function to each patient group
                result['days_since_last_migraine'] = result.groupby(self.patient_id_col).apply(
                    calc_days_since_last_migraine
                ).explode().values
                
                # Calculate migraine frequency features
                def calc_migraine_frequency(group):
                    # Calculate 30-day rolling sum of migraines
                    if len(group) > 1:
                        group = group.sort_values('_timestamp')
                        group['migraine_freq_30d'] = group[self.target_col].rolling('30D').sum()
                        return group['migraine_freq_30d']
                    else:
                        return pd.Series([np.nan] * len(group))
                
                # Apply the function to each patient group
                result['migraine_freq_30d'] = result.groupby(self.patient_id_col).apply(
                    calc_migraine_frequency
                ).values
                
                # Clean up temporary timestamp column
                result = result.drop(columns=['_timestamp'])
        
        return result
    
    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the operation.
        
        Returns:
            A dictionary of parameters
        """
        return {
            'feature_groups': self.feature_groups,
            'interaction_pairs': self.interaction_pairs,
            'temporal_cols': self.temporal_cols,
            'timestamp_col': self.timestamp_col,
            'target_col': self.target_col,
            'patient_id_col': self.patient_id_col,
            'polynomial_degree': self.polynomial_degree,
            'create_clusters': self.create_clusters,
            'n_clusters': self.n_clusters,
            'lag_features': self.lag_features
        }
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the parameters of the operation.
        
        Args:
            params: A dictionary of parameters
        """
        if 'feature_groups' in params:
            self.feature_groups = params['feature_groups']
        if 'interaction_pairs' in params:
            self.interaction_pairs = params['interaction_pairs']
        if 'temporal_cols' in params:
            self.temporal_cols = params['temporal_cols']
        if 'timestamp_col' in params:
            self.timestamp_col = params['timestamp_col']
        if 'target_col' in params:
            self.target_col = params['target_col']
        if 'patient_id_col' in params:
            self.patient_id_col = params['patient_id_col']
        if 'polynomial_degree' in params:
            self.polynomial_degree = params['polynomial_degree']
        if 'create_clusters' in params:
            self.create_clusters = params['create_clusters']
        if 'n_clusters' in params:
            self.n_clusters = params['n_clusters']
        if 'lag_features' in params:
            self.lag_features = params['lag_features']
    
    def get_quality_metrics(self, data: pd.DataFrame, transformed_data: pd.DataFrame) -> Dict[str, float]:
        """Get quality metrics for the transformation.
        
        Args:
            data: The original data
            transformed_data: The transformed data
            
        Returns:
            A dictionary of quality metrics
        """
        metrics = {
            'num_features_added': len(transformed_data.columns) - len(data.columns),
        }
        
        # Calculate feature importance statistics if available
        if self.feature_importances:
            # Get top features by importance
            sorted_importances = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_importances[:10]]
            
            metrics['top_features'] = top_features
            metrics['importance_concentration'] = sum([v for k, v in sorted_importances[:10]]) / sum(self.feature_importances.values())
        
        # Calculate correlation with target if available
        if self.target_col and self.target_col in transformed_data.columns:
            # Get numeric columns only
            numeric_data = transformed_data.select_dtypes(include=['number'])
            
            if self.target_col in numeric_data.columns:
                # Calculate correlations with target
                correlations = numeric_data.corr()[self.target_col].abs().sort_values(ascending=False)
                
                # Get top correlated features (excluding target itself)
                top_correlated = correlations[correlations.index != self.target_col].head(10)
                
                metrics['max_correlation_with_target'] = correlations[correlations.index != self.target_col].max()
                metrics['mean_correlation_with_target'] = correlations[correlations.index != self.target_col].mean()
                metrics['top_correlated_features'] = top_correlated.index.tolist()
        
        # Calculate feature redundancy metrics
        numeric_data = transformed_data.select_dtypes(include=['number'])
        if len(numeric_data.columns) > 1:
            # Calculate pairwise correlations
            corr_matrix = numeric_data.corr().abs()
            
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Calculate high correlation metrics
            high_corr_pairs = (upper_tri > 0.9).sum().sum()  # Count pairs with correlation > 0.9
            avg_corr = upper_tri.mean().mean()  # Average correlation across all pairs
            
            metrics['high_correlation_pairs'] = high_corr_pairs
            metrics['average_feature_correlation'] = avg_corr
        
        return metrics

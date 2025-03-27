"""
Enhanced Synthetic Patient Data Generator

This module extends the basic synthetic patient data generator with advanced capabilities:
1. Controlled drift simulation (sudden, gradual, recurring)
2. Expanded multimodal data generation (detailed physiological, environmental, behavioral)
3. Clinical performance metrics and evaluation
4. LLIF data structure compatibility

These enhancements support testing the MoE framework under realistic data conditions
and validating drift detection and adaptation capabilities.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import copy
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score

# Import the base generator
from utils.synthetic_patient_data import PatientDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPatientDataGenerator(PatientDataGenerator):
    """
    Enhanced synthetic patient data generator with drift simulation, 
    expanded multimodal data, and evaluation metrics.
    
    Inherits from the base PatientDataGenerator and extends its capabilities.
    """
    
    def __init__(self, 
                 output_dir: str = 'data/enhanced_synthetic_patients',
                 seed: int = 42,
                 n_features: int = 30):
        """
        Initialize the enhanced patient data generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to store generated data
        seed : int
            Random seed for reproducibility
        n_features : int
            Number of features to generate
        """
        # Initialize the parent class
        super().__init__(output_dir=output_dir, seed=seed, n_features=n_features)
        
        # Expanded feature categories with more detailed features
        self.feature_categories.update({
            'physiological': self.feature_categories['physiological'] + [
                'hrv', 'body_temperature', 'sleep_rem_percentage', 'sleep_deep_percentage', 
                'cortisol_level', 'inflammatory_markers', 'blood_glucose'
            ],
            'environmental': self.feature_categories['environmental'] + [
                'uv_index', 'air_quality_pm25', 'air_quality_ozone', 'pollen_tree',
                'pollen_grass', 'pollen_weed', 'altitude', 'weather_change_rate'
            ],
            'behavioral': self.feature_categories['behavioral'] + [
                'social_activity_hours', 'screen_blue_light_exposure', 'posture_score',
                'dietary_inflammatory_index', 'stress_management_activity'
            ]
        })
        
        # Add medication-specific features
        self.feature_categories['medication'] = [
            'triptan_usage', 'nsaid_usage', 'preventative_adherence', 
            'rescue_medication_count', 'days_since_last_dose'
        ]
        
        # Add subjective symptom logging
        self.feature_categories['subjective'] = [
            'reported_stress', 'reported_sleep_quality', 'mood_score',
            'prodrome_symptoms', 'aura_reported', 'pain_location'
        ]
        
        # Drift configuration options
        self.drift_types = {
            'sudden': {
                'description': 'Abrupt change in data distribution',
                'parameters': {
                    'magnitude': 0.5,  # How strong the drift is (0-1)
                    'duration': 0.3,   # Portion of timeline affected after drift onset
                    'features_affected': 0.4  # Portion of features affected
                }
            },
            'gradual': {
                'description': 'Slow progressive change in data distribution',
                'parameters': {
                    'magnitude': 0.4,  # Maximum drift magnitude
                    'rate': 0.05,      # Rate of change per time unit
                    'features_affected': 0.6  # Portion of features affected
                }
            },
            'recurring': {
                'description': 'Cyclical changes that repeat over time',
                'parameters': {
                    'magnitude': 0.3,  # Amplitude of the cyclical drift
                    'frequency': 0.1,  # Frequency of oscillation
                    'features_affected': 0.5  # Portion of features affected
                }
            },
            'none': {
                'description': 'No drift applied',
                'parameters': {}
            }
        }
        
        # Clinical relevance scoring parameters
        self.clinical_relevance = {
            'risk_factors': {
                'stress_level': 0.8,
                'sleep_quality': 0.7,
                'barometric_pressure': 0.6,
                'hrv': 0.7,
                'inflammatory_markers': 0.8,
                'medication_adherence': 0.9
            },
            'severity_weights': {
                'pain_level': {
                    'mild': 0.3,
                    'moderate': 0.6,
                    'severe': 1.0
                },
                'duration': {
                    'short': 0.4,
                    'medium': 0.7,
                    'long': 1.0
                },
                'associated_symptoms': {
                    'none': 0.0,
                    'mild': 0.4,
                    'moderate': 0.7,
                    'severe': 1.0
                }
            }
        }
        
        # Record of baseline models for performance tracking
        self.baseline_models = {}
        
        # LLIF output configuration
        self.llif_config = {
            'format_version': '1.0',
            'time_resolution': {
                'physiological': '5min',
                'environmental': '1hour',
                'behavioral': '1hour',
                'medication': '1day',
                'subjective': '6hour'
            }
        }

    def generate_enhanced_patient_set(self,
                                     num_patients: int = 10,
                                     time_periods: int = 60,
                                     samples_per_period: int = 6,
                                     drift_type: str = 'none',
                                     drift_start_time: float = 0.5,
                                     output_format: str = 'llif',
                                     include_evaluation: bool = True,
                                     include_visualization: bool = True) -> Dict[str, Any]:
        """
        Generate an enhanced set of patient data with controlled drift and multimodal features.
        
        Parameters:
        -----------
        num_patients : int
            Number of patients to generate
        time_periods : int
            Number of days to generate data for
        samples_per_period : int
            Base samples per day (will be adjusted for each modality)
        drift_type : str
            Type of drift to simulate: 'sudden', 'gradual', 'recurring', or 'none'
        drift_start_time : float
            When drift begins as a fraction of total time (0-1)
        output_format : str
            Output format: 'llif' (Low-Level Inference Format) or 'standard'
        include_evaluation : bool
            Whether to include evaluation metrics
        include_visualization : bool
            Whether to generate visualizations
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing generated data, metadata, and evaluation metrics
        """
        if drift_type not in self.drift_types:
            logger.warning(f"Unknown drift type '{drift_type}'. Using 'none' instead.")
            drift_type = 'none'
            
        # Prepare results container
        results = {
            'patients': [],
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'parameters': {
                    'num_patients': num_patients,
                    'time_periods': time_periods,
                    'samples_per_period': samples_per_period,
                    'drift_type': drift_type,
                    'drift_start_time': drift_start_time
                }
            }
        }
        
        patient_ids = []
        
        # Generate each patient
        for i in range(num_patients):
            # Generate a unique patient ID
            patient_id = f"patient_{i+1:03d}"
            patient_ids.append(patient_id)
            
            # Select a random profile type for this patient
            profile_type = random.choice(list(self.profile_types.keys()))
            
            logger.info(f"Generating enhanced data for {patient_id} with profile type: {profile_type}")
            
            # Generate demographics
            demographics = self._generate_demographics(patient_id, profile_type)
            
            # Generate expanded time series data with controlled drift
            data, targets, drift_metadata = self._generate_enhanced_timeseries(
                patient_id, 
                profile_type, 
                time_periods, 
                samples_per_period,
                drift_type,
                drift_start_time
            )
            
            # Generate subjective patient feedback
            feedback = self._generate_enhanced_feedback(
                patient_id, profile_type, data, targets, drift_metadata
            )
            
            # Calculate evaluation metrics if requested
            evaluation_metrics = None
            if include_evaluation:
                evaluation_metrics = self._calculate_evaluation_metrics(
                    patient_id, data, targets, drift_metadata
                )
            
            # Format output according to specified format
            if output_format.lower() == 'llif':
                formatted_data = self._format_as_llif(
                    patient_id, demographics, data, targets, feedback, 
                    drift_metadata, evaluation_metrics
                )
            else:
                formatted_data = {
                    'patient_id': patient_id,
                    'demographics': demographics,
                    'data': data.to_dict(orient='records'),
                    'targets': targets.tolist(),
                    'feedback': feedback,
                    'drift_metadata': drift_metadata
                }
                
                if evaluation_metrics:
                    formatted_data['evaluation_metrics'] = evaluation_metrics
            
            # Save the generated data
            self._save_enhanced_patient_data(
                patient_id, formatted_data, include_visualization
            )
            
            # Add to results
            results['patients'].append(formatted_data)
            
        # Create a summary file
        summary = self._create_enhanced_patient_summary(patient_ids, results)
        results['summary'] = summary
        
        return results
        
    def _generate_enhanced_timeseries(self,
                                      patient_id: str,
                                      profile_type: str,
                                      time_periods: int,
                                      samples_per_period: int,
                                      drift_type: str = 'none',
                                      drift_start_time: float = 0.5) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Generate enhanced time series data with controlled drift simulation.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        profile_type : str
            Type of patient profile
        time_periods : int
            Number of days to generate data for
        samples_per_period : int
            Base samples per day
        drift_type : str
            Type of drift to simulate
        drift_start_time : float
            When drift begins as a fraction of total time (0-1)
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, Dict]
            DataFrame of features, Series of targets, and drift metadata
        """
        # Get profile characteristics
        profile_info = self.profile_types[profile_type]
        key_features = profile_info['key_features']
        
        # Create base feature dictionary with all available features across categories
        all_features = []
        for category, features in self.feature_categories.items():
            all_features.extend(features)
        
        # Generate base means and variances for this patient
        feature_means = {}
        feature_vars = {}
        
        for feature in all_features:
            # Normal distribution for most features
            feature_means[feature] = random.uniform(40, 60)
            feature_vars[feature] = random.uniform(5, 15)
        
        # Generate time series data with different sampling rates per modality
        modality_samples = {
            'physiological': samples_per_period * 12,  # Every 5 minutes
            'environmental': samples_per_period,      # Every hour
            'behavioral': samples_per_period,         # Every hour
            'medication': 1,                          # Once per day
            'subjective': samples_per_period // 2     # Every ~2 hours
        }
        
        # Calculate drift parameters
        drift_params = {}
        drift_start_idx = None
        if drift_type != 'none':
            drift_config = self.drift_types[drift_type]['parameters']
            
            # Determine drift features
            num_features_affected = int(len(all_features) * drift_config.get('features_affected', 0.3))
            drift_features = random.sample(all_features, num_features_affected)
            
            # Calculate drift start index
            total_samples = time_periods * samples_per_period
            drift_start_idx = int(total_samples * drift_start_time)
            
            drift_params = {
                'type': drift_type,
                'affected_features': drift_features,
                'start_idx': drift_start_idx,
                'start_time': drift_start_time,
                'magnitude': drift_config.get('magnitude', 0.5),
                'config': drift_config
            }
            
            logger.info(f"Applying {drift_type} drift to {len(drift_features)} features starting at index {drift_start_idx}")
        
        # Create a combined dataframe for all modalities
        combined_data = []
        
        # Create timestamps
        start_date = datetime.now() - timedelta(days=time_periods)
        
        # Generate data for each modality
        for modality, features in self.feature_categories.items():
            samples_this_modality = modality_samples.get(modality, samples_per_period)
            
            for day in range(time_periods):
                for sample in range(samples_this_modality):
                    # Calculate the hour based on sampling rate
                    hour_increment = 24 / samples_this_modality
                    hour = int(hour_increment * sample)
                    minute = int((hour_increment * sample - hour) * 60)
                    
                    # Add some variability to the time
                    hour_var = random.uniform(-0.1, 0.1) * hour_increment
                    hour = max(0, min(23, hour + int(hour_var)))
                    minute = max(0, min(59, minute + int(hour_var * 60)))
                    
                    timestamp = start_date + timedelta(days=day, hours=hour, minutes=minute)
                    
                    # Calculate overall index for this sample (for drift application)
                    overall_idx = day * samples_per_period + (sample * samples_per_period // samples_this_modality)
                    
                    # Create sample
                    sample_data = {
                        'patient_id': patient_id,
                        'timestamp': timestamp,
                        'modality': modality
                    }
                    
                    # Add feature values
                    for feature in features:
                        # Base value with weekly pattern (if applicable)
                        day_of_week = timestamp.weekday()
                        if feature in ['stress_level', 'sleep_quality', 'work_hours']:
                            week_factor = 1.0 + 0.2 * (day_of_week < 5)  # Higher on weekdays
                        elif feature in ['social_activity_hours', 'alcohol_units']:
                            week_factor = 1.0 + 0.3 * (day_of_week >= 5)  # Higher on weekends
                        else:
                            week_factor = 1.0
                        
                        # Time-based trends (seasonal for some environmental features)
                        time_factor = overall_idx / (time_periods * samples_per_period)
                        if feature in ['temperature', 'humidity', 'uv_index']:
                            seasonal_factor = 1.0 + 0.15 * np.sin(time_factor * 2 * np.pi)
                        else:
                            seasonal_factor = 1.0
                        
                        # Base value with patterns
                        base_value = feature_means[feature] * week_factor * seasonal_factor
                        
                        # Apply drift if this feature is affected
                        drift_factor = 1.0
                        if drift_type != 'none' and feature in drift_params['affected_features'] and overall_idx >= drift_start_idx:
                            # Calculate drift effect based on type
                            if drift_type == 'sudden':
                                drift_factor = 1.0 + drift_params['magnitude']
                            elif drift_type == 'gradual':
                                progress = min(1.0, (overall_idx - drift_start_idx) / 
                                              (time_periods * samples_per_period * drift_params['config'].get('rate', 0.1)))
                                drift_factor = 1.0 + (drift_params['magnitude'] * progress)
                            elif drift_type == 'recurring':
                                cycles = (overall_idx - drift_start_idx) * drift_params['config'].get('frequency', 0.1)
                                drift_factor = 1.0 + (drift_params['magnitude'] * np.sin(cycles * 2 * np.pi))
                        
                        # Apply drift to the base value
                        base_value = base_value * drift_factor
                        
                        # Add random noise
                        noise = np.random.normal(0, np.sqrt(feature_vars[feature]))
                        value = base_value + noise
                        
                        # Ensure reasonable bounds
                        value = max(0, min(100, value))
                        sample_data[feature] = value
                    
                    # Apply profile-specific patterns to key features
                    for feature in key_features:
                        if feature in sample_data:
                            # Make key features more extreme
                            deviation = sample_data[feature] - feature_means[feature]
                            sample_data[feature] = feature_means[feature] + (deviation * 1.5)
                    
                    combined_data.append(sample_data)
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(combined_data)
        df = df.sort_values('timestamp')
        
        # Generate target values (migraine events)
        targets = self._generate_enhanced_targets(df, profile_type, key_features, drift_params)
        
        # Create metadata about the drift for evaluation
        drift_metadata = {
            'drift_applied': drift_type != 'none',
            'drift_type': drift_type,
            'drift_start_time': drift_start_time,
            'drift_start_idx': drift_start_idx,
            'drift_parameters': drift_params
        }
        
        return df, targets, drift_metadata
        
    def _generate_enhanced_targets(self,
                                   df: pd.DataFrame,
                                   profile_type: str,
                                   key_features: List[str],
                                   drift_params: Dict) -> pd.Series:
        """
        Generate enhanced target values (migraine events) with awareness of drift conditions.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature data
        profile_type : str
            Type of patient profile
        key_features : List[str]
            Key features for this profile type
        drift_params : Dict
            Parameters of the applied drift
            
        Returns:
        --------
        pd.Series
            Target values (1 for migraine, 0 for no migraine)
        """
        # Extract key features from the dataframe
        timestamps = df['timestamp'].unique()
        targets = pd.Series(0, index=df.index)
        
        # Group by day to determine daily migraine risk
        df['date'] = df['timestamp'].dt.date
        
        # Get unique dates
        dates = df['date'].unique()
        
        # Initialize empty lists to store migraine events and their severity
        migraine_events = []
        severity_scores = []
        
        # Calculate baseline risk factors
        risk_factors = {}
        for feature in key_features:
            if feature in df.columns:
                # Calculate risk as normalized deviation from mean
                mean_val = df[feature].mean()
                std_val = df[feature].std() if df[feature].std() > 0 else 1.0
                
                # For each feature, identify if higher or lower values are risky
                # For most features, higher values indicate risk
                if feature in ['sleep_quality', 'water_intake', 'meditation_minutes']:
                    # For these features, lower values indicate higher risk
                    risk_factors[feature] = lambda x, m=mean_val, s=std_val: max(0, (m - x) / s)
                else:
                    # For most features, higher values indicate higher risk
                    risk_factors[feature] = lambda x, m=mean_val, s=std_val: max(0, (x - m) / s)
        
        # Process each date
        for date in dates:
            # Get data for this date
            day_data = df[df['date'] == date]
            
            # Calculate risk scores for this day based on key features
            daily_risks = {}
            for feature, risk_func in risk_factors.items():
                if feature in day_data.columns:
                    # Calculate average risk for this feature on this day
                    feature_vals = day_data[feature].dropna()
                    if not feature_vals.empty:
                        feature_risk = np.mean([risk_func(x) for x in feature_vals])
                        daily_risks[feature] = feature_risk
            
            # Calculate overall risk as weighted sum
            if daily_risks:
                # Weight each feature by its clinical relevance
                weighted_risks = []
                for feature, risk in daily_risks.items():
                    # Get clinical relevance weight or default
                    weight = self.clinical_relevance['risk_factors'].get(feature, 0.5)
                    weighted_risks.append(risk * weight)
                
                # Overall risk for this day
                overall_risk = np.mean(weighted_risks) if weighted_risks else 0
                
                # Apply drift effect to risk calculation if applicable
                if drift_params and drift_params.get('type') != 'none':
                    # Find the first timestamp for this date
                    day_start = day_data['timestamp'].min()
                    
                    # Calculate day index in the full dataset
                    day_idx = (day_start - df['timestamp'].min()).total_seconds() / (24 * 3600)
                    
                    # Check if this day is after drift start
                    if drift_params.get('drift_start_idx') is not None:
                        drift_day = drift_params['drift_start_idx'] / (df['date'].nunique())
                        
                        if day_idx >= drift_day:
                            drift_type = drift_params['type']
                            magnitude = drift_params.get('magnitude', 0.5)
                            
                            # Modify risk based on drift type
                            if drift_type == 'sudden':
                                # Sudden increase in migraine risk
                                overall_risk *= (1 + magnitude)
                            elif drift_type == 'gradual':
                                # Gradual increase in risk proportional to time since drift
                                progress = min(1.0, (day_idx - drift_day) / 
                                              (df['date'].nunique() * drift_params.get('config', {}).get('rate', 0.1)))
                                overall_risk *= (1 + (magnitude * progress))
                            elif drift_type == 'recurring':
                                # Cyclical changes in risk
                                cycles = (day_idx - drift_day) * drift_params.get('config', {}).get('frequency', 0.1)
                                overall_risk *= (1 + (magnitude * np.sin(cycles * 2 * np.pi)))
                
                # Determine if migraine occurs based on risk
                # Higher risk = higher probability
                threshold = 0.15  # Base probability threshold
                probability = min(0.95, overall_risk * threshold)  # Cap at 95%
                
                has_migraine = random.random() < probability
                
                if has_migraine:
                    # Select a random time during this day for the migraine
                    migraine_time = random.choice(day_data['timestamp'].tolist())
                    
                    # Mark all records within 6-hour window as positive
                    migraine_window = 6  # hours
                    
                    # Calculate window boundaries
                    window_start = migraine_time - timedelta(hours=1)  # Prodrome
                    window_end = migraine_time + timedelta(hours=migraine_window - 1)  # Duration
                    
                    # Mark all records in window
                    for idx, row in day_data.iterrows():
                        if window_start <= row['timestamp'] <= window_end:
                            targets.loc[idx] = 1
                    
                    # Save event details for metadata
                    severity = min(1.0, overall_risk * 1.5)  # Scale to 0-1
                    migraine_events.append({
                        'timestamp': migraine_time.isoformat(),
                        'date': date.isoformat(),
                        'risk_score': overall_risk,
                        'severity': severity,
                        'contributing_features': {
                            feature: risk for feature, risk in daily_risks.items()
                        }
                    })
                    severity_scores.append(severity)
        
        # Save migraine events metadata to targets
        targets.attrs['migraine_events'] = migraine_events
        targets.attrs['avg_severity'] = np.mean(severity_scores) if severity_scores else 0
        targets.attrs['event_count'] = len(migraine_events)
        
        return targets
    
    def _calculate_evaluation_metrics(self,
                                      patient_id: str,
                                      data: pd.DataFrame,
                                      targets: pd.Series,
                                      drift_metadata: Dict) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for the generated data, especially focused on
        drift detection and clinical relevance.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        drift_metadata : Dict
            Metadata about the applied drift
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of evaluation metrics
        """
        metrics = {
            'statistical_metrics': {},
            'clinical_metrics': {},
            'drift_analysis': {}
        }
        
        # Split data into pre-drift and post-drift if drift was applied
        if drift_metadata.get('drift_applied', False) and drift_metadata.get('drift_start_idx') is not None:
            # Get indices before and after drift
            data['idx'] = range(len(data))
            pre_drift = data[data['idx'] < drift_metadata['drift_start_idx']]
            post_drift = data[data['idx'] >= drift_metadata['drift_start_idx']]
            
            # Calculate basic statistics pre and post drift
            metrics['drift_analysis']['pre_drift_stats'] = {
                'sample_count': len(pre_drift),
                'migraine_prevalence': targets.loc[pre_drift.index].mean(),
                'feature_means': {col: pre_drift[col].mean() for col in pre_drift.columns 
                                if col not in ['patient_id', 'timestamp', 'date', 'idx', 'modality']}
            }
            
            metrics['drift_analysis']['post_drift_stats'] = {
                'sample_count': len(post_drift),
                'migraine_prevalence': targets.loc[post_drift.index].mean(),
                'feature_means': {col: post_drift[col].mean() for col in post_drift.columns 
                                if col not in ['patient_id', 'timestamp', 'date', 'idx', 'modality']}
            }
            
            # Calculate drift magnitude for each feature
            drift_magnitudes = {}
            for feat in drift_metadata.get('drift_parameters', {}).get('affected_features', []):
                if feat in pre_drift.columns and feat in post_drift.columns:
                    pre_mean = pre_drift[feat].mean()
                    post_mean = post_drift[feat].mean()
                    if pre_mean != 0:
                        drift_magnitudes[feat] = abs((post_mean - pre_mean) / pre_mean)
                    else:
                        drift_magnitudes[feat] = abs(post_mean - pre_mean)
            
            metrics['drift_analysis']['measured_drift_magnitude'] = drift_magnitudes
            metrics['drift_analysis']['avg_drift_magnitude'] = np.mean(list(drift_magnitudes.values())) \
                                                               if drift_magnitudes else 0
        
        # Statistical metrics across the entire dataset
        metrics['statistical_metrics'] = {
            'sample_count': len(data),
            'migraine_prevalence': targets.mean(),
            'migraine_event_count': int(sum(targets)),
            'feature_means': {col: data[col].mean() for col in data.columns 
                             if col not in ['patient_id', 'timestamp', 'date', 'idx', 'modality']},
            'feature_std': {col: data[col].std() for col in data.columns 
                           if col not in ['patient_id', 'timestamp', 'date', 'idx', 'modality']}
        }
        
        # Calculate feature importance (correlation with target)
        feature_importances = {}
        for feat in data.columns:
            if feat not in ['patient_id', 'timestamp', 'date', 'idx', 'modality']:
                corr = np.corrcoef(data[feat], targets)[0, 1] if len(data[feat]) > 1 else 0
                if not np.isnan(corr):
                    feature_importances[feat] = abs(corr)
        
        metrics['statistical_metrics']['feature_importance'] = feature_importances
        
        # Clinical metrics - focus on migraine severity and clinical relevance
        # Extract migraine event metadata from targets.attrs if available
        migraine_events = getattr(targets, 'attrs', {}).get('migraine_events', [])
        avg_severity = getattr(targets, 'attrs', {}).get('avg_severity', 0)
        
        metrics['clinical_metrics'] = {
            'avg_migraine_severity': avg_severity,
            'migraine_frequency': len(migraine_events) / (data['date'].nunique() / 30)  # per month
        }
        
        # If drift was applied, calculate clinical impact of drift
        if drift_metadata.get('drift_applied', False) and drift_metadata.get('drift_start_idx') is not None:
            # Extract dates before and after drift
            pre_drift_dates = pre_drift['date'].unique()
            post_drift_dates = post_drift['date'].unique()
            
            # Calculate migraine frequency before and after drift
            pre_drift_events = [e for e in migraine_events 
                               if datetime.fromisoformat(e['date']).date() in pre_drift_dates]
            post_drift_events = [e for e in migraine_events 
                                if datetime.fromisoformat(e['date']).date() in post_drift_dates]
            
            pre_freq = len(pre_drift_events) / (len(pre_drift_dates) / 30)  # per month
            post_freq = len(post_drift_events) / (len(post_drift_dates) / 30)  # per month
            
            metrics['clinical_metrics']['pre_drift_frequency'] = pre_freq
            metrics['clinical_metrics']['post_drift_frequency'] = post_freq
            metrics['clinical_metrics']['frequency_change'] = (post_freq - pre_freq) / pre_freq if pre_freq > 0 else 0
            
            # Calculate severity change
            pre_severity = np.mean([e['severity'] for e in pre_drift_events]) if pre_drift_events else 0
            post_severity = np.mean([e['severity'] for e in post_drift_events]) if post_drift_events else 0
            
            metrics['clinical_metrics']['pre_drift_severity'] = pre_severity
            metrics['clinical_metrics']['post_drift_severity'] = post_severity
            metrics['clinical_metrics']['severity_change'] = (post_severity - pre_severity) / pre_severity \
                                                           if pre_severity > 0 else 0
        
        return metrics
    
    def _generate_enhanced_feedback(self,
                                    patient_id: str,
                                    profile_type: str,
                                    data: pd.DataFrame,
                                    targets: pd.Series,
                                    drift_metadata: Dict) -> List[Dict]:
        """
        Generate enhanced patient feedback based on data and migraine events.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        profile_type : str
            Type of patient profile
        data : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        drift_metadata : Dict
            Metadata about the applied drift
            
        Returns:
        --------
        List[Dict]
            List of feedback events with rich metadata
        """
        # Create feedback entries for each migraine event
        feedback_entries = []
        
        # Extract migraine events from targets attributes if available
        migraine_events = getattr(targets, 'attrs', {}).get('migraine_events', [])
        
        # If we don't have events from attributes, identify them from the targets
        if not migraine_events:
            # Group consecutive 1s in targets as single events
            event_dates = set()
            for idx, row in data.iterrows():
                if targets.loc[idx] == 1 and row['date'] not in event_dates:
                    event_dates.add(row['date'])
                    migraine_events.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'date': row['date'].isoformat()
                    })
        
        # Generate detailed feedback for each migraine event
        for event in migraine_events:
            # Determine if this event is post-drift
            post_drift = False
            if drift_metadata.get('drift_applied', False) and drift_metadata.get('drift_start_idx') is not None:
                event_time = datetime.fromisoformat(event['timestamp'])
                first_time = data['timestamp'].min()
                event_idx = (event_time - first_time).total_seconds() / (24 * 3600)
                post_drift = event_idx >= drift_metadata.get('drift_start_idx', 0)
            
            # Get the event timestamp and date
            event_time = datetime.fromisoformat(event['timestamp'])
            
            # Find data points near this event
            window_start = event_time - timedelta(hours=6)  # Include pre-migraine data
            window_end = event_time + timedelta(hours=12)   # Include during/post data
            
            # Find relevant data around the event
            event_window = data[(data['timestamp'] >= window_start) & (data['timestamp'] <= window_end)]
            
            # Extract feature values around the event
            feature_values = {}
            for category, features in self.feature_categories.items():
                category_data = event_window[event_window['modality'] == category]
                if not category_data.empty:
                    for feature in features:
                        if feature in category_data.columns:
                            feature_values[feature] = category_data[feature].mean()
            
            # Calculate severity based on the risk score or generate it
            severity = event.get('severity', 0.5)
            if severity == 0:
                severity = random.uniform(0.3, 1.0)  # Random severity if not available
            
            # Map severity to pain scale (0-10)
            pain_level = int(severity * 10)
            
            # Determine symptoms based on severity
            possible_symptoms = [
                'nausea', 'photophobia', 'phonophobia', 'vomiting', 'dizziness',
                'visual aura', 'numbness', 'tingling', 'difficulty concentrating',
                'fatigue', 'irritability', 'neck pain'
            ]
            
            # Number of symptoms based on severity
            num_symptoms = max(1, int(severity * len(possible_symptoms)))
            symptoms = random.sample(possible_symptoms, num_symptoms)
            
            # Calculate duration based on severity
            min_duration = 2  # hours
            max_duration = 72  # hours for severe cases
            duration = int(min_duration + severity * (max_duration - min_duration))
            
            # Add some variability to duration
            duration = max(1, int(duration * random.uniform(0.7, 1.3)))
            
            # Generate feedback entry
            entry = {
                'timestamp': event['timestamp'],
                'feedback_time': (event_time + timedelta(hours=duration + random.uniform(0, 6))).isoformat(),
                'event_type': 'migraine',
                'severity': severity,
                'pain_level': pain_level,  # 0-10 scale
                'duration_hours': duration,
                'symptoms': symptoms,
                'location': random.choice(['left', 'right', 'bilateral', 'frontal', 'occipital']),
                'triggers_reported': {},
                'medication_response': {},
                'notes': ''
            }
            
            # Add perceived triggers based on profile type
            profile_info = self.profile_types[profile_type]
            key_features = profile_info['key_features']
            
            # Select 1-3 features as perceived triggers
            num_triggers = random.randint(1, min(3, len(key_features)))
            trigger_features = random.sample(key_features, num_triggers)
            
            # Add trigger information
            for feature in trigger_features:
                if feature in feature_values:
                    value = feature_values[feature]
                    
                    # Convert technical feature name to user-friendly name
                    friendly_name = feature.replace('_', ' ').title()
                    
                    # Determine if the value is high or low
                    if feature in ['sleep_quality', 'water_intake']:
                        description = 'Low' if value < 50 else 'Normal'
                    else:
                        description = 'High' if value > 60 else 'Normal'
                    
                    entry['triggers_reported'][friendly_name] = description
            
            # Add medication response for moderate to severe migraines
            if pain_level >= 5:
                medications = {
                    'triptan': random.choice([True, False]),
                    'nsaid': random.choice([True, False]),
                    'acetaminophen': random.choice([True, False])
                }
                
                # Add effectiveness for each medication taken
                for med, taken in medications.items():
                    if taken:
                        # Lower effectiveness post-drift if applicable
                        base_effectiveness = random.uniform(0.3, 0.9)
                        
                        if post_drift and drift_metadata.get('type') in ['sudden', 'gradual']:
                            # Reduce medication effectiveness in drift scenarios
                            effectiveness = base_effectiveness * 0.7
                        else:
                            effectiveness = base_effectiveness
                            
                        entry['medication_response'][med] = {
                            'taken': True,
                            'timing': random.choice(['at onset', 'delayed']),
                            'effectiveness': min(1.0, effectiveness),
                            'side_effects': random.choice([None, 'drowsiness', 'nausea', 'dizziness']) \
                                            if random.random() < 0.3 else None
                        }
            
            # Generate freeform notes
            note_templates = [
                f"Migraine lasted about {duration} hours with {pain_level}/10 pain.",
                f"Had to lie down in a dark room. Pain was {pain_level}/10.",
                f"This migraine came on suddenly. Main symptoms were {', '.join(symptoms[:2])}.",
                f"Felt this one coming hours before. Pain reached {pain_level}/10 at peak."
            ]
            
            entry['notes'] = random.choice(note_templates)
            
            # Add drift-related notes if applicable
            if post_drift and random.random() < 0.4:
                drift_notes = [
                    "This migraine felt different from my usual pattern.",
                    "My usual triggers don't seem to explain this one.",
                    "My normal medication didn't work as well this time.",
                    "The pattern of my symptoms has been changing lately."
                ]
                entry['notes'] += " " + random.choice(drift_notes)
            
            feedback_entries.append(entry)
        
        return feedback_entries
    
    def _format_as_llif(self,
                        patient_id: str,
                        demographics: Dict,
                        data: pd.DataFrame,
                        targets: pd.Series,
                        feedback: List[Dict],
                        drift_metadata: Dict,
                        evaluation_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Format the generated data according to the Low-Level Inference Format (LLIF).
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        demographics : Dict
            Demographic data
        data : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        feedback : List[Dict]
            Feedback events
        drift_metadata : Dict
            Metadata about the applied drift
        evaluation_metrics : Dict, optional
            Calculated evaluation metrics
            
        Returns:
        --------
        Dict[str, Any]
            Data formatted according to LLIF specification
        """
        llif_data = {
            'metadata': {
                'format_version': self.llif_config['format_version'],
                'generation_time': datetime.now().isoformat(),
                'patient_id': patient_id,
                'data_type': 'synthetic',
                'drift_simulation': drift_metadata.get('drift_type', 'none')
            },
            'patient': {
                'demographics': demographics,
                'medical_history': demographics.get('medical_history', {})
            },
            'data_streams': {},
            'events': [],
            'feedback': feedback,
            'drift_metadata': drift_metadata
        }
        
        # Add evaluation metrics if available
        if evaluation_metrics:
            llif_data['evaluation_metrics'] = evaluation_metrics
        
        # Organize data by modality and time resolution
        for modality, resolution in self.llif_config['time_resolution'].items():
            # Filter data for this modality
            modality_data = data[data['modality'] == modality]
            
            if not modality_data.empty:
                # Get features for this modality
                modality_features = self.feature_categories.get(modality, [])
                
                # Create time series for each feature
                feature_series = {}
                for feature in modality_features:
                    if feature in modality_data.columns:
                        # Create series with timestamps and values
                        series = {
                            'name': feature,
                            'unit': 'normalized',  # Could be customized per feature
                            'resolution': resolution,
                            'timestamps': [ts.isoformat() for ts in modality_data['timestamp']],
                            'values': modality_data[feature].tolist()
                        }
                        feature_series[feature] = series
                
                llif_data['data_streams'][modality] = feature_series
        
        # Format migraine events
        migraine_events = getattr(targets, 'attrs', {}).get('migraine_events', [])
        
        for event in migraine_events:
            llif_event = {
                'event_type': 'migraine',
                'timestamp': event['timestamp'],
                'duration': random.randint(2, 72) * 3600,  # seconds
                'severity': event.get('severity', 0.5),
                'metadata': {
                    'contributing_features': event.get('contributing_features', {})
                }
            }
            llif_data['events'].append(llif_event)
        
        return llif_data
    
    def _save_enhanced_patient_data(self,
                                    patient_id: str,
                                    formatted_data: Dict[str, Any],
                                    include_visualization: bool = True) -> None:
        """
        Save the enhanced patient data to files.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        formatted_data : Dict[str, Any]
            Formatted patient data
        include_visualization : bool
            Whether to generate visualizations
        """
        # Create patient directory
        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete formatted data
        with open(patient_dir / 'patient_data.json', 'w') as f:
            json.dump(formatted_data, f, indent=2, default=str)
        
        # If the data is in LLIF format, extract components for easier access
        if 'data_streams' in formatted_data:
            # Extract demographics
            demographics = formatted_data.get('patient', {}).get('demographics', {})
            if demographics:
                with open(patient_dir / 'demographics.json', 'w') as f:
                    json.dump(demographics, f, indent=2)
            
            # Extract data streams into CSV files by modality
            for modality, features in formatted_data.get('data_streams', {}).items():
                modality_data = []
                
                for feature_name, feature_data in features.items():
                    timestamps = feature_data.get('timestamps', [])
                    values = feature_data.get('values', [])
                    
                    for i, (ts, val) in enumerate(zip(timestamps, values)):
                        modality_data.append({
                            'patient_id': patient_id,
                            'timestamp': ts,
                            'modality': modality,
                            feature_name: val
                        })
                
                if modality_data:
                    # Convert to DataFrame and save
                    df = pd.DataFrame(modality_data)
                    df.to_csv(patient_dir / f'{modality}_data.csv', index=False)
            
            # Extract events
            events = formatted_data.get('events', [])
            if events:
                with open(patient_dir / 'events.json', 'w') as f:
                    json.dump(events, f, indent=2, default=str)
            
            # Extract feedback
            feedback = formatted_data.get('feedback', [])
            if feedback:
                with open(patient_dir / 'feedback.json', 'w') as f:
                    json.dump(feedback, f, indent=2, default=str)
        
        # For standard format, extract components directly
        else:
            # Extract demographics
            demographics = formatted_data.get('demographics', {})
            if demographics:
                with open(patient_dir / 'demographics.json', 'w') as f:
                    json.dump(demographics, f, indent=2)
            
            # Extract data
            data = formatted_data.get('data', [])
            if data:
                pd.DataFrame(data).to_csv(patient_dir / 'timeseries_data.csv', index=False)
            
            # Extract feedback
            feedback = formatted_data.get('feedback', [])
            if feedback:
                with open(patient_dir / 'feedback.json', 'w') as f:
                    json.dump(feedback, f, indent=2, default=str)
        
        # Save drift metadata
        drift_metadata = formatted_data.get('drift_metadata', {})
        if drift_metadata:
            with open(patient_dir / 'drift_metadata.json', 'w') as f:
                json.dump(drift_metadata, f, indent=2, default=str)
        
        # Save evaluation metrics
        evaluation_metrics = formatted_data.get('evaluation_metrics', {})
        if evaluation_metrics:
            with open(patient_dir / 'evaluation_metrics.json', 'w') as f:
                json.dump(evaluation_metrics, f, indent=2, default=str)
        
        # Generate visualizations if requested
        if include_visualization:
            self._visualize_enhanced_patient_data(patient_id, formatted_data)
            
        logger.info(f"Saved enhanced data for patient {patient_id}")
    
    def create_patient_summary(self) -> Dict:
        """
        Create a summary of all generated patients with enhanced information.
        
        Returns:
        --------
        Dict
            Summary of generated patients
        """
        # Check if we have any stored patient data
        patient_ids = []
        results = {}
        
        # Collect all patient IDs from the output directory
        if self.output_dir.exists():
            for patient_dir in self.output_dir.glob("patient_*"):
                if patient_dir.is_dir():
                    patient_ids.append(patient_dir.name)
        
        # Use default empty dictionary if no data is available
        if not patient_ids:
            return {
                'generation_time': datetime.now().isoformat(),
                'total_patients': 0,
                'patient_list': [],
                'patients': {}
            }
        
        # Call the internal method with collected patient IDs
        return self._create_enhanced_patient_summary(patient_ids, results)
        
    def _create_enhanced_patient_summary(self, patient_ids: List[str], results: Dict) -> Dict:
        """
        Create a summary of all generated patients with enhanced information.
        
        Parameters:
        -----------
        patient_ids : List[str]
            List of patient identifiers
        results : Dict
            Complete results dictionary
            
        Returns:
        --------
        Dict
            Summary of generated patients
        """
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_patients': len(patient_ids),
            'patient_list': patient_ids,
            'drift_simulation': results.get('metadata', {}).get('parameters', {}).get('drift_type', 'none'),
            'patients': {}
        }
        
        # Collect key information for each patient
        for patient_data in results.get('patients', []):
            patient_id = patient_data.get('patient_id', '')
            if not patient_id:
                continue
                
            # Extract demographics
            if 'patient' in patient_data and 'demographics' in patient_data['patient']:
                demographics = patient_data['patient']['demographics']
            else:
                demographics = patient_data.get('demographics', {})
            
            # Extract migraine events
            events = patient_data.get('events', [])
            if not events and 'evaluation_metrics' in patient_data:
                events_count = patient_data['evaluation_metrics'].get('statistical_metrics', {}).get('migraine_event_count', 0)
            else:
                events_count = len(events)
            
            # Extract evaluation metrics if available
            eval_metrics = patient_data.get('evaluation_metrics', {})
            
            # Create patient summary
            summary['patients'][patient_id] = {
                'age': demographics.get('age'),
                'gender': demographics.get('gender'),
                'profile_type': demographics.get('profile_type'),
                'migraine_event_count': events_count,
                'avg_monthly_frequency': eval_metrics.get('clinical_metrics', {}).get('migraine_frequency', 0)
            }
            
            # Add drift-specific information if applicable
            if results.get('metadata', {}).get('parameters', {}).get('drift_type', 'none') != 'none':
                drift_impact = {
                    'frequency_change': eval_metrics.get('clinical_metrics', {}).get('frequency_change', 0),
                    'severity_change': eval_metrics.get('clinical_metrics', {}).get('severity_change', 0),
                    'avg_drift_magnitude': eval_metrics.get('drift_analysis', {}).get('avg_drift_magnitude', 0)
                }
                summary['patients'][patient_id]['drift_impact'] = drift_impact
        
        # Calculate overall metrics
        total_events = sum(p.get('migraine_event_count', 0) for p in summary['patients'].values())
        avg_frequency = np.mean([p.get('avg_monthly_frequency', 0) for p in summary['patients'].values()])
        
        summary['overall_metrics'] = {
            'total_migraine_events': total_events,
            'avg_monthly_frequency': avg_frequency
        }
        
        # Add drift summary if applicable
        if results.get('metadata', {}).get('parameters', {}).get('drift_type', 'none') != 'none':
            avg_frequency_change = np.mean([p.get('drift_impact', {}).get('frequency_change', 0) 
                                          for p in summary['patients'].values()])
            avg_severity_change = np.mean([p.get('drift_impact', {}).get('severity_change', 0) 
                                         for p in summary['patients'].values()])
            
            summary['drift_summary'] = {
                'drift_type': results.get('metadata', {}).get('parameters', {}).get('drift_type', 'none'),
                'avg_frequency_change': avg_frequency_change,
                'avg_severity_change': avg_severity_change,
            }
        
        # Save summary to file
        with open(self.output_dir / 'patient_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        return summary
    
    def _visualize_enhanced_patient_data(self, patient_id: str, formatted_data: Dict[str, Any]) -> None:
        """
        Create advanced visualizations of patient data with drift awareness.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        formatted_data : Dict[str, Any]
            Formatted patient data
        """
        # Create patient directory
        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data based on format
        data = None
        targets = None
        drift_metadata = formatted_data.get('drift_metadata', {})
        drift_applied = drift_metadata.get('drift_applied', False)
        drift_type = drift_metadata.get('drift_type', 'none')
        drift_start_idx = drift_metadata.get('drift_start_idx')
        
        # Extract events
        events = []
        if 'events' in formatted_data:
            events = formatted_data['events']
        
        # Get data from either LLIF or standard format
        if 'data_streams' in formatted_data:
            # LLIF format - combine data from all modalities
            combined_data = []
            timestamps = set()
            
            for modality, features in formatted_data.get('data_streams', {}).items():
                for feature_name, feature_data in features.items():
                    feature_timestamps = feature_data.get('timestamps', [])
                    feature_values = feature_data.get('values', [])
                    
                    for ts, val in zip(feature_timestamps, feature_values):
                        # Convert string timestamp to datetime
                        if isinstance(ts, str):
                            ts_dt = datetime.fromisoformat(ts)
                        else:
                            ts_dt = ts
                            
                        combined_data.append({
                            'patient_id': patient_id,
                            'timestamp': ts_dt,
                            'modality': modality,
                            feature_name: val
                        })
                        timestamps.add(ts_dt)
            
            if combined_data:
                # Convert to DataFrame
                data = pd.DataFrame(combined_data)
                
                # Create target series (1 for event timestamps, 0 otherwise)
                targets = pd.Series(0, index=data.index)
                
                # Mark event timestamps in targets
                for event in events:
                    event_time = event.get('timestamp')
                    if isinstance(event_time, str):
                        event_time = datetime.fromisoformat(event_time)
                    
                    # Find close timestamps in data
                    for idx, row in data.iterrows():
                        if abs((row['timestamp'] - event_time).total_seconds()) < 3600:  # Within 1 hour
                            targets.loc[idx] = 1
        else:
            # Standard format
            data_records = formatted_data.get('data', [])
            if data_records:
                data = pd.DataFrame(data_records)
                
                # Convert timestamp to datetime if it's a string
                if 'timestamp' in data.columns and isinstance(data['timestamp'].iloc[0], str):
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                # Get targets
                targets = pd.Series(formatted_data.get('targets', []))
        
        # Return if we couldn't extract valid data
        if data is None or targets is None or len(data) == 0:
            logger.warning(f"Could not extract data for visualization for patient {patient_id}")
            return
        
        # Create multiple visualizations
        self._create_timeseries_visualization(patient_id, data, targets, drift_metadata)
        self._create_feature_importance_visualization(patient_id, data, targets, drift_metadata)
        
        if drift_applied and drift_type != 'none':
            self._create_drift_visualization(patient_id, data, targets, drift_metadata)
        
        logger.info(f"Created enhanced visualizations for patient {patient_id}")
    
    def _create_timeseries_visualization(self, patient_id: str, data: pd.DataFrame, targets: pd.Series, 
                                        drift_metadata: Dict) -> None:
        """
        Create time series visualization of key features and migraine events.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        drift_metadata : Dict
            Metadata about the applied drift
        """
        patient_dir = self.output_dir / patient_id
        
        # Get key features from all categories (top 2 from each)
        key_features = []
        for category, features in self.feature_categories.items():
            # Get features that exist in data
            available_features = [f for f in features if f in data.columns]
            if available_features:
                # Sort by correlation with target
                feature_corrs = []
                for feat in available_features:
                    if len(data[feat]) > 1:
                        corr = np.corrcoef(data[feat], targets)[0, 1]
                        if not np.isnan(corr):
                            feature_corrs.append((feat, abs(corr)))
                
                # Sort by correlation (descending)
                feature_corrs.sort(key=lambda x: x[1], reverse=True)
                
                # Add top 2 features from this category
                top_features = [f[0] for f in feature_corrs[:2]]
                key_features.extend(top_features)
        
        # Create the figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 2]})
        
        # Plot 1: Key features over time
        feature_data = data.sort_values('timestamp')
        
        # Add visual indicator for drift start if applicable
        drift_applied = drift_metadata.get('drift_applied', False)
        drift_start_idx = drift_metadata.get('drift_start_idx')
        
        if drift_applied and drift_start_idx is not None:
            # Calculate the timestamp corresponding to drift start
            feature_data['idx'] = range(len(feature_data))
            drift_start_time = feature_data[feature_data['idx'] >= drift_start_idx]['timestamp'].min()
            
            if not pd.isna(drift_start_time):
                for ax in axs:
                    ax.axvline(x=drift_start_time, color='r', linestyle='--', alpha=0.7, linewidth=2)
                    ax.text(drift_start_time, ax.get_ylim()[1] * 0.95, f"Drift Start ({drift_metadata.get('drift_type')})", 
                           rotation=90, verticalalignment='top', color='r')
        
        # Plot key features
        for feature in key_features[:5]:  # Limit to top 5 features for clarity
            if feature in feature_data.columns:
                axs[0].plot(feature_data['timestamp'], feature_data[feature], 
                         label=feature.replace('_', ' ').title())
        
        axs[0].set_title(f"Key Features Over Time - Patient {patient_id}")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Feature Values")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: Migraine events
        event_data = data[targets > 0]
        axs[1].scatter(feature_data['timestamp'], targets, s=50, color='red', alpha=0.7)
        
        axs[1].set_title(f"Migraine Events - Patient {patient_id}")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Migraine Event")
        axs[1].set_ylim(-0.1, 1.1)
        axs[1].grid(True)
        
        # Plot 3: Clinical metrics if we have migraine events with time information
        if 'date' not in feature_data.columns:
            feature_data['date'] = feature_data['timestamp'].dt.date
        
        # Group by date and count migraine events
        daily_counts = feature_data.groupby('date').apply(lambda x: sum(targets.loc[x.index])).reset_index()
        daily_counts.columns = ['date', 'migraine_count']
        
        # Calculate 7-day rolling average
        if len(daily_counts) > 7:
            daily_counts['rolling_avg'] = daily_counts['migraine_count'].rolling(7).mean()
            
            # Plot rolling average
            axs[2].plot(daily_counts['date'], daily_counts['rolling_avg'], 
                     label='7-day Rolling Average', linewidth=2)
            
        # Plot daily counts as bars
        axs[2].bar(daily_counts['date'], daily_counts['migraine_count'], 
                 alpha=0.4, label='Daily Count')
        
        axs[2].set_title(f"Migraine Frequency Over Time - Patient {patient_id}")
        axs[2].set_xlabel("Date")
        axs[2].set_ylabel("Migraine Count")
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(patient_dir / 'timeseries_visualization.png')
        plt.close(fig)
    
    def _create_feature_importance_visualization(self, patient_id: str, data: pd.DataFrame, 
                                               targets: pd.Series, drift_metadata: Dict) -> None:
        """
        Create feature importance visualization.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        drift_metadata : Dict
            Metadata about the applied drift
        """
        patient_dir = self.output_dir / patient_id
        
        # Calculate feature correlations with migraine events
        correlations = {}
        for feature in data.columns:
            if feature not in ['patient_id', 'timestamp', 'date', 'idx', 'modality', 'target']:
                if len(data[feature]) > 1:
                    corr = np.corrcoef(data[feature], targets)[0, 1]
                    if not np.isnan(corr):
                        correlations[feature] = corr
        
        # Sort correlations by absolute value
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 15 features
        top_features = sorted_corrs[:15]
        
        # Create the plot
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        
        # Plot feature importance
        features = [x[0].replace('_', ' ').title() for x in top_features]
        importances = [x[1] for x in top_features]
        
        # Color based on correlation direction
        colors = ['g' if imp > 0 else 'r' for imp in importances]
        
        # Sort for horizontal bar chart (ascending for better display)
        features_sorted = [x for _, x in sorted(zip(abs(np.array(importances)), features))]
        importances_sorted = [x for _, x in sorted(zip(abs(np.array(importances)), importances))]
        colors_sorted = [x for _, x in sorted(zip(abs(np.array(importances)), colors))]
        
        # Create horizontal bar chart
        bars = ax.barh(features_sorted, importances_sorted, color=colors_sorted)
        
        # Add labels and title
        plt.xlabel('Correlation with Migraine Events')
        plt.title(f'Feature Importance for Patient {patient_id}')
        
        # Add vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add grid lines
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(patient_dir / 'feature_importance.png')
        plt.close(fig)
    
    def _create_drift_visualization(self, patient_id: str, data: pd.DataFrame, 
                                  targets: pd.Series, drift_metadata: Dict) -> None:
        """
        Create visualization specifically highlighting drift effects.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        drift_metadata : Dict
            Metadata about the applied drift
        """
        patient_dir = self.output_dir / patient_id
        
        # Only create this visualization if drift was applied
        if not drift_metadata.get('drift_applied', False) or drift_metadata.get('drift_type', 'none') == 'none':
            return
        
        # Get drift parameters
        drift_type = drift_metadata.get('drift_type', 'none')
        drift_start_idx = drift_metadata.get('drift_start_idx')
        
        if drift_start_idx is None:
            return
        
        # Split data into pre-drift and post-drift
        data['idx'] = range(len(data))
        pre_drift = data[data['idx'] < drift_start_idx]
        post_drift = data[data['idx'] >= drift_start_idx]
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Title for the entire figure
        fig.suptitle(f"Drift Analysis for Patient {patient_id} - {drift_type.title()} Drift", fontsize=16)
        
        # 1. Plot affected features before and after drift
        affected_features = drift_metadata.get('drift_parameters', {}).get('affected_features', [])
        
        # Select up to 3 affected features to visualize
        vis_features = affected_features[:3] if len(affected_features) > 3 else affected_features
        
        for feature in vis_features:
            if feature in data.columns:
                # Calculate moving average to smooth the visualization
                window_size = min(20, len(data) // 10)  # Adaptive window size
                data[f'{feature}_ma'] = data[feature].rolling(window=window_size, min_periods=1).mean()
                
                # Plot the moving average
                axs[0, 0].plot(data['timestamp'], data[f'{feature}_ma'], 
                            label=feature.replace('_', ' ').title())
        
        # Add vertical line for drift start
        if len(post_drift) > 0:
            drift_start_time = post_drift['timestamp'].min()
            axs[0, 0].axvline(x=drift_start_time, color='r', linestyle='--', linewidth=2)
            axs[0, 0].text(drift_start_time, axs[0, 0].get_ylim()[1], "Drift Start", 
                        rotation=90, verticalalignment='top', color='r')
        
        axs[0, 0].set_title("Key Affected Features")
        axs[0, 0].set_xlabel("Time")
        axs[0, 0].set_ylabel("Feature Value")
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # 2. Plot migraine frequency before and after drift
        # Safely add date column
        if len(pre_drift) > 0:
            pre_drift = pre_drift.copy()  # Avoid SettingWithCopyWarning
            pre_drift['date'] = pre_drift['timestamp'].dt.date
        
        if len(post_drift) > 0:
            post_drift = post_drift.copy()  # Avoid SettingWithCopyWarning
            post_drift['date'] = post_drift['timestamp'].dt.date
        
        # Group by date and calculate migraine frequency
        pre_daily_counts = pd.DataFrame(columns=['date', 'migraine_freq'])
        post_daily_counts = pd.DataFrame(columns=['date', 'migraine_freq'])
        
        if len(pre_drift) > 0:
            pre_daily_counts = pre_drift.groupby('date').apply(
                lambda x: sum(targets.loc[x.index]) / len(x) if len(x) > 0 else 0
            ).reset_index()
            pre_daily_counts.columns = ['date', 'migraine_freq']
        
        if len(post_drift) > 0:
            post_daily_counts = post_drift.groupby('date').apply(
                lambda x: sum(targets.loc[x.index]) / len(x) if len(x) > 0 else 0
            ).reset_index()
            post_daily_counts.columns = ['date', 'migraine_freq']
        
        # Calculate averages, handling empty dataframes and NaN values
        pre_avg = pre_daily_counts['migraine_freq'].mean() if (len(pre_daily_counts) > 0 and 
                                                           not pre_daily_counts['migraine_freq'].isna().all()) else 0
        post_avg = post_daily_counts['migraine_freq'].mean() if (len(post_daily_counts) > 0 and 
                                                             not post_daily_counts['migraine_freq'].isna().all()) else 0
        
        # Plot daily frequency if data is available
        if len(pre_daily_counts) > 0:
            axs[0, 1].scatter(pre_daily_counts['date'], pre_daily_counts['migraine_freq'], 
                           alpha=0.6, label='Pre-Drift', color='blue')
        
        if len(post_daily_counts) > 0:
            axs[0, 1].scatter(post_daily_counts['date'], post_daily_counts['migraine_freq'], 
                           alpha=0.6, label='Post-Drift', color='red')
        
        # Add horizontal lines for averages
        if not np.isnan(pre_avg):
            axs[0, 1].axhline(y=pre_avg, color='blue', linestyle='--', 
                          label=f'Pre-Drift Avg: {pre_avg:.3f}')
        
        if not np.isnan(post_avg):
            axs[0, 1].axhline(y=post_avg, color='red', linestyle='--', 
                          label=f'Post-Drift Avg: {post_avg:.3f}')
        
        axs[0, 1].set_title("Migraine Frequency Before and After Drift")
        axs[0, 1].set_xlabel("Date")
        axs[0, 1].set_ylabel("Daily Migraine Frequency")
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # 3. Feature distribution changes
        feature_to_analyze = vis_features[0] if vis_features else None
        
        if feature_to_analyze and feature_to_analyze in data.columns:
            # Check for valid pre and post drift data
            pre_data_valid = (len(pre_drift) > 0 and 
                             feature_to_analyze in pre_drift.columns and 
                             not pre_drift[feature_to_analyze].isna().all())
            
            post_data_valid = (len(post_drift) > 0 and 
                              feature_to_analyze in post_drift.columns and 
                              not post_drift[feature_to_analyze].isna().all())
            
            if pre_data_valid and post_data_valid:
                # Create histograms for pre and post drift
                # Drop NaN values before creating histogram
                pre_feature_data = pre_drift[feature_to_analyze].dropna()
                post_feature_data = post_drift[feature_to_analyze].dropna()
                
                if len(pre_feature_data) > 0:
                    axs[1, 0].hist(pre_feature_data, bins=20, alpha=0.5, label='Pre-Drift', 
                                color='blue', density=True)
                
                if len(post_feature_data) > 0:
                    axs[1, 0].hist(post_feature_data, bins=20, alpha=0.5, label='Post-Drift', 
                                color='red', density=True)
                
                axs[1, 0].set_title(f"Distribution Change: {feature_to_analyze.replace('_', ' ').title()}")
                axs[1, 0].set_xlabel("Value")
                axs[1, 0].set_ylabel("Density")
                axs[1, 0].legend()
                axs[1, 0].grid(True)
            else:
                axs[1, 0].text(0.5, 0.5, "Insufficient data for histogram", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[1, 0].transAxes)
                axs[1, 0].set_title("Feature Distribution (No Valid Data)")
        
        # 4. Create a visualization of drift magnitude for affected features
        drift_magnitudes = {}
        for feat in affected_features:
            if (feat in pre_drift.columns and feat in post_drift.columns and 
                len(pre_drift) > 0 and len(post_drift) > 0):
                
                # Get means with NaN handling
                pre_mean = pre_drift[feat].mean() if not pre_drift[feat].isna().all() else 0
                post_mean = post_drift[feat].mean() if not post_drift[feat].isna().all() else 0
                
                # Skip if either mean is NaN
                if np.isnan(pre_mean) or np.isnan(post_mean):
                    continue
                    
                if pre_mean != 0:
                    drift_magnitudes[feat.replace('_', ' ').title()] = abs((post_mean - pre_mean) / pre_mean)
                else:
                    drift_magnitudes[feat.replace('_', ' ').title()] = abs(post_mean - pre_mean)
        
        # Sort magnitudes and handle empty drift_magnitudes
        if drift_magnitudes:
            sorted_mags = sorted(drift_magnitudes.items(), key=lambda x: x[1], reverse=True)
            features_mag = [x[0] for x in sorted_mags[:10]]  # Top 10 features
            magnitudes = [x[1] for x in sorted_mags[:10]]  # Top 10 magnitudes
            
            # Create bar chart of drift magnitudes
            if features_mag and magnitudes:
                bars = axs[1, 1].barh(features_mag, magnitudes, color='purple')
            else:
                axs[1, 1].text(0.5, 0.5, "No significant drift detected", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[1, 1].transAxes)
        else:
            axs[1, 1].text(0.5, 0.5, "No valid drift data available", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axs[1, 1].transAxes)
        
        axs[1, 1].set_title("Drift Magnitude by Feature")
        axs[1, 1].set_xlabel("Relative Change")
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(patient_dir / 'drift_analysis.png')
        plt.close(fig)


class EnhancedSyntheticDataGenerator:
    """Enhanced synthetic data generator for MoE validation with real data"""
    
    def __init__(self, num_samples: int = 1000, drift_type: str = 'sudden', data_modality: str = 'mixed'):
        """
        Initialize the enhanced synthetic data generator.
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
        drift_type : str
            Type of drift to simulate ('sudden', 'gradual', 'recurring', 'none')
        data_modality : str
            Type of data to generate ('physiological', 'environmental', 'behavioral', 'mixed')
        """
        self.num_samples = num_samples
        self.drift_type = drift_type
        self.data_modality = data_modality
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize the patient data generator as the base
        self.base_generator = EnhancedPatientDataGenerator()
        
        # Feature categories by modality
        self.modality_features = {
            'physiological': [
                'heart_rate', 'blood_pressure', 'temperature', 'hrv',
                'body_temperature', 'sleep_rem_percentage', 'sleep_deep_percentage'
            ],
            'environmental': [
                'humidity', 'barometric_pressure', 'temperature', 
                'weather_change_rate', 'air_quality_pm25', 'uv_index'
            ],
            'behavioral': [
                'stress_level', 'sleep_quality', 'activity_level',
                'screen_blue_light_exposure', 'social_activity_hours'
            ],
            'medication': [
                'triptan_usage', 'nsaid_usage', 'preventative_adherence',
                'rescue_medication_count'
            ]
        }
        
        # Drift parameters for different types
        self.drift_params = {
            'sudden': {
                'magnitude': 0.5,
                'start_point': 0.5,  # Halfway through the data
                'features_affected': 0.4
            },
            'gradual': {
                'magnitude': 0.4,
                'rate': 0.05,
                'start_point': 0.3,
                'features_affected': 0.6
            },
            'recurring': {
                'magnitude': 0.3,
                'frequency': 0.1,
                'features_affected': 0.5
            },
            'none': {}
        }
    
    def generate_mirrored_data(self, feature_stats: dict, target_column: str, target_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate synthetic data mirroring the characteristics of real data.
        
        Parameters:
        -----------
        feature_stats : dict
            Statistics of features to mirror (means, stds, etc.)
        target_column : str
            Name of the target column
        target_ratio : float
            Ratio of positive cases in the target column
            
        Returns:
        --------
        pd.DataFrame
            Synthetic data mirroring real data characteristics
        """
        self.logger.info(f"Generating {self.num_samples} synthetic samples with {self.drift_type} drift")
        
        # Get the features to include based on modality
        if self.data_modality == 'mixed':
            # Use features from all modalities
            features = []
            for modality in self.modality_features:
                features.extend(self.modality_features[modality])
        else:
            # Use features from the specified modality
            features = self.modality_features.get(self.data_modality, [])
        
        # Create a dataframe with the required features
        df = pd.DataFrame(index=range(self.num_samples))
        
        # Generate timestamps spanning 30 days
        start_date = datetime.now() - timedelta(days=30)
        timestamps = [start_date + timedelta(minutes=i*15) for i in range(self.num_samples)]
        df['timestamp'] = timestamps
        
        # Set the target column with the desired ratio of positive cases
        target_values = np.zeros(self.num_samples)
        
        # Ensure target_ratio is valid (between 0 and 1)
        safe_target_ratio = min(max(0.0, target_ratio), 1.0)
        num_positive = int(self.num_samples * safe_target_ratio)
        
        # Use replace=True if we need more samples than available
        # This shouldn't happen with a proper ratio but adds robustness
        if num_positive >= self.num_samples:
            num_positive = int(self.num_samples * 0.5)  # Default to 50% if ratio is invalid
            self.logger.warning(f"Target ratio {target_ratio} too large, defaulting to 0.5")
            
        positive_indices = np.random.choice(
            range(self.num_samples), 
            size=num_positive, 
            replace=False
        )
        target_values[positive_indices] = 1
        df[target_column] = target_values
        
        # Generate feature data with appropriate drift characteristics
        for feature in features:
            # Generate base feature values
            if feature in feature_stats and 'mean' in feature_stats[feature] and 'std' in feature_stats[feature]:
                # Use statistics from real data if available
                mean = feature_stats[feature]['mean']
                std = feature_stats[feature]['std']
            else:
                # Use default values if not available
                mean = np.random.uniform(50, 100)
                std = np.random.uniform(5, 20)
            
            # Generate base feature values
            feature_values = np.random.normal(mean, std, self.num_samples)
            
            # Apply drift if specified
            if self.drift_type != 'none':
                feature_values = self._apply_drift(feature_values, feature)
            
            df[feature] = feature_values
        
        # Post-process to ensure data quality
        self._post_process_data(df, target_column)
        
        return df
    
    def _apply_drift(self, feature_values: np.ndarray, feature: str) -> np.ndarray:
        """
        Apply drift to feature values based on drift type.
        
        Parameters:
        -----------
        feature_values : np.ndarray
            Original feature values
        feature : str
            Feature name
            
        Returns:
        --------
        np.ndarray
            Feature values with drift applied
        """
        # Determine if this feature should be affected by drift
        # Use hash of feature name for consistent decision
        feature_hash = hash(feature) % 100 / 100.0
        if feature_hash > self.drift_params[self.drift_type].get('features_affected', 0.5):
            return feature_values
        
        # Make a copy to avoid modifying the original
        drifted_values = feature_values.copy()
        n_samples = len(feature_values)
        
        if self.drift_type == 'sudden':
            # Apply sudden drift after the start point
            start_idx = int(n_samples * self.drift_params['sudden']['start_point'])
            magnitude = self.drift_params['sudden']['magnitude']
            drifted_values[start_idx:] += magnitude * np.std(feature_values) * (1 if feature_hash > 0.5 else -1)
            
        elif self.drift_type == 'gradual':
            # Apply gradual drift that increases over time
            start_idx = int(n_samples * self.drift_params['gradual']['start_point'])
            max_magnitude = self.drift_params['gradual']['magnitude']
            rate = self.drift_params['gradual']['rate']
            
            for i in range(start_idx, n_samples):
                progress = (i - start_idx) / (n_samples - start_idx)  # 0 to 1
                current_magnitude = progress * max_magnitude
                drifted_values[i] += current_magnitude * np.std(feature_values) * (1 if feature_hash > 0.5 else -1)
                
        elif self.drift_type == 'recurring':
            # Apply cyclical drift pattern
            magnitude = self.drift_params['recurring']['magnitude']
            frequency = self.drift_params['recurring']['frequency'] * n_samples
            
            for i in range(n_samples):
                cycle_phase = np.sin(2 * np.pi * i / frequency)
                drifted_values[i] += magnitude * cycle_phase * np.std(feature_values)
        
        return drifted_values
    
    def _post_process_data(self, df: pd.DataFrame, target_column: str) -> None:
        """
        Post-process generated data for quality.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Generated dataframe
        target_column : str
            Name of the target column
        """
        # Ensure positive correlation between related features
        if 'stress_level' in df.columns and 'heart_rate' in df.columns:
            # Stress and heart rate should be positively correlated
            correlation = np.corrcoef(df['stress_level'], df['heart_rate'])[0, 1]
            if correlation < 0.3:
                # Adjust to ensure positive correlation
                df['heart_rate'] = df['heart_rate'] + 0.5 * df['stress_level']
        
        # Ensure target is related to key features
        target_triggers = [
            'stress_level', 'barometric_pressure', 'sleep_quality',
            'preventative_adherence', 'weather_change_rate'
        ]
        
        for trigger in target_triggers:
            if trigger in df.columns:
                # For each feature identified as a trigger, ensure some correlation with target
                for i in df[df[target_column] == 1].index:
                    # For positive cases, push the trigger feature values toward extremes
                    if np.random.random() > 0.3:  # 70% chance to adjust
                        mean = np.mean(df[trigger])
                        std = np.std(df[trigger])
                        # Push high for some features, low for others based on feature name hash
                        direction = 1 if hash(trigger) % 2 == 0 else -1
                        df.loc[i, trigger] = mean + direction * np.random.uniform(1.0, 2.0) * std

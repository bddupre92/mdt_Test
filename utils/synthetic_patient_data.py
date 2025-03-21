"""
Synthetic Patient Data Generator

This module generates synthetic patient data for testing the personalization layer
and patient profile adaptation features. It creates realistic patterns for various
patient profiles to thoroughly test the adaptation mechanisms.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatientDataGenerator:
    """
    Generates synthetic patient data with realistic patterns for testing
    personalization and adaptation features.
    """
    
    def __init__(self, 
                 output_dir: str = 'data/synthetic_patients',
                 seed: int = 42,
                 n_features: int = 20):
        """
        Initialize the patient data generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to store generated data
        seed : int
            Random seed for reproducibility
        n_features : int
            Number of features to generate
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.n_features = n_features
        
        # Define feature categories
        self.feature_categories = {
            'physiological': ['heart_rate', 'blood_pressure', 'sleep_quality', 'stress_level', 
                              'physio_marker_1', 'physio_marker_2', 'physio_marker_3'],
            'environmental': ['temperature', 'humidity', 'barometric_pressure', 'light_exposure',
                             'noise_level', 'air_quality', 'pollen_count'],
            'behavioral': ['caffeine_intake', 'water_intake', 'exercise_minutes', 'screen_time',
                          'meal_regularity', 'alcohol_units', 'medication_adherence']
        }
        
        # Define possible patient profile types
        self.profile_types = {
            'stress_sensitive': {
                'description': 'Patients sensitive to stress and anxiety',
                'key_features': ['stress_level', 'sleep_quality', 'heart_rate'],
                'feature_weights': {'physiological': 0.6, 'environmental': 0.2, 'behavioral': 0.2}
            },
            'weather_sensitive': {
                'description': 'Patients sensitive to weather changes',
                'key_features': ['barometric_pressure', 'humidity', 'temperature'],
                'feature_weights': {'physiological': 0.2, 'environmental': 0.7, 'behavioral': 0.1}
            },
            'lifestyle_sensitive': {
                'description': 'Patients sensitive to lifestyle factors',
                'key_features': ['caffeine_intake', 'water_intake', 'exercise_minutes', 'sleep_quality'],
                'feature_weights': {'physiological': 0.3, 'environmental': 0.1, 'behavioral': 0.6}
            },
            'mixed_triggers': {
                'description': 'Patients with mixed trigger patterns',
                'key_features': ['stress_level', 'barometric_pressure', 'caffeine_intake', 'sleep_quality'],
                'feature_weights': {'physiological': 0.4, 'environmental': 0.3, 'behavioral': 0.3}
            }
        }
        
        # Demographic options
        self.age_ranges = [(18, 30), (31, 45), (46, 60), (61, 80)]
        self.genders = ['male', 'female', 'non-binary']
        self.ethnicities = ['caucasian', 'african_american', 'hispanic', 'asian', 'other']
        self.comorbidities = ['none', 'hypertension', 'diabetes', 'depression', 'anxiety', 
                             'sleep_disorder', 'allergies']
    
    def generate_patient_set(self, 
                            num_patients: int = 10, 
                            time_periods: int = 30,
                            samples_per_period: int = 3,
                            include_feedback: bool = True,
                            visualize: bool = True) -> List[str]:
        """
        Generate a complete set of patient data.
        
        Parameters:
        -----------
        num_patients : int
            Number of patients to generate
        time_periods : int
            Number of days to generate data for
        samples_per_period : int
            Number of samples per day
        include_feedback : bool
            Whether to include simulated patient feedback
        visualize : bool
            Whether to generate visualizations
            
        Returns:
        --------
        List[str]
            List of patient IDs
        """
        patient_ids = []
        
        for i in range(num_patients):
            # Generate a unique patient ID
            patient_id = f"patient_{i+1:03d}"
            patient_ids.append(patient_id)
            
            # Select a random profile type for this patient
            profile_type = random.choice(list(self.profile_types.keys()))
            
            logger.info(f"Generating data for {patient_id} with profile type: {profile_type}")
            
            # Generate demographics
            demographics = self._generate_demographics(patient_id, profile_type)
            
            # Generate time series data
            data, targets = self._generate_patient_timeseries(
                patient_id, 
                profile_type, 
                time_periods, 
                samples_per_period
            )
            
            # Generate feedback if requested
            feedback = None
            if include_feedback:
                feedback = self._generate_patient_feedback(patient_id, profile_type, data, targets)
            
            # Save the generated data
            self._save_patient_data(patient_id, demographics, data, targets, feedback)
            
            # Generate visualizations if requested
            if visualize:
                self._visualize_patient_data(patient_id, data, targets)
        
        # Create a summary file
        self._create_patient_summary(patient_ids)
        
        return patient_ids
    
    def _generate_demographics(self, patient_id: str, profile_type: str) -> Dict:
        """
        Generate demographic data for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        profile_type : str
            Type of patient profile
            
        Returns:
        --------
        Dict
            Demographic data
        """
        age_range = random.choice(self.age_ranges)
        age = random.randint(age_range[0], age_range[1])
        
        gender = random.choice(self.genders)
        ethnicity = random.choice(self.ethnicities)
        
        # Assign random comorbidities (0-3)
        num_comorbidities = random.randint(0, 3)
        patient_comorbidities = random.sample(self.comorbidities, num_comorbidities)
        if 'none' in patient_comorbidities and len(patient_comorbidities) > 1:
            patient_comorbidities.remove('none')
        
        # Generate migraine history
        migraine_years = min(age - 10, random.randint(1, 20)) if age > 15 else random.randint(1, 5)
        avg_frequency = round(random.uniform(1, 8), 1)  # Average migraines per month
        
        # Generate medication history
        medications = []
        if random.random() > 0.3:  # 70% chance of being on medication
            num_meds = random.randint(1, 3)
            med_options = ['triptan', 'nsaid', 'antidepressant', 'beta_blocker', 'anticonvulsant', 'cgrp_inhibitor']
            medications = random.sample(med_options, num_meds)
        
        demographics = {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'comorbidities': patient_comorbidities,
            'migraine_history': {
                'years_with_migraine': migraine_years,
                'avg_monthly_frequency': avg_frequency,
                'typical_duration_hours': random.randint(4, 72)
            },
            'medications': medications,
            'profile_type': profile_type
        }
        
        return demographics
    
    def _generate_patient_timeseries(self, 
                                    patient_id: str, 
                                    profile_type: str, 
                                    time_periods: int,
                                    samples_per_period: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate time series data for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        profile_type : str
            Type of patient profile
        time_periods : int
            Number of days to generate data for
        samples_per_period : int
            Number of samples per day
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            DataFrames containing features and target values
        """
        # Get profile characteristics
        profile_info = self.profile_types[profile_type]
        key_features = profile_info['key_features']
        
        # Create base feature dictionary with all available features
        all_features = []
        for category in self.feature_categories.values():
            all_features.extend(category)
        
        # Generate base means and variances for this patient
        feature_means = {}
        feature_vars = {}
        
        for feature in all_features:
            # Normal distribution for most features
            feature_means[feature] = random.uniform(40, 60)
            feature_vars[feature] = random.uniform(5, 15)
        
        # Generate time series data
        total_samples = time_periods * samples_per_period
        data = []
        
        # Create timestamps
        start_date = datetime.now() - timedelta(days=time_periods)
        timestamps = []
        
        for day in range(time_periods):
            for sample in range(samples_per_period):
                # Generate timestamp with some variability in time of day
                hour = int(8 + (16 * sample / samples_per_period) + random.uniform(-1, 1))
                hour = max(0, min(23, hour))  # Ensure hour is valid
                
                timestamp = start_date + timedelta(days=day, hours=hour)
                timestamps.append(timestamp)
        
        # Generate feature values for each timestamp
        for i, timestamp in enumerate(timestamps):
            # Add some trends and patterns over time
            time_factor = i / total_samples  # Ranges from 0 to 1 over the whole period
            
            # Create a sample for this timestamp
            sample = {'patient_id': patient_id, 'timestamp': timestamp}
            
            # Add noise to each feature
            for feature in all_features:
                # Add time-based trends for some features
                if feature in ['stress_level', 'sleep_quality']:
                    # Weekly cyclic pattern (work week stress)
                    day_of_week = timestamp.weekday()
                    week_factor = 1.0 + 0.2 * (day_of_week < 5)  # Higher on weekdays
                else:
                    week_factor = 1.0
                
                # Base value with trend and weekly pattern
                base_value = feature_means[feature] * (1 + 0.1 * np.sin(time_factor * 2 * np.pi)) * week_factor
                
                # Add random noise
                noise = np.random.normal(0, np.sqrt(feature_vars[feature]))
                value = base_value + noise
                
                # Ensure reasonable bounds
                value = max(0, min(100, value))
                sample[feature] = value
            
            # Apply profile-specific patterns to key features
            for feature in key_features:
                if feature in sample:
                    # Make key features more extreme
                    deviation = sample[feature] - feature_means[feature]
                    sample[feature] = feature_means[feature] + (deviation * 1.5)
            
            data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Generate target values (migraine events)
        targets = self._generate_target_values(df, profile_type, key_features)
        
        return df, targets
    
    def _generate_target_values(self, 
                               df: pd.DataFrame, 
                               profile_type: str,
                               key_features: List[str]) -> pd.Series:
        """
        Generate target values (migraine events) based on feature data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature data
        profile_type : str
            Type of patient profile
        key_features : List[str]
            Key features for this profile type
            
        Returns:
        --------
        pd.Series
            Target values (1 for migraine, 0 for no migraine)
        """
        # Initialize target column
        targets = np.zeros(len(df))
        
        # Calculate a risk score based on key features
        risk_scores = np.zeros(len(df))
        
        for feature in key_features:
            if feature in df.columns:
                # Normalize the feature
                normalized = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
                
                # For some features, higher values mean higher risk
                if feature in ['stress_level', 'caffeine_intake', 'screen_time', 'heart_rate']:
                    risk_factor = normalized
                # For some features, lower values mean higher risk
                elif feature in ['sleep_quality', 'water_intake', 'exercise_minutes']:
                    risk_factor = 1 - normalized
                # For environmental factors, extreme values (high or low) mean higher risk
                elif feature in ['temperature', 'humidity', 'barometric_pressure']:
                    risk_factor = 2 * abs(normalized - 0.5)
                else:
                    risk_factor = normalized
                
                # Add to risk score with some random weight
                feature_weight = random.uniform(0.5, 1.5)
                risk_scores += risk_factor * feature_weight
        
        # Normalize risk scores to 0-1 range
        risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())
        
        # Add some randomness
        risk_scores = risk_scores * 0.8 + np.random.random(len(df)) * 0.2
        
        # Add time-dependent correlation (migraines are more likely if previous values were high)
        for i in range(1, len(df)):
            risk_scores[i] = risk_scores[i] * 0.8 + risk_scores[i-1] * 0.2
        
        # Generate migraine events based on risk scores
        migraine_threshold = 0.7  # Threshold for migraine event
        
        for i in range(len(df)):
            if risk_scores[i] > migraine_threshold:
                # Create a migraine event
                targets[i] = 1
                
                # Add cooldown period after a migraine (lower likelihood for a while)
                cooldown = min(10, len(df) - i - 1)
                for j in range(1, cooldown):
                    if i + j < len(df):
                        risk_scores[i + j] *= (0.5 + j / cooldown / 2)  # Gradually return to normal
        
        return pd.Series(targets)
    
    def _generate_patient_feedback(self, 
                                  patient_id: str, 
                                  profile_type: str,
                                  data: pd.DataFrame,
                                  targets: pd.Series) -> List[Dict]:
        """
        Generate simulated patient feedback based on data and migraine events.
        
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
            
        Returns:
        --------
        List[Dict]
            List of feedback events
        """
        feedback_events = []
        
        # Get timestamps where migraines occurred
        migraine_indices = np.where(targets == 1)[0]
        
        # Generate feedback for some migraine events (not all, simulating incomplete feedback)
        for idx in migraine_indices:
            if random.random() < 0.7:  # 70% chance of providing feedback
                timestamp = data.iloc[idx]['timestamp']
                
                # Get key features for this profile
                key_features = self.profile_types[profile_type]['key_features']
                
                # Generate trigger sensitivity feedback
                triggers = {}
                for feature in key_features:
                    if feature in data.columns:
                        # Calculate how extreme the value was
                        value = data.iloc[idx][feature]
                        mean = data[feature].mean()
                        std = data[feature].std()
                        
                        # Z-score as a measure of how unusual this value was
                        z_score = abs(value - mean) / std if std > 0 else 0
                        
                        # If value was unusual, include as a trigger
                        if z_score > 1.0:
                            trigger_certainty = min(1.0, z_score / 3)
                            triggers[feature] = round(trigger_certainty, 2)
                
                # Generate symptom severity (random but correlated with trigger strength)
                avg_trigger_strength = sum(triggers.values()) / len(triggers) if triggers else 0.5
                severity = {
                    'pain_level': round(random.uniform(5, 10) * avg_trigger_strength, 1),
                    'nausea': random.random() > 0.3,  # 70% chance of nausea
                    'aura': random.random() > 0.7,    # 30% chance of aura
                    'duration_hours': round(random.uniform(4, 48) * avg_trigger_strength, 1)
                }
                
                # Generate treatment effectiveness
                treatment_effectiveness = {
                    'medication_effectiveness': round(random.uniform(0.2, 0.9), 2),
                    'onset_to_treatment_hours': round(random.uniform(0.5, 6), 1)
                }
                
                feedback = {
                    'timestamp': timestamp.isoformat(),
                    'migraine_confirmed': True,
                    'trigger_sensitivity': triggers,
                    'symptom_severity': severity,
                    'treatment_effectiveness': treatment_effectiveness
                }
                
                feedback_events.append(feedback)
        
        # Add some false negative feedback (patient had migraine but model didn't predict)
        false_neg_count = random.randint(0, 3)
        non_migraine_indices = np.where(targets == 0)[0]
        
        if len(non_migraine_indices) > 0 and false_neg_count > 0:
            false_neg_indices = np.random.choice(non_migraine_indices, 
                                                size=min(false_neg_count, len(non_migraine_indices)), 
                                                replace=False)
            
            for idx in false_neg_indices:
                timestamp = data.iloc[idx]['timestamp']
                
                feedback = {
                    'timestamp': timestamp.isoformat(),
                    'migraine_confirmed': True,
                    'model_prediction': False,
                    'comment': "Had a migraine but it wasn't predicted"
                }
                
                feedback_events.append(feedback)
        
        # Add some false positive feedback (model predicted migraine but patient didn't have one)
        false_pos_count = random.randint(0, 3)
        
        if len(migraine_indices) > 0 and false_pos_count > 0:
            # Choose some indices where we predicted migraines
            false_pos_candidates = random.sample(list(migraine_indices), 
                                              min(false_pos_count, len(migraine_indices)))
            
            for idx in false_pos_candidates:
                if random.random() < 0.3:  # Only 30% chance as we don't want too many false positives
                    timestamp = data.iloc[idx]['timestamp']
                    
                    feedback = {
                        'timestamp': timestamp.isoformat(),
                        'migraine_confirmed': False,
                        'model_prediction': True,
                        'comment': "Model predicted a migraine but I didn't have one"
                    }
                    
                    feedback_events.append(feedback)
        
        return feedback_events
    
    def _save_patient_data(self, 
                          patient_id: str, 
                          demographics: Dict, 
                          data: pd.DataFrame,
                          targets: pd.Series,
                          feedback: Optional[List] = None):
        """
        Save the generated patient data to files.
        
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
        feedback : List, optional
            Feedback events
        """
        # Create patient directory
        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(exist_ok=True)
        
        # Save demographics
        with open(patient_dir / 'demographics.json', 'w') as f:
            json.dump(demographics, f, indent=2)
        
        # Add target to data
        data['target'] = targets
        
        # Save data to CSV
        data.to_csv(patient_dir / 'timeseries_data.csv', index=False)
        
        # Save feedback if available
        if feedback is not None:
            with open(patient_dir / 'feedback.json', 'w') as f:
                json.dump(feedback, f, indent=2)
        
        logger.info(f"Saved data for patient {patient_id}")
    
    def _visualize_patient_data(self, patient_id: str, data: pd.DataFrame, targets: pd.Series):
        """
        Create visualizations of patient data.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Feature data
        targets : pd.Series
            Target values
        """
        patient_dir = self.output_dir / patient_id
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot key features
        key_features = []
        for features in self.feature_categories.values():
            key_features.extend(features[:2])  # Take first 2 from each category
        
        # Filter to keep only those in the dataset
        key_features = [f for f in key_features if f in data.columns][:5]  # Take top 5
        
        # Convert timestamps to datetime if they're strings
        if isinstance(data['timestamp'].iloc[0], str):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Plot 1: Key features over time
        for feature in key_features:
            if feature in data.columns:
                axs[0].plot(data['timestamp'], data[feature], label=feature)
        
        axs[0].set_title(f"Key Features Over Time - Patient {patient_id}")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Feature Value")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: Migraine events
        axs[1].plot(data['timestamp'], targets, 'ro', markersize=8)
        axs[1].set_title(f"Migraine Events - Patient {patient_id}")
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Migraine Event")
        axs[1].set_ylim(-0.1, 1.1)
        axs[1].grid(True)
        
        # Plot 3: Feature correlations with migraine events
        correlations = {}
        for feature in data.columns:
            if feature not in ['patient_id', 'timestamp', 'target'] and feature in data.columns:
                corr = data[feature].corr(targets)
                if not np.isnan(corr):
                    correlations[feature] = corr
        
        # Sort correlations
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [x[0] for x in sorted_corrs[:10]]  # Top 10 correlated features
        corrs = [x[1] for x in sorted_corrs[:10]]
        
        # Create correlation plot
        bars = axs[2].barh(features, corrs)
        
        # Color bars based on correlation (positive or negative)
        for i, bar in enumerate(bars):
            if corrs[i] < 0:
                bar.set_color('r')
            else:
                bar.set_color('g')
        
        axs[2].set_title(f"Feature Correlations with Migraine Events - Patient {patient_id}")
        axs[2].set_xlabel("Correlation Coefficient")
        axs[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(patient_dir / 'visualization.png')
        plt.close(fig)
        
        logger.info(f"Created visualization for patient {patient_id}")
    
    def _create_patient_summary(self, patient_ids: List[str]):
        """
        Create a summary of all generated patients.
        
        Parameters:
        -----------
        patient_ids : List[str]
            List of patient identifiers
        """
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_patients': len(patient_ids),
            'patient_list': patient_ids,
            'patients': {}
        }
        
        # Collect demographics for each patient
        for patient_id in patient_ids:
            demo_file = self.output_dir / patient_id / 'demographics.json'
            if demo_file.exists():
                with open(demo_file, 'r') as f:
                    demographics = json.load(f)
                    
                # Extract key information for summary
                summary['patients'][patient_id] = {
                    'age': demographics.get('age'),
                    'gender': demographics.get('gender'),
                    'profile_type': demographics.get('profile_type'),
                    'avg_monthly_frequency': demographics.get('migraine_history', {}).get('avg_monthly_frequency')
                }
        
        # Save summary
        with open(self.output_dir / 'patient_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Created summary for {len(patient_ids)} patients")
    
    def load_patient_data(self, patient_id: str) -> Tuple[Dict, pd.DataFrame, List]:
        """
        Load previously generated patient data.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
            
        Returns:
        --------
        Tuple[Dict, pd.DataFrame, List]
            Demographics, time series data, and feedback
        """
        patient_dir = self.output_dir / patient_id
        
        if not patient_dir.exists():
            logger.error(f"No data found for patient {patient_id}")
            return None, None, None
        
        # Load demographics
        demographics = None
        demo_file = patient_dir / 'demographics.json'
        if demo_file.exists():
            with open(demo_file, 'r') as f:
                demographics = json.load(f)
        
        # Load time series data
        data = None
        data_file = patient_dir / 'timeseries_data.csv'
        if data_file.exists():
            data = pd.read_csv(data_file)
            
            # Convert timestamp to datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Load feedback
        feedback = None
        feedback_file = patient_dir / 'feedback.json'
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                feedback = json.load(f)
        
        return demographics, data, feedback


if __name__ == "__main__":
    generator = PatientDataGenerator(output_dir='data/synthetic_patients')
    
    # Generate a set of patients with different profiles
    patient_ids = generator.generate_patient_set(
        num_patients=10,
        time_periods=60,  # 60 days
        samples_per_period=3,  # 3 samples per day
        include_feedback=True,
        visualize=True
    )
    
    print(f"Generated data for {len(patient_ids)} patients")
    
    # Example of loading a patient's data
    if patient_ids:
        demographics, data, feedback = generator.load_patient_data(patient_ids[0])
        if data is not None:
            print(f"Loaded data for {patient_ids[0]} with {len(data)} records and {sum(data['target'])} migraine events")

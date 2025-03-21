"""
MoE Personalization Testing Module

This module tests the patient profile personalization layer 
by creating synthetic patient data with diverse characteristics
and testing the adaptation of expert weights based on patient-specific factors.
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import personalization components
from core.personalization_layer import PersonalizationLayer
from core.moe_validation_enhancements import MoEValidationEnhancer

# Import explainability components to leverage existing framework
from explainability.explainer_factory import ExplainerFactory
from explainability.base_explainer import BaseExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticPatientGenerator:
    """
    Generates synthetic patient data with controllable characteristics.
    """
    
    def __init__(self, n_patients=10, seed=42):
        """
        Initialize the synthetic patient generator.
        
        Parameters:
        -----------
        n_patients : int
            Number of synthetic patients to generate
        seed : int
            Random seed for reproducibility
        """
        self.n_patients = n_patients
        self.seed = seed
        np.random.seed(seed)
        self.patient_ids = [f"patient_{i}" for i in range(n_patients)]
        
    def generate_demographic_data(self):
        """
        Generate synthetic demographic data for patients.
        
        Returns:
        --------
        Dict[str, Dict]
            Dictionary of patient demographics keyed by patient_id
        """
        demographics = {}
        
        # Age categories
        age_categories = ["18-30", "31-45", "46-60", "61+"]
        age_probs = [0.2, 0.3, 0.3, 0.2]
        
        # Gender categories
        gender_categories = ["Male", "Female", "Other"]
        gender_probs = [0.48, 0.48, 0.04]
        
        # Generate demographics for each patient
        for patient_id in self.patient_ids:
            age = np.random.choice(age_categories, p=age_probs)
            gender = np.random.choice(gender_categories, p=gender_probs)
            
            # Create demographic profile
            demographics[patient_id] = {
                "age_group": age,
                "gender": gender,
                "migraine_history_years": np.random.randint(1, 20),
                "migraine_frequency_per_month": np.random.randint(1, 15)
            }
            
        return demographics
        
    def generate_physiological_data(self, patient_id, n_days=30):
        """
        Generate synthetic physiological data for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        n_days : int
            Number of days of data to generate
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with physiological data
        """
        # Generate timestamps
        base_date = pd.Timestamp('2025-01-01')
        timestamps = [base_date + pd.Timedelta(days=i) for i in range(n_days)]
        
        # Base features
        data = {
            'timestamp': timestamps,
            'patient_id': [patient_id] * n_days,
            'physio_heart_rate': np.random.normal(75, 10, n_days),
            'physio_blood_pressure_systolic': np.random.normal(120, 15, n_days),
            'physio_blood_pressure_diastolic': np.random.normal(80, 10, n_days),
            'physio_sleep_hours': np.random.normal(7, 1.5, n_days),
            'physio_stress_level': np.random.uniform(1, 10, n_days).round(1)
        }
        
        # Introduce patient-specific patterns
        patient_num = int(patient_id.split('_')[1])
        
        if patient_num % 3 == 0:
            # Heart rate sensitive patient
            data['physio_heart_rate'] = np.random.normal(85, 15, n_days)  # Higher mean and variance
            
        if patient_num % 3 == 1:
            # Blood pressure sensitive patient
            data['physio_blood_pressure_systolic'] = np.random.normal(135, 20, n_days)  # Higher BP
            
        if patient_num % 3 == 2:
            # Sleep sensitive patient
            data['physio_sleep_hours'] = np.random.normal(5.5, 2, n_days)  # Less sleep, more variance
        
        return pd.DataFrame(data)
    
    def generate_environmental_data(self, patient_id, n_days=30):
        """
        Generate synthetic environmental data for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        n_days : int
            Number of days of data to generate
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with environmental data
        """
        # Generate timestamps
        base_date = pd.Timestamp('2025-01-01')
        timestamps = [base_date + pd.Timedelta(days=i) for i in range(n_days)]
        
        # Base features
        data = {
            'timestamp': timestamps,
            'patient_id': [patient_id] * n_days,
            'temperature': np.random.normal(70, 10, n_days),
            'humidity': np.random.normal(60, 15, n_days),
            'barometric_pressure': np.random.normal(1013, 10, n_days),
            'light_exposure': np.random.uniform(100, 1000, n_days)
        }
        
        # Introduce patient-specific patterns
        patient_num = int(patient_id.split('_')[1])
        
        if patient_num % 4 == 0:
            # Temperature sensitive patient
            data['temperature'] = np.random.normal(85, 15, n_days)  # Higher temperature
            
        if patient_num % 4 == 1:
            # Pressure sensitive patient
            data['barometric_pressure'] = np.random.normal(990, 20, n_days)  # Lower pressure
            
        if patient_num % 4 == 2:
            # Humidity sensitive patient
            data['humidity'] = np.random.normal(80, 15, n_days)  # Higher humidity
        
        return pd.DataFrame(data)
    
    def generate_behavioral_data(self, patient_id, n_days=30):
        """
        Generate synthetic behavioral data for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        n_days : int
            Number of days of data to generate
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with behavioral data
        """
        # Generate timestamps
        base_date = pd.Timestamp('2025-01-01')
        timestamps = [base_date + pd.Timedelta(days=i) for i in range(n_days)]
        
        # Base features
        data = {
            'timestamp': timestamps,
            'patient_id': [patient_id] * n_days,
            'behavior_exercise_minutes': np.random.normal(30, 20, n_days).round(),
            'behavior_screen_time_hours': np.random.normal(4, 2, n_days).round(1),
            'behavior_caffeine_mg': np.random.normal(200, 100, n_days).round(),
            'behavior_alcohol_drinks': np.random.poisson(0.5, n_days),
            'behavior_meal_regularity': np.random.uniform(1, 10, n_days).round(1)
        }
        
        # Introduce patient-specific patterns
        patient_num = int(patient_id.split('_')[1])
        
        if patient_num % 5 == 0:
            # Caffeine sensitive patient
            data['behavior_caffeine_mg'] = np.random.normal(350, 150, n_days).round()
            
        if patient_num % 5 == 1:
            # Screen time sensitive patient
            data['behavior_screen_time_hours'] = np.random.normal(8, 3, n_days).round(1)
            
        if patient_num % 5 == 2:
            # Alcohol sensitive patient
            data['behavior_alcohol_drinks'] = np.random.poisson(2, n_days)
            
        if patient_num % 5 == 3:
            # Exercise sensitive patient
            data['behavior_exercise_minutes'] = np.random.normal(10, 15, n_days).round()
        
        return pd.DataFrame(data)
    
    def generate_migraine_data(self, patient_id, merged_data, sensitivity_factors=None):
        """
        Generate synthetic migraine data based on other factors.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        merged_data : pd.DataFrame
            Merged physiological, environmental, and behavioral data
        sensitivity_factors : Dict[str, float], optional
            Feature-specific sensitivity factors
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added migraine probability and occurrence
        """
        if sensitivity_factors is None:
            # Default sensitivity - depends on patient number to simulate different profiles
            patient_num = int(patient_id.split('_')[1])
            
            if patient_num % 3 == 0:
                # Physiologically triggered migraines
                sensitivity_factors = {
                    'physio_stress_level': 0.4,
                    'physio_sleep_hours': -0.3,
                    'temperature': 0.2,
                    'behavior_caffeine_mg': 0.1
                }
            elif patient_num % 3 == 1:
                # Environmentally triggered migraines
                sensitivity_factors = {
                    'temperature': 0.3,
                    'barometric_pressure': -0.4,
                    'humidity': 0.2,
                    'physio_stress_level': 0.1
                }
            else:
                # Behaviorally triggered migraines
                sensitivity_factors = {
                    'behavior_alcohol_drinks': 0.4,
                    'behavior_caffeine_mg': 0.3,
                    'behavior_screen_time_hours': 0.2,
                    'physio_sleep_hours': -0.1
                }
        
        # Normalize all factors to 0-1 range for probability calculation
        data = merged_data.copy()
        
        # Calculate base probability
        base_prob = 0.1  # 10% base probability
        migraine_prob = np.ones(len(data)) * base_prob
        
        # Adjust probability based on sensitivity factors
        for feature, sensitivity in sensitivity_factors.items():
            if feature in data.columns:
                # Normalize the feature
                feature_values = data[feature].values
                if feature_values.max() > feature_values.min():
                    normalized = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                else:
                    normalized = np.zeros_like(feature_values)
                
                # For negative sensitivity, invert the effect
                if sensitivity < 0:
                    normalized = 1 - normalized
                    sensitivity = abs(sensitivity)
                
                # Add to probability based on sensitivity
                migraine_prob += normalized * abs(sensitivity)
        
        # Ensure probability is between 0 and 1
        migraine_prob = np.clip(migraine_prob, 0, 1)
        
        # Generate migraine occurrence
        migraine_occurrence = np.random.random(len(data)) < migraine_prob
        
        # Add to dataframe
        data['migraine_probability'] = migraine_prob
        data['migraine_occurrence'] = migraine_occurrence.astype(int)
        data['target'] = data['migraine_probability']  # Use probability as target for regression
        
        return data
    
    def generate_complete_patient_dataset(self, patient_id, n_days=30):
        """
        Generate a complete dataset for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        n_days : int
            Number of days of data to generate
            
        Returns:
        --------
        pd.DataFrame
            Complete patient dataset
        """
        # Generate individual data types
        physio_data = self.generate_physiological_data(patient_id, n_days)
        env_data = self.generate_environmental_data(patient_id, n_days)
        behav_data = self.generate_behavioral_data(patient_id, n_days)
        
        # Merge datasets
        merged_data = pd.merge(physio_data, env_data, on=['timestamp', 'patient_id'])
        merged_data = pd.merge(merged_data, behav_data, on=['timestamp', 'patient_id'])
        
        # Generate migraine data
        complete_data = self.generate_migraine_data(patient_id, merged_data)
        
        return complete_data
    
    def generate_all_patient_data(self, n_days=30):
        """
        Generate data for all patients.
        
        Parameters:
        -----------
        n_days : int
            Number of days of data to generate per patient
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of patient datasets keyed by patient_id
        Dict[str, Dict]
            Dictionary of patient demographics keyed by patient_id
        """
        patient_data = {}
        demographics = self.generate_demographic_data()
        
        for patient_id in self.patient_ids:
            patient_data[patient_id] = self.generate_complete_patient_dataset(patient_id, n_days)
            
        return patient_data, demographics


class MockExpert:
    """
    Mock expert model for MoE testing.
    """
    
    def __init__(self, expert_id, specialty=None):
        """
        Initialize the mock expert.
        
        Parameters:
        -----------
        expert_id : int
            Expert identifier
        specialty : str, optional
            Expert specialty (e.g., 'physiological', 'environmental', 'behavioral')
        """
        self.expert_id = expert_id
        self.specialty = specialty
        self.trained = False
        self.coefficients = {}
    
    def train(self, X, y):
        """
        Train the mock expert.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : array-like
            Target data
            
        Returns:
        --------
        self : MockExpert
            Trained expert
        """
        # Simulate training by assigning random coefficients
        for col in X.columns:
            self.coefficients[col] = np.random.uniform(-1, 1)
        
        self.trained = True
        logger.info(f"Expert {self.expert_id} trained with specialty {self.specialty}")
        return self
    
    def predict(self, X):
        """
        Generate predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
            
        Returns:
        --------
        array-like
            Predictions
        """
        if not self.trained:
            raise RuntimeError(f"Expert {self.expert_id} not trained")
        
        # Generate predictions
        pred = np.zeros(len(X))
        for col in X.columns:
            if col in self.coefficients:
                pred += X[col].values * self.coefficients[col]
        
        return pred
    
    def get_feature_importance(self):
        """
        Get feature importance.
        
        Returns:
        --------
        Dict[str, float]
            Feature importance
        """
        return {col: abs(coef) for col, coef in self.coefficients.items()}


class MockGatingNetwork:
    """
    Mock gating network for MoE testing.
    """
    
    def __init__(self):
        """Initialize the mock gating network."""
        self.trained = False
        self.experts = {}
        self.base_weights = {}
    
    def train(self, X, y, experts):
        """
        Train the gating network.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        y : array-like
            Target data
        experts : Dict[int, MockExpert]
            Dictionary of experts
            
        Returns:
        --------
        self : MockGatingNetwork
            Trained gating network
        """
        self.experts = experts
        
        # Assign base weights based on feature categories
        n_experts = len(experts)
        self.base_weights = {expert_id: 1.0/n_experts for expert_id in experts}
        
        self.trained = True
        logger.info(f"Gating network trained with {len(experts)} experts")
        return self
    
    def predict_weights(self, X, patient_id=None):
        """
        Predict expert weights for input features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature data
        patient_id : str, optional
            Patient identifier for personalization
            
        Returns:
        --------
        List[Dict[int, float]]
            Predicted weights for each sample
        """
        if not self.trained:
            raise RuntimeError("Gating network not trained")
        
        # Get feature categories present in data
        has_physio = any('physio' in col for col in X.columns)
        has_env = any(col in ['temperature', 'humidity', 'barometric_pressure'] for col in X.columns)
        has_behav = any('behavior' in col for col in X.columns)
        
        # Adjust weights based on feature categories
        weights = dict(self.base_weights)
        
        # Slightly increase weights for experts matching feature categories
        for expert_id, expert in self.experts.items():
            if expert.specialty == 'physiological' and has_physio:
                weights[expert_id] *= 1.2
            elif expert.specialty == 'environmental' and has_env:
                weights[expert_id] *= 1.2
            elif expert.specialty == 'behavioral' and has_behav:
                weights[expert_id] *= 1.2
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {k: v/weight_sum for k, v in weights.items()}
        
        # Return same weights for all samples (for simplicity)
        return [weights for _ in range(len(X))]


class MockExplainabilityEngine:
    """
    Mock explainability engine for testing, leveraging the existing framework.
    """
    
    def __init__(self):
        """Initialize the mock explainability engine."""
        # Try to use the actual explainer factory if available
        try:
            self.explainer_factory = ExplainerFactory()
            self.using_real_explainers = True
            logger.info("Using real explainer factory from existing framework")
        except Exception as e:
            self.using_real_explainers = False
            logger.warning(f"Could not initialize real explainer factory: {str(e)}")
    
    def explain_model(self, model, X):
        """
        Generate feature importance for a model.
        
        Parameters:
        -----------
        model : object
            Model to explain
        X : pd.DataFrame
            Feature data
            
        Returns:
        --------
        Dict[str, float]
            Feature importance
        """
        # Try to use a real explainer if available
        if self.using_real_explainers:
            try:
                explainer = self.explainer_factory.create_explainer('feature_importance')
                return explainer.explain(model, X)
            except Exception as e:
                logger.warning(f"Real explainer failed: {str(e)}. Using mock explainer.")
        
        # Fall back to mock implementation
        if hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        
        # Create mock feature importance
        feature_names = X.columns
        importance = np.random.uniform(0, 1, len(feature_names))
        importance = importance / np.sum(importance)  # Normalize
        
        return dict(zip(feature_names, importance))


def run_personalization_test():
    """
    Run the personalization test.
    
    Returns:
    --------
    Dict[str, Any]
        Test results
    """
    logger.info("Starting personalization test")
    
    # Generate synthetic patient data
    generator = SyntheticPatientGenerator(n_patients=5)
    patient_data, demographics = generator.generate_all_patient_data(n_days=30)
    
    # Create experts with different specialties
    experts = {
        0: MockExpert(0, specialty='physiological'),
        1: MockExpert(1, specialty='environmental'),
        2: MockExpert(2, specialty='behavioral')
    }
    
    # Train experts on all patient data combined
    combined_data = pd.concat(list(patient_data.values()))
    
    # Train each expert on its specialty features
    for expert_id, expert in experts.items():
        if expert.specialty == 'physiological':
            cols = [col for col in combined_data.columns if 'physio' in col or col in ['patient_id', 'timestamp', 'target']]
        elif expert.specialty == 'environmental':
            cols = [col for col in combined_data.columns if col in ['temperature', 'humidity', 'barometric_pressure', 'light_exposure', 'patient_id', 'timestamp', 'target']]
        elif expert.specialty == 'behavioral':
            cols = [col for col in combined_data.columns if 'behavior' in col or col in ['patient_id', 'timestamp', 'target']]
        else:
            cols = combined_data.columns
        
        X = combined_data[cols].drop(columns=['patient_id', 'timestamp', 'target'], errors='ignore')
        y = combined_data['target']
        
        expert.train(X, y)
    
    # Create gating network
    gating_network = MockGatingNetwork()
    X_train = combined_data.drop(columns=['patient_id', 'timestamp', 'target', 'migraine_probability', 'migraine_occurrence'], errors='ignore')
    y_train = combined_data['target']
    gating_network.train(X_train, y_train, experts)
    
    # Create explainability engine
    explainer = MockExplainabilityEngine()
    
    # Create personalization layer
    results_dir = Path('results/personalization_test')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    personalizer = PersonalizationLayer(
        results_dir=str(results_dir),
        adaptation_rate=0.3,
        profile_update_threshold=0.1
    )
    
    # Test personalization for all patients
    personalization_results = {}
    
    for patient_id, data in patient_data.items():
        logger.info(f"Testing personalization for patient {patient_id}")
        
        # Create patient profile
        profile = personalizer.create_patient_profile(
            patient_id=patient_id,
            demographic_data=demographics[patient_id],
            initial_data=data,
            explainer=explainer
        )
        
        # Get base weights from gating network
        X_test = data.drop(columns=['patient_id', 'timestamp', 'target', 'migraine_probability', 'migraine_occurrence'], errors='ignore')
        base_weights = gating_network.predict_weights(X_test)[0]  # Get first sample's weights
        
        # Personalize weights
        personalized_weights = personalizer.personalize_expert_weights(
            patient_id=patient_id,
            base_weights=base_weights,
            features=X_test
        )
        
        # Store results
        patient_result = {
            'base_weights': base_weights,
            'personalized_weights': personalized_weights,
            'profile': profile
        }
        
        # Calculate weight differences
        weight_diffs = {expert_id: personalized_weights[expert_id] - base_weights[expert_id] 
                      for expert_id in base_weights}
        patient_result['weight_differences'] = weight_diffs
        
        personalization_results[patient_id] = patient_result
    
    # Save overall results
    overall_results = {
        'timestamp': datetime.now().isoformat(),
        'n_patients': len(patient_data),
        'patient_results': personalization_results
    }
    
    with open(results_dir / 'personalization_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        import json
        def convert_numpy(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj
        
        json.dump(overall_results, f, indent=2, default=convert_numpy)
    
    # Create visualization of personalization effect
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    patient_ids = list(personalization_results.keys())
    expert_ids = [0, 1, 2]  # Assuming these are the expert IDs
    
    # Create grouped bar chart showing before/after weights
    bar_width = 0.35
    index = np.arange(len(patient_ids))
    
    for i, expert_id in enumerate(expert_ids):
        base_weights = [personalization_results[pid]['base_weights'][expert_id] for pid in patient_ids]
        pers_weights = [personalization_results[pid]['personalized_weights'][expert_id] for pid in patient_ids]
        
        plt.bar(index + (i-1)*bar_width, base_weights, bar_width, 
                label=f'Expert {expert_id} Base', alpha=0.6)
        plt.bar(index + i*bar_width, pers_weights, bar_width, 
                label=f'Expert {expert_id} Personalized')
    
    plt.xlabel('Patient ID')
    plt.ylabel('Expert Weight')
    plt.title('Personalization Effect on Expert Weights')
    plt.xticks(index, patient_ids)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(results_dir / 'personalization_effect.png')
    logger.info(f"Saved personalization effect visualization to {results_dir / 'personalization_effect.png'}")
    
    return overall_results


if __name__ == "__main__":
    results = run_personalization_test()
    logger.info("Personalization test completed")
    print(f"Results saved to {Path('results/personalization_test')}")

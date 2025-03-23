#!/usr/bin/env python3
"""
Functional test script for patient adaptation features.

This script performs end-to-end testing of:
1. Patient profile creation and management
2. Adaptive thresholds
3. Contextual adjustments
4. Online profile updates
5. Feedback-based adaptation

Visualizations are saved to the results directory.
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.personalization_layer import PersonalizationLayer
from core.patient_profile_adapter import PatientProfileAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(num_patients=4, samples_per_patient=50):
    """Create synthetic test data for multiple patients."""
    data = []
    
    for patient_id in range(1, num_patients + 1):
        # Create patient-specific baseline values with some randomness
        base_heart_rate = np.random.randint(65, 85)
        base_stress = np.random.randint(30, 60)
        base_sleep = np.random.randint(6, 9)
        
        for i in range(samples_per_patient):
            # Add some variability for each patient's readings
            heart_rate = base_heart_rate + np.random.normal(0, 10)
            stress = base_stress + np.random.normal(0, 15)
            sleep = max(1, min(10, base_sleep + np.random.normal(0, 2)))
            
            # Weather data with seasonal patterns
            temp = 15 + 10 * np.sin(i / 30 * np.pi) + np.random.normal(0, 5)
            pressure = 1000 + np.random.normal(0, 10)
            humidity = 50 + np.random.normal(0, 20)
            
            # Other factors
            menstruation = 1 if i % 28 < 5 and np.random.random() < 0.8 else 0
            medication = 1 if stress > 70 or heart_rate > 90 else 0
            
            # Create timestamp
            timestamp = datetime.now() - timedelta(days=samples_per_patient-i)
            
            # Generate record
            record = {
                'patient_id': f'patient_{patient_id}',
                'timestamp': timestamp.isoformat(),
                'heart_rate': float(heart_rate),
                'stress_level': float(stress),
                'sleep_quality': float(sleep),
                'temperature': float(temp),
                'weather_pressure': float(pressure),
                'humidity': float(humidity),
                'menstruation': int(menstruation),
                'medication_taken': int(medication)
            }
            data.append(record)
    
    return pd.DataFrame(data)


def create_mock_explainer():
    """Create a mock explainer for testing."""
    class MockExplainer:
        def __init__(self):
            self.model = None
        
        def set_model(self, model):
            self.model = model
            
        def explain(self, X):
            if self.model is None:
                raise ValueError("Model not set. Call set_model() first.")
            # Return dummy feature importance
            return {
                'feature_importance': {
                    'heart_rate': 0.3,
                    'stress_level': 0.25,
                    'sleep_quality': 0.2,
                    'weather_pressure': 0.15,
                    'medication_taken': 0.1
                }
            }
    
    return MockExplainer()


def run_adaptation_test(output_dir='results/adaptation_test'):
    """Run a functional test of patient adaptation features."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    logger.info("Generating test data")
    data = create_test_data()
    
    # Split data by patient
    patient_data = {}
    for patient_id, group in data.groupby('patient_id'):
        patient_data[patient_id] = group.sort_values('timestamp').reset_index(drop=True)
    
    # Create personalization layer and patient profile adapter
    personalization_dir = os.path.join(output_dir, 'personalization')
    Path(personalization_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to use real explainer if available
        from explainability.explainer_factory import ExplainerFactory
        explainer = ExplainerFactory.create_explainer('shap')
    except Exception as e:
        logger.warning(f"Real explainer failed: {str(e)}. Using mock explainer.")
        explainer = create_mock_explainer()
    
    # Initialize components
    personalization_layer = PersonalizationLayer(
        results_dir=personalization_dir,
        adaptation_rate=0.2,
        threshold_adaptation_rate=0.2  # Make it more responsive for our test
    )
    
    patient_profile_adapter = PatientProfileAdapter(
        personalization_layer=personalization_layer
    )
    
    # Track adaptation effects
    original_predictions = {}
    adapted_predictions = {}
    threshold_changes = {}
    
    # Process each patient
    for patient_id, data in patient_data.items():
        logger.info(f"Testing personalization for patient {patient_id}")
        
        # Split data into training and testing
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Create or load patient profile with the explainer
        profile = personalization_layer.create_patient_profile(
            patient_id, 
            initial_data=train_data.head(10),  # Use a small sample of data for initialization
            explainer=explainer
        )
        
        # Initial threshold
        initial_threshold = personalization_layer.get_adaptive_threshold(patient_id)
        threshold_changes[patient_id] = [initial_threshold]
        
        # Train a simple model (we're just testing the adaptation framework)
        # In a real scenario, this would be a proper ML model
        class SimpleModel:
            def __init__(self):
                self.coef_ = [0.3, 0.25, 0.2, 0.15, 0.1]
                self.feature_names_in_ = ['heart_rate', 'stress_level', 'sleep_quality', 
                                         'weather_pressure', 'medication_taken']
                # Add feature_importances_ attribute for explainers that look for it
                self.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
            
            def __call__(self, X):
                """Make model callable for SHAP compatibility."""
                # Convert to numpy array if it's a DataFrame
                if isinstance(X, pd.DataFrame):
                    # Try to reorder columns to match feature_names_in_
                    try:
                        X = X[self.feature_names_in_]
                    except:
                        pass  # If columns don't match, just use as is
                    X = X.values
                
                # Generate probabilities directly
                return self.predict_proba(X)[:, 1]  # Return positive class probability
            
            def predict_proba(self, X):
                # Simple probability calculation based on feature values
                # This is just for testing purposes
                probs = []
                
                # Handle both DataFrame and numpy array inputs
                if isinstance(X, pd.DataFrame):
                    for _, row in X.iterrows():
                        # Safely convert feature values to float if they're strings
                        try:
                            heart_rate = float(row['heart_rate'])
                            stress_level = float(row['stress_level'])
                            sleep_quality = float(row['sleep_quality'])
                            
                            # Scale heart_rate to be higher when > 80
                            hr_factor = 0.4 if heart_rate > 80 else 0.1
                            # Scale stress to be higher when > 60
                            stress_factor = 0.3 if stress_level > 60 else 0.1
                            # Scale sleep to be higher when < 5
                            sleep_factor = 0.3 if sleep_quality < 5 else 0.1
                        except (ValueError, TypeError):
                            # Default values if conversion fails
                            hr_factor = 0.2
                            stress_factor = 0.2
                            sleep_factor = 0.2
                        
                        p = 0.2 + hr_factor + stress_factor + sleep_factor
                        probs.append([1-p, p])  # [no_migraine, migraine]
                else:
                    # Handle numpy array (assume columns are in expected order)
                    for i in range(len(X)):
                        row = X[i]
                        # Default values if we can't map features
                        hr_factor = 0.2
                        stress_factor = 0.2
                        sleep_factor = 0.2
                        
                        if len(row) >= 5:  # Make sure we have enough features
                            # Safely convert values to float
                            try:
                                heart_rate = float(row[0])
                                stress_level = float(row[1])
                                sleep_quality = float(row[2])
                                
                                hr_factor = 0.4 if heart_rate > 80 else 0.1
                                stress_factor = 0.3 if stress_level > 60 else 0.1
                                sleep_factor = 0.3 if sleep_quality < 5 else 0.1
                            except (ValueError, TypeError):
                                # Default values if conversion fails
                                hr_factor = 0.2
                                stress_factor = 0.2
                                sleep_factor = 0.2
                        
                        p = 0.2 + hr_factor + stress_factor + sleep_factor
                        probs.append([1-p, p])
                        
                return np.array(probs)
        
        model = SimpleModel()
        if hasattr(explainer, 'set_model'):
            explainer.set_model(model)
        
        # Process each day's data to simulate online updates
        original_pred_list = []
        adapted_pred_list = []
        
        for i in range(len(test_data)):
            day_data = test_data.iloc[[i]]
            
            # Get predictions before adaptation
            probs = model.predict_proba(day_data[model.feature_names_in_])[:, 1]
            original_pred_list.append(float(probs[0]))
            
            # Update the profile online
            features_for_adaptation = day_data.copy()
            prediction_results = {
                'probabilities': probs,
                'true_positives': 1 if i % 10 == 0 else 0,  # Simulate some TP
                'false_positives': 1 if i % 15 == 0 else 0,  # Simulate some FP
                'true_negatives': 1 if i % 7 == 0 else 0,    # Simulate some TN
                'false_negatives': 1 if i % 20 == 0 else 0   # Simulate some FN
            }
            
            # Apply online update
            result = patient_profile_adapter.update_profile_online(
                patient_id, 
                features_for_adaptation, 
                prediction_results
            )
            
            # Apply contextual adjustments
            adj_probs = []
            for j, prob in enumerate(probs):
                adjustment = personalization_layer.apply_contextual_adjustments(
                    patient_id, 
                    prob, 
                    day_data.iloc[[j]]
                )
                
                if isinstance(adjustment, dict) and 'adjusted_proba' in adjustment:
                    adj_probs.append(adjustment['adjusted_proba'])
                else:
                    adj_probs.append(prob)
            
            adapted_pred_list.append(float(adj_probs[0]))
            
            # Update threshold periodically
            if i % 5 == 0:
                new_threshold = personalization_layer.update_adaptive_threshold(
                    patient_id, prediction_results
                )
                threshold_changes[patient_id].append(new_threshold)
            
            # Simulate feedback occasionally
            if i % 10 == 0:
                feedback = {
                    'was_accurate': i % 15 != 0,  # Sometimes it's inaccurate
                    'actual_severity': np.random.randint(1, 10),
                    'reported_symptoms': ['pain', 'sensitivity'] if i % 2 == 0 else ['nausea'],
                    'feedback_text': 'The prediction was helpful' if i % 15 != 0 else 'The prediction was not accurate'
                }
                
                personalization_layer.update_profile_from_feedback(patient_id, feedback)
        
        original_predictions[patient_id] = original_pred_list
        adapted_predictions[patient_id] = adapted_pred_list
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Plot 1: Personalization effect (original vs. adapted predictions)
    plt.figure(figsize=(12, 8))
    for i, patient_id in enumerate(patient_data.keys()):
        plt.subplot(len(patient_data), 1, i+1)
        plt.plot(original_predictions[patient_id], label='Original Predictions', alpha=0.7)
        plt.plot(adapted_predictions[patient_id], label='Adapted Predictions', alpha=0.7)
        plt.title(f'Patient {patient_id} - Adaptation Effect')
        plt.xlabel('Observation')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adaptation_effect.png'))
    
    # Plot 2: Threshold changes over time
    plt.figure(figsize=(12, 8))
    for i, patient_id in enumerate(patient_data.keys()):
        plt.subplot(len(patient_data), 1, i+1)
        plt.plot(threshold_changes[patient_id], marker='o')
        plt.title(f'Patient {patient_id} - Threshold Adaptation')
        plt.xlabel('Update')
        plt.ylabel('Threshold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_adaptation.png'))
    
    # Save results as JSON for future reference
    with open(os.path.join(output_dir, 'adaptation_results.json'), 'w') as f:
        json.dump({
            'original_predictions': {k: [float(x) for x in v] for k, v in original_predictions.items()},
            'adapted_predictions': {k: [float(x) for x in v] for k, v in adapted_predictions.items()},
            'threshold_changes': {k: [float(x) for x in v] for k, v in threshold_changes.items()}
        }, f, indent=2)
    
    logger.info(f"Adaptation test completed\nResults saved to {output_dir}")
    return True


if __name__ == "__main__":
    run_adaptation_test()

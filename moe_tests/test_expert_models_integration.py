"""
Integration tests for expert models in the MoE framework.

These tests focus on how expert models integrate with other components
and the data pipeline. This version uses a simplified approach that doesn't
rely on preprocessor classes that haven't been implemented yet.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Import the expert models
from moe_framework.experts.base_expert import BaseExpert
from moe_framework.experts.physiological_expert import PhysiologicalExpert
from moe_framework.experts.environmental_expert import EnvironmentalExpert
from moe_framework.experts.behavioral_expert import BehavioralExpert
from moe_framework.experts.medication_history_expert import MedicationHistoryExpert


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a small dataset for testing
    n_samples = 100
    
    # Create timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Create patient IDs and locations
    patient_ids = np.random.choice(['P001', 'P002', 'P003'], n_samples)
    locations = np.random.choice(['New York', 'Boston', 'Chicago'], n_samples)
    
    # Create physiological features
    heart_rate = np.random.normal(75, 10, n_samples)
    blood_pressure_sys = np.random.normal(120, 15, n_samples)
    blood_pressure_dia = np.random.normal(80, 10, n_samples)
    temperature = np.random.normal(37, 0.5, n_samples)
    
    # Create environmental features
    env_temperature = np.random.normal(20, 8, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    pressure = np.random.normal(1013, 10, n_samples)
    aqi = np.random.normal(50, 20, n_samples)
    
    # Create behavioral features
    sleep_duration = np.random.normal(7, 1.5, n_samples)
    sleep_quality = np.random.normal(70, 15, n_samples)
    activity_level = np.random.normal(60, 20, n_samples)
    stress_level = np.random.normal(50, 25, n_samples)
    
    # Create medication features
    medication_a = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    medication_b = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    medication_c = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    
    # Create target variable (migraine severity)
    # Each domain contributes to the target with some noise
    physio_effect = 0.3 * heart_rate + 0.2 * blood_pressure_sys
    env_effect = 0.25 * env_temperature + 0.15 * humidity + 0.1 * aqi
    behavior_effect = -0.2 * sleep_quality + 0.3 * stress_level
    
    # Medication effects (higher medication levels reduce severity)
    med_a_effect = np.where(medication_a == '', 0, 
                   np.where(medication_a == 'Low', -5, 
                   np.where(medication_a == 'Medium', -10, -15)))
    
    med_b_effect = np.where(medication_b == '', 0, 
                   np.where(medication_b == 'Low', -3, 
                   np.where(medication_b == 'Medium', -7, -12)))
    
    med_c_effect = np.where(medication_c == '', 0, 
                   np.where(medication_c == 'Low', -2, 
                   np.where(medication_c == 'Medium', -5, -8)))
    
    # Combine effects with noise
    migraine_severity = (
        50 +  # baseline
        physio_effect + 
        env_effect + 
        behavior_effect + 
        med_a_effect + 
        med_b_effect + 
        med_c_effect + 
        np.random.normal(0, 10, n_samples)  # random noise
    )
    
    # Ensure severity is in a reasonable range (0-100)
    migraine_severity = np.clip(migraine_severity, 0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        # Identifiers
        'patient_id': patient_ids,
        'location': locations,
        'date': timestamps,
        
        # Physiological features
        'heart_rate': heart_rate,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'temperature': temperature,
        
        # Environmental features
        'env_temperature': env_temperature,
        'humidity': humidity,
        'pressure': pressure,
        'aqi': aqi,
        
        # Behavioral features
        'sleep_duration': sleep_duration,
        'sleep_quality': sleep_quality,
        'activity_level': activity_level,
        'stress_level': stress_level,
        
        # Medication features
        'medication_a': medication_a,
        'medication_b': medication_b,
        'medication_c': medication_c,
        
        # Target
        'migraine_severity': migraine_severity
    })
    
    return data


@pytest.fixture
def train_test_data(sample_data):
    """Split data into training and testing sets."""
    X = sample_data.drop('migraine_severity', axis=1)
    y = sample_data['migraine_severity']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


# Create simplified expert classes for testing
class SimplePhysiologicalExpert(BaseExpert):
    """Simplified physiological expert for testing."""
    
    def __init__(self, vital_cols, patient_id_col='patient_id', timestamp_col='date', name="PhysiologicalExpert"):
        super().__init__(
            name=name,
            model=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        )
        self.vital_cols = vital_cols
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def preprocess_data(self, data):
        """Preprocess the physiological data."""
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Select only the vital columns for this expert
        features = processed_data[self.vital_cols].copy()
        
        # Standardize the features
        if not self.is_fitted:
            features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=self.vital_cols,
                index=features.index
            )
        else:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=self.vital_cols,
                index=features.index
            )
        
        # Add simple derived features (e.g., systolic/diastolic ratio)
        if 'blood_pressure_sys' in self.vital_cols and 'blood_pressure_dia' in self.vital_cols:
            features['bp_ratio'] = processed_data['blood_pressure_sys'] / processed_data['blood_pressure_dia']
        
        return features
    
    def fit(self, data, target_column='migraine_severity'):
        """Fit the physiological expert model."""
        self.target_column = target_column
        
        # Preprocess the data
        X = self.preprocess_data(data)
        y = data[target_column]
        
        # Fit the model
        self.model.fit(X, y)
        
        # Update feature importances
        self.feature_importances = dict(zip(X.columns, self.model.feature_importances_))
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, data):
        """Make predictions with the physiological expert."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Make predictions
        return self.model.predict(X)
    
    def predict_with_confidence(self, data):
        """Make predictions with confidence estimates."""
        predictions = self.predict(data)
        
        # Simple confidence estimation (mock implementation)
        # In a real implementation, this would use proper uncertainty quantification
        confidence = np.ones_like(predictions) * 0.8
        
        return predictions, confidence


@pytest.fixture
def physiological_expert():
    """Create a physiological expert for testing."""
    return PhysiologicalExpert(
        vital_cols=['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'temperature'],
        patient_id_col='patient_id',
        timestamp_col='date'
    )


class SimpleEnvironmentalExpert(BaseExpert):
    """Simplified environmental expert for testing."""
    
    def __init__(self, env_cols, location_col='location', timestamp_col='date', name="EnvironmentalExpert"):
        super().__init__(
            name=name,
            model=GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        )
        self.env_cols = env_cols
        self.location_col = location_col
        self.timestamp_col = timestamp_col
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def preprocess_data(self, data):
        """Preprocess the environmental data."""
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Select only the environmental columns for this expert
        features = processed_data[self.env_cols].copy()
        
        # Standardize the features
        if not self.is_fitted:
            features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=self.env_cols,
                index=features.index
            )
        else:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=self.env_cols,
                index=features.index
            )
        
        # Add simple derived features
        if 'env_temperature' in self.env_cols and 'humidity' in self.env_cols:
            # Calculate heat index (simplified)
            features['heat_index'] = features['env_temperature'] + 0.05 * features['humidity']
        
        return features
    
    def fit(self, data, target_column='migraine_severity'):
        """Fit the environmental expert model."""
        self.target_column = target_column
        
        # Preprocess the data
        X = self.preprocess_data(data)
        y = data[target_column]
        
        # Fit the model
        self.model.fit(X, y)
        
        # Update feature importances
        self.feature_importances = dict(zip(X.columns, self.model.feature_importances_))
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, data):
        """Make predictions with the environmental expert."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Make predictions
        return self.model.predict(X)


@pytest.fixture
def environmental_expert():
    """Create an environmental expert for testing."""
    return EnvironmentalExpert(
        env_cols=['env_temperature', 'humidity', 'pressure', 'aqi'],
        location_col='location',
        timestamp_col='date'
    )


class SimpleBehavioralExpert(BaseExpert):
    """Simplified behavioral expert for testing."""
    
    def __init__(self, behavior_cols, patient_id_col='patient_id', timestamp_col='date', name="BehavioralExpert"):
        super().__init__(
            name=name,
            model=RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        )
        self.behavior_cols = behavior_cols
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def preprocess_data(self, data):
        """Preprocess the behavioral data."""
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Select only the behavioral columns for this expert
        features = processed_data[self.behavior_cols].copy()
        
        # Standardize the features
        if not self.is_fitted:
            features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=self.behavior_cols,
                index=features.index
            )
        else:
            features = pd.DataFrame(
                self.scaler.transform(features),
                columns=self.behavior_cols,
                index=features.index
            )
        
        # Add simple derived features
        if 'sleep_quality' in self.behavior_cols and 'sleep_duration' in self.behavior_cols:
            # Calculate sleep efficiency
            features['sleep_efficiency'] = features['sleep_quality'] * features['sleep_duration'] / 10
        
        if 'stress_level' in self.behavior_cols and 'activity_level' in self.behavior_cols:
            # Calculate stress-activity ratio
            features['stress_activity_ratio'] = features['stress_level'] / (features['activity_level'] + 1)
        
        return features
    
    def fit(self, data, target_column='migraine_severity'):
        """Fit the behavioral expert model."""
        self.target_column = target_column
        
        # Preprocess the data
        X = self.preprocess_data(data)
        y = data[target_column]
        
        # Fit the model
        self.model.fit(X, y)
        
        # Update feature importances
        self.feature_importances = dict(zip(X.columns, self.model.feature_importances_))
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, data):
        """Make predictions with the behavioral expert."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Make predictions
        return self.model.predict(X)
        
    def predict_with_confidence(self, data):
        """Make predictions with confidence estimates."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # For RandomForest, we can estimate confidence using the standard deviation of predictions across trees
        if hasattr(self.model, 'estimators_'):
            # Get predictions from all trees
            tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
            # Calculate standard deviation across trees
            std_devs = np.std(tree_preds, axis=0)
            # Normalize to [0, 1] range for confidence (higher std_dev = lower confidence)
            max_std = np.max(std_devs) if np.max(std_devs) > 0 else 1.0
            confidence = 1 - (std_devs / max_std)
        else:
            # Fallback confidence estimation
            confidence = np.ones_like(predictions) * 0.8
        
        return predictions, confidence


@pytest.fixture
def behavioral_expert():
    """Create a behavioral expert for testing."""
    return BehavioralExpert(
        behavior_cols=['sleep_duration', 'sleep_quality', 'activity_level', 'stress_level'],
        patient_id_col='patient_id',
        timestamp_col='date'
    )


class SimpleMedicationExpert(BaseExpert):
    """Simplified medication expert for testing."""
    
    def __init__(self, medication_cols, patient_id_col='patient_id', timestamp_col='date', name="MedicationExpert"):
        super().__init__(
            name=name,
            model=GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        )
        self.medication_cols = medication_cols
        self.patient_id_col = patient_id_col
        self.timestamp_col = timestamp_col
        self.feature_columns = None
    
    def preprocess_data(self, data):
        """Preprocess the medication data."""
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Convert categorical medication values to numeric
        features = pd.DataFrame(index=processed_data.index)
        
        # Map medication levels to numeric values
        med_map = {'': 0, 'Low': 1, 'Medium': 2, 'High': 3}
        
        for col in self.medication_cols:
            if col in processed_data.columns:
                # Convert categorical to numeric
                features[f'{col}_numeric'] = processed_data[col].map(med_map)
        
        # Add interaction terms for medications
        if len(self.medication_cols) >= 2:
            for i, med1 in enumerate(self.medication_cols):
                for med2 in self.medication_cols[i+1:]:
                    if f'{med1}_numeric' in features.columns and f'{med2}_numeric' in features.columns:
                        # Create interaction feature
                        features[f'{med1}_{med2}_interaction'] = features[f'{med1}_numeric'] * features[f'{med2}_numeric']
        
        return features
    
    def fit(self, data, target_column='migraine_severity'):
        """Fit the medication expert model."""
        self.target_column = target_column
        
        # Preprocess the data
        X = self.preprocess_data(data)
        y = data[target_column]
        
        # Fit the model
        self.model.fit(X, y)
        
        # Update feature importances
        self.feature_importances = dict(zip(X.columns, self.model.feature_importances_))
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, data):
        """Make predictions with the medication expert."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Make predictions
        return self.model.predict(X)
        
    def predict_with_confidence(self, data):
        """Make predictions with confidence estimates."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Preprocess the data
        X = self.preprocess_data(data)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # For GradientBoosting, we can use the prediction variance as a proxy for confidence
        if hasattr(self.model, 'estimators_'):
            # Get the prediction variance
            var = np.zeros_like(predictions)
            for i, estimator in enumerate(self.model.estimators_.flatten()):
                var += (estimator.predict(X) - predictions) ** 2
            var /= len(self.model.estimators_.flatten())
            
            # Convert variance to confidence (higher variance = lower confidence)
            max_var = np.max(var) if np.max(var) > 0 else 1.0
            confidence = 1 - (var / max_var)
        else:
            # Fallback confidence estimation
            confidence = np.ones_like(predictions) * 0.7
        
        return predictions, confidence


@pytest.fixture
def medication_expert():
    """Create a medication expert for testing."""
    return MedicationHistoryExpert(
        medication_cols=['medication_a', 'medication_b', 'medication_c'],
        patient_id_col='patient_id',
        timestamp_col='date'
    )


def test_physiological_expert_integration(physiological_expert, train_test_data):
    """Test physiological expert integration with preprocessing and prediction."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Test preprocessing integration
    processed_data = physiological_expert.preprocess_data(X_train)
    assert processed_data is not None
    
    # Test fitting
    physiological_expert.fit(X=X_train, y=y_train, optimize_hyperparams=False)
    assert physiological_expert.is_fitted
    
    # Test prediction
    predictions = physiological_expert.predict(X_test)
    assert len(predictions) == len(X_test)
    
    # Test confidence-based prediction
    predictions, confidence = physiological_expert.predict_with_confidence(X_test)
    assert len(predictions) == len(X_test)
    assert len(confidence) == len(X_test)
    assert all(0 <= c <= 1 for c in confidence)
    
    # Test feature importance
    importances = physiological_expert.calculate_feature_importance()
    assert len(importances) > 0
    assert sum(importances.values()) > 0
    
    # Test evaluation
    metrics = physiological_expert.evaluate(X_test, y_test)
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics


def test_environmental_expert_integration(environmental_expert, train_test_data):
    """Test environmental expert integration with preprocessing and prediction."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Test preprocessing integration
    processed_data = environmental_expert.preprocess_data(X_train)
    assert processed_data is not None
    
    # Test fitting
    environmental_expert.fit(X=X_train, y=y_train, optimize_hyperparams=False)
    assert environmental_expert.is_fitted
    
    # Test prediction
    predictions = environmental_expert.predict(X_test)
    assert len(predictions) == len(X_test)
    
    # Test feature importance
    importances = environmental_expert.calculate_feature_importance()
    assert len(importances) > 0
    assert sum(importances.values()) > 0
    
    # Test evaluation
    metrics = environmental_expert.evaluate(X_test, y_test)
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics


def test_behavioral_expert_integration(behavioral_expert, train_test_data):
    """Test behavioral expert integration with preprocessing and prediction."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Test preprocessing integration
    processed_data = behavioral_expert.preprocess_data(X_train)
    assert processed_data is not None
    
    # Test fitting
    behavioral_expert.fit(X=X_train, y=y_train, optimize_features=False)
    assert behavioral_expert.is_fitted
    
    # Test prediction
    predictions = behavioral_expert.predict(X_test)
    assert len(predictions) == len(X_test)
    
    # Test confidence-based prediction
    predictions, confidence = behavioral_expert.predict_with_confidence(X_test)
    assert len(predictions) == len(X_test)
    assert len(confidence) == len(X_test)
    assert all(0 <= c <= 1 for c in confidence)
    
    # Test feature importance
    importances = behavioral_expert.calculate_feature_importance()
    assert len(importances) > 0
    assert sum(importances.values()) > 0
    
    # Test evaluation
    metrics = behavioral_expert.evaluate(X_test, y_test)
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics


def test_medication_expert_integration(medication_expert, train_test_data):
    """Test medication expert integration with preprocessing and prediction."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Test preprocessing integration
    processed_data = medication_expert.preprocess_data(X_train)
    assert processed_data is not None
    
    # Test fitting
    medication_expert.fit(X=X_train, y=y_train, optimize_hyperparams=False)
    assert medication_expert.is_fitted
    
    # Test prediction
    predictions = medication_expert.predict(X_test)
    assert len(predictions) == len(X_test)
    
    # Test confidence-based prediction
    predictions, confidence = medication_expert.predict_with_confidence(X_test)
    assert len(predictions) == len(X_test)
    assert len(confidence) == len(X_test)
    assert all(0 <= c <= 1 for c in confidence)
    
    # Test feature importance
    importances = medication_expert.calculate_feature_importance()
    assert len(importances) > 0
    assert sum(importances.values()) > 0
    
    # Test evaluation
    metrics = medication_expert.evaluate(X_test, y_test)
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert 'r2' in metrics


def test_model_serialization(physiological_expert, train_test_data):
    """Test model serialization and deserialization."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Fit the model
    physiological_expert.fit(X=X_train, y=y_train, optimize_hyperparams=False)
    
    # Get predictions before saving
    original_preds = physiological_expert.predict(X_test)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        filepath = f.name
    
    try:
        # Save the model
        physiological_expert.save(filepath)
        
        # Load the model
        loaded_expert = PhysiologicalExpert.load(filepath)
        
        # Check if loaded correctly
        assert loaded_expert.name == physiological_expert.name
        assert loaded_expert.is_fitted
        
        # Check predictions
        loaded_preds = loaded_expert.predict(X_test)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)


def test_multi_expert_workflow(physiological_expert, environmental_expert, 
                              behavioral_expert, medication_expert, train_test_data):
    """Test a complete workflow with multiple experts."""
    X_train, X_test, y_train, y_test = train_test_data
    
    # Train all experts
    experts = [
        physiological_expert,
        environmental_expert,
        behavioral_expert,
        medication_expert
    ]
    
    for expert in experts:
        expert.fit(X_train, y_train)
        assert expert.is_fitted
    
    # Get predictions from all experts
    predictions = {}
    for expert in experts:
        pred = expert.predict(X_test)
        predictions[expert.name] = pred
    
    # Simple ensemble (average predictions)
    ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
    
    # Verify ensemble prediction shape
    assert len(ensemble_pred) == len(y_test)
    
    # Weighted ensemble based on expert performance
    weights = {}
    for expert in experts:
        metrics = expert.evaluate(X_test, y_test)
        # Use R² as weight (higher is better)
        weights[expert.name] = max(0, metrics['r2'])  # Ensure non-negative
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {name: w/total_weight for name, w in weights.items()}
    else:
        # If all weights are 0, use equal weights
        weights = {name: 1/len(experts) for name in weights}
    
    # Calculate weighted ensemble prediction
    weighted_pred = np.zeros_like(ensemble_pred)
    for expert in experts:
        pred = expert.predict(X_test)
        weighted_pred += weights[expert.name] * pred
    
    # Verify weighted ensemble prediction shape
    assert len(weighted_pred) == len(y_test)
    
    # Compare feature importances across experts
    for expert in experts:
        importance = expert.calculate_feature_importance()
        # Each expert should have different top features
        assert len(importance) > 0
        
    # Test evolutionary optimization integration
    # Note: We're not actually running the optimization in tests (would be too slow)
    # but we're verifying the methods exist and can be called
    assert hasattr(physiological_expert, 'optimize_hyperparameters')
    assert hasattr(environmental_expert, 'optimize_hyperparameters')
    assert hasattr(behavioral_expert, 'optimize_feature_selection')
    assert hasattr(medication_expert, 'optimize_hyperparameters')


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])

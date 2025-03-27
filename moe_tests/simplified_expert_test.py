"""
Simplified integration tests for expert models in the MoE framework.

These tests focus on the core functionality of expert models without relying on
preprocessor classes that haven't been implemented yet.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Import the base expert to test
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from moe_framework.experts.base_expert import BaseExpert


class SimpleExpert(BaseExpert):
    """A simple implementation of BaseExpert for testing purposes."""
    
    def __init__(self, name="SimpleExpert", model=None, feature_columns=None, 
                 target_column=None, metadata=None):
        """Initialize the simple expert."""
        if model is None:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        super().__init__(name=name, model=model, 
                         feature_columns=feature_columns,
                         target_column=target_column,
                         metadata=metadata)
        
        self.scaler = StandardScaler()
    
    def preprocess(self, data):
        """Preprocess the data for the expert model."""
        # Make a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # If feature columns not specified, use all numeric columns except target
        if self.feature_columns is None:
            # Get all numeric columns
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target column if it's in numeric_cols
            if self.target_column in numeric_cols:
                numeric_cols.remove(self.target_column)
            # Also exclude date column if present
            if 'date' in numeric_cols:
                numeric_cols.remove('date')
            self.feature_columns = numeric_cols
        
        # Handle categorical columns
        categorical_cols = [col for col in processed_data.columns 
                           if processed_data[col].dtype == 'object' and col != 'date']
        
        # One-hot encode categorical columns
        for col in categorical_cols:
            dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=True)
            processed_data = pd.concat([processed_data.drop(col, axis=1), dummies], axis=1)
        
        # Get updated numeric columns (after one-hot encoding)
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        # Standardize numeric features if there are any
        if numeric_cols:
            if not self.is_fitted:
                processed_data[numeric_cols] = self.scaler.fit_transform(processed_data[numeric_cols])
            else:
                processed_data[numeric_cols] = self.scaler.transform(processed_data[numeric_cols])
        
        # Extract features for model
        X = processed_data.drop([self.target_column, 'date', 'patient_id', 'location'], axis=1, errors='ignore')
        
        return X
    
    def fit(self, data, target_column=None, **kwargs):
        """Fit the expert model to the data."""
        if target_column is not None:
            self.target_column = target_column
        
        if self.target_column is None:
            raise ValueError("Target column must be specified")
        
        # Preprocess the data
        X = self.preprocess(data)
        y = data[self.target_column]
        
        # Fit the model
        self.model.fit(X, y)
        
        # Update feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = dict(zip(X.columns, self.model.feature_importances_))
        
        # Calculate quality metrics
        y_pred = self.model.predict(X)
        self.quality_metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, data, **kwargs):
        """Make predictions using the expert model."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        # Preprocess the data
        X = self.preprocess(data)
        
        # Make predictions
        return self.model.predict(X)
    
    def evaluate(self, data, metrics=None, **kwargs):
        """Evaluate the expert model on the data."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
        
        if self.target_column is None:
            raise ValueError("Target column must be specified")
        
        # Preprocess the data
        X = self.preprocess(data)
        y = data[self.target_column]
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        evaluation = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        return evaluation
        
    def save(self, path):
        """Save the expert model to a file."""
        # Create a dictionary with all the necessary attributes
        state = {
            'name': self.name,
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'metadata': self.metadata,
            'feature_importances': self.feature_importances,
            'quality_metrics': self.quality_metrics,
            'is_fitted': self.is_fitted,
            'scaler': self.scaler
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        return path
    
    @classmethod
    def load(cls, path):
        """Load an expert model from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance with the loaded attributes
        expert = cls(
            name=state['name'],
            model=state['model'],
            feature_columns=state['feature_columns'],
            target_column=state['target_column'],
            metadata=state['metadata']
        )
        
        # Restore the other attributes
        expert.feature_importances = state['feature_importances']
        expert.quality_metrics = state['quality_metrics']
        expert.is_fitted = state['is_fitted']
        expert.scaler = state['scaler']
        
        return expert


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
    # We don't need to separate X and y since our expert model handles that
    train_data, test_data = train_test_split(
        sample_data, test_size=0.2, random_state=42
    )
    
    return train_data, test_data


@pytest.fixture
def simple_expert():
    """Create a simple expert for testing."""
    return SimpleExpert(
        name="SimpleExpert",
        feature_columns=None,  # Will be determined during preprocessing
        target_column='migraine_severity'
    )


def test_expert_initialization(simple_expert):
    """Test expert model initialization."""
    assert simple_expert.name == "SimpleExpert"
    assert simple_expert.target_column == 'migraine_severity'
    assert simple_expert.is_fitted is False
    assert isinstance(simple_expert.model, RandomForestRegressor)


def test_expert_fit_predict(simple_expert, train_test_data):
    """Test expert model fitting and prediction."""
    X_train, X_test = train_test_data
    
    # Fit the model
    simple_expert.fit(X_train)
    
    # Check that the model is fitted
    assert simple_expert.is_fitted is True
    
    # Check that feature importances are calculated
    assert len(simple_expert.feature_importances) > 0
    
    # Check that quality metrics are calculated
    assert 'mse' in simple_expert.quality_metrics
    assert 'mae' in simple_expert.quality_metrics
    assert 'r2' in simple_expert.quality_metrics
    
    # Make predictions
    predictions = simple_expert.predict(X_test)
    
    # Check predictions
    assert len(predictions) == len(X_test)
    assert isinstance(predictions, np.ndarray)


def test_expert_evaluation(simple_expert, train_test_data):
    """Test expert model evaluation."""
    X_train, X_test = train_test_data
    
    # Fit the model
    simple_expert.fit(X_train)
    
    # Evaluate the model
    evaluation = simple_expert.evaluate(X_test)
    
    # Check evaluation metrics
    assert 'mse' in evaluation
    assert 'mae' in evaluation
    assert 'r2' in evaluation


def test_expert_serialization(simple_expert, train_test_data):
    """Test expert model serialization and deserialization."""
    X_train, X_test = train_test_data
    
    # Fit the model
    simple_expert.fit(X_train)
    
    # Create a temporary directory for saving the model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'simple_expert.pkl')
        
        # Save the model
        simple_expert.save(model_path)
        
        # Check that the model file exists
        assert os.path.exists(model_path)
        
        # Load the model
        loaded_expert = SimpleExpert.load(model_path)
        
        # Check that the loaded model has the same attributes
        assert loaded_expert.name == simple_expert.name
        assert loaded_expert.target_column == simple_expert.target_column
        assert loaded_expert.is_fitted is True
        
        # Make predictions with the loaded model
        original_predictions = simple_expert.predict(X_test)
        loaded_predictions = loaded_expert.predict(X_test)
        
        # Check that the predictions are the same
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])

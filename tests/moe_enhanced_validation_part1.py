#!/usr/bin/env python3
"""
MoE Enhanced Validation Framework - Part 1
-----------------------------------------
A comprehensive validation framework for the Mixture-of-Experts system,
focusing on Meta_Optimizer, Meta_Learner, and their integration with the gating network.

This framework includes:
1. Component-specific test cases
2. Drift detection and adaptation testing
3. Explainability validation
4. Gating network integration tests
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
parent_dir = str(Path(__file__).parent.parent.absolute())
results_dir = Path(parent_dir) / "results" / "moe_validation"
results_dir.mkdir(parents=True, exist_ok=True)

#############################
# Synthetic Data Generators #
#############################

class SyntheticDataGenerator:
    """Base generator for all synthetic data."""
    
    def __init__(self, seed=42):
        """Initialize the synthetic data generator."""
        self.seed = seed
        np.random.seed(seed)
        
    def generate_physiological_data(self, n_samples=100, n_features=5):
        """Generate synthetic physiological data."""
        logger.info(f"Generating {n_samples} samples of physiological data with {n_features} features")
        
        # Features like heart rate, blood pressure, etc.
        feature_names = [f"physio_{i}" for i in range(n_features)]
        
        # Generate random data with correlations
        mean = np.random.uniform(60, 120, n_features)
        cov = np.random.uniform(0.1, 0.5, (n_features, n_features))
        cov = cov @ cov.T  # ensure positive semi-definite
        np.fill_diagonal(cov, np.random.uniform(1, 5, n_features))
        
        data = np.random.multivariate_normal(mean, cov, n_samples)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=feature_names)
        
        # Add time dimension for temporal data
        timestamps = pd.date_range(start=datetime.now(), periods=n_samples, freq='h')
        df['timestamp'] = timestamps
        
        logger.info(f"Generated physiological data with shape {df.shape}")
        return df
    
    def generate_environmental_data(self, n_samples=100, n_features=3):
        """Generate synthetic environmental data."""
        logger.info(f"Generating {n_samples} samples of environmental data")
        
        # Environmental features (weather, pollution, etc.)
        feature_names = ['temperature', 'humidity', 'barometric_pressure', 'air_quality', 'light_intensity']
        feature_names = feature_names[:n_features]
        
        # Generate random environmental data
        timestamps = pd.date_range(start=datetime.now(), periods=n_samples, freq='h')
        
        # Create base values
        base_values = {
            'temperature': np.random.uniform(10, 30),
            'humidity': np.random.uniform(30, 80),
            'barometric_pressure': np.random.uniform(990, 1030),
            'air_quality': np.random.uniform(30, 150),
            'light_intensity': np.random.uniform(0, 1000)
        }
        
        # Generate with daily patterns
        data = {}
        for feature in feature_names:
            # Add daily cycles with noise
            hours = np.array([(t.hour + t.minute/60) for t in timestamps])
            if feature == 'temperature':
                # Temperature peaks in afternoon
                cycle = 5 * np.sin(2 * np.pi * (hours - 14) / 24)
            elif feature == 'humidity':
                # Humidity peaks at night/early morning
                cycle = 15 * np.sin(2 * np.pi * (hours - 4) / 24)
            elif feature == 'light_intensity':
                # Light follows daylight hours
                cycle = 500 * np.sin(2 * np.pi * (hours - 12) / 24)
                cycle[cycle < 0] = 0  # No negative light
            else:
                # Other features have milder cycles
                cycle = 3 * np.sin(2 * np.pi * hours / 24)
                
            # Add noise
            noise = np.random.normal(0, abs(base_values[feature]) * 0.05, n_samples)
            
            # Final values
            data[feature] = base_values[feature] + cycle + noise
        
        # Create dataframe
        df = pd.DataFrame(data)
        df['timestamp'] = timestamps
        
        logger.info(f"Generated environmental data with shape {df.shape}")
        return df
    
    def generate_migraine_events(self, n_samples=100, n_events=10, physiological_data=None, environmental_data=None):
        """Generate synthetic migraine events based on physiological and environmental data."""
        logger.info(f"Generating {n_events} migraine events")
        
        # Generate random timestamps
        timestamps = pd.date_range(start=datetime.now(), periods=n_samples, freq='h')
        
        if physiological_data is not None and environmental_data is not None:
            # Create migraine events with some correlation to input data
            # Merge data on timestamp
            merged_data = pd.merge(
                physiological_data, 
                environmental_data, 
                on='timestamp', 
                how='inner',
                suffixes=('_physio', '_env')
            )
            
            # Create a simple model to generate migraine probability
            # High temp + high pressure + high physio_0 increases probability
            if 'temperature' in merged_data.columns and 'barometric_pressure' in merged_data.columns:
                temp_norm = (merged_data['temperature'] - merged_data['temperature'].mean()) / merged_data['temperature'].std()
                pressure_norm = (merged_data['barometric_pressure'] - merged_data['barometric_pressure'].mean()) / merged_data['barometric_pressure'].std()
                physio_norm = (merged_data['physio_0'] - merged_data['physio_0'].mean()) / merged_data['physio_0'].std() if 'physio_0' in merged_data.columns else 0
                
                # Probability model (simple)
                migraine_prob = 0.1 + 0.3 * (temp_norm + pressure_norm + physio_norm)
                migraine_prob = np.clip(migraine_prob, 0.01, 0.5)  # Limit probability range
                
                # Generate events based on probability
                events = np.random.binomial(1, migraine_prob)
                event_indices = np.where(events == 1)[0]
                
                # If we don't have enough events, add some
                if len(event_indices) < n_events:
                    additional_indices = np.random.choice(
                        [i for i in range(len(merged_data)) if i not in event_indices],
                        n_events - len(event_indices),
                        replace=False
                    )
                    event_indices = np.concatenate([event_indices, additional_indices])
                
                # If we have too many events, sample
                if len(event_indices) > n_events:
                    event_indices = np.random.choice(event_indices, n_events, replace=False)
                
                event_timestamps = merged_data.iloc[event_indices]['timestamp']
            else:
                # Randomly select events if we don't have the expected columns
                event_indices = np.random.choice(len(merged_data), min(n_events, len(merged_data)), replace=False)
                event_timestamps = merged_data.iloc[event_indices]['timestamp']
        else:
            # Randomly select events if no input data
            event_indices = np.random.choice(n_samples, n_events, replace=False)
            event_timestamps = timestamps[event_indices]
        
        # Create event dataframe
        events = pd.DataFrame({
            'timestamp': event_timestamps,
            'intensity': np.random.uniform(1, 10, len(event_timestamps)),
            'duration_hours': np.random.uniform(1, 48, len(event_timestamps))
        })
        
        logger.info(f"Generated {len(events)} migraine events")
        return events
    
    def generate_concept_drift_data(self, n_samples=500, drift_point=250, drift_magnitude=1.0):
        """Generate data with concept drift at a specified point."""
        logger.info(f"Generating synthetic data with concept drift at sample {drift_point}")
        
        # Initialize data structures
        X = np.zeros((n_samples, 5))
        y = np.zeros(n_samples)
        
        # Generate features
        for i in range(5):
            X[:, i] = np.random.normal(0, 1, n_samples)
        
        # Pre-drift relationship: y is related to the first two features
        for i in range(drift_point):
            y[i] = 2*X[i, 0] + X[i, 1] + np.random.normal(0, 0.5)
        
        # Post-drift relationship: y is now related to different features with different weights
        for i in range(drift_point, n_samples):
            # Change the relationship based on drift_magnitude
            y[i] = drift_magnitude*(X[i, 2] + 3*X[i, 3]) + np.random.normal(0, 0.5)
        
        # Create a DataFrame with timestamp
        timestamps = pd.date_range(start=datetime.now(), periods=n_samples, freq='h')
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        df['timestamp'] = timestamps
        
        logger.info(f"Generated concept drift data with shape {df.shape}, drift at index {drift_point}")
        return df

####################
# Mock Components #
####################

class MockExpert:
    """Mock expert model for MoE validation."""
    
    def __init__(self, expert_id, specialty='general'):
        """Initialize the mock expert."""
        self.expert_id = expert_id
        self.specialty = specialty
        self.accuracy = np.random.uniform(0.7, 0.95)
        self.trained = False
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        
    def train(self, X, y):
        """Train the expert model."""
        logger.info(f"Training Expert {self.expert_id} ({self.specialty})")
        # Train the actual model
        self.model.fit(X, y)
        self.trained = True
        # Slightly improve accuracy after training
        self.accuracy += np.random.uniform(0.01, 0.05)
        if self.accuracy > 0.99:
            self.accuracy = 0.99
            
        return self
    
    def predict(self, X):
        """Make predictions with the expert model."""
        if not self.trained:
            raise RuntimeError(f"Expert {self.expert_id} not trained")
            
        # Make actual predictions
        predictions = self.model.predict(X)
        
        # Add noise inversely proportional to accuracy
        noise = np.random.normal(0, (1 - self.accuracy) * 2, len(X))
        predictions = predictions + noise
        
        return predictions

class MockMetaOptimizer:
    """Mock Meta_Optimizer for validation testing."""
    
    def __init__(self):
        """Initialize the mock Meta_Optimizer."""
        self.optimizers = {
            'DE': {'speed': 0.7, 'quality': 0.8, 'robustness': 0.75},
            'PSO': {'speed': 0.8, 'quality': 0.75, 'robustness': 0.7},
            'ES': {'speed': 0.75, 'quality': 0.85, 'robustness': 0.8},
            'GWO': {'speed': 0.85, 'quality': 0.7, 'robustness': 0.85}
        }
        self.history = []
        
    def select_optimizer(self, problem_features):
        """Select the best optimizer for a given problem."""
        # Simple selection based on problem features
        if problem_features.get('dimension', 10) > 10:
            # High-dimensional problems favor ES
            chosen = 'ES'
        elif problem_features.get('multimodality', 0.5) > 0.7:
            # Highly multimodal problems favor GWO
            chosen = 'GWO'
        elif problem_features.get('ruggedness', 0.5) < 0.3:
            # Smooth problems favor PSO
            chosen = 'PSO'
        else:
            # Default to DE for balanced performance
            chosen = 'DE'
            
        self.history.append({
            'problem': problem_features,
            'selected': chosen,
            'timestamp': datetime.now()
        })
        
        return chosen
    
    def optimize(self, problem, max_evals=100):
        """Run optimization on the given problem."""
        # Select optimizer
        selected = self.select_optimizer(problem)
        
        # Simulate optimization
        logger.info(f"Running {selected} optimizer for {max_evals} evaluations")
        time.sleep(0.2)
        
        # Generate mock result
        solution = np.random.uniform(-1, 1, problem.get('dimension', 2))
        score = np.random.uniform(0, 1) * (1 - self.optimizers[selected]['quality'])
        
        return {
            'solution': solution,
            'score': score,
            'optimizer': selected,
            'evals': max_evals
        }

class MockMetaLearner:
    """Mock Meta_Learner for validation testing."""
    
    def __init__(self):
        """Initialize the mock Meta_Learner."""
        self.experts = {}
        self.performance_history = {}
        
    def register_expert(self, expert_id, expert):
        """Register an expert with the Meta_Learner."""
        self.experts[expert_id] = {
            'expert': expert,
            'specialty': expert.specialty,
            'accuracy': expert.accuracy
        }
        self.performance_history[expert_id] = []
        
    def predict_weights(self, features):
        """Predict weights for experts based on input features."""
        if not self.experts:
            raise ValueError("No experts registered with Meta_Learner")
            
        weights = {}
        
        # Generate weights based on expert specialty and features
        for expert_id, expert_info in self.experts.items():
            specialty = expert_info['specialty']
            accuracy = expert_info['accuracy']
            
            # Base weight on accuracy
            weight = accuracy
            
            # Adjust weight based on specialty match to features
            if specialty == 'physiological' and features.get('has_physiological', False):
                weight *= 1.2
            elif specialty == 'behavioral' and features.get('has_behavioral', False):
                weight *= 1.2
            elif specialty == 'environmental' and features.get('has_environmental', False):
                weight *= 1.2
                
            # Normalize to 0-1 range
            weight = min(1.0, weight)
            
            weights[expert_id] = weight
            
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
            
        return weights
    
    def update_performance(self, expert_id, performance_metric):
        """Update the performance history for an expert."""
        if expert_id not in self.performance_history:
            raise ValueError(f"Expert {expert_id} not registered")
            
        self.performance_history[expert_id].append({
            'timestamp': datetime.now(),
            'metric': performance_metric
        })
        
        # Update accuracy in expert info
        if expert_id in self.experts:
            # Exponential moving average
            current = self.experts[expert_id]['accuracy']
            self.experts[expert_id]['accuracy'] = 0.8 * current + 0.2 * performance_metric

class MockDriftDetector:
    """Mock drift detector for testing."""
    
    def __init__(self, window_size=10, threshold=0.05):
        """Initialize the mock drift detector."""
        self.window_size = window_size
        self.threshold = threshold
        self.reference_data = None
        self.last_drift_time = None
        
    def set_reference(self, data):
        """Set the reference data for drift comparison."""
        self.reference_data = data
        
    def detect_drift(self, current_data):
        """Detect drift between current data and reference data."""
        if self.reference_data is None:
            raise ValueError("Reference data must be set before detecting drift")
            
        # Simple drift detection based on mean shift
        ref_mean = np.mean(self.reference_data, axis=0)
        cur_mean = np.mean(current_data, axis=0)
        
        # Calculate mean shift as percent of reference standard deviation
        ref_std = np.std(self.reference_data, axis=0)
        ref_std[ref_std == 0] = 1e-6  # Avoid division by zero
        
        # Calculate normalized shift
        mean_shift = np.abs(cur_mean - ref_mean) / ref_std
        
        # Total shift (average across features)
        total_shift = np.mean(mean_shift)
        
        # Detect drift if shift exceeds threshold
        drift_detected = total_shift > self.threshold
        
        # Additional info
        info = {
            'mean_shift': total_shift,
            'feature_shifts': mean_shift,
            'threshold': self.threshold
        }
        
        if drift_detected:
            self.last_drift_time = datetime.now()
            
        return drift_detected, total_shift, info

class MockGatingNetwork:
    """Mock gating network for MoE system."""
    
    def __init__(self, meta_learner=None):
        """Initialize the mock gating network."""
        self.meta_learner = meta_learner
        self.trained = False
        
    def train(self, X, y, experts):
        """Train the gating network."""
        logger.info("Training gating network")
        
        # Register experts with meta_learner if available
        if self.meta_learner:
            for expert_id, expert in experts.items():
                self.meta_learner.register_expert(expert_id, expert)
                
        self.trained = True
        return self
    
    def predict_weights(self, X):
        """Predict expert weights for input features."""
        if not self.trained:
            raise RuntimeError("Gating network not trained")
            
        # Get feature set
        n_samples = len(X)
        
        if self.meta_learner:
            # Use meta_learner to predict weights
            # For each sample, determine feature types present
            weights_per_sample = []
            
            for i in range(n_samples):
                # Detect feature types based on column names
                features = {}
                if any('physio' in col for col in X.columns):
                    features['has_physiological'] = True
                if any(col in ['temperature', 'humidity', 'barometric_pressure'] for col in X.columns):
                    features['has_environmental'] = True
                if any('behavior' in col for col in X.columns):
                    features['has_behavioral'] = True
                    
                # Get weights from meta_learner
                weights = self.meta_learner.predict_weights(features)
                weights_per_sample.append(weights)
                
            return weights_per_sample
        else:
            # Equal weighting if no meta_learner
            return [{expert_id: 1.0/n_experts for expert_id in range(n_experts)} 
                    for _ in range(n_samples)]

class MockExplainabilityEngine:
    """Mock explainability engine for testing."""
    
    def __init__(self):
        """Initialize the mock explainability engine."""
        pass
        
    def explain_model(self, model, X):
        """Generate mock feature importance for a model."""
        # Create mock feature importance
        if hasattr(X, 'columns'):
            feature_names = X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        importance = np.random.uniform(0, 1, len(feature_names))
        # Normalize importance
        importance = importance / np.sum(importance)
        
        return dict(zip(feature_names, importance))
    
    def explain_prediction(self, model, X, sample_idx=0):
        """Explain a single prediction."""
        if hasattr(X, 'iloc'):
            sample = X.iloc[sample_idx]
        else:
            sample = X[sample_idx]
            
        # Generate mock explanation
        explanation = {
            'sample_values': sample,
            'contribution': np.random.uniform(-1, 1, len(sample))
        }
        
        return explanation
    
    def generate_importance_plot(self, importance_dict, title="Feature Importance", output_file=None):
        """Generate a feature importance plot."""
        # Sort importance
        sorted_imp = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(10, 6))
        plt.barh(list(sorted_imp.keys()), list(sorted_imp.values()))
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
            return output_file
        else:
            return plt

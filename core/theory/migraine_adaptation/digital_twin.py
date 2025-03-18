"""
Digital Twin Foundation for Migraine Prediction
============================================

This module provides the mathematical foundation for patient digital twins,
enabling personalized simulation and intervention testing for migraine prediction.

Key Features:
1. Patient state modeling using multimodal data
2. Dynamic model updating with new observations
3. Intervention simulation and outcome prediction
4. Model accuracy assessment and validation
5. Uncertainty quantification in predictions

The implementation uses a combination of statistical modeling, physiological 
signal processing, and machine learning techniques to create accurate digital
representations of individual patients.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from core.theory.multimodal_integration import ModalityData
from core.theory.temporal_modeling.causal_inference import CausalInferenceAnalyzer
from . import DigitalTwinFoundation

class PatientState:
    """Represents the current state of a patient in the digital twin."""
    
    def __init__(self,
                 physiological_state: Dict[str, np.ndarray],
                 trigger_state: Dict[str, float],
                 symptom_state: Dict[str, float],
                 timestamp: datetime,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize patient state.
        
        Parameters
        ----------
        physiological_state : Dict[str, np.ndarray]
            Current physiological measurements
        trigger_state : Dict[str, float]
            Current trigger levels
        symptom_state : Dict[str, float]
            Current symptom intensities
        timestamp : datetime
            Time of the state
        metadata : Optional[Dict[str, Any]]
            Additional state metadata
        """
        self.physiological_state = physiological_state
        self.trigger_state = trigger_state
        self.symptom_state = symptom_state
        self.timestamp = timestamp
        self.metadata = metadata or {}
        
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for modeling."""
        features = []
        
        # Add physiological features
        for signal in self.physiological_state.values():
            if len(signal.shape) == 1:
                # Single channel signals (ECG, SC)
                if len(signal) == 1:
                    # Single value
                    features.extend([
                        float(signal[0]),  # mean
                        0.0,               # std
                        float(signal[0]),  # min
                        float(signal[0])   # max
                    ])
                else:
                    # Multiple values
                    features.extend([
                        np.mean(signal),
                        np.std(signal),
                        np.min(signal),
                        np.max(signal)
                    ])
            else:
                # Multi-channel signals (EEG)
                features.extend([
                    np.mean(signal),
                    np.std(signal.flatten())
                ])
        
        # Add trigger levels
        features.extend(list(self.trigger_state.values()))
        
        # Add symptom intensities
        features.extend(list(self.symptom_state.values()))
        
        return np.array(features)
    
    @classmethod
    def from_vector(cls,
                   vector: np.ndarray,
                   feature_map: Dict[str, slice],
                   timestamp: datetime,
                   metadata: Optional[Dict[str, Any]] = None) -> 'PatientState':
        """
        Create PatientState from feature vector.
        
        Parameters
        ----------
        vector : np.ndarray
            Feature vector
        feature_map : Dict[str, slice]
            Mapping of features to vector slices
        timestamp : datetime
            Time of the state
        metadata : Optional[Dict[str, Any]]
            Additional state metadata
            
        Returns
        -------
        PatientState
            Reconstructed patient state
        """
        physiological_state = {}
        trigger_state = {}
        symptom_state = {}
        
        for name, slice_idx in feature_map.items():
            if name.startswith('phys_'):
                physiological_state[name[5:]] = vector[slice_idx]
            elif name.startswith('trigger_'):
                # Extract single value for triggers
                trigger_state[name[8:]] = float(vector[slice_idx].item())
            elif name.startswith('symptom_'):
                # Extract single value for symptoms
                symptom_state[name[8:]] = float(vector[slice_idx].item())
        
        return cls(
            physiological_state=physiological_state,
            trigger_state=trigger_state,
            symptom_state=symptom_state,
            timestamp=timestamp,
            metadata=metadata
        )

class DigitalTwinModel(DigitalTwinFoundation):
    """Implementation of patient digital twin for migraine prediction."""
    
    def __init__(self,
                 state_prediction_horizon: int = 24,  # hours
                 update_window: int = 12,  # hours
                 confidence_threshold: float = 0.8):
        """
        Initialize digital twin model.
        
        Parameters
        ----------
        state_prediction_horizon : int
            How far ahead to predict patient state (hours)
        update_window : int
            How often to update the model (hours)
        confidence_threshold : float
            Minimum confidence for predictions
        """
        self.state_prediction_horizon = state_prediction_horizon
        self.update_window = update_window
        self.confidence_threshold = confidence_threshold
        
        # Initialize model components
        self.state_transition_model = None
        self.intervention_response_model = None
        self.anomaly_detector = None
        self.feature_scaler = StandardScaler()
        self.causal_analyzer = CausalInferenceAnalyzer(pd.DataFrame())
        
        # State tracking
        self.current_state = None
        self.state_history = []
        self.feature_map = {}
        self.model_metadata = {}
    
    def initialize_twin(self,
                       patient_history: Dict[str, np.ndarray],
                       patient_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize digital twin from patient history.
        
        Parameters
        ----------
        patient_history : Dict[str, np.ndarray]
            Historical patient data across modalities
        patient_metadata : Dict[str, Any]
            Patient metadata including demographics and medical history
            
        Returns
        -------
        Dict[str, Any]
            Initialized digital twin model state
        """
        # Process historical data
        processed_states = self._process_historical_data(
            patient_history,
            patient_metadata
        )
        
        # Create feature mapping
        self._create_feature_mapping(processed_states[0])
        
        # Convert states to feature vectors
        feature_vectors = np.array([
            state.to_vector() for state in processed_states
        ])
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(feature_vectors)
        
        # Initialize state transition model
        self.state_transition_model = self._initialize_state_model(scaled_features)
        
        # Initialize intervention response model
        self.intervention_response_model = self._initialize_response_model(
            scaled_features,
            patient_metadata
        )
        
        # Initialize anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        ).fit(scaled_features)
        
        # Set current state
        self.current_state = processed_states[-1]
        self.state_history = processed_states
        
        # Store metadata
        self.model_metadata = {
            'patient_id': patient_metadata.get('patient_id'),
            'initialization_time': datetime.now(),
            'last_update_time': datetime.now(),
            'n_historical_states': len(processed_states),
            'feature_dimension': feature_vectors.shape[1]
        }
        
        return self.get_model_state()
    
    def update_twin(self,
                   twin_model: Dict[str, Any],
                   new_observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Update digital twin with new observations.
        
        Parameters
        ----------
        twin_model : Dict[str, Any]
            Current digital twin model
        new_observations : Dict[str, np.ndarray]
            New observations to incorporate
            
        Returns
        -------
        Dict[str, Any]
            Updated digital twin model
        """
        # Load model state if different from current
        if twin_model.get('model_id') != id(self):
            self._load_model_state(twin_model)
        
        # Process new observations
        new_states = self._process_observations(new_observations)
        
        # Return current state if no new observations
        if not new_states:
            return self.get_model_state()
        
        # Convert to feature vectors
        new_vectors = np.array([
            state.to_vector() for state in new_states
        ])
        
        # Handle empty data
        if new_vectors.size == 0:
            return self.get_model_state()
            
        scaled_vectors = self.feature_scaler.transform(new_vectors)
        
        # Check for anomalies
        anomaly_scores = self.anomaly_detector.score_samples(scaled_vectors)
        valid_indices = anomaly_scores > -0.5  # Higher score = more normal
        
        if np.any(valid_indices):
            # Update state transition model
            self._update_state_model(scaled_vectors[valid_indices])
            
            # Update response model if interventions present
            if 'interventions' in new_observations:
                self._update_response_model(
                    scaled_vectors[valid_indices],
                    new_observations['interventions']
                )
            
            # Update current state and history
            self.current_state = new_states[-1]
            self.state_history.extend(new_states)
            
            # Update metadata
            self.model_metadata['last_update_time'] = datetime.now()
            self.model_metadata['n_historical_states'] += len(new_states)
        
        return self.get_model_state()
    
    def simulate_intervention(self,
                            twin_model: Dict[str, Any],
                            intervention: Dict[str, Any],
                            simulation_duration: float) -> Dict[str, np.ndarray]:
        """
        Simulate intervention effect on digital twin.
        
        Parameters
        ----------
        twin_model : Dict[str, Any]
            Digital twin model
        intervention : Dict[str, Any]
            Intervention to simulate
        simulation_duration : float
            Duration of simulation in hours
            
        Returns
        -------
        Dict[str, np.ndarray]
            Simulated responses to the intervention
        """
        # Load model state if different from current
        if twin_model.get('model_id') != id(self):
            self._load_model_state(twin_model)
        
        # Initialize simulation
        n_steps = int(simulation_duration / self.update_window)
        current_state_vector = self.current_state.to_vector()
        scaled_state = self.feature_scaler.transform([current_state_vector])[0]
        
        # Prepare results containers
        state_trajectories = []
        confidence_scores = []
        
        # Run simulation
        for step in range(n_steps):
            # Predict next state
            next_state = self._predict_next_state(
                scaled_state,
                intervention
            )
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(
                scaled_state,
                next_state,
                step * self.update_window
            )
            
            # Store results
            state_trajectories.append(next_state)
            confidence_scores.append(confidence)
            
            # Update for next step
            scaled_state = next_state
            
            # Stop if confidence too low
            if confidence < self.confidence_threshold:
                break
        
        # Convert results
        trajectories = np.array(state_trajectories)
        unscaled_trajectories = self.feature_scaler.inverse_transform(trajectories)
        
        # Organize results
        results = {
            'state_trajectories': unscaled_trajectories,
            'confidence_scores': np.array(confidence_scores),
            'simulation_times': np.arange(len(confidence_scores)) * self.update_window,
            'intervention_effects': self._analyze_intervention_effects(
                unscaled_trajectories,
                intervention
            )
        }
        
        return results
    
    def assess_twin_accuracy(self,
                           twin_model: Dict[str, Any],
                           actual_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Assess digital twin accuracy against actual data.
        
        Parameters
        ----------
        twin_model : Dict[str, Any]
            Digital twin model
        actual_data : Dict[str, np.ndarray]
            Actual patient data for comparison
            
        Returns
        -------
        Dict[str, float]
            Accuracy metrics for the digital twin
        """
        # Load model state if different from current
        if twin_model.get('model_id') != id(self):
            self._load_model_state(twin_model)
        
        # Process actual data
        actual_states = self._process_observations(actual_data)
        if not actual_states:
            return {
                'mse': 0.0,
                'mae': 0.0,
                'r2': 0.0,
                'temporal_correlation': 0.0,
                'feature_metrics': {}
            }
            
        actual_vectors = np.array([
            state.to_vector() for state in actual_states
        ])
        scaled_actual = self.feature_scaler.transform(actual_vectors)
        
        # Generate predictions
        predictions = []
        scaled_state = self.feature_scaler.transform([self.current_state.to_vector()])[0]
        
        for _ in range(len(actual_states)):
            next_state = self._predict_next_state(scaled_state, None)
            predictions.append(next_state)
            scaled_state = next_state
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        mse = np.mean((predictions - scaled_actual) ** 2)
        mae = np.mean(np.abs(predictions - scaled_actual))
        
        # Calculate R² with handling for constant predictions
        ss_tot = np.sum((scaled_actual - scaled_actual.mean()) ** 2)
        ss_res = np.sum((scaled_actual - predictions) ** 2)
        r2 = 1 - ss_res / (ss_tot if ss_tot > 0 else 1)
        
        # Calculate temporal correlations with handling for constant arrays
        temporal_correlations = []
        for i in range(predictions.shape[1]):
            pred_std = np.std(predictions[:, i])
            actual_std = np.std(scaled_actual[:, i])
            
            if pred_std > 0 and actual_std > 0:
                try:
                    corr = stats.pearsonr(predictions[:, i], scaled_actual[:, i])[0]
                    if not np.isnan(corr):
                        temporal_correlations.append(corr)
                except:
                    continue
        
        temporal_correlation = np.mean(temporal_correlations) if temporal_correlations else 0.0
        
        # Calculate feature-specific metrics
        feature_metrics = {}
        for name, slice_idx in self.feature_map.items():
            feature_mse = np.mean((predictions[:, slice_idx] - scaled_actual[:, slice_idx]) ** 2)
            feature_metrics[name] = float(feature_mse)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'temporal_correlation': float(temporal_correlation),
            'feature_metrics': feature_metrics
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state."""
        return {
            'model_id': id(self),
            'current_state': self.current_state,
            'model_metadata': self.model_metadata.copy(),
            'feature_map': self.feature_map.copy()
        }
    
    def _process_historical_data(self,
                               patient_history: Dict[str, np.ndarray],
                               patient_metadata: Dict[str, Any]) -> List[PatientState]:
        """Process historical data into patient states."""
        states = []
        timestamps = patient_history.get('timestamps', np.array([]))
        
        if len(timestamps) == 0:
            return states
        
        for i in range(len(timestamps)):
            physiological_state = {}
            trigger_state = {}
            symptom_state = {}
            
            # Extract physiological data
            for key, data in patient_history.items():
                if key.startswith('phys_'):
                    if len(data.shape) > 1:
                        # Multi-channel data (e.g., EEG)
                        physiological_state[key[5:]] = data[i, :]
                    else:
                        # Single-channel data (e.g., ECG, SC)
                        physiological_state[key[5:]] = np.array([data[i]])
                elif key.startswith('trigger_'):
                    trigger_state[key[8:]] = float(data[i])
                elif key.startswith('symptom_'):
                    symptom_state[key[8:]] = float(data[i])
            
            state = PatientState(
                physiological_state=physiological_state,
                trigger_state=trigger_state,
                symptom_state=symptom_state,
                timestamp=timestamps[i],
                metadata={'patient_id': patient_metadata.get('patient_id')}
            )
            states.append(state)
        
        return states
    
    def _create_feature_mapping(self, initial_state: PatientState):
        """Create mapping between features and vector indices."""
        vector = initial_state.to_vector()
        current_idx = 0
        
        # Map physiological features
        for name, signal in initial_state.physiological_state.items():
            if len(signal.shape) == 1:
                # Single channel signals (ECG, SC)
                feature_len = 4  # mean, std, min, max
            else:
                # Multi-channel signals (EEG)
                feature_len = 2  # mean, std
            
            # Create slice for this feature
            self.feature_map[f'phys_{name}'] = slice(
                current_idx,
                current_idx + feature_len
            )
            current_idx += feature_len
        
        # Map trigger features
        for name in initial_state.trigger_state:
            self.feature_map[f'trigger_{name}'] = slice(
                current_idx,
                current_idx + 1
            )
            current_idx += 1
        
        # Map symptom features
        for name in initial_state.symptom_state:
            self.feature_map[f'symptom_{name}'] = slice(
                current_idx,
                current_idx + 1
            )
            current_idx += 1
    
    def _initialize_state_model(self, scaled_features: np.ndarray) -> GaussianProcessRegressor:
        """Initialize state transition model."""
        # Define kernel for temporal correlations
        kernel = (
            ConstantKernel() *
            RBF(length_scale=np.ones(scaled_features.shape[1])) +
            WhiteKernel()
        )
        
        # Create and fit model
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=42
        )
        
        # Fit to predict next state from current
        model.fit(
            scaled_features[:-1],
            scaled_features[1:]
        )
        
        return model
    
    def _initialize_response_model(self,
                                 scaled_features: np.ndarray,
                                 patient_metadata: Dict[str, Any]) -> GaussianProcessRegressor:
        """Initialize intervention response model."""
        # Define kernel for intervention responses
        kernel = (
            ConstantKernel() *
            RBF(length_scale=np.ones(scaled_features.shape[1] + 1)) +
            WhiteKernel()
        )
        
        # Create model
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            random_state=42
        )
        
        # Initial fit with dummy data if no intervention history
        dummy_interventions = np.zeros((scaled_features.shape[0] - 1, 1))
        model.fit(
            np.hstack([scaled_features[:-1], dummy_interventions]),
            scaled_features[1:]
        )
        
        return model
    
    def _predict_next_state(self,
                           current_state: np.ndarray,
                           intervention: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Predict next state given current state and intervention."""
        # Reshape for prediction
        X = current_state.reshape(1, -1)
        
        if intervention is None:
            # Use state transition model
            next_state, _ = self.state_transition_model.predict(
                X,
                return_std=True
            )
        else:
            # Include intervention effect
            intervention_vector = self._encode_intervention(intervention)
            X_with_intervention = np.hstack([X, intervention_vector.reshape(1, -1)])
            
            next_state, _ = self.intervention_response_model.predict(
                X_with_intervention,
                return_std=True
            )
        
        return next_state[0]
    
    def _calculate_prediction_confidence(self,
                                      current_state: np.ndarray,
                                      predicted_state: np.ndarray,
                                      prediction_horizon: float) -> float:
        """Calculate confidence in state prediction."""
        # Get prediction variance
        _, std = self.state_transition_model.predict(
            current_state.reshape(1, -1),
            return_std=True
        )
        
        # Calculate normalized confidence score
        confidence = 1.0 / (1.0 + np.mean(std))
        
        # Decay confidence with prediction horizon
        time_factor = np.exp(-prediction_horizon / (2 * self.state_prediction_horizon))
        confidence *= time_factor
        
        return float(confidence)
    
    def _analyze_intervention_effects(self,
                                    trajectories: np.ndarray,
                                    intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the effects of an intervention on state trajectories."""
        # Calculate baseline trajectory without intervention
        baseline = self._predict_baseline_trajectory(len(trajectories))
        
        # Calculate effect sizes
        effects = {
            'immediate': float(np.mean(np.abs(trajectories[0] - baseline[0]))),
            'sustained': float(np.mean(np.abs(trajectories - baseline))),
            'peak': float(np.max(np.abs(trajectories - baseline))),
            'time_to_peak': float(np.argmax(np.abs(trajectories - baseline)) * self.update_window)
        }
        
        # Analyze feature-specific effects
        feature_effects = {}
        for name, slice_idx in self.feature_map.items():
            effect = np.mean(np.abs(trajectories[:, slice_idx] - baseline[:, slice_idx]))
            feature_effects[name] = float(effect)
        
        effects['feature_effects'] = feature_effects
        return effects
    
    def _predict_baseline_trajectory(self, n_steps: int) -> np.ndarray:
        """Predict baseline trajectory without intervention."""
        trajectories = []
        current = self.feature_scaler.transform([self.current_state.to_vector()])[0]
        
        for _ in range(n_steps):
            next_state = self._predict_next_state(current, None)
            trajectories.append(next_state)
            current = next_state
        
        return np.array(trajectories)
    
    def _encode_intervention(self, intervention: Dict[str, Any]) -> np.ndarray:
        """Encode intervention as feature vector."""
        # Basic encoding - can be extended for more complex interventions
        return np.array([float(intervention.get('intensity', 1.0))])
    
    def _update_state_model(self, new_data: np.ndarray):
        """Update state transition model with new data."""
        if len(new_data) > 1:
            # Partial fit with new data
            self.state_transition_model.fit(
                new_data[:-1],
                new_data[1:]
            )
    
    def _update_response_model(self,
                             new_states: np.ndarray,
                             interventions: np.ndarray):
        """Update intervention response model."""
        if len(new_states) > 1 and len(interventions) == len(new_states) - 1:
            # Combine states with interventions
            X = np.hstack([new_states[:-1], interventions])
            
            # Update model
            self.intervention_response_model.fit(X, new_states[1:])
    
    def _process_observations(self,
                            observations: Dict[str, np.ndarray]) -> List[PatientState]:
        """Process new observations into patient states."""
        states = []
        timestamps = observations.get('timestamps', np.array([]))
        
        if len(timestamps) == 0:
            return states
        
        for i in range(len(timestamps)):
            physiological_state = {}
            trigger_state = {}
            symptom_state = {}
            
            # Extract data
            for key, data in observations.items():
                if key.startswith('phys_'):
                    if len(data.shape) > 1:
                        # Multi-channel data (e.g., EEG)
                        physiological_state[key[5:]] = data[i, :]
                    else:
                        # Single-channel data (e.g., ECG, SC)
                        physiological_state[key[5:]] = np.array([data[i]])
                elif key.startswith('trigger_'):
                    trigger_state[key[8:]] = float(data[i])
                elif key.startswith('symptom_'):
                    symptom_state[key[8:]] = float(data[i])
            
            # Ensure all expected features are present
            for feature in self.feature_map:
                if feature.startswith('phys_') and feature[5:] not in physiological_state:
                    # Add dummy physiological data
                    if feature == 'phys_eeg':
                        physiological_state[feature[5:]] = np.zeros(4)  # 4-channel EEG
                    else:
                        physiological_state[feature[5:]] = np.array([0.0])  # Single channel
                elif feature.startswith('trigger_') and feature[8:] not in trigger_state:
                    trigger_state[feature[8:]] = 0.0
                elif feature.startswith('symptom_') and feature[8:] not in symptom_state:
                    symptom_state[feature[8:]] = 0.0
            
            state = PatientState(
                physiological_state=physiological_state,
                trigger_state=trigger_state,
                symptom_state=symptom_state,
                timestamp=timestamps[i],
                metadata=self.model_metadata.copy()
            )
            states.append(state)
        
        return states
    
    def _load_model_state(self, model_state: Dict[str, Any]):
        """Load model state from dictionary."""
        self.current_state = model_state['current_state']
        self.model_metadata = model_state['model_metadata'].copy()
        self.feature_map = model_state['feature_map'].copy() 
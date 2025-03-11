"""
State Space Models for Physiological Time Series Data.

This module provides theoretical components for state space modeling of physiological time series data,
including linear and non-linear state space models, Kalman filtering, hidden Markov models,
and particle filtering techniques relevant to migraine prediction.
"""

import numpy as np
from scipy import stats
import scipy.linalg as linalg
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import systematic_resample
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

# Optional import of hmm package
try:
    import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from core.theory.base import TheoryComponent


class StateSpaceModeler(TheoryComponent):
    """
    Modeler for theoretical state space properties in physiological time series.
    
    This class provides methods for analyzing and modeling the state space
    dynamics of physiological time series data relevant to migraine prediction,
    including Kalman filtering, hidden Markov models, and particle filtering.
    """
    
    MODEL_METRICS = {
        "types": {
            "linear": "Linear state space models (Kalman filter)",
            "extended": "Extended Kalman filter for nonlinear systems",
            "unscented": "Unscented Kalman filter for highly nonlinear systems",
            "hmm": "Hidden Markov models for discrete state systems",
            "particle": "Particle filters for complex non-Gaussian systems"
        },
        "estimation_methods": ["filtering", "smoothing", "prediction"],
        "criteria": ["aic", "bic", "likelihood", "rmse", "mae"]
    }
    
    def __init__(self, data_type: str = "general", model_type: str = "linear", description: str = ""):
        """
        Initialize the state space modeler with data type and model type.
        
        Args:
            data_type: Type of physiological data (e.g., "eeg", "hrv", "general")
            model_type: Type of state space model to use (e.g., "linear", "hmm")
            description: Optional description
        """
        super().__init__(description)
        self.data_type = data_type.lower()
        self.model_type = model_type.lower()
        self.current_model = None
        self.state_dim = None
        self.obs_dim = None
        
        # Default parameters for different model types
        self.model_parameters = self._initialize_model_parameters(model_type)
        
    def _initialize_model_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        Initialize default parameters for the given model type.
        
        Args:
            model_type: Type of state space model
            
        Returns:
            Dictionary of default parameters
        """
        if model_type == "linear":
            return {
                "state_dim": 2,          # Default state dimension
                "process_noise": 0.01,   # Default process noise variance
                "measurement_noise": 0.1, # Default measurement noise variance
                "dt": 1.0                # Default time step
            }
        elif model_type == "extended":
            return {
                "state_dim": 2,
                "process_noise": 0.01,
                "measurement_noise": 0.1,
                "dt": 1.0,
                "linearization_method": "jacobian"
            }
        elif model_type == "unscented":
            return {
                "state_dim": 2,
                "process_noise": 0.01,
                "measurement_noise": 0.1,
                "dt": 1.0,
                "alpha": 0.1,            # UKF spread parameter
                "beta": 2.0,             # UKF distribution parameter
                "kappa": 0.0             # UKF secondary scaling parameter
            }
        elif model_type == "hmm":
            return {
                "n_states": 3,           # Default number of hidden states
                "covariance_type": "full",
                "n_iter": 100            # Default max iterations for EM algorithm
            }
        elif model_type == "particle":
            return {
                "n_particles": 100,      # Default number of particles
                "state_dim": 2,
                "process_noise": 0.01,
                "measurement_noise": 0.1,
                "resampling_threshold": 0.5
            }
        else:
            # Default to linear model parameters
            return {
                "state_dim": 2,
                "process_noise": 0.01,
                "measurement_noise": 0.1,
                "dt": 1.0
            }
            
    def analyze(self, time_series: np.ndarray, parameters: Dict[str, Any] = None,
                estimation_method: str = "filtering") -> Dict[str, Any]:
        """
        Analyze time series using state space modeling.
        
        Args:
            time_series: Time series data array
            parameters: Optional parameters to override defaults
            estimation_method: Method for state estimation ("filtering", "smoothing", "prediction")
            
        Returns:
            Dictionary containing state space analysis results
        """
        # Update parameters if provided
        if parameters:
            self.model_parameters.update(parameters)
            
        # Basic validation
        if len(time_series) < 3:
            raise ValueError("Time series must contain at least 3 data points")
        
        # Reshape time series if needed (ensure it's a column vector for observation)
        if time_series.ndim == 1:
            observations = time_series.reshape(-1, 1)
        else:
            observations = time_series
            
        self.obs_dim = observations.shape[1] if observations.ndim > 1 else 1
        
        # Initialize state space model based on type
        if self.model_type == "linear":
            results = self._analyze_linear_kalman(observations, estimation_method)
        elif self.model_type == "extended":
            results = self._analyze_extended_kalman(observations, estimation_method)
        elif self.model_type == "unscented":
            results = self._analyze_unscented_kalman(observations, estimation_method)
        elif self.model_type == "hmm":
            results = self._analyze_hmm(observations)
        elif self.model_type == "particle":
            results = self._analyze_particle_filter(observations)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Add model evaluation metrics
        results["evaluation"] = self._evaluate_model(observations, results)
        
        # Add theoretical insights
        results["theoretical_insights"] = self._get_theoretical_insights(results)
        
        return results
        
    def _analyze_linear_kalman(self, observations: np.ndarray, 
                              estimation_method: str) -> Dict[str, Any]:
        """
        Analyze time series using linear Kalman filter.
        
        Args:
            observations: Time series observations
            estimation_method: Method for state estimation
            
        Returns:
            Dictionary containing Kalman filter results
        """
        # Extract parameters
        state_dim = self.model_parameters["state_dim"]
        dt = self.model_parameters["dt"]
        process_noise = self.model_parameters["process_noise"]
        measurement_noise = self.model_parameters["measurement_noise"]
        
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=state_dim, dim_z=self.obs_dim)
        
        # State transition matrix (constant velocity model by default)
        kf.F = np.array([
            [1, dt],
            [0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([[1, 0]])
        
        # Process noise
        kf.Q = Q_discrete_white_noise(dim=state_dim, dt=dt, var=process_noise)
        
        # Measurement noise
        kf.R = np.eye(self.obs_dim) * measurement_noise
        
        # Initial state and covariance
        kf.x = np.zeros((state_dim, 1))
        kf.P = np.eye(state_dim) * 100  # High initial uncertainty
        
        # Run filter
        filtered_states = []
        filtered_covs = []
        
        for z in observations:
            if estimation_method == "filtering":
                kf.predict()
                kf.update(z)
            elif estimation_method == "prediction":
                kf.predict()
                # For prediction, don't update with measurement
            elif estimation_method == "smoothing":
                kf.predict()
                kf.update(z)
                # Smoothing requires a separate step after all filtering
            
            filtered_states.append(kf.x.copy())
            filtered_covs.append(kf.P.copy())
            
        filtered_states = np.array(filtered_states)
        
        # If smoothing requested, perform RTS smoothing
        if estimation_method == "smoothing":
            smoothed_states, smoothed_covs = self._rts_smoother(
                filtered_states, filtered_covs, kf.F, kf.Q)
            states = smoothed_states
        else:
            states = filtered_states
            
        # Compile results
        results = {
            "model_type": "linear_kalman",
            "states": states,
            "observations": observations,
            "parameters": {
                "F": kf.F.copy(),
                "H": kf.H.copy(),
                "Q": kf.Q.copy(),
                "R": kf.R.copy()
            },
            "state_names": [f"state_{i}" for i in range(state_dim)],
            "method": estimation_method
        }
        
        # Store model for later use
        self.current_model = kf
        
        return results
        
    def _analyze_extended_kalman(self, observations: np.ndarray, 
                                 estimation_method: str) -> Dict[str, Any]:
        """
        Analyze time series using extended Kalman filter for nonlinear systems.
        
        Args:
            observations: Time series observations
            estimation_method: Method for state estimation
            
        Returns:
            Dictionary containing extended Kalman filter results
        """
        # Basic implementation - in a real system this would use a proper nonlinear model
        # For simplicity, we'll use a stub implementation here
        return {
            "model_type": "extended_kalman",
            "states": np.zeros((len(observations), self.model_parameters["state_dim"])),
            "observations": observations,
            "note": "Extended Kalman Filter implementation placeholder - requires specific nonlinear system definition",
            "method": estimation_method
        }
        
    def _analyze_unscented_kalman(self, observations: np.ndarray, 
                                 estimation_method: str) -> Dict[str, Any]:
        """
        Analyze time series using unscented Kalman filter for highly nonlinear systems.
        
        Args:
            observations: Time series observations
            estimation_method: Method for state estimation
            
        Returns:
            Dictionary containing unscented Kalman filter results
        """
        # Basic implementation - in a real system this would use a proper nonlinear model
        # For simplicity, we'll use a stub implementation here
        return {
            "model_type": "unscented_kalman",
            "states": np.zeros((len(observations), self.model_parameters["state_dim"])),
            "observations": observations,
            "note": "Unscented Kalman Filter implementation placeholder - requires specific nonlinear system definition",
            "method": estimation_method
        }
    
    def _analyze_hmm(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Analyze time series using hidden Markov model.
        
        Args:
            observations: Time series observations
            
        Returns:
            Dictionary containing HMM results
        """
        # Extract parameters
        n_states = self.model_parameters["n_states"]
        covariance_type = self.model_parameters["covariance_type"]
        n_iter = self.model_parameters["n_iter"]
        
        # Check if hmm package is available
        if not HMM_AVAILABLE:
            # Return basic implementation with a warning
            return {
                "model_type": "hmm",
                "states": np.zeros((len(observations), 1)),  # Discrete state estimates
                "observations": observations,
                "parameters": {
                    "n_states": n_states,
                    "covariance_type": covariance_type,
                    "n_iter": n_iter
                },
                "warning": "HMM package not available. Install the hmm package to use this feature.",
                "note": "HMM implementation placeholder - requires specific library integration"
            }
        
        # Basic implementation - would use a proper HMM library in production
        return {
            "model_type": "hmm",
            "states": np.zeros((len(observations), 1)),  # Discrete state estimates
            "observations": observations,
            "parameters": {
                "n_states": n_states,
                "covariance_type": covariance_type,
                "n_iter": n_iter
            },
            "note": "HMM implementation placeholder - requires specific library integration"
        }
    
    def _analyze_particle_filter(self, observations: np.ndarray) -> Dict[str, Any]:
        """
        Analyze time series using particle filter.
        
        Args:
            observations: Time series observations
            
        Returns:
            Dictionary containing particle filter results
        """
        # Extract parameters
        n_particles = self.model_parameters["n_particles"]
        state_dim = self.model_parameters["state_dim"]
        process_noise = self.model_parameters["process_noise"]
        measurement_noise = self.model_parameters["measurement_noise"]
        
        # Basic implementation - would use a more sophisticated approach in production
        return {
            "model_type": "particle_filter",
            "states": np.zeros((len(observations), state_dim)),
            "observations": observations,
            "parameters": {
                "n_particles": n_particles,
                "state_dim": state_dim,
                "process_noise": process_noise,
                "measurement_noise": measurement_noise
            },
            "note": "Particle filter implementation placeholder - requires detailed system model"
        }
    
    def _rts_smoother(self, filtered_states: np.ndarray, filtered_covs: List[np.ndarray],
                     F: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Rauch-Tung-Striebel smoother for Kalman filtering.
        
        Args:
            filtered_states: States from forward Kalman filter
            filtered_covs: Covariances from forward Kalman filter
            F: State transition matrix
            Q: Process noise covariance
            
        Returns:
            Tuple of smoothed states and covariances
        """
        n = len(filtered_states)
        state_dim = filtered_states.shape[1]
        
        # Initialize smoothed estimates with the final filtered estimates
        smoothed_states = filtered_states.copy()
        smoothed_covs = filtered_covs.copy()
        
        # Run the smoother backwards
        for t in range(n-2, -1, -1):
            # Predicted covariance for t+1
            P_pred = F @ filtered_covs[t] @ F.T + Q
            
            # Smoother gain
            K = filtered_covs[t] @ F.T @ np.linalg.inv(P_pred)
            
            # Smooth the state and covariance
            smoothed_states[t] = (filtered_states[t] + 
                                 K @ (smoothed_states[t+1] - F @ filtered_states[t]))
            smoothed_covs[t] = (filtered_covs[t] + 
                               K @ (smoothed_covs[t+1] - P_pred) @ K.T)
        
        return smoothed_states, smoothed_covs
            
    def _evaluate_model(self, observations: np.ndarray, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the state space model performance.
        
        Args:
            observations: Original observations
            results: Model results including estimated states
            
        Returns:
            Dictionary of evaluation metrics
        """
        states = results["states"]
        
        # For simplicity, assume the first state component corresponds to the observation
        predicted = states[:, 0] if states.ndim > 1 else states
        actual = observations.flatten()
        
        # Calculate standard metrics
        residuals = actual - predicted
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # Calculate information criteria if possible (simplified)
        n_params = self.model_parameters.get("state_dim", 2) * 2  # Simplified
        n_samples = len(observations)
        
        # Log-likelihood (simplified Gaussian assumption)
        log_likelihood = -n_samples/2 * np.log(2*np.pi*mse) - n_samples/2
        
        # AIC and BIC
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic
        }
            
    def _get_theoretical_insights(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate theoretical insights from state space analysis.
        
        Args:
            results: State space analysis results
            
        Returns:
            List of theoretical insights
        """
        model_type = results["model_type"]
        evaluation = results.get("evaluation", {})
        
        insights = []
        
        # General insights based on model type
        if model_type == "linear_kalman":
            insights.append("Linear Kalman filtering assumes Gaussian noise and linear system dynamics.")
            insights.append("The process follows a Markovian evolution where future states depend only on the current state.")
            
            # Check if the system is stable/observable/controllable
            if "parameters" in results:
                F = results["parameters"].get("F")
                if F is not None:
                    eigenvalues = np.linalg.eigvals(F)
                    if np.all(np.abs(eigenvalues) < 1):
                        insights.append("The system is stable as all eigenvalues of the state transition matrix have magnitude less than 1.")
                    else:
                        insights.append("The system may be unstable as some eigenvalues of the state transition matrix have magnitude greater than or equal to 1.")
        
        elif model_type == "hmm":
            insights.append("Hidden Markov Models represent the system as transitioning between discrete latent states.")
            insights.append("HMMs assume the observation at time t depends only on the hidden state at time t.")
            
        elif model_type == "particle_filter":
            insights.append("Particle filtering uses Monte Carlo approximation to represent non-Gaussian state distributions.")
            insights.append("Particle methods are particularly useful for multi-modal posterior distributions and highly nonlinear systems.")
        
        # Insights based on evaluation metrics
        if "rmse" in evaluation:
            if evaluation["rmse"] < 0.1:
                insights.append("The model achieves high accuracy with low RMSE, suggesting a good fit to the observed dynamics.")
            elif evaluation["rmse"] > 1.0:
                insights.append("High RMSE suggests the model may not fully capture the underlying dynamics or noise characteristics.")
                
        if "aic" in evaluation and "bic" in evaluation:
            insights.append(f"Information criteria (AIC: {evaluation['aic']:.2f}, BIC: {evaluation['bic']:.2f}) can be used for model selection.")
            
        # Physiological-specific insights for migraine data
        if self.data_type == "eeg":
            insights.append("State space modeling of EEG can capture transitions between different brain states relevant to migraine phases.")
        elif self.data_type in ["ecg", "hrv"]:
            insights.append("Cardiovascular dynamics during migraine attacks may show distinct state transitions detectable via state space modeling.")
            
        return insights
        
    def predict(self, time_series: np.ndarray, horizon: int, 
               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict future values and uncertainty bounds for the time series.
        
        Args:
            time_series: Historical time series data
            horizon: Number of steps to predict into the future
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        # First analyze the time series to fit the model
        results = self.analyze(time_series)
        
        # Get the model type
        model_type = results["model_type"]
        
        # Generate predictions based on model type
        if model_type == "linear_kalman":
            return self._predict_kalman(time_series, horizon, confidence_level)
        elif model_type == "hmm":
            return self._predict_hmm(time_series, horizon, confidence_level)
        elif model_type == "particle_filter":
            return self._predict_particle(time_series, horizon, confidence_level)
        else:
            # Generic prediction method for other model types
            return self._predict_generic(time_series, results, horizon, confidence_level)
    
    def _predict_kalman(self, time_series: np.ndarray, horizon: int, 
                        confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using Kalman filter.
        
        Args:
            time_series: Historical time series data
            horizon: Number of steps to predict
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        # Assume self.current_model contains a fitted KalmanFilter
        if self.current_model is None:
            raise ValueError("Model has not been fit. Call analyze() first.")
            
        kf = self.current_model
        
        # Make a copy of the current state and covariance
        x = kf.x.copy()
        P = kf.P.copy()
        
        # Generate predictions
        predictions = []
        covariances = []
        
        for _ in range(horizon):
            # Predict next state
            x = kf.F @ x
            P = kf.F @ P @ kf.F.T + kf.Q
            
            # Expected observation
            predicted_obs = kf.H @ x
            predictions.append(predicted_obs.flatten()[0])  # Extract scalar prediction
            covariances.append(kf.H @ P @ kf.H.T + kf.R)
            
        predictions = np.array(predictions)
        
        # Compute confidence intervals
        z_value = stats.norm.ppf((1 + confidence_level) / 2)
        std_devs = np.array([np.sqrt(np.diag(cov))[0] for cov in covariances])  # Extract scalar standard deviation
        
        # Lower and upper bounds
        lower_bound = predictions - z_value * std_devs
        upper_bound = predictions + z_value * std_devs
        
        confidence_intervals = np.stack([lower_bound, upper_bound], axis=1)
        
        return predictions, confidence_intervals
    
    def _predict_hmm(self, time_series: np.ndarray, horizon: int, 
                    confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Stub implementation for HMM prediction."""
        # Check if hmm package is available
        if not HMM_AVAILABLE:
            # Just return zeros with a warning printed
            print("Warning: HMM package not available. Install the hmm package to use this feature.")
            
        # Placeholder implementation
        predictions = np.zeros(horizon)
        confidence_intervals = np.zeros((horizon, 2))
        return predictions, confidence_intervals
    
    def _predict_particle(self, time_series: np.ndarray, horizon: int, 
                       confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Stub implementation for particle filter prediction."""
        # Placeholder implementation
        predictions = np.zeros(horizon)
        confidence_intervals = np.zeros((horizon, 2))
        return predictions, confidence_intervals
    
    def _predict_generic(self, time_series: np.ndarray, results: Dict[str, Any], 
                      horizon: int, confidence_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """Stub implementation for generic prediction."""
        # Placeholder implementation
        predictions = np.zeros(horizon)
        confidence_intervals = np.zeros((horizon, 2))
        return predictions, confidence_intervals
        
    def compare_models(self, time_series: np.ndarray, 
                      model_types: List[str] = None) -> Dict[str, Any]:
        """
        Compare different state space model types on the same time series.
        
        Args:
            time_series: Time series data array
            model_types: List of model types to compare (defaults to all types)
            
        Returns:
            Dictionary containing comparison results
        """
        if model_types is None:
            model_types = list(self.MODEL_METRICS["types"].keys())
            
        # Store original model type
        original_type = self.model_type
        
        # Results for each model
        results = {}
        for model_type in model_types:
            self.model_type = model_type
            try:
                model_results = self.analyze(time_series)
                results[model_type] = {
                    "evaluation": model_results["evaluation"],
                    "theoretical_insights": model_results["theoretical_insights"]
                }
            except Exception as e:
                results[model_type] = {
                    "error": str(e)
                }
                
        # Restore original model type
        self.model_type = original_type
        
        # Determine best model based on AIC/BIC
        best_model = None
        best_aic = float('inf')
        for model_type, model_results in results.items():
            if "evaluation" in model_results and "aic" in model_results["evaluation"]:
                if model_results["evaluation"]["aic"] < best_aic:
                    best_aic = model_results["evaluation"]["aic"]
                    best_model = model_type
                    
        comparison = {
            "model_results": results,
            "best_model": best_model,
            "comparison_insights": self._generate_comparison_insights(results)
        }
        
        return comparison
    
    def _generate_comparison_insights(self, model_results: Dict[str, Any]) -> List[str]:
        """
        Generate insights from model comparison.
        
        Args:
            model_results: Results from different models
            
        Returns:
            List of comparison insights
        """
        insights = [
            "State space model selection should consider the trade-off between model complexity and fit quality.",
            "Information criteria (AIC/BIC) penalize model complexity to prevent overfitting."
        ]
        
        # Add specific insights based on results
        linear_results = model_results.get("linear", {}).get("evaluation", {})
        hmm_results = model_results.get("hmm", {}).get("evaluation", {})
        
        if "rmse" in linear_results and "rmse" in hmm_results:
            if linear_results["rmse"] < hmm_results["rmse"]:
                insights.append("Linear state space models achieve lower RMSE, suggesting the system dynamics may be approximately linear.")
            else:
                insights.append("HMM achieves lower RMSE, suggesting discrete state transitions may better characterize the system.")
                
        # Add physiological insights
        if self.data_type == "eeg":
            insights.append("EEG during migraine may exhibit both continuous dynamics (Kalman) and discrete state transitions (HMM).")
        elif self.data_type in ["ecg", "hrv"]:
            insights.append("Cardiovascular responses during migraine may show nonlinear dynamics requiring more sophisticated state space models.")
            
        return insights
            
    def get_formal_definition(self) -> str:
        """
        Get the formal mathematical definition of state space models.
        
        Returns:
            A string containing the formal mathematical definition
        """
        if self.model_type == "linear":
            return """
            Linear State Space Model Formal Definition:
            
            A linear state space model is defined by the following equations:
            
            State equation: x_t = Fx_{t-1} + w_t, where w_t ~ N(0, Q)
            Observation equation: z_t = Hx_t + v_t, where v_t ~ N(0, R)
            
            where:
            - x_t is the state vector at time t
            - z_t is the observation vector at time t
            - F is the state transition matrix
            - H is the observation matrix
            - w_t is the process noise with covariance Q
            - v_t is the observation noise with covariance R
            
            The Kalman filter provides the optimal state estimate under linear-Gaussian assumptions.
            """
        elif self.model_type == "hmm":
            return """
            Hidden Markov Model Formal Definition:
            
            A hidden Markov model is defined by:
            
            P(s_t | s_{t-1}) - State transition probability
            P(o_t | s_t) - Emission probability
            
            where:
            - s_t is the hidden state at time t
            - o_t is the observation at time t
            
            The joint probability of a sequence of states and observations is:
            P(s_1:T, o_1:T) = P(s_1)P(o_1|s_1)∏_{t=2}^T P(s_t|s_{t-1})P(o_t|s_t)
            """
        else:
            return f"Formal definition for {self.model_type} state space model not provided" 
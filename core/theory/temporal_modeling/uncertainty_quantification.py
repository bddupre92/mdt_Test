"""
Uncertainty Quantification for Physiological Time Series Data.

This module provides theoretical components for quantifying uncertainty in physiological time series analysis,
including Bayesian inference, confidence intervals, prediction intervals, and error propagation
techniques relevant to migraine prediction.
"""

import numpy as np
from scipy import stats
from scipy.stats import norm, t, chi2
from sklearn.utils import resample
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import pandas as pd

from core.theory.base import TheoryComponent


class UncertaintyQuantifier(TheoryComponent):
    """
    Quantifier for theoretical uncertainty in physiological time series.
    
    This class provides methods for analyzing and quantifying uncertainty
    in physiological time series data relevant to migraine prediction,
    including Bayesian inference, confidence intervals, and error propagation.
    """
    
    UNCERTAINTY_METRICS = {
        "methods": {
            "bayesian": "Bayesian inference with posterior distributions",
            "frequentist": "Frequentist confidence intervals",
            "bootstrap": "Bootstrap-based uncertainty estimation",
            "monte_carlo": "Monte Carlo error propagation"
        },
        "confidence_levels": [0.68, 0.90, 0.95, 0.99],  # Common confidence levels
        "interval_types": ["confidence", "prediction", "credible", "tolerance"],
        "error_types": ["aleatory", "epistemic", "total"]
    }
    
    DEFAULT_PARAMETERS = {
        "confidence_level": 0.95,
        "n_resamples": 1000,
        "n_samples": 1000,
        "prior_type": "uninformative",
        "mcmc_samples": 1000,
        "burn_in": 100,
        "distribution": "normal",
        "two_sided": True,
        "method": "percentile",
        "error_model": "gaussian"
    }
    
    def __init__(self, data_type: str = "general", method: str = "bayesian", description: str = ""):
        """
        Initialize the uncertainty quantifier.
        
        Args:
            data_type: Type of physiological data (e.g., "eeg", "hrv", "migraine", "general")
            method: Method for uncertainty quantification ("bayesian", "frequentist", etc.)
            description: Optional description
        """
        super().__init__(description)
        self.data_type = data_type.lower()
        self.method = method.lower()
        
        # Initialize parameters with defaults and method-specific settings
        self.parameters = self.DEFAULT_PARAMETERS.copy()
        self.parameters.update(self._initialize_parameters(method))
        
    def _initialize_parameters(self, method: str) -> Dict[str, Any]:
        """
        Initialize default parameters for the given method.
        
        Args:
            method: Uncertainty quantification method
            
        Returns:
            Dictionary of default parameters
        """
        if method == "bayesian":
            return {
                "prior_type": "uninformative",
                "mcmc_samples": 1000,
                "burn_in": 100,
                "credible_level": 0.95
            }
        elif method == "frequentist":
            return {
                "confidence_level": 0.95,
                "distribution": "normal",
                "two_sided": True
            }
        elif method == "bootstrap":
            return {
                "n_resamples": 1000,
                "confidence_level": 0.95,
                "method": "percentile"
            }
        elif method == "monte_carlo":
            return {
                "n_samples": 1000,
                "confidence_level": 0.95,
                "error_model": "gaussian"
            }
        else:
            return {
                "confidence_level": 0.95,
                "method": "default"
            }
            
    def compute_confidence_interval(self, data: np.ndarray, confidence_level: float = None,
                                  method: str = None) -> Dict[str, Any]:
        """
        Compute confidence interval for the mean of the data.
        
        Args:
            data: Input data array
            confidence_level: Confidence level (default from parameters)
            method: Method to use (default from initialization)
            
        Returns:
            Dictionary containing confidence interval results
        """
        if confidence_level is None:
            confidence_level = self.parameters["confidence_level"]
        if method is None:
            method = self.method
            
        # Basic validation
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to compute confidence interval")
            
        if method == "bayesian":
            return self._compute_bayesian_interval(data, confidence_level)
        elif method == "bootstrap":
            return self._compute_bootstrap_interval(data, confidence_level)
        else:  # frequentist
            return self._compute_frequentist_interval(data, confidence_level)
            
    def compute_prediction_interval(self, data: np.ndarray, confidence_level: float = None,
                                 method: str = None) -> Dict[str, Any]:
        """
        Compute prediction interval for future observations.
        
        Args:
            data: Input data array
            confidence_level: Confidence level (default from parameters)
            method: Method to use (default from initialization)
            
        Returns:
            Dictionary containing prediction interval results
        """
        if confidence_level is None:
            confidence_level = self.parameters["confidence_level"]
        if method is None:
            method = self.method
            
        # Basic validation
        if len(data) < 2:
            raise ValueError("Need at least 2 data points to compute prediction interval")
            
        # Compute mean and standard deviation
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # Compute prediction interval
        if method == "frequentist":
            # Use t-distribution for prediction interval
            t_value = t.ppf((1 + confidence_level) / 2, df=n-1)
            margin = t_value * std * np.sqrt(1 + 1/n)
            lower = mean - margin
            upper = mean + margin
        else:  # bootstrap or bayesian
            # Use bootstrap for prediction interval
            intervals = self._compute_bootstrap_prediction_interval(data, confidence_level)
            lower = intervals["lower"]
            upper = intervals["upper"]
            
        return {
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence_level,
            "method": method,
            "n_samples": n
        }
        
    def propagate_uncertainty(self, data: np.ndarray, function: Callable,
                            uncertainties: np.ndarray) -> Dict[str, Any]:
        """
        Propagate uncertainties through a function using Monte Carlo simulation.
        
        Args:
            data: Input data array
            function: Function to propagate uncertainties through
            uncertainties: Array of uncertainties for each data point
            
        Returns:
            Dictionary containing uncertainty propagation results
        """
        n_samples = self.parameters.get("n_samples", 1000)
        
        # Generate Monte Carlo samples
        samples = np.zeros((n_samples, len(data)))
        for i in range(len(data)):
            samples[:, i] = np.random.normal(data[i], uncertainties[i], n_samples)
            
        # Propagate through function
        results = np.array([function(sample) for sample in samples])
        
        # Compute statistics
        mean = np.mean(results)
        std = np.std(results)
        
        # Compute confidence interval
        confidence_level = self.parameters["confidence_level"]
        lower, upper = np.percentile(results, [(1-confidence_level)/2*100, 
                                             (1+confidence_level)/2*100])
        
        return {
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence_level,
            "n_samples": n_samples
        }
        
    def compute_error_decomposition(self, data: np.ndarray, model_predictions: np.ndarray,
                                  uncertainties: np.ndarray) -> Dict[str, Any]:
        """
        Decompose total error into aleatory and epistemic uncertainty.
        
        Args:
            data: Observed data array
            model_predictions: Model predictions array
            uncertainties: Uncertainty estimates array
            
        Returns:
            Dictionary containing error decomposition results
        """
        # Compute residuals
        residuals = data - model_predictions
        
        # Aleatory uncertainty (irreducible, random variation)
        aleatory = np.mean(uncertainties**2)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = np.mean(residuals**2) - aleatory
        
        # Total uncertainty
        total = aleatory + epistemic
        
        # Compute relative contributions
        aleatory_fraction = aleatory / total
        epistemic_fraction = epistemic / total
        
        return {
            "total_uncertainty": total,
            "aleatory_uncertainty": aleatory,
            "epistemic_uncertainty": epistemic,
            "aleatory_fraction": aleatory_fraction,
            "epistemic_fraction": epistemic_fraction
        }
        
    def _compute_frequentist_interval(self, data: np.ndarray, 
                                    confidence_level: float) -> Dict[str, Any]:
        """
        Compute frequentist confidence interval.
        
        Args:
            data: Input data array
            confidence_level: Confidence level
            
        Returns:
            Dictionary containing interval results
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # Use t-distribution for small samples
        t_value = t.ppf((1 + confidence_level) / 2, df=n-1)
        margin = t_value * std / np.sqrt(n)
        
        return {
            "mean": mean,
            "std": std,
            "lower": mean - margin,
            "upper": mean + margin,
            "confidence_level": confidence_level,
            "method": "frequentist",
            "n_samples": n
        }
        
    def _compute_bayesian_interval(self, data: np.ndarray, 
                                 confidence_level: float) -> Dict[str, Any]:
        """
        Compute Bayesian credible interval.
        
        Args:
            data: Input data array
            confidence_level: Confidence level
            
        Returns:
            Dictionary containing interval results
        """
        # Simple implementation using normal-normal conjugate prior
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        
        # Use uninformative prior
        prior_mean = mean
        prior_std = std * 10  # Wide prior
        
        # Compute posterior parameters
        posterior_var = 1 / (1/prior_std**2 + n/std**2)
        posterior_mean = posterior_var * (prior_mean/prior_std**2 + n*mean/std**2)
        
        # Compute credible interval
        z_value = norm.ppf((1 + confidence_level) / 2)
        margin = z_value * np.sqrt(posterior_var)
        
        return {
            "mean": posterior_mean,
            "std": np.sqrt(posterior_var),
            "lower": posterior_mean - margin,
            "upper": posterior_mean + margin,
            "confidence_level": confidence_level,
            "method": "bayesian",
            "n_samples": n
        }
        
    def _compute_bootstrap_interval(self, data: np.ndarray, 
                                  confidence_level: float) -> Dict[str, Any]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            data: Input data array
            confidence_level: Confidence level
            
        Returns:
            Dictionary containing interval results
        """
        n_resamples = self.parameters["n_resamples"]
        bootstrap_means = []
        
        for _ in range(n_resamples):
            resample_data = resample(data)
            bootstrap_means.append(np.mean(resample_data))
            
        # Compute percentile interval
        lower, upper = np.percentile(bootstrap_means, 
                                   [(1-confidence_level)/2*100, 
                                    (1+confidence_level)/2*100])
        
        return {
            "mean": np.mean(data),
            "std": np.std(bootstrap_means),
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence_level,
            "method": "bootstrap",
            "n_samples": len(data),
            "n_resamples": n_resamples
        }
        
    def _compute_bootstrap_prediction_interval(self, data: np.ndarray,
                                            confidence_level: float) -> Dict[str, Any]:
        """
        Compute bootstrap prediction interval.
        
        Args:
            data: Input data array
            confidence_level: Confidence level
            
        Returns:
            Dictionary containing interval results
        """
        n_resamples = self.parameters["n_resamples"]
        predictions = []
        
        for _ in range(n_resamples):
            # Resample with replacement
            resample_data = resample(data)
            # Generate new observation from the resampled distribution
            predictions.append(np.random.choice(resample_data))
            
        # Compute percentile interval
        lower, upper = np.percentile(predictions, 
                                   [(1-confidence_level)/2*100, 
                                    (1+confidence_level)/2*100])
        
        return {
            "lower": lower,
            "upper": upper,
            "predictions": predictions
        }
        
    def get_formal_definition(self) -> str:
        """
        Get the formal mathematical definition of uncertainty quantification methods.
        
        Returns:
            A string containing the formal mathematical definition
        """
        if self.method == "bayesian":
            return """
            Bayesian Uncertainty Quantification Formal Definition:
            
            For a parameter θ and data D, the posterior distribution is:
            
            P(θ|D) = P(D|θ)P(θ) / P(D)
            
            where:
            - P(θ|D) is the posterior probability of θ given D
            - P(D|θ) is the likelihood of D given θ
            - P(θ) is the prior probability of θ
            - P(D) is the marginal likelihood
            
            Credible intervals [a, b] satisfy:
            P(a ≤ θ ≤ b|D) = 1 - α
            
            where α is the significance level.
            """
        elif self.method == "frequentist":
            return """
            Frequentist Uncertainty Quantification Formal Definition:
            
            For a sample mean x̄ and standard error SE, the confidence interval is:
            
            [x̄ - t_{α/2,n-1}·SE, x̄ + t_{α/2,n-1}·SE]
            
            where:
            - t_{α/2,n-1} is the t-distribution critical value
            - SE = s/√n for sample standard deviation s and size n
            
            The interpretation is that (1-α)% of such intervals contain the true parameter.
            """
        elif self.method == "bootstrap":
            return """
            Bootstrap Uncertainty Quantification Formal Definition:
            
            For B bootstrap samples, each estimate θ̂*_b is computed from resampled data D*_b.
            
            The bootstrap confidence interval is:
            [q_{α/2}, q_{1-α/2}]
            
            where q_p is the p-th quantile of the bootstrap distribution of θ̂*_b.
            
            This approximates the sampling distribution without parametric assumptions.
            """
        else:
            return "Formal definition not provided for the selected uncertainty quantification method."

    def analyze(self, time_series: np.ndarray, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze uncertainty in time series data.
        
        Args:
            time_series: Time series data array
            parameters: Optional parameters to override defaults
            
        Returns:
            Dictionary containing uncertainty analysis results
        """
        # Update parameters if provided
        if parameters:
            self.parameters.update(parameters)
            
        # Basic validation
        if len(time_series) < 2:
            raise ValueError("Need at least 2 data points for uncertainty analysis")
            
        # Compute confidence interval
        confidence_interval = self.compute_confidence_interval(time_series)
        
        # Compute prediction interval
        prediction_interval = self.compute_prediction_interval(time_series)
        
        # Compute basic error metrics
        mean = np.mean(time_series)
        std = np.std(time_series, ddof=1)
        
        # Compile results
        results = {
            "method": self.method,
            "data_type": self.data_type,
            "statistics": {
                "mean": mean,
                "std": std,
                "n_samples": len(time_series)
            },
            "confidence_interval": confidence_interval,
            "prediction_interval": prediction_interval,
            "parameters": self.parameters
        }
        
        # Add theoretical insights
        results["theoretical_insights"] = self._get_theoretical_insights(results)
        
        return results
        
    def _get_theoretical_insights(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate theoretical insights from uncertainty analysis.
        
        Args:
            results: Uncertainty analysis results
            
        Returns:
            List of theoretical insights
        """
        insights = []
        
        # General insights about uncertainty
        insights.append("Uncertainty quantification is essential for reliable predictions and decision-making.")
        
        # Method-specific insights
        if self.method == "bayesian":
            insights.append("Bayesian methods provide a natural framework for updating uncertainty with new data.")
            insights.append("Credible intervals have a direct probability interpretation, unlike confidence intervals.")
            
        elif self.method == "frequentist":
            insights.append("Frequentist methods rely on sampling distributions and repeated sampling assumptions.")
            insights.append("Confidence intervals are properties of the procedure, not the parameter.")
            
        elif self.method == "bootstrap":
            insights.append("Bootstrap methods make minimal distributional assumptions.")
            insights.append("Resampling provides empirical estimates of sampling distributions.")
            
        # Data-specific insights
        if self.data_type == "migraine":
            insights.append("Uncertainty in migraine prediction must account for both physiological and behavioral variability.")
            insights.append("Prediction intervals are particularly important for anticipating migraine onset windows.")
            
        elif self.data_type in ["eeg", "ecg", "hrv"]:
            insights.append("Physiological signals often exhibit both measurement and intrinsic biological uncertainty.")
            insights.append("Signal-to-noise ratios affect the width of confidence and prediction intervals.")
            
        # Add insights about interval widths
        if "confidence_interval" in results and "prediction_interval" in results:
            ci_width = results["confidence_interval"]["upper"] - results["confidence_interval"]["lower"]
            pi_width = results["prediction_interval"]["upper"] - results["prediction_interval"]["lower"]
            
            if pi_width > ci_width:
                insights.append("Prediction intervals are wider than confidence intervals, reflecting additional uncertainty in individual predictions.")
                
        return insights 
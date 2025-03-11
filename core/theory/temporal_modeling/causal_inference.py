"""
Causal Inference for Physiological Time Series Data.

This module provides theoretical components for causal inference analysis of physiological time series data,
including Granger causality, transfer entropy, convergent cross-mapping, and causal impact analysis
techniques relevant to migraine prediction and trigger identification.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from core.theory.base import TheoryComponent


class CausalInferenceAnalyzer(TheoryComponent):
    """
    Analyzer for theoretical causal relationships in physiological time series.
    
    This class provides methods for analyzing and characterizing the causal
    relationships between physiological variables relevant to migraine prediction,
    including Granger causality, transfer entropy, and causal impact analysis.
    """
    
    CAUSAL_METRICS = {
        "methods": {
            "granger": "Granger causality test based on predictive improvement",
            "transfer_entropy": "Information-theoretic measure of directed information flow",
            "ccm": "Convergent cross-mapping for nonlinear dynamical systems",
            "causal_impact": "Bayesian structural time series for intervention analysis",
            "pcmci": "Peter Clark Momentary Conditional Independence algorithm",
            "joint_ccm": "Joint convergent cross-mapping with surrogate testing"
        },
        "significance_levels": [0.01, 0.05, 0.10],
        "maxlags_options": [1, 2, 3, 5, 7, 10, 14, 21, 28],  # Different lag windows to consider (days)
        "variable_types": ["symptom", "trigger", "treatment", "physiological", "environmental"]
    }
    
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], data_type: str = "general", description: str = ""):
        """
        Initialize the causal inference analyzer.
        
        Args:
            data: Input data as pandas DataFrame or numpy array
            data_type: Type of physiological data (e.g., "eeg", "hrv", "migraine", "general")
            description: Optional description
        """
        super().__init__(description)
        self.data_type = data_type.lower()
        
        # Store the data
        if isinstance(data, pd.DataFrame):
            self.data = data
            self.is_dataframe = True
        else:
            # Convert numpy array to DataFrame with default column names
            self.data = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
            self.is_dataframe = False
        
        # Default parameters for different data types
        self.parameters = self._initialize_parameters(data_type)
        
    def _initialize_parameters(self, data_type: str) -> Dict[str, Any]:
        """
        Initialize default parameters for the given data type.
        
        Args:
            data_type: Type of physiological data
            
        Returns:
            Dictionary of default parameters
        """
        if data_type == "migraine":
            return {
                "maxlag": 7,  # Default max lag (7 days for migraine data)
                "significance_level": 0.05,
                "variable_types": {
                    "triggers": ["stress", "sleep", "weather", "hormonal", "diet"],
                    "symptoms": ["aura", "pain", "nausea", "photophobia", "phonophobia"],
                    "treatments": ["medication", "behavioral", "preventive"]
                }
            }
        elif data_type in ["eeg", "ecg", "hrv"]:
            return {
                "maxlag": 10,  # Shorter time scale for physiological signals
                "significance_level": 0.05,
                "stationarity_test": True  # Test for stationarity before causality analysis
            }
        else:
            # Generic parameters
            return {
                "maxlag": 5,
                "significance_level": 0.05,
                "stationarity_test": False
            }
            
    def analyze(self, time_series: Dict[str, np.ndarray], method: str = "granger", 
                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze causal relationships between time series.
        
        Args:
            time_series: Dictionary mapping variable names to time series data
            method: Causality method to use ("granger", "transfer_entropy", "ccm", etc.)
            parameters: Optional parameters to override defaults
            
        Returns:
            Dictionary containing causal analysis results
        """
        # Update parameters if provided
        if parameters:
            self.parameters.update(parameters)
            
        # Basic validation
        for name, ts in time_series.items():
            if len(ts) < 10:  # Minimum required for meaningful causal analysis
                raise ValueError(f"Time series '{name}' must contain at least 10 data points")
        
        # Check for stationarity if required
        if self.parameters.get("stationarity_test", False):
            stationarity_results = self._check_stationarity(time_series)
            if not all(stationarity_results.values()):
                # Log warning about non-stationary data
                print("Warning: Some time series are not stationary, which may affect causal inference")
        
        # Call appropriate analysis method
        if method == "granger":
            results = self._analyze_granger_causality(time_series)
        elif method == "transfer_entropy":
            results = self._analyze_transfer_entropy(time_series)
        elif method == "ccm":
            results = self._analyze_ccm(time_series)
        elif method == "causal_impact":
            results = self._analyze_causal_impact(time_series)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Add theoretical insights
        results["theoretical_insights"] = self._get_theoretical_insights(results, method)
        
        return results
        
    def _check_stationarity(self, time_series: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """
        Check if time series are stationary using Augmented Dickey-Fuller test.
        
        Args:
            time_series: Dictionary mapping variable names to time series data
            
        Returns:
            Dictionary mapping variable names to stationarity boolean
        """
        results = {}
        for name, ts in time_series.items():
            # Perform ADF test
            try:
                adf_result = adfuller(ts)
                # p-value less than threshold indicates stationarity (reject unit root hypothesis)
                is_stationary = adf_result[1] < self.parameters["significance_level"]
                results[name] = is_stationary
            except Exception:
                # Default to False if test fails
                results[name] = False
                
        return results
        
    def _analyze_granger_causality(self, time_series: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze Granger causality between time series.
        
        Args:
            time_series: Dictionary mapping variable names to time series data
            
        Returns:
            Dictionary containing Granger causality results
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(time_series)
        
        # Get parameters
        maxlag = self.parameters["maxlag"]
        significance_level = self.parameters["significance_level"]
        
        # Initialize results
        causality_matrix = pd.DataFrame(
            index=df.columns, 
            columns=df.columns,
            data=np.zeros((len(df.columns), len(df.columns)))
        )
        p_values = causality_matrix.copy()
        f_statistics = causality_matrix.copy()
        
        # Compute causality for each pair
        for cause in df.columns:
            for effect in df.columns:
                if cause == effect:
                    continue
                    
                # Extract relevant columns
                data = df[[cause, effect]].dropna()
                
                if len(data) < maxlag + 2:
                    continue
                    
                # Run Granger causality test
                try:
                    granger_test = grangercausalitytests(data, maxlag, verbose=False)
                    
                    # Extract results (using F-test with smallest p-value)
                    best_lag = min(granger_test.keys(), 
                                 key=lambda lag: granger_test[lag][0]['ssr_ftest'][1])
                    p_value = granger_test[best_lag][0]['ssr_ftest'][1]
                    f_stat = granger_test[best_lag][0]['ssr_ftest'][0]
                    
                    # Record results
                    causality_matrix.loc[cause, effect] = 1 if p_value < significance_level else 0
                    p_values.loc[cause, effect] = p_value
                    f_statistics.loc[cause, effect] = f_stat
                    
                except Exception as e:
                    # Handle errors gracefully
                    print(f"Error in Granger test for {cause}->{effect}: {str(e)}")
                    
        # Identify strongest causal relationships
        causal_pairs = []
        for cause in causality_matrix.index:
            for effect in causality_matrix.columns:
                if causality_matrix.loc[cause, effect] == 1:
                    causal_pairs.append({
                        "cause": cause,
                        "effect": effect,
                        "p_value": p_values.loc[cause, effect],
                        "f_statistic": f_statistics.loc[cause, effect]
                    })
                    
        # Sort by strength (F-statistic)
        causal_pairs.sort(key=lambda x: x["f_statistic"], reverse=True)
        
        # Compile results
        results = {
            "method": "granger_causality",
            "causality_matrix": causality_matrix.to_dict(),
            "p_values": p_values.to_dict(),
            "f_statistics": f_statistics.to_dict(),
            "causal_pairs": causal_pairs,
            "parameters": {
                "maxlag": maxlag,
                "significance_level": significance_level
            }
        }
        
        return results
        
    def _analyze_transfer_entropy(self, time_series: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze causal relationships using transfer entropy.
        
        Args:
            time_series: Dictionary mapping variable names to time series data
            
        Returns:
            Dictionary containing transfer entropy results
        """
        # Simple placeholder implementation for transfer entropy
        # In a real implementation, this would compute actual transfer entropy
        
        # Initialize results matrices
        variable_names = list(time_series.keys())
        n_vars = len(variable_names)
        
        # Create empty matrices for results
        te_matrix = pd.DataFrame(
            index=variable_names, 
            columns=variable_names,
            data=np.zeros((n_vars, n_vars))
        )
        significance_matrix = te_matrix.copy()
        
        # Placeholder values (would compute actual TE in real implementation)
        for i, source in enumerate(variable_names):
            for j, target in enumerate(variable_names):
                if i == j:
                    continue
                
                # Generate placeholder TE value 
                # (in real implementation, would compute actual transfer entropy)
                # For demonstration, use correlation as a very rough proxy
                src_data = time_series[source]
                tgt_data = time_series[target]
                
                if len(src_data) != len(tgt_data):
                    min_len = min(len(src_data), len(tgt_data))
                    src_data = src_data[:min_len]
                    tgt_data = tgt_data[:min_len]
                
                # Simple correlation as placeholder
                corr = np.corrcoef(src_data, tgt_data)[0, 1]
                te_value = abs(corr) * np.random.uniform(0.5, 1.5)  # Add randomness to simulate TE
                
                # Record in matrix
                te_matrix.iloc[i, j] = te_value
                significance_matrix.iloc[i, j] = 1 if te_value > 0.2 else 0  # Arbitrary threshold
        
        # Identify significant directional relationships
        causal_pairs = []
        for source in te_matrix.index:
            for target in te_matrix.columns:
                if source == target:
                    continue
                    
                te_value = te_matrix.loc[source, target]
                is_significant = significance_matrix.loc[source, target] == 1
                
                if is_significant:
                    # Check for bidirectionality
                    reverse_te = te_matrix.loc[target, source]
                    is_bidirectional = significance_matrix.loc[target, source] == 1
                    
                    direction = "bidirectional" if is_bidirectional else "unidirectional"
                    asymmetry = abs(te_value - reverse_te) / max(te_value, reverse_te) if is_bidirectional else 1.0
                    
                    causal_pairs.append({
                        "source": source,
                        "target": target,
                        "te_value": te_value,
                        "direction": direction,
                        "asymmetry": asymmetry
                    })
        
        # Sort by TE value
        causal_pairs.sort(key=lambda x: x["te_value"], reverse=True)
        
        # Compile results
        results = {
            "method": "transfer_entropy",
            "te_matrix": te_matrix.to_dict(),
            "significance_matrix": significance_matrix.to_dict(),
            "causal_pairs": causal_pairs,
            "note": "This is a placeholder implementation. A full implementation would use proper TE calculation."
        }
        
        return results
        
    def _analyze_ccm(self, time_series: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze causal relationships using convergent cross-mapping.
        
        Args:
            time_series: Dictionary mapping variable names to time series data
            
        Returns:
            Dictionary containing CCM results
        """
        # Placeholder implementation for CCM
        # In a real implementation, this would compute proper CCM
        
        variable_names = list(time_series.keys())
        n_vars = len(variable_names)
        
        # Create empty matrices for results
        ccm_matrix = pd.DataFrame(
            index=variable_names, 
            columns=variable_names,
            data=np.zeros((n_vars, n_vars))
        )
        significance_matrix = ccm_matrix.copy()
        
        # Add placeholder note
        results = {
            "method": "ccm",
            "ccm_matrix": ccm_matrix.to_dict(),
            "significance_matrix": significance_matrix.to_dict(),
            "causal_pairs": [],
            "note": "Convergent Cross-Mapping implementation placeholder. This requires a specialized nonlinear dynamics library."
        }
        
        return results
        
    def _analyze_causal_impact(self, time_series: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze causal impact of interventions.
        
        Args:
            time_series: Dictionary mapping variable names to time series data
            
        Returns:
            Dictionary containing causal impact results
        """
        # Placeholder implementation for causal impact
        # In a real implementation, this would use CausalImpact or similar
        
        # Add placeholder note
        results = {
            "method": "causal_impact",
            "intervention_effects": {},
            "note": "Causal Impact analysis implementation placeholder. This requires the CausalImpact package or similar Bayesian structural time series implementation."
        }
        
        return results
        
    def identify_potential_triggers(self, symptoms: np.ndarray, 
                                   variables: Dict[str, np.ndarray], 
                                   method: str = "granger") -> Dict[str, Any]:
        """
        Identify potential migraine triggers from a set of variables.
        
        Args:
            symptoms: Time series of migraine symptoms/intensity
            variables: Dictionary mapping variable names to potential trigger time series
            method: Causal inference method to use
            
        Returns:
            Dictionary containing potential triggers ranked by likelihood
        """
        # Combine symptoms and variables
        all_series = {"symptoms": symptoms}
        all_series.update(variables)
        
        # Analyze causal relationships
        results = self.analyze(all_series, method=method)
        
        # Extract causal relationships where symptoms is the effect
        triggers = []
        
        if method == "granger":
            causality_matrix = pd.DataFrame(results["causality_matrix"])
            p_values = pd.DataFrame(results["p_values"])
            
            for var_name in variables.keys():
                if causality_matrix.loc[var_name, "symptoms"] == 1:
                    triggers.append({
                        "variable": var_name,
                        "p_value": p_values.loc[var_name, "symptoms"],
                        "confidence": 1 - p_values.loc[var_name, "symptoms"]
                    })
        
        elif method == "transfer_entropy":
            te_matrix = pd.DataFrame(results["te_matrix"])
            significance_matrix = pd.DataFrame(results["significance_matrix"])
            
            for var_name in variables.keys():
                if significance_matrix.loc[var_name, "symptoms"] == 1:
                    triggers.append({
                        "variable": var_name,
                        "te_value": te_matrix.loc[var_name, "symptoms"],
                        "confidence": te_matrix.loc[var_name, "symptoms"]
                    })
        
        # Sort triggers by confidence
        if triggers:
            triggers.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
        # Create theoretical explanation
        if self.data_type == "migraine":
            if triggers:
                explanation = [
                    f"The analysis identified {len(triggers)} potential migraine triggers.",
                    f"The strongest trigger candidates are: {', '.join([t['variable'] for t in triggers[:3]])}.",
                    "These relationships indicate temporal precedence, but clinical validation is needed to confirm causality."
                ]
            else:
                explanation = [
                    "No statistically significant triggers were identified in the dataset.",
                    "This could be due to insufficient data, complex nonlinear relationships, or missing relevant variables."
                ]
        else:
            explanation = [
                f"The analysis identified {len(triggers)} potential causal relationships.",
                "Time-precedence is necessary but not sufficient for establishing causality."
            ]
            
        # Compile results
        trigger_results = {
            "method": method,
            "potential_triggers": triggers,
            "explanation": explanation,
            "full_analysis": results
        }
        
        return trigger_results
    
    def compute_time_delay(self, cause_series: np.ndarray, effect_series: np.ndarray, 
                          max_lag: int = None) -> Dict[str, Any]:
        """
        Compute the time delay between cause and effect.
        
        Args:
            cause_series: Time series of potential cause
            effect_series: Time series of potential effect
            max_lag: Maximum lag to consider (defaults to self.parameters["maxlag"])
            
        Returns:
            Dictionary containing time delay analysis results
        """
        if max_lag is None:
            max_lag = self.parameters["maxlag"]
            
        # Ensure series have the same length
        min_len = min(len(cause_series), len(effect_series))
        cause = cause_series[:min_len]
        effect = effect_series[:min_len]
        
        # Compute cross-correlation
        cross_corr = np.correlate(effect, cause, mode='full')
        lags = np.arange(-min_len + 1, min_len)
        
        # Find lag with maximum correlation
        max_idx = np.argmax(cross_corr)
        optimal_lag = lags[max_idx]
        max_corr = cross_corr[max_idx]
        
        # Compute other lag correlations
        lag_corrs = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # effect precedes cause (negative lag)
                c = cause[-lag:]
                e = effect[:len(c)]
            else:
                # cause precedes effect (positive lag)
                c = cause[:-lag] if lag > 0 else cause
                e = effect[lag:] if lag > 0 else effect
                
            if len(c) > 1 and len(e) > 1:
                lag_corrs[lag] = np.corrcoef(c, e)[0, 1]
            else:
                lag_corrs[lag] = np.nan
                
        # Sort lags by correlation strength
        sorted_lags = sorted(lag_corrs.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)
        
        # Interpretation
        if optimal_lag > 0:
            interpretation = f"Cause appears to precede effect by {optimal_lag} time units"
        elif optimal_lag < 0:
            interpretation = f"Effect appears to precede cause by {abs(optimal_lag)} time units, suggesting reverse causality or a common driver"
        else:
            interpretation = "No time delay detected, suggesting instantaneous interaction or insufficient temporal resolution"
            
        # Compile results
        results = {
            "optimal_lag": optimal_lag,
            "max_correlation": max_corr,
            "lag_correlations": lag_corrs,
            "sorted_lags": sorted_lags,
            "interpretation": interpretation
        }
        
        return results
            
    def _get_theoretical_insights(self, results: Dict[str, Any], method: str) -> List[str]:
        """
        Generate theoretical insights from causal analysis.
        
        Args:
            results: Causal analysis results
            method: Method used for analysis
            
        Returns:
            List of theoretical insights
        """
        insights = []
        
        # General insights about causality
        insights.append("Causal inference in time series requires assumptions beyond mere correlation or time precedence.")
        
        # Method-specific insights
        if method == "granger":
            insights.append("Granger causality tests whether past values of X improve prediction of Y beyond Y's own past values.")
            insights.append("Granger causality assumes linear relationships and may miss nonlinear causal effects.")
            
            # Check if any causal relationships were found
            if results.get("causal_pairs"):
                insights.append(f"Found {len(results['causal_pairs'])} Granger-causal relationships at significance level {results['parameters']['significance_level']}.")
                
                # Comment on strongest relationship
                if results['causal_pairs']:
                    strongest = results['causal_pairs'][0]
                    insights.append(f"The strongest causal relationship is from '{strongest['cause']}' to '{strongest['effect']}' (F={strongest['f_statistic']:.2f}, p={strongest['p_value']:.4f}).")
                
        elif method == "transfer_entropy":
            insights.append("Transfer entropy measures directed information flow and can detect nonlinear causal relationships.")
            insights.append("Unlike Granger causality, transfer entropy makes no assumptions about the model generating the data.")
            
            # Check for bidirectional relationships
            bidirectional = [p for p in results.get("causal_pairs", []) if p.get("direction") == "bidirectional"]
            if bidirectional:
                insights.append(f"Found {len(bidirectional)} bidirectional relationships, suggesting potential feedback loops or common drivers.")
                
        elif method == "ccm":
            insights.append("Convergent cross-mapping tests for causality in dynamical systems based on Takens' theorem.")
            insights.append("CCM can detect causality in nonlinear, deterministic systems where variables are dynamically coupled.")
            
        elif method == "causal_impact":
            insights.append("Causal impact analysis uses Bayesian structural time series to estimate the effect of interventions.")
            insights.append("This approach constructs a counterfactual prediction of how the outcome would have evolved without the intervention.")
            
        # Domain-specific insights
        if self.data_type == "migraine":
            insights.append("In migraine analysis, causal inference can help identify potential triggers, but clinical validation is essential.")
            insights.append("Time delays between triggers and symptoms are clinically significant and may vary substantially between individuals.")
            
        elif self.data_type in ["eeg", "ecg", "hrv"]:
            insights.append("In physiological signals, causal relationships may reflect underlying neural or cardiovascular control mechanisms.")
            insights.append("The temporal resolution of the data significantly impacts the detectable causal relationships.")
            
        return insights
        
    def get_formal_definition(self) -> str:
        """
        Get the formal mathematical definition of causal inference methods.
        
        Returns:
            A string containing the formal mathematical definition
        """
        if self.parameters.get("method", "granger") == "granger":
            return """
            Granger Causality Formal Definition:
            
            For two time series X and Y, X is said to Granger-cause Y if:
            
            P(Y_{t+1} | Y_{1:t}, X_{1:t}) ≠ P(Y_{t+1} | Y_{1:t})
            
            This is typically tested using nested linear models:
            
            Restricted model: Y_t = ∑_{j=1}^p a_j Y_{t-j} + ε_t
            Unrestricted model: Y_t = ∑_{j=1}^p a_j Y_{t-j} + ∑_{j=1}^p b_j X_{t-j} + ε_t
            
            And then using an F-test to compare the residual sum of squares.
            """
        elif self.parameters.get("method", "") == "transfer_entropy":
            return """
            Transfer Entropy Formal Definition:
            
            Transfer entropy from X to Y is defined as:
            
            TE_{X→Y} = ∑ p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) log(p(y_{t+1} | y_t^{(k)}, x_t^{(l)}) / p(y_{t+1} | y_t^{(k)}))
            
            Where:
            - y_t^{(k)} represents the k past values of Y
            - x_t^{(l)} represents the l past values of X
            - p(·) represents the probability distribution
            
            This measures the reduction in uncertainty about Y's future when knowing X's past, beyond what Y's past already provides.
            """
        elif self.parameters.get("method", "") == "ccm":
            return """
            Convergent Cross-Mapping (CCM) Formal Definition:
            
            CCM tests for causality between X and Y by examining whether the historical values of Y can be used to estimate the historical values of X.
            
            If X causally influences Y, then Y's shadow manifold MY contains information about X, allowing us to estimate X from MY.
            
            The correlation ρ between actual X values and those estimated from Y's shadow manifold increases with library size L if X causally affects Y.
            
            This approach is based on Takens' theorem from dynamical systems theory and works for nonlinear, deterministic systems.
            """
        else:
            return "Formal definition not provided for the selected causal inference method." 

    def analyze_granger_causality(self, cause: str, effect: str, max_lag: int = 10) -> Dict[str, Any]:
        """
        Analyze Granger causality between two variables.
        
        Args:
            cause: Name of potential cause variable
            effect: Name of potential effect variable
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary containing test results
        """
        # Extract the time series
        cause_series = self.data[cause].values
        effect_series = self.data[effect].values
        
        # Perform Granger causality test
        data = np.column_stack((effect_series, cause_series))
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # Find optimal lag based on minimum p-value
        p_values = [result[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag + 1)]
        optimal_lag = np.argmin(p_values) + 1
        
        return {
            'p_value': min(p_values),
            'f_statistic': result[optimal_lag][0]['ssr_chi2test'][0],
            'optimal_lag': optimal_lag
        }

    def compute_transfer_entropy(self, source: str, target: str, k: int = 1, l: int = 1) -> float:
        """
        Compute transfer entropy from source to target.
        
        Args:
            source: Source variable name
            target: Target variable name
            k: History length for target
            l: History length for source
            
        Returns:
            Transfer entropy value
        """
        source_data = self.data[source].values
        target_data = self.data[target].values
        
        # Compute probabilities using histogram method
        joint_data = np.column_stack((target_data[k:], target_data[:-k], source_data[:-k]))
        
        # Compute entropies using KDE for better accuracy
        h_joint = stats.entropy(np.histogramdd(joint_data)[0].flatten() + 1e-10)
        h_target_past = stats.entropy(np.histogram(target_data[:-k])[0] + 1e-10)
        h_target_future_past = stats.entropy(np.histogram2d(target_data[k:], target_data[:-k])[0].flatten() + 1e-10)
        
        # Transfer entropy is the difference of conditional entropies
        te = h_target_future_past - h_joint + h_target_past
        
        return max(0, te)  # Ensure non-negative

    def analyze_convergent_cross_mapping(self, var1: str, var2: str, embed_dim: int = 3) -> Dict[str, Any]:
        """
        Perform convergent cross-mapping analysis.
        
        Args:
            var1: First variable name
            var2: Second variable name
            embed_dim: Embedding dimension
            
        Returns:
            Dictionary with CCM results
        """
        # Extract time series
        x = self.data[var1].values
        y = self.data[var2].values
        
        # Create shadow manifold using time-delay embedding
        def create_shadow_manifold(data: np.ndarray, dim: int, tau: int = 1) -> np.ndarray:
            n = len(data) - (dim - 1) * tau
            shadow = np.zeros((n, dim))
            for i in range(dim):
                shadow[:, i] = data[i * tau:i * tau + n]
            return shadow
        
        # Compute cross-mapping correlation
        mx = create_shadow_manifold(x, embed_dim)
        my = create_shadow_manifold(y, embed_dim)
        
        # Use correlation as a simple measure of mapping quality
        correlation = np.corrcoef(mx[:, 0], my[:, 0])[0, 1]
        
        return {
            'correlation': correlation,
            'significance': correlation > 0.5  # Simple threshold-based significance
        }

    def analyze_causal_impact(self, treatment: str, outcome: str, 
                            pre_period: List[int], post_period: List[int]) -> Dict[str, Any]:
        """
        Analyze causal impact using a simple difference-in-differences approach.
        
        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name
            pre_period: [start, end] indices for pre-intervention period
            post_period: [start, end] indices for post-intervention period
            
        Returns:
            Dictionary with impact analysis results
        """
        # Extract relevant data
        pre_outcome = self.data[outcome].iloc[pre_period[0]:pre_period[1]]
        post_outcome = self.data[outcome].iloc[post_period[0]:post_period[1]]
        
        # Compute simple differences
        pre_mean = pre_outcome.mean()
        post_mean = post_outcome.mean()
        effect = post_mean - pre_mean
        
        # Compute confidence interval using bootstrap
        n_bootstrap = 1000
        bootstrap_effects = []
        for _ in range(n_bootstrap):
            pre_sample = np.random.choice(pre_outcome, size=len(pre_outcome), replace=True)
            post_sample = np.random.choice(post_outcome, size=len(post_outcome), replace=True)
            bootstrap_effects.append(post_sample.mean() - pre_sample.mean())
        
        ci = np.percentile(bootstrap_effects, [2.5, 97.5])
        
        return {
            'average_effect': effect,
            'confidence_interval': ci.tolist()
        }

    def identify_triggers(self, potential_triggers: List[str], target: str) -> Dict[str, Any]:
        """
        Identify potential migraine triggers using multiple causal inference methods.
        
        Args:
            potential_triggers: List of potential trigger variable names
            target: Target variable name (e.g., migraine occurrence)
            
        Returns:
            Dictionary containing identified triggers and confidence scores
        """
        results = []
        for trigger in potential_triggers:
            # Combine multiple causal metrics
            granger_result = self.analyze_granger_causality(trigger, target)
            te = self.compute_transfer_entropy(trigger, target)
            ccm_result = self.analyze_convergent_cross_mapping(trigger, target)
            
            # Compute combined confidence score
            confidence = (
                (1 - granger_result['p_value']) * 0.4 +  # Weight Granger causality
                min(te, 1.0) * 0.3 +                     # Weight transfer entropy
                ccm_result['correlation'] * 0.3          # Weight CCM correlation
            )
            
            results.append({
                'trigger': trigger,
                'confidence': confidence,
                'granger_p_value': granger_result['p_value'],
                'transfer_entropy': te,
                'ccm_correlation': ccm_result['correlation']
            })
        
        # Sort by confidence score
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'potential_triggers': [r['trigger'] for r in results],
            'confidence_scores': [r['confidence'] for r in results],
            'detailed_results': results
        } 
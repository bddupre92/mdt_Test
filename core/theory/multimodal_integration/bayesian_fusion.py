"""Bayesian Fusion for Multimodal Integration.

This module implements Bayesian approaches to fusing data from multiple sources,
particularly for physiological signals and contextual information relevant to migraine prediction.

Key features:
1. Bayesian model averaging for combining predictions from different modalities
2. Hierarchical Bayesian modeling for handling data at different temporal resolutions
3. Prior knowledge incorporation from domain expertise
4. Posterior distribution analysis for uncertainty quantification
5. Dynamic updating of fusion parameters as reliability changes
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from scipy import stats
from functools import reduce
import warnings

from .. import base
from . import DataFusionMethod, ModalityData, FusionResult

class BayesianFusion(DataFusionMethod):
    """Bayesian approach to fusing multiple data sources."""
    
    def __init__(self, 
                 fusion_type: str = 'model_averaging',
                 prior_type: str = 'flat',
                 prior_params: Optional[Dict[str, Any]] = None,
                 mcmc_samples: int = 1000,
                 burn_in: int = 100,
                 random_state: Optional[int] = None):
        """Initialize Bayesian fusion model.
        
        Args:
            fusion_type: Type of Bayesian fusion to use
                - 'model_averaging': Bayesian model averaging
                - 'hierarchical': Hierarchical Bayesian modeling
                - 'dynamic': Dynamic Bayesian updating
            prior_type: Type of prior to use
                - 'flat': Flat prior (uniform)
                - 'normal': Normal prior
                - 'informative': Informative prior based on domain knowledge
            prior_params: Parameters for the prior distribution
            mcmc_samples: Number of MCMC samples for posterior estimation
            burn_in: Number of burn-in samples to discard
            random_state: Random seed for reproducibility
        """
        self.fusion_type = fusion_type
        self.prior_type = prior_type
        self.prior_params = prior_params or {}
        self.mcmc_samples = mcmc_samples
        self.burn_in = burn_in
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize fusion parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize fusion parameters based on fusion type."""
        if self.fusion_type == 'model_averaging':
            # Initialize weights for model averaging
            self.weights = None  # Will be determined during fusion
            
        elif self.fusion_type == 'hierarchical':
            # Parameters for hierarchical model
            self.level_params = self.prior_params.get('level_params', {
                'alpha': 1.0,  # Shape parameter for gamma distribution
                'beta': 1.0    # Rate parameter for gamma distribution
            })
            
        elif self.fusion_type == 'dynamic':
            # Parameters for dynamic updating
            self.update_rate = self.prior_params.get('update_rate', 0.1)
            self.forget_factor = self.prior_params.get('forget_factor', 0.95)
            
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def validate_inputs(self, *data_sources: Union[np.ndarray, ModalityData]) -> bool:
        """Validate input data sources for Bayesian fusion.
        
        Args:
            *data_sources: Data sources to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If data sources are invalid or incompatible
        """
        if len(data_sources) < 2:
            raise ValueError("At least two data sources are required for fusion")
        
        # Check if all inputs are either numpy arrays or ModalityData objects
        for i, source in enumerate(data_sources):
            if not isinstance(source, (np.ndarray, ModalityData)):
                raise ValueError(f"Data source {i} must be a numpy array or ModalityData object")
        
        # For model averaging, check if all inputs have the same shape in axis 0
        if self.fusion_type == 'model_averaging':
            # Extract data arrays
            data_arrays = [source.data if isinstance(source, ModalityData) else source 
                           for source in data_sources]
            
            # Check if all arrays have the same number of samples (rows)
            sample_counts = [arr.shape[0] for arr in data_arrays]
            if len(set(sample_counts)) > 1:
                raise ValueError("All data sources must have the same number of samples for model averaging")
        
        return True
    
    def fuse(self, *data_sources: Union[np.ndarray, ModalityData], **kwargs) -> FusionResult:
        """Fuse multiple data sources using Bayesian approaches.
        
        Args:
            *data_sources: Data sources to fuse
            **kwargs: Additional parameters
                - reliability_scores: Dict mapping source index to reliability (0-1)
                - prior_strength: Strength of the prior (higher = stronger prior)
                - output_type: Type of output ('point' or 'distribution')
            
        Returns:
            FusionResult object containing fused data and metadata
        """
        # Validate inputs
        self.validate_inputs(*data_sources)
        
        # Extract kwargs
        reliability_scores = kwargs.get('reliability_scores', None)
        prior_strength = kwargs.get('prior_strength', 1.0)
        output_type = kwargs.get('output_type', 'point')
        
        # Extract data arrays
        data_arrays = [source.data if isinstance(source, ModalityData) else source 
                       for source in data_sources]
        
        # Apply appropriate fusion method
        if self.fusion_type == 'model_averaging':
            fused_data, uncertainty, weights = self._bayesian_model_averaging(
                data_arrays, reliability_scores, prior_strength
            )
            
            # Create source contributions dictionary
            source_contributions = {f"source_{i}": weight for i, weight in enumerate(weights)}
            
        elif self.fusion_type == 'hierarchical':
            fused_data, uncertainty, level_weights = self._hierarchical_fusion(
                data_arrays, reliability_scores, prior_strength
            )
            
            # Create source contributions dictionary
            source_contributions = {f"level_{i}": weight for i, weight in enumerate(level_weights)}
            
        elif self.fusion_type == 'dynamic':
            fused_data, uncertainty, update_info = self._dynamic_fusion(
                data_arrays, reliability_scores, prior_strength
            )
            
            # Create source contributions dictionary
            source_contributions = {
                "update_magnitude": update_info['magnitude'],
                "adaptation_rate": update_info['rate']
            }
        
        # Create metadata
        metadata = {
            'fusion_type': self.fusion_type,
            'prior_type': self.prior_type,
            'output_type': output_type,
            'num_sources': len(data_sources),
            'mcmc_samples': self.mcmc_samples if output_type == 'distribution' else None
        }
        
        # Create and return FusionResult
        return FusionResult(
            fused_data=fused_data,
            uncertainty=uncertainty,
            source_contributions=source_contributions,
            metadata=metadata
        )
    
    def _bayesian_model_averaging(self, 
                                 data_arrays: List[np.ndarray], 
                                 reliability_scores: Optional[Dict[int, float]] = None,
                                 prior_strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Bayesian model averaging across multiple data sources.
        
        Args:
            data_arrays: List of data arrays to fuse
            reliability_scores: Dictionary mapping source index to reliability score (0-1)
            prior_strength: Strength of the prior (higher = stronger prior)
            
        Returns:
            Tuple containing:
                - Fused data array
                - Uncertainty estimates
                - Model weights
        """
        n_sources = len(data_arrays)
        n_samples = data_arrays[0].shape[0]
        
        # Initialize weights based on reliability scores or use uniform weights
        if reliability_scores is not None:
            # Convert reliability scores to weights
            weights = np.ones(n_sources)
            for idx, score in reliability_scores.items():
                if 0 <= idx < n_sources:
                    weights[idx] = max(0.01, score)  # Ensure minimum weight
            
            # Normalize weights
            weights = weights / np.sum(weights)
        else:
            # Use uniform weights
            weights = np.ones(n_sources) / n_sources
        
        # Apply prior strength
        prior = np.ones(n_sources) / n_sources  # Uniform prior
        weights = (weights + prior_strength * prior) / (1 + prior_strength)
        
        # Compute weighted average
        weighted_arrays = [weights[i] * arr for i, arr in enumerate(data_arrays)]
        fused_data = sum(weighted_arrays)
        
        # Compute uncertainty (weighted standard deviation)
        # First calculate squared deviations from fused data
        squared_devs = [(arr - fused_data) ** 2 for arr in data_arrays]
        
        # Weighted average of squared deviations
        variance = sum(weights[i] * dev for i, dev in enumerate(squared_devs))
        uncertainty = np.sqrt(variance)
        
        return fused_data, uncertainty, weights
    
    def _hierarchical_fusion(self, 
                            data_arrays: List[np.ndarray],
                            reliability_scores: Optional[Dict[int, float]] = None,
                            prior_strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform hierarchical Bayesian fusion across multiple data sources.
        
        Args:
            data_arrays: List of data arrays to fuse
            reliability_scores: Dictionary mapping source index to reliability score (0-1)
            prior_strength: Strength of the prior (higher = stronger prior)
            
        Returns:
            Tuple containing:
                - Fused data array
                - Uncertainty estimates
                - Level weights
        """
        n_sources = len(data_arrays)
        n_samples = data_arrays[0].shape[0]
        
        # Extract alpha and beta parameters
        alpha = self.level_params['alpha']
        beta = self.level_params['beta']
        
        # Initialize precision parameters (inverse variance)
        if reliability_scores is not None:
            # Convert reliability scores to precision
            tau = np.ones(n_sources)
            for idx, score in reliability_scores.items():
                if 0 <= idx < n_sources:
                    # Higher reliability = higher precision
                    tau[idx] = alpha / beta * max(0.01, score) / (1 - min(0.99, score))
        else:
            # Sample from prior
            tau = np.random.gamma(alpha, 1/beta, size=n_sources)
        
        # Adjust precision with prior strength
        tau = tau * prior_strength
        
        # Calculate weights proportional to precision
        weights = tau / np.sum(tau)
        
        # Compute weighted average
        weighted_arrays = [weights[i] * arr for i, arr in enumerate(data_arrays)]
        fused_data = sum(weighted_arrays)
        
        # Compute uncertainty
        # In hierarchical model, uncertainty decreases with more sources
        total_precision = np.sum(tau)
        uncertainty = 1.0 / np.sqrt(total_precision) * np.ones_like(fused_data)
        
        return fused_data, uncertainty, weights
    
    def _dynamic_fusion(self, 
                       data_arrays: List[np.ndarray],
                       reliability_scores: Optional[Dict[int, float]] = None,
                       prior_strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Perform dynamic Bayesian fusion with temporal adaptation.
        
        Args:
            data_arrays: List of data arrays to fuse
            reliability_scores: Dictionary mapping source index to reliability score (0-1)
            prior_strength: Strength of the prior (higher = stronger prior)
            
        Returns:
            Tuple containing:
                - Fused data array
                - Uncertainty estimates
                - Update information dictionary
        """
        n_sources = len(data_arrays)
        n_samples = data_arrays[0].shape[0]
        
        # Adjust update rate based on prior strength
        effective_update_rate = self.update_rate / (1 + prior_strength)
        
        # Initialize state if not already initialized
        if not hasattr(self, 'current_state'):
            # Use first source as initial state
            self.current_state = data_arrays[0].copy()
            self.current_uncertainty = np.ones_like(self.current_state)
        
        # Create updated state through weighted combination
        if reliability_scores is not None:
            # Convert reliability scores to weights
            weights = np.ones(n_sources)
            for idx, score in reliability_scores.items():
                if 0 <= idx < n_sources:
                    weights[idx] = max(0.01, score)
            
            # Normalize weights
            weights = weights / np.sum(weights)
        else:
            # Use uniform weights
            weights = np.ones(n_sources) / n_sources
        
        # Compute weighted source data
        weighted_data = sum(weights[i] * arr for i, arr in enumerate(data_arrays))
        
        # Update state using exponential forgetting
        update_magnitude = np.mean(np.abs(weighted_data - self.current_state))
        self.current_state = (self.forget_factor * self.current_state + 
                              effective_update_rate * weighted_data)
        
        # Normalize to maintain scale
        normalization = self.forget_factor + effective_update_rate
        self.current_state = self.current_state / normalization
        
        # Update uncertainty estimate
        source_uncertainties = [np.abs(arr - self.current_state) for arr in data_arrays]
        avg_uncertainty = sum(weights[i] * unc for i, unc in enumerate(source_uncertainties))
        
        # Combine with previous uncertainty using temporal decay
        self.current_uncertainty = (self.forget_factor * self.current_uncertainty +
                                    effective_update_rate * avg_uncertainty) / normalization
        
        # Create update information
        update_info = {
            'magnitude': update_magnitude,
            'rate': effective_update_rate,
            'forget_factor': self.forget_factor
        }
        
        return self.current_state, self.current_uncertainty, update_info
    
    def get_posterior_samples(self, 
                             *data_sources: Union[np.ndarray, ModalityData],
                             n_samples: Optional[int] = None) -> np.ndarray:
        """Generate posterior samples from the Bayesian fusion model.
        
        Args:
            *data_sources: Data sources to fuse
            n_samples: Number of posterior samples to generate (default: self.mcmc_samples)
            
        Returns:
            Array of posterior samples
        """
        # Validate inputs
        self.validate_inputs(*data_sources)
        
        # Use default number of samples if not specified
        if n_samples is None:
            n_samples = self.mcmc_samples
        
        # Extract data arrays
        data_arrays = [source.data if isinstance(source, ModalityData) else source 
                       for source in data_sources]
        
        # Perform fusion to get point estimate and uncertainty
        fused_data, uncertainty, _ = self._bayesian_model_averaging(data_arrays)
        
        # Generate samples from posterior distribution
        # For simplicity, assume normal posterior
        posterior_samples = np.random.normal(
            loc=fused_data,
            scale=uncertainty,
            size=(n_samples,) + fused_data.shape
        )
        
        return posterior_samples[self.burn_in:]  # Discard burn-in samples
    
    def update_fusion_parameters(self, 
                                performance_metrics: Dict[str, Any],
                                learning_rate: float = 0.1) -> None:
        """Update fusion parameters based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            learning_rate: Rate at which to update parameters
            
        Returns:
            None (updates parameters in-place)
        """
        # Only implemented for dynamic fusion
        if self.fusion_type != 'dynamic':
            warnings.warn(f"Parameter updating not implemented for fusion type '{self.fusion_type}'")
            return
        
        # Extract relevant metrics
        prediction_error = performance_metrics.get('prediction_error', 0.0)
        
        # Adapt forget factor based on error
        # Higher error = lower forget factor (faster adaptation)
        if prediction_error > 0:
            error_scale = min(1.0, prediction_error)
            self.forget_factor = max(0.5, min(0.99, 
                                             self.forget_factor - learning_rate * error_scale))
            
        # Adapt update rate based on error
        # Higher error = higher update rate
        self.update_rate = max(0.01, min(0.5, 
                                        self.update_rate + learning_rate * prediction_error))

class HierarchicalBayesianFusion(BayesianFusion):
    """Specialized implementation of hierarchical Bayesian fusion."""
    
    def __init__(self, 
                 hierarchy_levels: int = 2,
                 level_params: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Initialize hierarchical Bayesian fusion model.
        
        Args:
            hierarchy_levels: Number of levels in hierarchy
            level_params: Parameters for each hierarchy level
            **kwargs: Additional parameters for BayesianFusion
        """
        # Set fusion type to hierarchical
        kwargs['fusion_type'] = 'hierarchical'
        
        # Initialize level parameters
        self.hierarchy_levels = hierarchy_levels
        self.level_params = level_params or {}
        
        # Default level parameters if not provided
        if 'alpha' not in self.level_params:
            self.level_params['alpha'] = [1.0] * hierarchy_levels
        if 'beta' not in self.level_params:
            self.level_params['beta'] = [1.0] * hierarchy_levels
            
        # Pass level params to parent class through prior_params
        kwargs['prior_params'] = {'level_params': self.level_params}
        
        # Initialize parent class
        super().__init__(**kwargs)
    
    def fuse_hierarchical(self,
                         data_hierarchy: List[List[np.ndarray]],
                         **kwargs) -> FusionResult:
        """Fuse data with explicit hierarchy structure.
        
        Args:
            data_hierarchy: List of lists, where each inner list contains
                            data sources at the same hierarchy level
            **kwargs: Additional parameters for fusion
            
        Returns:
            FusionResult object
        """
        if len(data_hierarchy) != self.hierarchy_levels:
            raise ValueError(f"Expected {self.hierarchy_levels} hierarchy levels, "
                            f"got {len(data_hierarchy)}")
        
        # Initialize with bottom level
        current_level_data = data_hierarchy[0]
        level_results = []
        
        # Fuse each level, from bottom to top
        for level in range(self.hierarchy_levels):
            # Use specific level parameters
            alpha = self.level_params['alpha'][level]
            beta = self.level_params['beta'][level]
            
            # If we're above the bottom level, include results from previous level
            if level > 0:
                current_level_data = data_hierarchy[level] + [prev_result.fused_data]
            
            # Fuse the current level
            level_result = self.fuse(*current_level_data, **kwargs)
            level_results.append(level_result)
            
            # Store result for next level
            prev_result = level_result
        
        # Return the top-level result
        return level_results[-1]

class DynamicBayesianFusion(BayesianFusion):
    """Specialized implementation of dynamic Bayesian fusion with temporal adaptation."""
    
    def __init__(self,
                 update_rate: float = 0.1,
                 forget_factor: float = 0.95,
                 change_detection_threshold: float = 0.3,
                 **kwargs):
        """Initialize dynamic Bayesian fusion model.
        
        Args:
            update_rate: Rate at which to update the state
            forget_factor: Exponential forgetting factor
            change_detection_threshold: Threshold for detecting changes
            **kwargs: Additional parameters for BayesianFusion
        """
        # Set fusion type to dynamic
        kwargs['fusion_type'] = 'dynamic'
        
        # Store dynamic parameters
        self.update_rate = update_rate
        self.forget_factor = forget_factor
        self.change_detection_threshold = change_detection_threshold
        
        # Initialize state tracking
        self.previous_state = None
        self.state_history = []
        self.change_points = []
        
        # Pass dynamic params to parent class through prior_params
        kwargs['prior_params'] = {
            'update_rate': update_rate,
            'forget_factor': forget_factor
        }
        
        # Initialize parent class
        super().__init__(**kwargs)
    
    def fuse_sequential(self,
                       data_sequence: List[List[np.ndarray]],
                       **kwargs) -> List[FusionResult]:
        """Fuse a sequence of data observations with temporal adaptation.
        
        Args:
            data_sequence: List of data sets, where each data set is a list
                          of data sources at the same time point
            **kwargs: Additional parameters for fusion
            
        Returns:
            List of FusionResult objects, one for each time point
        """
        results = []
        
        # Process each time point
        for t, data_sources in enumerate(data_sequence):
            # Fuse the current data sources
            result = self.fuse(*data_sources, **kwargs)
            
            # Detect changes if we have previous state
            if self.previous_state is not None:
                change_magnitude = np.mean(np.abs(result.fused_data - self.previous_state))
                
                if change_magnitude > self.change_detection_threshold:
                    self.change_points.append(t)
                    
                    # Adaptive update rate based on change magnitude
                    self.update_rate = min(0.5, self.update_rate * (1 + change_magnitude))
                else:
                    # Gradually decrease update rate during stable periods
                    self.update_rate = max(0.01, self.update_rate * 0.99)
                
                # Update parameters
                self.update_fusion_parameters({'prediction_error': change_magnitude})
            
            # Store current state
            self.previous_state = result.fused_data.copy()
            self.state_history.append(result)
            
            # Add to results
            results.append(result)
        
        return results 
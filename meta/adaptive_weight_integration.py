"""
Adaptive Weight Integration Module

This module provides integration between the Meta_Learner, DriftDetector, and 
expert models for quality-aware adaptive weighting.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from meta_optimizer.drift_detection.drift_detector import DriftDetector

# Configure logging
logger = logging.getLogger(__name__)

class AdaptiveWeightIntegration:
    """
    Coordinates the integration between Meta_Learner and quality-aware weighting 
    based on data quality and drift detection.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        drift_threshold: float = 0.05,
        significance_level: float = 0.05,
        drift_method: str = 'ks',
        quality_impact: float = 0.4,
        drift_impact: float = 0.3,
        history_weight: float = 0.7
    ):
        """
        Initialize the adaptive weight integration.
        
        Args:
            window_size: Size of window for drift detection
            drift_threshold: Threshold for drift detection
            significance_level: Significance level for statistical tests
            drift_method: Detection method (ks, ad, error_rate, window_comparison)
            quality_impact: Impact factor of quality on expert weights (0-1)
            drift_impact: Impact factor of drift on expert weights (0-1)
            history_weight: Weight given to historical quality metrics
        """
        self.drift_detector = DriftDetector(
            window_size=window_size,
            drift_threshold=drift_threshold,
            significance_level=significance_level,
            method=drift_method
        )
        
        self.quality_impact = quality_impact
        self.drift_impact = drift_impact
        self.history_weight = history_weight
        self.domain_data_quality = {}
        self.reference_windows = {}
        
    def register_meta_learner(self, meta_learner):
        """
        Register a Meta_Learner with this integration.
        
        Args:
            meta_learner: Meta_Learner instance to register
        """
        self.meta_learner = meta_learner
        meta_learner.drift_detector = self.drift_detector
        
        # Inject quality-aware methods if they don't exist
        if not hasattr(meta_learner, 'domain_data_quality'):
            meta_learner.domain_data_quality = self.domain_data_quality
            
        logger.info(f"Registered Meta_Learner with AdaptiveWeightIntegration")
    
    def update_quality_metrics(self, X: np.ndarray, y: np.ndarray, domain: str):
        """
        Update data quality metrics for a specific domain.
        
        Args:
            X: Feature data
            y: Target values
            domain: Data domain (physiological, behavioral, environmental, etc.)
        
        Returns:
            Dict of updated quality metrics
        """
        if domain not in self.domain_data_quality:
            self.domain_data_quality[domain] = {
                'completeness': 0.0,
                'consistency': 0.0,
                'recency': 1.0,
                'drift_stability': 1.0,  # Higher is better (less drift)
                'overall': 0.0
            }
            
        quality = self.domain_data_quality[domain]
        
        # Calculate completeness (percentage of non-missing values)
        missing_rate = np.isnan(X).sum() / X.size if X.size > 0 else 0
        quality['completeness'] = 1.0 - missing_rate
        
        # Calculate consistency (variance in feature scales)
        feature_stds = np.nanstd(X, axis=0) if X.shape[0] > 1 else np.zeros(X.shape[1])
        if len(feature_stds) > 0 and not np.all(np.isnan(feature_stds)):
            # Coefficient of variation between features indicates inconsistency
            valid_stds = feature_stds[~np.isnan(feature_stds)]
            if len(valid_stds) > 0 and np.mean(valid_stds) > 0:
                cv = np.std(valid_stds) / np.mean(valid_stds)
                quality['consistency'] = max(0.0, 1.0 - min(1.0, cv))
        
        # Recency is always 1.0 for new data
        quality['recency'] = 1.0
        
        # Check for drift if we have a reference window
        if domain in self.reference_windows:
            reference_window = self.reference_windows[domain]
            try:
                is_drift, drift_score, _ = self.drift_detector.detect_drift(
                    current_window_X=X
                )
                
                # Update drift stability score
                if is_drift:
                    # Exponential decay based on history weight
                    quality['drift_stability'] = quality['drift_stability'] * self.history_weight + \
                                                (1.0 - drift_score) * (1.0 - self.history_weight)
                    logger.info(f"Drift detected in {domain} data, adjusting stability score to {quality['drift_stability']:.3f}")
            except Exception as e:
                logger.error(f"Error in drift detection for {domain}: {str(e)}")
        else:
            # Store as reference window for this domain
            self.reference_windows[domain] = X
        
        # Calculate overall quality score
        quality['overall'] = (quality['completeness'] * 0.3 + 
                             quality['consistency'] * 0.2 + 
                             quality['recency'] * 0.2 + 
                             quality['drift_stability'] * 0.3)
        
        # Update Meta_Learner if registered
        if hasattr(self, 'meta_learner') and self.meta_learner is not None:
            self.meta_learner.domain_data_quality[domain] = quality
            
        logger.debug(f"{domain} quality metrics: {quality}")
        return quality
    
    def adjust_weights_by_quality(self, weights: Dict[int, float], experts: Dict[int, Any]) -> Dict[int, float]:
        """
        Adjust weights based on data quality metrics.
        
        Args:
            weights: Base weights dictionary mapping expert IDs to weights
            experts: Dictionary mapping expert IDs to expert objects
            
        Returns:
            Dictionary of quality-adjusted weights
        """
        adjusted_weights = weights.copy()
        
        for expert_id, expert in experts.items():
            specialty = getattr(expert, 'specialty', 'general')
            
            # Get quality score for this domain if available
            domain_quality = self.domain_data_quality.get(specialty, {}).get('overall', 0.8)
            
            # Quality score ranges from 0-1, transform to adjustment factor
            # A quality of 1.0 gives full weight, 0.0 reduces weight significantly
            quality_factor = 0.5 + (0.5 * domain_quality)
            
            # Apply quality adjustment
            adjusted_weights[expert_id] *= (1.0 - self.quality_impact + (self.quality_impact * quality_factor))
            
            logger.debug(f"Expert {expert_id} ({specialty}) quality adjustment: {domain_quality:.2f} → factor: {quality_factor:.2f}")
            
        return adjusted_weights
    
    def prepare_context_with_quality(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context dictionary with quality metrics for weight prediction.
        
        Args:
            base_context: Base context dictionary with data type flags
            
        Returns:
            Enhanced context with quality metrics and recent data windows
        """
        enhanced_context = base_context.copy()
        
        # Add quality metrics
        enhanced_context['quality_metrics'] = {
            domain: metrics.get('overall', 0.8) 
            for domain, metrics in self.domain_data_quality.items()
        }
        
        # Add recent data for drift detection
        enhanced_context['recent_data'] = {
            domain: {
                'reference_window': self.reference_windows.get(domain),
                'current_window': None  # Will be populated during use
            }
            for domain in self.reference_windows.keys()
        }
        
        return enhanced_context
    
    def update_for_domain(self, X: np.ndarray, y: np.ndarray, domain: str):
        """
        Update both quality metrics and drift detection for a specific domain.
        
        Args:
            X: Feature data
            y: Target values
            domain: Data domain
            
        Returns:
            Dict of quality metrics and bool indicating if drift was detected
        """
        # Update quality metrics
        quality = self.update_quality_metrics(X, y, domain)
        
        # Check for drift
        is_drift = False
        if domain in self.reference_windows:
            try:
                is_drift, drift_score, p_value = self.drift_detector.detect_drift(
                    current_window_X=X
                )
                
                if is_drift:
                    logger.info(f"Drift detected in {domain} data: score={drift_score:.3f}, p-value={p_value:.3f}")
                    
                    # Update reference window after detecting drift
                    self.reference_windows[domain] = X
            except Exception as e:
                logger.error(f"Error in drift detection for {domain}: {str(e)}")
        else:
            # Store initial reference window
            self.reference_windows[domain] = X
        
        # If Meta_Learner is registered, update its state
        if hasattr(self, 'meta_learner') and self.meta_learner is not None:
            # Update Meta_Learner's drift detector if needed
            if self.meta_learner.drift_detector is None:
                self.meta_learner.drift_detector = self.drift_detector
        
        return quality, is_drift

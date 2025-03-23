"""Pattern Recognition Framework

This module provides tools for pattern recognition in physiological signals,
including feature extraction and pattern classification.

The framework consists of two main components:
1. Feature Extraction - For extracting meaningful features from raw signals
2. Pattern Classification - For identifying and classifying patterns in the extracted features

Components
----------
- FeatureExtractor: Base class for all feature extraction methods
- PatternClassifier: Base class for all pattern classification methods
- TimeDomainFeatures: Time-domain feature extraction
- FrequencyDomainFeatures: Frequency-domain feature extraction
- StatisticalFeatures: Statistical feature computation
- PhysiologicalFeatures: Physiological signal-specific features

Example
-------
>>> from core.theory.pattern_recognition import FeatureExtractor, PatternClassifier
>>> extractor = FeatureExtractor()
>>> classifier = PatternClassifier()
>>> features = extractor.extract(signal)
>>> patterns = classifier.classify(features)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

# Version of the pattern recognition module
__version__ = '0.1.0'

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def validate_input(self, signal: np.ndarray) -> bool:
        """Validate input signal.
        
        Args:
            signal: Input signal to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If signal is invalid
        """
        pass
    
    @abstractmethod
    def extract(self, signal: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Extract features from the signal.
        
        Args:
            signal: Input physiological signal
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of extracted features
        """
        pass

# Import feature extractors
from .feature_extraction import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    StatisticalFeatures,
    PhysiologicalFeatures
)

# Import pattern classifiers
from .pattern_classification import (
    PatternClassifier,
    BinaryClassifier,
    EnsembleClassifier,
    ProbabilisticClassifier
)

__all__ = [
    # Base classes
    'FeatureExtractor',
    'PatternClassifier',
    # Feature extractors
    'TimeDomainFeatures',
    'FrequencyDomainFeatures',
    'StatisticalFeatures',
    'PhysiologicalFeatures',
    # Pattern classifiers
    'BinaryClassifier',
    'EnsembleClassifier',
    'ProbabilisticClassifier'
] 
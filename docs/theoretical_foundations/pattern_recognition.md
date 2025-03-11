# Pattern Recognition Framework

This document provides a comprehensive overview of the pattern recognition framework implemented in the Meta Optimizer system. The framework is designed to analyze physiological signals and identify patterns relevant to migraine prediction.

## Overview

The pattern recognition framework consists of two main components:
1. Feature Extraction
2. Pattern Classification

Each component provides specialized tools for transforming raw physiological signals into meaningful features and classifying them into relevant patterns.

## 1. Feature Extraction

### Theory
The feature extraction component implements various methods to extract meaningful features from physiological signals:

- **Time-domain Analysis**: Statistical and morphological features in the time domain
- **Frequency-domain Analysis**: Spectral and wavelet-based features in the frequency domain
- **Statistical Analysis**: Distribution and variability measures
- **Physiological Analysis**: Signal-specific biomarkers for ECG, EEG, EMG, PPG, GSR, and respiratory signals

### Implementation

The feature extraction framework is built around the abstract `FeatureExtractor` interface:

```python
from core.theory.pattern_recognition import FeatureExtractor

class MyCustomExtractor(FeatureExtractor):
    def validate_input(self, signal):
        # Validate input signal
        pass
        
    def extract(self, signal, **kwargs):
        # Extract features from signal
        pass
```

The framework includes the following specialized feature extractors:

#### Time-domain Features (`TimeDomainFeatures`)
- Peak-to-peak amplitude
- Zero crossings
- Mean absolute value
- Waveform length
- Slope sign changes
- RMS amplitude
- Integrated EMG
- Variance
- Max/min amplitude

```python
from core.theory.pattern_recognition import TimeDomainFeatures

# Initialize with default parameters (window_size=256, overlap=0.5)
extractor = TimeDomainFeatures()

# OR specify custom parameters
extractor = TimeDomainFeatures(
    window_size=512,
    overlap=0.75,
    features=['mean_amplitude', 'peak_to_peak', 'zero_crossings']
)

# Extract features
features = extractor.extract(signal)
```

#### Frequency-domain Features (`FrequencyDomainFeatures`)
- Power spectral density
- Spectral centroid
- Spectral bandwidth
- Dominant frequency
- Band powers (delta, theta, alpha, beta, gamma)
- Spectral flatness
- Median frequency
- Power ratio
- Spectral edge frequency

```python
from core.theory.pattern_recognition import FrequencyDomainFeatures

# Initialize with required parameters
extractor = FrequencyDomainFeatures(sampling_rate=100.0)

# OR with custom parameters
extractor = FrequencyDomainFeatures(
    sampling_rate=250.0,
    window_size=512,
    overlap=0.5,
    features=['spectral_centroid', 'dominant_frequency', 'band_powers'],
    freq_bands={'low': (1, 10), 'mid': (10, 30), 'high': (30, 50)}
)

# Extract features
features = extractor.extract(signal)
```

#### Statistical Features (`StatisticalFeatures`)
- Mean and standard deviation
- Skewness and kurtosis
- Median and IQR
- Entropy measures
- Percentiles (25th, 50th, 75th, 90th, 95th)
- Coefficient of variation
- RMS and energy
- Max/min ratio
- Mean/median absolute deviation

```python
from core.theory.pattern_recognition import StatisticalFeatures

# Initialize with default parameters
extractor = StatisticalFeatures()

# OR specify custom parameters
extractor = StatisticalFeatures(
    window_size=256,
    overlap=0.5,
    features=['mean', 'std', 'skewness', 'kurtosis', 'entropy']
)

# Extract features
features = extractor.extract(signal)
```

#### Physiological Features (`PhysiologicalFeatures`)
Signal-specific features for various physiological signals:

##### ECG (Electrocardiogram)
- Heart rate
- RR intervals
- Heart rate variability (time and frequency domain)
- QRS duration
- QT interval

##### EEG (Electroencephalogram)
- Band powers (delta, theta, alpha, beta, gamma)
- Spectral edge frequency
- Hjorth parameters (activity, mobility, complexity)

##### EMG (Electromyogram)
- RMS amplitude
- Mean absolute value
- Waveform length
- Zero crossings
- Slope sign changes
- Frequency median

##### PPG (Photoplethysmogram)
- Pulse rate
- Peak amplitude
- Reflection index

##### GSR (Galvanic Skin Response)
- Skin conductance level (SCL)
- Skin conductance response (SCR)
- SCR amplitude, rise time, and recovery time

##### RESP (Respiratory)
- Respiratory rate
- Tidal volume
- Inspiration/expiration time
- Minute ventilation

```python
from core.theory.pattern_recognition import PhysiologicalFeatures

# Initialize with required parameters
extractor = PhysiologicalFeatures(
    signal_type='ecg',    # Options: 'ecg', 'eeg', 'emg', 'ppg', 'gsr', 'resp'
    sampling_rate=100.0
)

# OR with custom parameters
extractor = PhysiologicalFeatures(
    signal_type='eeg',
    sampling_rate=250.0,
    window_size=512,
    overlap=0.75
)

# Extract features
features = extractor.extract(signal)
```

### Common Usage Patterns

#### Windowed Analysis
All feature extractors support windowed analysis with overlap:

```python
extractor = TimeDomainFeatures(window_size=256, overlap=0.5)
features = extractor.extract(long_signal)  # Features will be extracted from overlapping windows
```

#### Multi-channel Support
All feature extractors support multi-channel signals:

```python
# Signal shape: (n_samples, n_channels)
multi_channel_signal = np.column_stack([ecg_signal, resp_signal, gsr_signal])
features = extractor.extract(multi_channel_signal)
```

#### Feature Selection
You can select specific features to extract:

```python
extractor = StatisticalFeatures(features=['mean', 'std', 'entropy'])
features = extractor.extract(signal)  # Only extracts mean, std, and entropy
```

## 2. Pattern Classification

### Theory
The pattern classification component implements various classification methods:

- **Binary Classification**: \[ P(y|x) = \frac{1}{1 + e^{-f(x)}} \]
- **Ensemble Methods**: \[ H(x) = \sum_{i=1}^{M} w_i h_i(x) \]
- **Probabilistic Classification**: \[ p(\theta|D) \propto p(D|\theta)p(\theta) \]
- **Uncertainty Estimation**: Bootstrap and ensemble variance

### Implementation

The pattern classification framework is built around the abstract `PatternClassifier` interface:

```python
from core.theory.pattern_recognition import PatternClassifier

class MyCustomClassifier(PatternClassifier):
    def fit(self, X, y):
        # Train the classifier
        pass
        
    def predict(self, X):
        # Make predictions
        pass
```

The framework includes the following specialized classifiers:

#### Binary Classifier

Supports multiple algorithms:
- Random Forest
- Support Vector Machine
- Neural Network
- Gradient Boosting

```python
from core.theory.pattern_recognition import BinaryClassifier

# Initialize with default algorithm (random forest)
clf = BinaryClassifier()

# OR specify algorithm and parameters
clf = BinaryClassifier(
    algorithm='rf',  # Options: 'rf', 'svm', 'mlp', 'gb'
    n_estimators=100,
    max_depth=10
)

# Train classifier
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

#### Ensemble Classifier

Combines multiple base classifiers:
- Weighted voting
- Probability averaging
- Custom weighting schemes

```python
from core.theory.pattern_recognition import EnsembleClassifier, BinaryClassifier

# Create base classifiers
clf1 = BinaryClassifier(algorithm='rf')
clf2 = BinaryClassifier(algorithm='svm')
clf3 = BinaryClassifier(algorithm='mlp')

# Initialize ensemble with different classifiers
ensemble = EnsembleClassifier(
    classifiers=[clf1, clf2, clf3],
    weights=[0.5, 0.3, 0.2]  # Optional weights
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)
```

#### Probabilistic Classifier

Uncertainty-aware classification:
- Bootstrap aggregation
- Uncertainty estimation
- Confidence scoring

```python
from core.theory.pattern_recognition import ProbabilisticClassifier

# Initialize with a base classifier
base_clf = BinaryClassifier(algorithm='rf')
prob_clf = ProbabilisticClassifier(
    base_classifier=base_clf,
    n_bootstrap=100
)

# Train classifier
prob_clf.fit(X_train, y_train)

# Make predictions with uncertainty
probabilities = prob_clf.predict_proba(X_test)
uncertainties = prob_clf.get_uncertainty(X_test)
```

## Integration Guidelines

### Complete Feature Extraction Pipeline

```python
import numpy as np
from core.theory.pattern_recognition import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    StatisticalFeatures,
    PhysiologicalFeatures
)

# Create multi-channel signal (example)
t = np.linspace(0, 10, 1000)
ecg = np.sin(2 * np.pi * 1.1 * t) + 0.1 * np.random.randn(len(t))  # Simulated ECG
resp = np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.random.randn(len(t))  # Simulated respiration

# Initialize extractors
time_features = TimeDomainFeatures(window_size=200, overlap=0.5)
freq_features = FrequencyDomainFeatures(sampling_rate=100, window_size=200, overlap=0.5)
stat_features = StatisticalFeatures(window_size=200, overlap=0.5)
ecg_features = PhysiologicalFeatures(signal_type='ecg', sampling_rate=100)
resp_features = PhysiologicalFeatures(signal_type='resp', sampling_rate=100)

# Extract features
ecg_time_features = time_features.extract(ecg)
ecg_freq_features = freq_features.extract(ecg)
ecg_stat_features = stat_features.extract(ecg)
ecg_phys_features = ecg_features.extract(ecg)
resp_phys_features = resp_features.extract(resp)

# Combine features
combined_features = np.concatenate([
    ecg_time_features['rms'],
    ecg_freq_features['dominant_frequency'],
    ecg_stat_features['skewness'],
    ecg_phys_features['heart_rate'],
    resp_phys_features['respiratory_rate']
], axis=1)
```

### Complete Classification Pipeline

```python
from core.theory.pattern_recognition import (
    BinaryClassifier,
    EnsembleClassifier,
    ProbabilisticClassifier
)
from sklearn.model_selection import train_test_split

# Create dataset from features (example)
X = combined_features
y = np.random.randint(0, 2, size=len(X))  # Binary classification (0/1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create ensemble of classifiers
classifiers = [
    BinaryClassifier(algorithm='rf'),
    BinaryClassifier(algorithm='svm'),
    BinaryClassifier(algorithm='mlp')
]
ensemble = EnsembleClassifier(classifiers=classifiers)

# Add uncertainty estimation
prob_clf = ProbabilisticClassifier(base_classifier=ensemble, n_bootstrap=50)
prob_clf.fit(X_train, y_train)

# Get predictions with uncertainty
predictions = prob_clf.predict(X_test)
probabilities = prob_clf.predict_proba(X_test)
uncertainties = prob_clf.get_uncertainty(X_test)

# Evaluate performance
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.3f}")
```

## Best Practices

### Feature Extraction
1. Always validate input signals before extraction
2. Use appropriate window sizes for different signal types (ECG, EEG, etc.)
3. Consider physiological relevance of features for specific use cases
4. Handle missing or corrupted data appropriately
5. Use multi-channel support for correlated signals (e.g., multiple EEG channels)

### Pattern Classification
1. Scale features before classification to avoid dominance of high-magnitude features
2. Use cross-validation for model evaluation and hyperparameter tuning
3. Consider ensemble methods for robust predictions
4. Quantify uncertainty in predictions, especially for medical applications
5. Interpret models using feature importance or SHAP values

## Performance Considerations

### Computational Efficiency
- Time-domain feature extraction: O(n) time complexity
- Frequency-domain feature extraction: O(n log n) time complexity due to FFT
- Statistical feature extraction: O(n) to O(n log n) for different statistics
- Physiological feature extraction: Varies by signal type, generally O(n) to O(n log n)

### Memory Usage
- Feature extraction: Memory usage scales with window size, overlap, and number of features
- Classification: Model size scales with algorithm complexity and dataset size
- Ensemble methods: Linear increase in memory with number of base classifiers

## Testing and Validation

### Unit Tests
All components include comprehensive unit tests in the `tests/theory/` directory:

```bash
# Run all feature extraction tests
python -m tests.theory.run_feature_extraction_tests

# Run individual component tests
python -m tests.theory.test_time_domain_features
python -m tests.theory.test_frequency_domain_features
python -m tests.theory.test_statistical_features
python -m tests.theory.test_physiological_features
```

### Integration Tests
Test complete pipelines with synthetic and real data using the provided example pipelines.

## Future Extensions

1. **Advanced Feature Extraction**
   - Deep learning-based feature extraction
   - Time-frequency representations (e.g., spectrograms, scalograms)
   - Nonlinear feature extraction (e.g., fractal dimensions)

2. **Enhanced Classification**
   - Multi-task learning for predicting related variables
   - Online learning for adapting to non-stationary signals
   - Active learning for optimizing label acquisition

3. **Uncertainty Quantification**
   - Bayesian neural networks for uncertainty estimation
   - Conformal prediction for rigorous prediction intervals
   - Out-of-distribution detection for anomaly detection

4. **Integration Features**
   - Automated feature selection and importance ranking
   - Hyperparameter optimization for classification algorithms
   - Model interpretability tools for clinical decision support 
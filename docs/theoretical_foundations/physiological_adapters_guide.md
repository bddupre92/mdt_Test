# Physiological Signal Adapters Guide

## Introduction

The Physiological Signal Adapters module provides specialized components for processing various physiological signals in the context of migraine prediction and monitoring. This guide documents the design, implementation, and usage of these adapters, which form a critical part of the migraine digital twin system.

Physiological signals provide valuable information about a patient's state and can serve as early indicators of migraine onset. However, these signals require specialized processing due to their unique characteristics, noise patterns, and information content.

## Signal Processing Framework

### Architecture Overview

The physiological signal processing framework is based on a modular adapter pattern, where each type of physiological signal has a dedicated adapter class implementing a common interface. This design allows for:

1. **Consistent interface**: All adapters follow the same interface, making it easy to process different signal types using a unified approach.
2. **Signal-specific processing**: Each adapter implements specialized processing algorithms tailored to the characteristics of its signal type.
3. **Quality assessment**: Built-in quality evaluation to detect and handle poor-quality signals.
4. **Extensibility**: New signal types can be added by implementing new adapter classes.

### Core Interface

All physiological signal adapters implement the `PhysiologicalSignalAdapter` abstract base class, which defines three key methods:

```python
class PhysiologicalSignalAdapter(ABC):
    @abstractmethod
    def preprocess(self, signal_data: np.ndarray, sampling_rate: float, **kwargs) -> np.ndarray:
        """Preprocess the raw physiological signal data."""
        pass
    
    @abstractmethod
    def extract_features(self, preprocessed_data: np.ndarray, sampling_rate: float, **kwargs) -> Dict[str, np.ndarray]:
        """Extract migraine-relevant features from the preprocessed signal."""
        pass
    
    @abstractmethod
    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """Assess the quality of the physiological signal."""
        pass
```

### Processing Pipeline

The standard processing pipeline for physiological signals consists of:

1. **Quality assessment**: Evaluate the raw signal quality to determine if it's suitable for analysis.
2. **Preprocessing**: Clean and prepare the signal by removing noise, correcting baseline drift, and enhancing relevant components.
3. **Feature extraction**: Compute meaningful features from the preprocessed signal that are relevant to migraine prediction.
4. **Integration**: Combine extracted features with other data sources for multimodal analysis.

## Signal-Specific Implementations

### ECG/HRV Adapter

The `ECGAdapter` processes electrocardiogram (ECG) signals and extracts heart rate variability (HRV) features, which have been shown to correlate with autonomic nervous system activity and stress levels.

#### Preprocessing

ECG preprocessing includes:
- Bandpass filtering (0.5-40 Hz by default) to remove baseline wander and high-frequency noise
- Baseline correction using median filtering
- Optional R-peak enhancement

#### Feature Extraction

The ECG adapter extracts the following HRV features:
- RR intervals: Time between consecutive heartbeats
- SDNN: Standard deviation of NN (normal-to-normal) intervals
- RMSSD: Root mean square of successive differences
- pNN50: Percentage of consecutive RR intervals differing by more than 50ms

#### Quality Assessment

ECG quality is assessed based on:
- Signal-to-noise ratio
- R-peak detectability
- Presence of artifacts or excessive noise

### EEG Adapter

The `EEGAdapter` processes electroencephalogram (EEG) signals, which measure brain electrical activity and can reveal patterns associated with migraine phases.

#### Preprocessing

EEG preprocessing includes:
- Notch filtering to remove power line interference (50/60 Hz)
- Bandpass filtering to isolate relevant frequency ranges
- Artifact detection and handling

#### Feature Extraction

The EEG adapter extracts the following features:
- Power in frequency bands (delta, theta, alpha, beta, gamma)
- Spectral edge frequency
- Spectral entropy
- Signal complexity measures

#### Quality Assessment

EEG quality is assessed based on:
- Signal-to-noise ratio
- Line noise interference
- Presence of movement artifacts
- Signal stationarity

### Skin Conductance Adapter

The `SkinConductanceAdapter` processes electrodermal activity (EDA) signals, which reflect sympathetic nervous system activation and can indicate stress or emotional arousal.

#### Preprocessing

Skin conductance preprocessing includes:
- Low-pass filtering to remove high-frequency noise
- Trend removal to correct for baseline drift
- Signal smoothing to enhance skin conductance responses (SCRs)

#### Feature Extraction

The skin conductance adapter extracts the following features:
- Tonic skin conductance level (SCL)
- Phasic skin conductance responses (SCRs)
- SCR amplitude and rise time
- SCR frequency and density

#### Quality Assessment

Skin conductance quality is assessed based on:
- Signal range appropriateness
- Noise level
- Response detectability
- Contact quality indicators

### Respiratory Adapter

The `RespiratoryAdapter` processes respiratory signals, which provide information about breathing patterns that may change before or during migraines.

#### Preprocessing

Respiratory signal preprocessing includes:
- Bandpass filtering to isolate respiratory frequencies
- Trend removal for baseline correction
- Optional resampling for consistent analysis

#### Feature Extraction

The respiratory adapter extracts the following features:
- Respiratory rate
- Respiratory depth/amplitude
- Inspiratory/expiratory ratio
- Respiratory variability
- Respiratory pattern regularity

#### Quality Assessment

Respiratory signal quality is assessed based on:
- Signal amplitude appropriateness
- Respiratory cycle detectability
- Noise level and artifact presence
- Signal continuity

### Temperature Adapter

The `TemperatureAdapter` processes peripheral temperature signals, which can reflect autonomic nervous system changes associated with migraine.

#### Preprocessing

Temperature signal preprocessing includes:
- Low-pass filtering to remove high-frequency noise
- Outlier detection and handling
- Trend analysis for relative changes

#### Feature Extraction

The temperature adapter extracts the following features:
- Absolute temperature values
- Temperature gradients and rates of change
- Temporal variability measures
- Extremity temperature differentials (when multiple sensors are available)

#### Quality Assessment

Temperature signal quality is assessed based on:
- Signal stability
- Presence of environmental artifacts
- Sensor contact quality
- Physiological plausibility

## Integration with Core Components

### Integration with Multimodal Framework

The physiological signal adapters are designed to integrate seamlessly with the multimodal integration framework:

1. **Data Preparation**: Adapters preprocess raw signals and extract features that serve as input to the multimodal integration layer.

2. **Quality Metadata**: Quality scores from the adapters are passed to the reliability modeling component of the multimodal integration framework, allowing for weighted fusion based on signal quality.

3. **Temporal Alignment**: Adapters handle signal-specific temporal issues before data is temporally aligned in the multimodal layer.

4. **Feature Standardization**: Each adapter ensures that extracted features are properly scaled and normalized for cross-modal compatibility.

### Integration with Digital Twin

The physiological adapters support the digital twin framework by:

1. **Patient State Representation**: Providing processed physiological features that represent key aspects of the patient's current physiological state.

2. **Temporal Prediction**: Enabling the digital twin to track physiological changes over time for predictive modeling.

3. **Intervention Simulation**: Supporting the simulation of how interventions might affect physiological parameters based on historical data patterns.

4. **Personalization**: Adapting processing parameters based on individual patient characteristics stored in the digital twin model.

### Integration with Feature Interaction Analysis

The adapters support feature interaction analysis by:

1. **Cross-Modal Correlation**: Providing cleaned and normalized physiological features that can be correlated with features from other modalities.

2. **Temporal Pattern Detection**: Extracting meaningful temporal patterns that can be analyzed for lead/lag relationships with other modalities.

3. **Trigger Response Analysis**: Enabling the analysis of physiological responses to potential migraine triggers.

## Extending with New Signal Types

### Creating a New Adapter

To add support for a new physiological signal type, follow these steps:

1. **Create a new class** that inherits from `PhysiologicalSignalAdapter`.

```python
class NewSignalAdapter(PhysiologicalSignalAdapter):
    def preprocess(self, signal_data, sampling_rate, **kwargs):
        # Implement signal-specific preprocessing
        pass
    
    def extract_features(self, preprocessed_data, sampling_rate, **kwargs):
        # Implement feature extraction
        pass
    
    def assess_quality(self, signal_data, sampling_rate):
        # Implement quality assessment
        pass
```

2. **Implement preprocessing** tailored to the signal characteristics, including appropriate filtering, artifact removal, and signal enhancement techniques.

3. **Implement feature extraction** to calculate features relevant to migraine prediction. Consider both time-domain and frequency-domain features.

4. **Implement quality assessment** to evaluate signal quality on a scale from 0 to 1.

### Best Practices for Adapter Implementation

1. **Validate inputs** to ensure signal data and sampling rate are valid.

2. **Document parameters** thoroughly, especially kwargs that control processing behavior.

3. **Handle edge cases** such as very short signals, signals with missing data, or extreme values.

4. **Optimize for performance** while maintaining accuracy, especially for real-time applications.

5. **Include unit tests** that verify the adapter works correctly with both ideal and challenging signal data.

### Integration Guidelines

When integrating a new adapter into the system:

1. **Register the adapter** with the appropriate factory or registry if applicable.

2. **Document the adapter's requirements**, such as expected signal characteristics, sampling rate ranges, and any hardware-specific considerations.

3. **Provide default configuration parameters** that work well for typical signal characteristics.

4. **Include example usage** demonstrating how to process signals with the new adapter.

5. **Define feature semantics** clearly so that downstream components can correctly interpret the extracted features.

## Example Usage

### Basic Signal Processing

```python
# Create an ECG adapter
ecg_adapter = ECGAdapter()

# Assess signal quality
quality_score = ecg_adapter.assess_quality(ecg_signal, sampling_rate=250)

# Only process if quality is acceptable
if quality_score > 0.7:
    # Preprocess the signal
    preprocessed_ecg = ecg_adapter.preprocess(ecg_signal, sampling_rate=250)
    
    # Extract features
    ecg_features = ecg_adapter.extract_features(preprocessed_ecg, sampling_rate=250)
    
    # Use features for downstream analysis
    print(f"SDNN: {ecg_features['sdnn']}, RMSSD: {ecg_features['rmssd']}")
```

### Processing Multiple Signal Types

```python
# Create adapters
ecg_adapter = ECGAdapter()
eeg_adapter = EEGAdapter()
eda_adapter = SkinConductanceAdapter()

# Process each signal type
ecg_features = process_signal(ecg_adapter, ecg_signal, ecg_sampling_rate)
eeg_features = process_signal(eeg_adapter, eeg_signal, eeg_sampling_rate)
eda_features = process_signal(eda_adapter, eda_signal, eda_sampling_rate)

# Combine features for multimodal analysis
all_features = {
    'ecg': ecg_features,
    'eeg': eeg_features,
    'eda': eda_features
}

# Helper function
def process_signal(adapter, signal, sampling_rate):
    if adapter.assess_quality(signal, sampling_rate) > 0.6:
        preprocessed = adapter.preprocess(signal, sampling_rate)
        return adapter.extract_features(preprocessed, sampling_rate)
    else:
        return None  # Signal quality too low
```

## Conclusion

The physiological signal adapters provide a robust and extensible framework for processing various types of physiological signals in the context of migraine prediction. By standardizing the interface while allowing for signal-specific processing, the framework supports complex multimodal analysis while maintaining modularity and clarity.

Future enhancements may include additional signal types, more sophisticated feature extraction methods, and adaptive processing parameters based on individual patient characteristics and environmental contexts. 
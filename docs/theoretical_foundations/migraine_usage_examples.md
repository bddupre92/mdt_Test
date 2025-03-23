# Migraine Digital Twin System: Usage Examples

This document provides practical examples of how to use the Migraine Digital Twin system's key features and components. Each example includes detailed code snippets and explanations.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Physiological Signal Processing](#physiological-signal-processing)
3. [Feature Interaction Analysis](#feature-interaction-analysis)
4. [Trigger Identification](#trigger-identification)
5. [Digital Twin Creation and Usage](#digital-twin-creation-and-usage)
6. [Multimodal Integration](#multimodal-integration)
7. [Advanced Use Cases](#advanced-use-cases)

## Basic Setup

### Installing Dependencies

```python
pip install migraine-dt
```

### Basic Configuration

```python
from migraine_dt import MigraineDigitalTwin
from migraine_dt.config import Config

# Create configuration
config = Config(
    data_dir="path/to/data",
    model_type="personalized",
    enable_real_time=True
)

# Initialize the system
mdt = MigraineDigitalTwin(config)
```

## Physiological Signal Processing

### Processing ECG Data

```python
from migraine_dt.adapters import ECGAdapter
from migraine_dt.utils import load_ecg_data

# Load ECG data
ecg_data = load_ecg_data("path/to/ecg.csv")

# Create and configure ECG adapter
ecg_adapter = ECGAdapter(
    sampling_rate=250,  # Hz
    window_size=300,    # seconds
    overlap=0.5         # 50% overlap
)

# Process ECG data
processed_ecg = ecg_adapter.process(ecg_data)
hrv_features = ecg_adapter.extract_hrv_features()

# Access specific HRV metrics
rmssd = hrv_features.rmssd
sdnn = hrv_features.sdnn
```

### Processing EEG Data

```python
from migraine_dt.adapters import EEGAdapter
from migraine_dt.utils import load_eeg_data

# Load EEG data
eeg_data = load_eeg_data("path/to/eeg.csv")

# Create and configure EEG adapter
eeg_adapter = EEGAdapter(
    channels=['Fp1', 'Fp2', 'F3', 'F4'],
    sampling_rate=256,  # Hz
    notch_freq=50      # Hz
)

# Process EEG data
processed_eeg = eeg_adapter.process(eeg_data)
eeg_features = eeg_adapter.extract_features(
    bands=['alpha', 'beta', 'theta', 'delta']
)
```

## Feature Interaction Analysis

### Analyzing Cross-Modal Interactions

```python
from migraine_dt.analysis import FeatureInteractionAnalyzer

# Create analyzer
analyzer = FeatureInteractionAnalyzer()

# Add physiological features
analyzer.add_features(
    hrv_features,
    feature_type='hrv',
    timestamp=hrv_features.timestamp
)

analyzer.add_features(
    eeg_features,
    feature_type='eeg',
    timestamp=eeg_features.timestamp
)

# Analyze interactions
interactions = analyzer.analyze_interactions(
    window_size='1h',
    min_correlation=0.3
)

# Get significant interactions
significant = analyzer.get_significant_interactions(
    p_value_threshold=0.05
)
```

## Trigger Identification

### Detecting Environmental Triggers

```python
from migraine_dt.triggers import TriggerAnalyzer
from migraine_dt.data import EnvironmentalData

# Load environmental data
env_data = EnvironmentalData.from_csv("path/to/environmental.csv")

# Create trigger analyzer
trigger_analyzer = TriggerAnalyzer(
    sensitivity=0.8,
    temporal_window='2h'
)

# Add data sources
trigger_analyzer.add_environmental_data(env_data)
trigger_analyzer.add_physiological_features(hrv_features)

# Identify triggers
triggers = trigger_analyzer.identify_triggers()

# Generate trigger profile
profile = trigger_analyzer.generate_trigger_profile()
```

## Digital Twin Creation and Usage

### Creating a Personalized Digital Twin

```python
from migraine_dt.twin import DigitalTwinBuilder
from migraine_dt.data import PatientData

# Load patient data
patient_data = PatientData.from_files(
    medical_history="path/to/history.json",
    physiological_data="path/to/physio/",
    environmental_data="path/to/env/"
)

# Create digital twin builder
builder = DigitalTwinBuilder()

# Configure and build twin
digital_twin = builder\
    .add_patient_data(patient_data)\
    .add_physiological_adapters([ecg_adapter, eeg_adapter])\
    .add_trigger_analyzer(trigger_analyzer)\
    .set_update_frequency('1h')\
    .build()

# Initialize twin
digital_twin.initialize()
```

### Using the Digital Twin for Prediction

```python
# Get current risk assessment
risk = digital_twin.assess_risk()

# Predict next 24 hours
predictions = digital_twin.predict_next_hours(24)

# Simulate intervention
intervention_impact = digital_twin.simulate_intervention(
    intervention_type="medication",
    dosage="100mg",
    timing="immediate"
)
```

## Multimodal Integration

### Integrating Multiple Data Sources

```python
from migraine_dt.integration import MultimodalIntegrator
from migraine_dt.data import WeatherData, ActivityData

# Create integrator
integrator = MultimodalIntegrator()

# Add data sources
integrator.add_source(
    hrv_features,
    source_type='physiological',
    reliability=0.9
)

integrator.add_source(
    WeatherData.from_api(),
    source_type='environmental',
    reliability=0.8
)

integrator.add_source(
    ActivityData.from_wearable(),
    source_type='behavioral',
    reliability=0.7
)

# Perform integration
integrated_state = integrator.integrate(
    timestamp='now',
    window_size='30m'
)
```

## Advanced Use Cases

### Real-time Monitoring System

```python
from migraine_dt.monitoring import RealTimeMonitor
from migraine_dt.alerts import AlertManager

# Create real-time monitor
monitor = RealTimeMonitor(digital_twin)

# Configure alert manager
alert_manager = AlertManager(
    alert_threshold=0.7,
    notification_channels=['email', 'mobile']
)

# Start real-time monitoring
monitor.start(
    update_frequency='5m',
    alert_manager=alert_manager
)

# Register callback for high-risk situations
@monitor.on_high_risk
def handle_high_risk(risk_level, triggers):
    print(f"High risk detected: {risk_level}")
    print(f"Contributing triggers: {triggers}")
```

### Batch Processing Historical Data

```python
from migraine_dt.processing import BatchProcessor
from migraine_dt.validation import CrossValidator

# Create batch processor
processor = BatchProcessor(digital_twin)

# Process historical data
results = processor.process_historical(
    start_date="2023-01-01",
    end_date="2023-12-31",
    batch_size='1d'
)

# Validate results
validator = CrossValidator(k_folds=5)
validation_metrics = validator.validate(results)
```

### Custom Feature Extraction

```python
from migraine_dt.features import FeatureExtractor
import numpy as np

class CustomFeatureExtractor(FeatureExtractor):
    def extract(self, data):
        return {
            'custom_mean': np.mean(data),
            'custom_std': np.std(data),
            'custom_peaks': len(self.find_peaks(data))
        }
    
    def find_peaks(self, data):
        # Custom peak detection logic
        pass

# Use custom extractor
custom_extractor = CustomFeatureExtractor()
features = custom_extractor.extract(processed_ecg)
```

### Extending the Digital Twin

```python
from migraine_dt.twin import DigitalTwinExtension

class CustomTwinExtension(DigitalTwinExtension):
    def initialize(self):
        # Extension initialization logic
        pass
    
    def update(self, new_data):
        # Custom update logic
        pass
    
    def get_state(self):
        # Return extension-specific state
        pass

# Add extension to digital twin
extension = CustomTwinExtension()
digital_twin.add_extension(extension)
```

## Best Practices and Tips

1. Always validate input data before processing:
```python
from migraine_dt.validation import DataValidator

validator = DataValidator()
if validator.validate(input_data):
    # Process data
    pass
else:
    # Handle validation errors
    pass
```

2. Use appropriate error handling:
```python
from migraine_dt.exceptions import MDTError

try:
    result = digital_twin.process(data)
except MDTError as e:
    logger.error(f"Processing error: {e}")
    # Handle error appropriately
```

3. Implement proper cleanup:
```python
# Using context manager
with RealTimeMonitor(digital_twin) as monitor:
    monitor.start()
    # Monitor will be automatically cleaned up

# Manual cleanup
monitor = RealTimeMonitor(digital_twin)
try:
    monitor.start()
finally:
    monitor.cleanup()
```

4. Optimize performance for large datasets:
```python
from migraine_dt.optimization import DataOptimizer

optimizer = DataOptimizer()
optimized_data = optimizer.optimize(
    large_dataset,
    chunk_size='1h',
    parallel=True
)
```

This documentation provides a comprehensive set of examples for using the Migraine Digital Twin system. For more detailed information about specific components, please refer to the API Reference documentation. 
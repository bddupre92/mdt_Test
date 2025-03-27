# Domain-Specific Preprocessing for Migraine Data

This module provides specialized preprocessing operations for migraine-related data, extending the base preprocessing pipeline with domain-specific knowledge and transformations.

## Overview

The domain-specific preprocessing module contains a set of specialized preprocessing operations designed to handle migraine-related data. These operations focus on extracting meaningful features from various data sources, including:

- Medication data
- Symptom data
- Physiological signals
- Environmental factors
- Comorbidity information
- Temporal patterns
- Advanced feature engineering

## Key Components

### Base Classes

- `PreprocessingOperation`: Abstract base class for all preprocessing operations (defined in `preprocessing_pipeline.py`)
- `PreprocessingPipeline`: Class for chaining multiple preprocessing operations together (defined in `preprocessing_pipeline.py`)

### Domain-Specific Operations

1. **MedicationNormalizer**
   - Normalizes medication names, dosages, and frequencies
   - Maps various medication names to standardized forms
   - Extracts numerical dosage values and units

2. **SymptomExtractor**
   - Extracts symptoms from text data using a symptom dictionary
   - Identifies symptom severity when available
   - Creates binary symptom indicators and severity scores

3. **TemporalPatternExtractor**
   - Extracts temporal features from timestamp data
   - Identifies cyclical patterns (daily, weekly, monthly)
   - Calculates migraine frequency and temporal trends

4. **PhysiologicalSignalProcessor**
   - Processes physiological signals (heart rate, blood pressure, etc.)
   - Normalizes vital signs based on patient-specific baselines
   - Extracts variability and anomaly features

5. **ComorbidityAnalyzer**
   - Analyzes comorbid conditions and their relationship to migraines
   - Calculates comorbidity burden scores
   - Identifies interactions between comorbidities

6. **EnvironmentalTriggerAnalyzer**
   - Analyzes environmental factors as potential migraine triggers
   - Processes weather, pollution, and light exposure data
   - Identifies deviations from baseline that may trigger migraines

7. **AdvancedFeatureEngineer**
   - Implements advanced feature engineering techniques
   - Creates interaction features between related variables
   - Applies clustering to identify patient subgroups
   - Generates polynomial features for non-linear relationships
   - Extracts temporal lag features and rolling statistics

## Usage

### Basic Usage

```python
from data.preprocessing_pipeline import PreprocessingPipeline
from data.domain_specific_preprocessing import (
    MedicationNormalizer,
    SymptomExtractor,
    EnvironmentalTriggerAnalyzer
)

# Initialize preprocessing operations
medication_normalizer = MedicationNormalizer(
    medication_cols=['medication_name'],
    dosage_cols=['dosage'],
    frequency_cols=['frequency']
)

symptom_extractor = SymptomExtractor(
    text_cols=['symptom_notes'],
    symptom_dict=symptom_dictionary
)

environmental_analyzer = EnvironmentalTriggerAnalyzer(
    weather_cols=['temperature', 'pressure'],
    pollution_cols=['aqi'],
    light_cols=['light_exposure'],
    timestamp_col='date'
)

# Create pipeline
pipeline = PreprocessingPipeline(
    operations=[
        medication_normalizer,
        symptom_extractor,
        environmental_analyzer
    ],
    name="Migraine_Preprocessing_Pipeline"
)

# Fit and transform data
pipeline.fit(training_data)
transformed_data = pipeline.transform(new_data)
```

### Complete Example

See the `examples/preprocessing_pipeline_example.py` file for a complete example of how to use all the domain-specific preprocessing operations in a pipeline.

## Testing

Unit tests for all preprocessing operations are available in the `tests/test_domain_specific_preprocessing.py` file. You can run the tests using the `run_preprocessing_tests.py` script:

```bash
python run_preprocessing_tests.py
```

## Integration with MoE Framework

These domain-specific preprocessing operations are designed to integrate seamlessly with the Mixture of Experts (MoE) framework. They provide specialized features that can be used by different expert models, including:

- PhysiologicalExpert
- EnvironmentalExpert
- BehavioralExpert
- MedicationHistoryExpert

The preprocessing operations follow the same interface as the base preprocessing operations, making them compatible with the existing preprocessing pipeline and MoE framework.

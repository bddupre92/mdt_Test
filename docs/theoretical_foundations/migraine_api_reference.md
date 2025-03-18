# Migraine Prediction API Reference

## Introduction

This document provides a comprehensive reference for the core APIs in the migraine prediction framework. It details the interfaces between components, function signatures, parameter specifications, return values, and usage considerations.

The API is organized around several key modules:

1. **Physiological Signal Processing**: Interfaces for processing different physiological signals
2. **Feature Interaction Analysis**: APIs for analyzing relationships between different features
3. **Trigger Identification**: Interfaces for identifying and analyzing migraine triggers
4. **Digital Twin**: APIs for patient digital twin modeling and simulation
5. **Multimodal Integration**: Interfaces for integrating data from multiple sources
6. **Temporal Modeling**: APIs for time-series analysis and causal inference

## Core Abstract Interfaces

### PhysiologicalSignalAdapter

Base interface for all physiological signal adapters.

```python
class PhysiologicalSignalAdapter(ABC):
    @abstractmethod
    def preprocess(self, signal_data: np.ndarray, sampling_rate: float, **kwargs) -> np.ndarray:
        """
        Preprocess the raw physiological signal data.
        
        Parameters
        ----------
        signal_data : np.ndarray
            The raw signal data
        sampling_rate : float
            The sampling rate of the signal in Hz
        **kwargs : dict
            Additional preprocessing parameters
            
        Returns
        -------
        np.ndarray
            The preprocessed signal
        """
        pass
    
    @abstractmethod
    def extract_features(self, preprocessed_data: np.ndarray, sampling_rate: float, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract migraine-relevant features from the preprocessed signal.
        
        Parameters
        ----------
        preprocessed_data : np.ndarray
            The preprocessed signal data
        sampling_rate : float
            The sampling rate of the signal in Hz
        **kwargs : dict
            Additional feature extraction parameters
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of extracted features
        """
        pass
    
    @abstractmethod
    def assess_quality(self, signal_data: np.ndarray, sampling_rate: float) -> float:
        """
        Assess the quality of the physiological signal.
        
        Parameters
        ----------
        signal_data : np.ndarray
            The signal data to assess
        sampling_rate : float
            The sampling rate of the signal in Hz
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        pass
```

**Usage Considerations:**
- Implement this interface for each type of physiological signal
- Use the `preprocess` method before `extract_features`
- Check the quality with `assess_quality` before processing to avoid poor results
- Consider signal-specific parameters through the `**kwargs` arguments

### MigraineFeatureInteractionAnalyzer

Interface for analyzing interactions between different features relevant to migraine prediction.

```python
class MigraineFeatureInteractionAnalyzer(ABC):
    @abstractmethod
    def analyze_prodrome_indicators(self, 
                                   data_sources: Dict[str, ModalityData], 
                                   time_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Analyze prodrome phase indicators across multiple data sources.
        
        Parameters
        ----------
        data_sources : Dict[str, ModalityData]
            Dictionary of data sources keyed by modality name
        time_window : Optional[Tuple[float, float]]
            Optional time window for analysis (start_time, end_time)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of prodrome analysis results
        """
        pass
    
    @abstractmethod
    def detect_trigger_interactions(self, 
                                   triggers: Dict[str, np.ndarray],
                                   physiological_responses: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Detect interactions between multiple triggers and physiological responses.
        
        Parameters
        ----------
        triggers : Dict[str, np.ndarray]
            Dictionary of trigger time series keyed by trigger name
        physiological_responses : Dict[str, np.ndarray]
            Dictionary of physiological response time series
            
        Returns
        -------
        Dict[str, float]
            Interaction strengths between triggers and responses
        """
        pass
    
    @abstractmethod
    def rank_feature_importance(self, 
                              features: Dict[str, np.ndarray], 
                              migraine_occurrences: np.ndarray) -> List[Tuple[str, float]]:
        """
        Rank features by importance for migraine prediction.
        
        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Dictionary of feature time series
        migraine_occurrences : np.ndarray
            Binary array indicating migraine occurrences
            
        Returns
        -------
        List[Tuple[str, float]]
            Ranked list of (feature_name, importance_score) tuples
        """
        pass
```

**Usage Considerations:**
- Use `analyze_prodrome_indicators` to identify early warning signs before migraine onset
- Apply `detect_trigger_interactions` to understand how triggers and physiological responses relate
- Employ `rank_feature_importance` to prioritize features for prediction models

### MigraineTriggerAnalyzer

Interface for analyzing migraine triggers.

```python
class MigraineTriggerAnalyzer(ABC):
    @abstractmethod
    def identify_triggers(self,
                         symptom_data: Dict[str, np.ndarray],
                         potential_triggers: Dict[str, np.ndarray],
                         timestamps: np.ndarray,
                         context_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Identify potential migraine triggers from time series data.
        
        Parameters
        ----------
        symptom_data : Dict[str, np.ndarray]
            Time series of migraine symptoms
        potential_triggers : Dict[str, np.ndarray]
            Time series of potential trigger factors
        timestamps : np.ndarray
            Timestamps for the data points
        context_data : Optional[Dict[str, np.ndarray]]
            Additional contextual data
            
        Returns
        -------
        Dict[str, Any]
            Identified triggers with confidence scores and relationships
        """
        pass
    
    @abstractmethod
    def analyze_trigger_sensitivity(self,
                                  trigger_data: np.ndarray,
                                  symptom_data: Dict[str, np.ndarray],
                                  baseline_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Analyze sensitivity thresholds for triggers.
        
        Parameters
        ----------
        trigger_data : np.ndarray
            Time series of trigger measurements
        symptom_data : Dict[str, np.ndarray]
            Time series of symptoms
        baseline_period : Optional[Tuple[datetime, datetime]]
            Period for baseline comparison
            
        Returns
        -------
        Dict[str, Any]
            Sensitivity analysis results
        """
        pass
```

**Usage Considerations:**
- Use `identify_triggers` to discover potential migraine triggers from time series data
- Apply `analyze_trigger_sensitivity` to determine threshold levels for triggers
- Provide timestamps aligned with the data to enable temporal analysis
- Include contextual data when available to improve trigger identification

### DigitalTwinFoundation

Interface for the patient digital twin modeling and simulation.

```python
class DigitalTwinFoundation(ABC):
    @abstractmethod
    def initialize_twin(self, 
                      patient_history: Dict[str, np.ndarray],
                      patient_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize a digital twin model from patient history.
        
        Parameters
        ----------
        patient_history : Dict[str, np.ndarray]
            Historical patient data across modalities
        patient_metadata : Dict[str, Any]
            Patient metadata including demographics and medical history
            
        Returns
        -------
        Dict[str, Any]
            Initialized digital twin model
        """
        pass
    
    @abstractmethod
    def update_twin(self, 
                  twin_model: Dict[str, Any],
                  new_observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Update a digital twin model with new observations.
        
        Parameters
        ----------
        twin_model : Dict[str, Any]
            Current digital twin model
        new_observations : Dict[str, np.ndarray]
            New observations to incorporate
            
        Returns
        -------
        Dict[str, Any]
            Updated digital twin model
        """
        pass
    
    @abstractmethod
    def simulate_intervention(self, 
                            twin_model: Dict[str, Any],
                            intervention: Dict[str, Any],
                            simulation_duration: float) -> Dict[str, np.ndarray]:
        """
        Simulate the effect of an intervention on the digital twin.
        
        Parameters
        ----------
        twin_model : Dict[str, Any]
            Digital twin model
        intervention : Dict[str, Any]
            Intervention to simulate
        simulation_duration : float
            Duration of simulation in hours
            
        Returns
        -------
        Dict[str, np.ndarray]
            Simulated responses to the intervention
        """
        pass
    
    @abstractmethod
    def assess_twin_accuracy(self, 
                           twin_model: Dict[str, Any],
                           actual_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Assess the accuracy of the digital twin against actual patient data.
        
        Parameters
        ----------
        twin_model : Dict[str, Any]
            Digital twin model
        actual_data : Dict[str, np.ndarray]
            Actual patient data for comparison
            
        Returns
        -------
        Dict[str, float]
            Accuracy metrics for the digital twin
        """
        pass
```

**Usage Considerations:**
- Use `initialize_twin` to create a new digital twin from historical patient data
- Apply `update_twin` periodically with new observations to keep the twin current
- Use `simulate_intervention` to test potential treatments before applying them to the patient
- Employ `assess_twin_accuracy` to validate the twin against actual patient outcomes

## Multimodal Integration Interfaces

### ModalityData

Container class for data from a specific modality.

```python
class ModalityData:
    def __init__(self, 
                 data: np.ndarray, 
                 modality_type: str,
                 timestamps: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 reliability: Optional[float] = None):
        """
        Initialize data for a specific modality.
        
        Parameters
        ----------
        data : np.ndarray
            Data array for the modality
        modality_type : str
            Type of modality (e.g., 'ecg', 'weather', 'sleep')
        timestamps : Optional[np.ndarray]
            Timestamp array aligned with data (if applicable)
        metadata : Optional[Dict[str, Any]]
            Additional information about the data
        reliability : Optional[float]
            Reliability score (0-1) if known
        """
        self.data = data
        self.modality_type = modality_type
        self.timestamps = timestamps
        self.metadata = metadata or {}
        self.reliability = reliability
```

**Usage Considerations:**
- Use this class to encapsulate data from different sources
- Include timestamps whenever possible for temporal alignment
- Add metadata to provide context for the data
- Set reliability if known, or leave it to be determined by reliability models

### DataFusionMethod

Interface for methods that fuse data from multiple modalities.

```python
class DataFusionMethod(ABC):
    @abstractmethod
    def fuse(self, *data_sources: Union[np.ndarray, ModalityData],
             weights: Optional[Dict[str, float]] = None,
             **kwargs) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Fuse multiple data sources into a unified representation.
        
        Parameters
        ----------
        *data_sources : Union[np.ndarray, ModalityData]
            Variable number of data sources to fuse
        weights : Optional[Dict[str, float]]
            Optional dictionary of weights for each modality
        **kwargs : dict
            Additional keyword arguments for specific fusion methods
            
        Returns
        -------
        Union[np.ndarray, Dict[str, Any]]
            Fused data representation
        """
        pass
    
    @abstractmethod
    def assess_fusion_quality(self, fused_data: Union[np.ndarray, Dict[str, Any]],
                            *original_sources: Union[np.ndarray, ModalityData]) -> float:
        """
        Assess the quality of the fusion result.
        
        Parameters
        ----------
        fused_data : Union[np.ndarray, Dict[str, Any]]
            The result of the fusion operation
        *original_sources : Union[np.ndarray, ModalityData]
            Original data sources used in fusion
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        pass
```

**Usage Considerations:**
- Use this interface to combine information from multiple sources
- Provide weights to prioritize more reliable or informative sources
- Assess fusion quality to determine the reliability of the fused result
- Consider specific fusion methods based on data characteristics

### MultimodalMissingDataHandler

Interface for handling missing data in multimodal datasets.

```python
class MultimodalMissingDataHandler(ABC):
    @abstractmethod
    def detect_missing_patterns(self, *data_sources: Union[np.ndarray, ModalityData]) -> Dict[str, Any]:
        """
        Detect patterns in missing data across modalities.
        
        Parameters
        ----------
        *data_sources : Union[np.ndarray, ModalityData]
            Variable number of data sources to analyze
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - patterns: Identified missing data patterns
            - frequencies: Pattern occurrence frequencies
            - modality_stats: Missing data statistics per modality
            - temporal_stats: Temporal distribution of missing data
        """
        pass
    
    @abstractmethod
    def impute(self, *data_sources: Union[np.ndarray, ModalityData], **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Impute missing values in multimodal data.
        
        Parameters
        ----------
        *data_sources : Union[np.ndarray, ModalityData]
            Variable number of data sources with missing values
        **kwargs : dict
            Additional arguments for specific imputation methods
            
        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            - List of imputed data arrays
            - Array of imputation uncertainty estimates
        """
        pass
```

**Usage Considerations:**
- Use `detect_missing_patterns` to understand patterns in missing data
- Apply `impute` to fill in missing values based on cross-modal relationships
- Consider uncertainty estimates when using imputed values
- Be aware that imputation quality may depend on the underlying patterns

## Temporal Modeling Interfaces

### CausalInferenceAnalyzer

Analyzer for causal relationships in time series data.

```python
class CausalInferenceAnalyzer(TheoryComponent):
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], data_type: str = "general", description: str = ""):
        """
        Initialize the causal inference analyzer.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Input data as pandas DataFrame or numpy array
        data_type : str
            Type of physiological data (e.g., "eeg", "hrv", "migraine", "general")
        description : str
            Optional description
        """
        # Implementation details...
    
    def test_granger_causality(self, cause: str, effect: str, max_lag: int = 10) -> Dict[str, Any]:
        """
        Test for Granger causality between two variables.
        
        Parameters
        ----------
        cause : str
            Name of potential causal variable
        effect : str
            Name of potential effect variable
        max_lag : int
            Maximum lag to consider
            
        Returns
        -------
        Dict[str, Any]
            Granger causality test results including:
            - p_values: P-values for each lag
            - f_stats: F-statistics for each lag
            - optimal_lag: Lag with the strongest causal relationship
            - significant: Whether causality is significant at any lag
        """
        # Implementation details...
    
    def calculate_transfer_entropy(self, source: str, target: str, k: int = 1, l: int = 1) -> float:
        """
        Calculate transfer entropy from source to target.
        
        Parameters
        ----------
        source : str
            Name of source variable
        target : str
            Name of target variable
        k : int
            History length for target
        l : int
            History length for source
            
        Returns
        -------
        float
            Transfer entropy value
        """
        # Implementation details...
```

**Usage Considerations:**
- Use this analyzer to investigate potential causal relationships
- Consider testing multiple lag values to find optimal temporal relationships
- Compare Granger causality with transfer entropy for robust causal assessment
- Be cautious about inferring causality from observational data

## Data Structures

### TriggerEvent

Data structure representing a trigger event.

```python
@dataclass
class TriggerEvent:
    """Represents a trigger event with its characteristics."""
    trigger_type: str
    timestamp: datetime
    intensity: float
    duration: Optional[timedelta] = None
    confidence: float = 1.0
    associated_symptoms: List[str] = None
    context: Dict[str, Any] = None
```

**Usage Considerations:**
- Use this structure to represent discrete trigger occurrences
- Include intensity information whenever possible
- Specify duration for triggers that extend over time
- Include associated symptoms and context for comprehensive analysis

### TriggerProfile

Data structure representing a personalized trigger profile.

```python
@dataclass
class TriggerProfile:
    """Represents a personalized trigger profile."""
    trigger_sensitivities: Dict[str, float]
    interaction_effects: Dict[str, float]
    temporal_patterns: Dict[str, Dict[str, Any]]
    threshold_ranges: Dict[str, Tuple[float, float]]
    confidence_scores: Dict[str, float]
```

**Usage Considerations:**
- Use this structure to store a comprehensive patient-specific trigger profile
- Refer to `trigger_sensitivities` for individual trigger responses
- Check `interaction_effects` for understanding trigger combinations
- Consult `temporal_patterns` for time-dependent sensitivity variations
- Use `confidence_scores` to assess reliability of different profile aspects

## Error Handling

The framework employs several approaches to error handling:

1. **Input Validation**
   - All public methods should validate their inputs
   - Types are enforced through type annotations
   - Value ranges are checked for numerical parameters
   - `ValueError` is raised for invalid inputs

2. **Missing Data Handling**
   - Missing data is detected before processing
   - Methods return partial results when possible
   - NaN values are handled explicitly
   - Quality metrics reflect missing data impacts

3. **Error Types**
   - `ValueError`: For invalid parameter values
   - `TypeError`: For incorrect parameter types
   - `NotImplementedError`: For abstract methods
   - `RuntimeError`: For execution failures
   - `DataQualityError`: For insufficient data quality

4. **Quality Assessment**
   - Quality assessment functions return scores between 0 and 1
   - Results can be filtered by quality thresholds
   - Confidence scores accompany analytical results
   - Uncertainty estimates are provided where applicable

## API Version Control

The API follows semantic versioning:

- **Major version** (x.0.0): Breaking API changes
- **Minor version** (0.x.0): New functionality, backwards compatible
- **Patch version** (0.0.x): Bug fixes, backwards compatible

Each module includes a `__version__` attribute to indicate its current version.

## Integration Examples

### Processing Physiological Signals

```python
def process_ecg_signal(ecg_data: np.ndarray, sampling_rate: float) -> Dict[str, np.ndarray]:
    """Process ECG signal and extract HRV features."""
    # Create ECG adapter
    adapter = ECGAdapter()
    
    # Check signal quality
    quality = adapter.assess_quality(ecg_data, sampling_rate)
    if quality < 0.6:
        print(f"Warning: Low signal quality ({quality:.2f})")
    
    # Preprocess signal
    preprocessed = adapter.preprocess(ecg_data, sampling_rate)
    
    # Extract features
    features = adapter.extract_features(preprocessed, sampling_rate)
    
    # Add quality metadata
    for feature_name in features:
        features[f"{feature_name}_quality"] = quality
    
    return features
```

### Analyzing Feature Interactions

```python
def analyze_cross_modal_patterns(ecg_features: Dict[str, np.ndarray], 
                               skin_features: Dict[str, np.ndarray],
                               migraine_events: np.ndarray, 
                               timestamps: np.ndarray) -> Dict[str, Any]:
    """Analyze interactions between ECG, skin conductance, and migraine events."""
    # Create interaction analyzer
    analyzer = FeatureInteractionAnalyzer()
    
    # Prepare modality data
    modalities = {
        'ecg': ModalityData(
            data=np.column_stack([ecg_features[key] for key in ecg_features]),
            modality_type='ecg',
            timestamps=timestamps,
            metadata={'feature_names': list(ecg_features.keys())}
        ),
        'skin': ModalityData(
            data=np.column_stack([skin_features[key] for key in skin_features]),
            modality_type='eda',
            timestamps=timestamps,
            metadata={'feature_names': list(skin_features.keys())}
        )
    }
    
    # Analyze prodrome indicators
    prodrome_results = analyzer.analyze_prodrome_indicators(modalities)
    
    # Analyze feature importance
    # Combine features from both modalities
    all_features = {**ecg_features, **skin_features}
    importance = analyzer.rank_feature_importance(all_features, migraine_events)
    
    return {
        'prodrome_indicators': prodrome_results['indicators'],
        'feature_importance': importance,
        'cross_modal_patterns': prodrome_results['cross_modal_interactions']
    }
```

### Creating and Using a Digital Twin

```python
def create_patient_digital_twin(patient_history: Dict[str, np.ndarray], 
                              patient_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Create a digital twin for a migraine patient."""
    # Create digital twin foundation
    twin_foundation = PatientDigitalTwin()
    
    # Initialize twin from patient history
    twin = twin_foundation.initialize_twin(patient_history, patient_metadata)
    
    # Analyze triggers and create trigger profile
    trigger_analyzer = TriggerIdentificationAnalyzer()
    symptom_data = {'headache': patient_history['headache_severity']}
    potential_triggers = {
        'stress': patient_history['stress_level'],
        'sleep': patient_history['sleep_quality'],
        'weather': patient_history['barometric_pressure']
    }
    timestamps = patient_history['timestamps']
    
    trigger_results = trigger_analyzer.identify_triggers(
        symptom_data=symptom_data,
        potential_triggers=potential_triggers,
        timestamps=timestamps
    )
    
    # Update twin with trigger information
    twin['trigger_profile'] = trigger_results
    
    return twin
```

## Conclusion

This API reference provides a comprehensive overview of the interfaces, function signatures, parameters, return values, and usage considerations for the migraine prediction framework. Developers should refer to specific module documentation for more detailed information on particular components. 
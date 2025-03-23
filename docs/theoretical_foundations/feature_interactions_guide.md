# Feature Interactions Guide

## Introduction

The Feature Interactions module provides specialized components for analyzing how different physiological signals, environmental factors, and behavioral patterns interact and potentially contribute to migraine onset. This guide documents the design, implementation, and usage of these interaction analysis components, which are critical for understanding the complex relationships between various migraine triggers and responses.

Migraine is a multifactorial condition with complex interactions between different triggers and physiological responses. The Feature Interactions framework enables the systematic analysis of these relationships, helping to identify:

1. **Prodrome Indicators**: Early warning signs before migraine onset
2. **Cross-Modal Correlations**: Relationships between different data sources
3. **Temporal Patterns**: How features evolve over time
4. **Feature Importance**: Which factors are most predictive of migraine events
5. **Trigger Synergies**: How multiple triggers may interact to increase migraine risk

## Interaction Analysis Framework

### Architecture Overview

The feature interaction analysis framework is designed to analyze complex relationships between different data sources and their association with migraine events. The architecture follows these key principles:

1. **Modality Independence**: The framework can work with any data modality, including physiological signals, environmental data, behavioral patterns, etc.
2. **Temporal Analysis**: Explicit modeling of time-based relationships, including lead-lag effects and temporal evolution
3. **Statistical Rigor**: Solid statistical foundation for identifying significant interactions
4. **Multivariate Analysis**: Consideration of how multiple features interact simultaneously
5. **Causal Modeling**: Integration with causal inference to move beyond correlation

### Core Interface

All feature interaction analyzers implement the `MigraineFeatureInteractionAnalyzer` abstract base class, which defines three key methods:

```python
class MigraineFeatureInteractionAnalyzer(ABC):
    @abstractmethod
    def analyze_prodrome_indicators(self, 
                                   data_sources: Dict[str, ModalityData], 
                                   time_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Analyze prodrome phase indicators across multiple data sources."""
        pass
    
    @abstractmethod
    def detect_trigger_interactions(self, 
                                   triggers: Dict[str, np.ndarray],
                                   physiological_responses: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Detect interactions between multiple triggers and physiological responses."""
        pass
    
    @abstractmethod
    def rank_feature_importance(self, 
                              features: Dict[str, np.ndarray], 
                              migraine_occurrences: np.ndarray) -> List[Tuple[str, float]]:
        """Rank features by importance for migraine prediction."""
        pass
```

### Analysis Pipeline

The standard analysis pipeline for feature interactions consists of:

1. **Data Preparation**: Aligning time series from different sources and preprocessing
2. **Prodrome Analysis**: Identifying early indicators that precede migraine onset
3. **Cross-Modal Analysis**: Analyzing relationships between different data modalities
4. **Trigger Interaction Analysis**: Detecting how triggers interact with physiological responses
5. **Feature Importance Ranking**: Determining which features are most predictive of migraines

## Cross-Modal Correlation Analysis Methods

### Correlation Analysis

The framework provides several methods for analyzing correlations between different data modalities:

#### Pairwise Correlation

```python
def _calculate_modal_correlation(self, data1: np.ndarray, data2: np.ndarray) -> float:
    """Calculate correlation between two data modalities."""
    # Ensure data is clean for correlation
    mask = ~(np.isnan(data1) | np.isnan(data2))
    if np.sum(mask) < 2:
        return 0.0
        
    # Calculate correlation
    correlation, p_value = stats.pearsonr(data1[mask], data2[mask])
    
    # Return correlation if significant, otherwise 0
    return correlation if p_value < self.significance_threshold else 0.0
```

This method calculates Pearson correlation coefficients between pairs of time series, handling missing data and assessing statistical significance.

#### Cross-Correlation Matrix

The framework builds a comprehensive cross-correlation matrix between all features:

```python
def _analyze_cross_modal_interactions(self, aligned_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze interactions between different modalities."""
    # Extract feature names and data
    feature_names = list(aligned_features.keys())
    n_features = len(feature_names)
    
    # Initialize correlation matrix
    correlation_matrix = np.zeros((n_features, n_features))
    
    # Calculate pairwise correlations
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                correlation_matrix[i, j] = self._calculate_modal_correlation(
                    aligned_features[feature_names[i]],
                    aligned_features[feature_names[j]]
                )
    
    # Identify connected components in the correlation network
    G = nx.Graph()
    G.add_nodes_from(range(n_features))
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(correlation_matrix[i, j]) > self.min_correlation:
                G.add_edge(i, j, weight=abs(correlation_matrix[i, j]))
    
    connected_components = list(nx.connected_components(G))
    
    return {
        'correlation_matrix': correlation_matrix,
        'modalities': feature_names,
        'connected_components': [
            [feature_names[i] for i in component]
            for component in connected_components
        ]
    }
```

This analysis reveals clusters of features that co-vary, potentially indicating common underlying physiological processes.

### Mutual Information Analysis

Beyond linear correlation, the framework uses mutual information to capture non-linear relationships:

```python
# Mutual information calculation
mi_score = mutual_info_regression(X[:, [feature_idx]], migraine_occurrences)[0]
```

Mutual information quantifies how much knowing one variable reduces uncertainty about another, capturing both linear and non-linear dependencies.

### Information Flow Analysis

The framework implements Granger causality analysis to identify potential causal relationships:

```python
def _calculate_granger_causality(self, feature: np.ndarray, target: np.ndarray) -> float:
    """Calculate Granger causality from feature to target."""
    # Prepare data for causal inference
    data = pd.DataFrame({
        'feature': feature,
        'target': target
    })
    
    # Update causal analyzer with new data
    self.causal_analyzer.update_data(data)
    
    # Calculate Granger causality
    max_lag = min(12, len(feature) // 4)  # Maximum lag based on data size
    causality_results = self.causal_analyzer.test_granger_causality(
        'feature', 'target', max_lag=max_lag
    )
    
    # Return the maximum F-statistic across lags
    if causality_results and 'f_stats' in causality_results:
        return max(causality_results['f_stats'])
    return 0.0
```

This analysis helps determine whether changes in one feature precede and help predict changes in another, supporting causal inference.

## Temporal Pattern Detection Approaches

### Time Lag Analysis

The framework analyzes time lags between potential triggers and physiological responses:

```python
def _find_optimal_time_lag(self, trigger: np.ndarray, response: np.ndarray) -> int:
    """Find the optimal time lag between trigger and response."""
    max_lag = min(self.max_time_lag, len(trigger) // 3)
    correlations = []
    
    for lag in range(max_lag):
        if lag < len(trigger):
            corr = np.corrcoef(trigger[:-lag], response[lag:])[0, 1] if lag > 0 else np.corrcoef(trigger, response)[0, 1]
            correlations.append((lag, corr))
    
    # Return lag with highest correlation
    return max(correlations, key=lambda x: abs(x[1]))[0] if correlations else 0
```

This method identifies the temporal offset that maximizes correlation between a trigger and response, helping to establish the typical delay between cause and effect.

### Periodicity Detection

The framework can detect cyclical patterns in physiological signals and symptoms:

```python
def _detect_periodicity(self, data: np.ndarray) -> Dict[str, float]:
    """Detect periodic patterns in time series data."""
    # Check for sufficient data
    if len(data) < 24:
        return {'has_periodicity': False, 'period': 0, 'strength': 0}
        
    # Calculate autocorrelation
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Find peaks in autocorrelation
    peaks, _ = signal.find_peaks(autocorr, height=0)
    
    if len(peaks) < 2:
        return {'has_periodicity': False, 'period': 0, 'strength': 0}
    
    # Calculate average period and strength
    periods = np.diff(peaks)
    avg_period = np.mean(periods)
    strength = autocorr[peaks[0]] / autocorr[0]
    
    return {
        'has_periodicity': True,
        'period': avg_period,
        'strength': strength
    }
```

This analysis can identify daily, weekly, or monthly cycles in physiological parameters or symptoms, which is particularly relevant for hormonal migraine triggers.

### Changepoint Detection

The framework identifies significant shifts in time series data:

```python
def _detect_changepoints(self, data: np.ndarray) -> List[int]:
    """Detect points where the time series shows significant changes."""
    # Calculate cumulative sum
    cumsum = np.cumsum(data - np.mean(data))
    
    # Calculate CUSUM statistic
    s_pos = np.maximum(0, cumsum)
    s_neg = np.maximum(0, -cumsum)
    
    # Find changepoints
    threshold = np.std(data) * 1.5
    pos_changepoints = np.where(np.diff(s_pos > threshold))[0]
    neg_changepoints = np.where(np.diff(s_neg > threshold))[0]
    
    return sorted(np.concatenate([pos_changepoints, neg_changepoints]))
```

Changepoint detection helps identify when physiological parameters undergo significant transitions, which may indicate prodrome onset or response to triggers.

### Temporal Lead Analysis

The framework analyzes whether changes in one feature consistently precede changes in another:

```python
def _analyze_temporal_lead(self, feature: np.ndarray, target: np.ndarray) -> float:
    """Analyze if feature changes consistently lead target changes."""
    # Calculate feature gradient
    feature_grad = np.gradient(feature)
    target_grad = np.gradient(target)
    
    # Calculate cross-correlation with different lags
    max_lag = min(24, len(feature) // 4)  # Maximum lag to consider
    cross_corr = []
    
    for lag in range(max_lag):
        if lag >= len(feature_grad):
            break
        shifted_corr = np.corrcoef(feature_grad[:-lag], target_grad[lag:])[0, 1] if lag > 0 else np.corrcoef(feature_grad, target_grad)[0, 1]
        cross_corr.append((lag, shifted_corr))
    
    # Find lag with maximum correlation
    best_lag, best_corr = max(cross_corr, key=lambda x: abs(x[1])) if cross_corr else (0, 0)
    
    # Score based on correlation and lag
    lead_score = abs(best_corr) * (1 - best_lag/max_lag) if best_lag > 0 else 0
    
    return lead_score
```

This analysis helps identify early warning signs by determining which features change before migraine onset.

## Integration with Prediction Models

### Feature Importance for Prediction

The feature interaction analysis provides ranked feature importance for migraine prediction:

```python
def rank_feature_importance(self, 
                          features: Dict[str, np.ndarray], 
                          migraine_occurrences: np.ndarray) -> List[Tuple[str, float]]:
    """Rank features by importance for migraine prediction."""
    importance_scores = []
    
    # Prepare feature matrix
    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names])
    
    # Calculate importance using multiple methods
    for feature_idx, feature_name in enumerate(feature_names):
        # Mutual information
        mi_score = mutual_info_regression(X[:, [feature_idx]], migraine_occurrences)[0]
        
        # Correlation
        corr = abs(stats.pearsonr(X[:, feature_idx], migraine_occurrences)[0])
        
        # Granger causality
        granger_score = self._calculate_granger_causality(
            X[:, feature_idx], 
            migraine_occurrences
        )
        
        # Temporal lead analysis
        lead_score = self._analyze_temporal_lead(
            X[:, feature_idx],
            migraine_occurrences
        )
        
        # Combine scores
        combined_score = (mi_score * 0.4 + corr * 0.3 + 
                          granger_score * 0.2 + lead_score * 0.1)
        
        importance_scores.append((feature_name, combined_score))
    
    # Sort by importance score in descending order
    return sorted(importance_scores, key=lambda x: x[1], reverse=True)
```

This ranked feature list can be used to select the most informative features for predictive models, improving model performance by focusing on the most relevant signals.

### Prodrome Indicator Integration

The prodrome indicator analysis identifies early warning signs that can be used in prediction models:

```python
def analyze_prodrome_indicators(self, 
                              data_sources: Dict[str, ModalityData], 
                              time_window: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Analyze prodrome phase indicators across multiple data sources."""
    # Implementation details...
    
    return {
        'indicators': indicators,
        'temporal_patterns': temporal_patterns,
        'significance': significance_scores,
        'feature_importance': feature_importance,
        'cross_modal_interactions': cross_modal
    }
```

These prodrome indicators can be integrated into early warning systems to predict migraine onset before symptoms become severe.

### Trigger Interaction Models

The trigger interaction analysis identifies how multiple triggers may interact:

```python
def _analyze_multi_trigger_interactions(self,
                                  triggers: Dict[str, np.ndarray],
                                  responses: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze interactions between multiple triggers."""
    # Implementation details...
    
    return {
        'synergy_scores': synergy_scores,
        'interaction_graph': interaction_data
    }
```

This information can be used to create more sophisticated prediction models that account for how multiple triggers may combine to increase migraine risk beyond what individual triggers would suggest.

### Model Feature Engineering

The framework supports advanced feature engineering for prediction models:

1. **Temporal Features**: Creating lagged features based on optimal time lags
2. **Interaction Features**: Creating new features that capture interactions between triggers
3. **Trend Features**: Extracting rate-of-change and trend information
4. **Frequency Features**: Capturing cyclical patterns in the data

These engineered features can significantly improve the performance of prediction models by incorporating domain knowledge about migraine dynamics.

## Example Usage

### Basic Feature Importance Analysis

```python
# Create feature interaction analyzer
analyzer = FeatureInteractionAnalyzer(
    significance_threshold=0.05,
    min_correlation=0.2,
    max_time_lag=48  # 48 hours
)

# Analyze feature importance
feature_importance = analyzer.rank_feature_importance(
    features={
        'stress_level': stress_data,
        'sleep_quality': sleep_data,
        'weather_pressure': pressure_data,
        'hrv_sdnn': hrv_data
    },
    migraine_occurrences=migraine_events
)

# Print ranked features
for feature_name, importance in feature_importance:
    print(f"{feature_name}: {importance:.3f}")
```

### Analyzing Prodrome Indicators

```python
# Create data sources dictionary
data_sources = {
    'ecg': ModalityData(
        data=ecg_data,
        timestamps=ecg_timestamps,
        metadata={'sampling_rate': 250, 'quality': 0.95}
    ),
    'skin_conductance': ModalityData(
        data=eda_data,
        timestamps=eda_timestamps,
        metadata={'sampling_rate': 32, 'quality': 0.87}
    ),
    'self_report': ModalityData(
        data=symptom_data,
        timestamps=symptom_timestamps,
        metadata={'type': 'ordinal', 'scale': 5}
    )
}

# Set time window for analysis (24 hours before migraine onset)
time_window = (migraine_onset - 24*3600, migraine_onset)

# Analyze prodrome indicators
prodrome_results = analyzer.analyze_prodrome_indicators(
    data_sources=data_sources,
    time_window=time_window
)

# Extract significant indicators
for indicator in prodrome_results['indicators']:
    print(f"Prodrome indicator: {indicator['modality']}.{indicator['feature']}")
    print(f"  Significance: {indicator['significance']:.3f}")
    print(f"  Temporal pattern: {indicator['temporal_pattern']}")
```

### Detecting Trigger Interactions

```python
# Define triggers and physiological responses
triggers = {
    'stress': stress_data,
    'sleep_deficit': sleep_deficit_data,
    'caffeine': caffeine_intake_data
}

physiological_responses = {
    'hrv': hrv_data,
    'cortisol': cortisol_data,
    'vasoconstriction': vasoconstriction_data
}

# Analyze trigger interactions
interactions = analyzer.detect_trigger_interactions(
    triggers=triggers,
    physiological_responses=physiological_responses
)

# Print interaction results
for interaction_key, interaction_data in interactions.items():
    if interaction_key != 'multi_trigger':
        print(f"Interaction: {interaction_key}")
        print(f"  Correlation: {interaction_data['correlation']:.3f}")
        print(f"  Time lag: {interaction_data['time_lag']} hours")
    else:
        print("Multi-trigger interactions:")
        for trigger_pair, synergy in interaction_data['synergy_scores'].items():
            print(f"  {trigger_pair}: {synergy:.3f}")
```

## Conclusion

The Feature Interactions framework provides a sophisticated approach to analyzing the complex relationships between different physiological signals, environmental factors, and behavioral patterns that contribute to migraine onset. By leveraging advanced statistical and temporal analysis techniques, it enables the identification of prodrome indicators, important features, and trigger interactions that can significantly improve migraine prediction and management.

Future enhancements to the framework may include:

1. **Advanced causal discovery** algorithms to better distinguish causation from correlation
2. **Deep learning approaches** to capture more complex non-linear interactions
3. **Transfer learning** capabilities to leverage insights across patients while maintaining personalization
4. **Interactive visualization** tools to help patients and clinicians understand feature interactions
5. **Reinforcement learning** integration for adaptive intervention recommendation

By continuing to refine these analysis techniques, the framework will provide increasingly valuable insights into migraine mechanisms and support more effective personalized interventions. 
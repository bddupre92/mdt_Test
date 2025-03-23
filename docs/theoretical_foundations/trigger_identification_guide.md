# Trigger Identification Guide

## Introduction

The Trigger Identification module provides a sophisticated framework for identifying, analyzing, and managing migraine triggers. This guide documents the theoretical foundation, implementation, and usage of the trigger identification components, which form a critical part of the migraine digital twin system.

Migraine is a complex neurological condition with numerous potential triggers that vary significantly between individuals. The ability to accurately identify these triggers and understand their relationships with migraine events is essential for effective management and prevention. The Trigger Identification framework provides systematic approaches to:

1. **Identify Potential Triggers**: Discover factors that may contribute to migraine onset
2. **Analyze Causal Relationships**: Determine the strength of association between triggers and migraines
3. **Establish Sensitivity Thresholds**: Quantify how much of a trigger is needed to increase migraine risk
4. **Detect Temporal Patterns**: Identify when triggers are most likely to provoke migraines
5. **Analyze Trigger Interactions**: Understand how multiple triggers may interact
6. **Generate Personalized Profiles**: Create individualized trigger profiles for patients

## Causal Framework for Trigger Analysis

### Theoretical Foundation

The causal framework for migraine trigger analysis is built on principles of causal inference, which seeks to move beyond correlation to establish causal relationships. The framework employs:

1. **Granger Causality**: Testing whether past values of a trigger help predict future migraine events
2. **Transfer Entropy**: Measuring information transfer from trigger to migraine events
3. **Counterfactual Analysis**: Estimating what would happen in the absence of the trigger
4. **Time-Lagged Relationships**: Analyzing the temporal dynamics between triggers and migraines
5. **Natural Experiments**: Leveraging natural variations in trigger exposure

### Core Analysis Methods

The `TriggerIdentificationAnalyzer` implements several key methods for causal analysis:

```python
def _analyze_causal_relationship(self,
                               trigger_data: np.ndarray,
                               symptom_data: Dict[str, np.ndarray],
                               timestamps: np.ndarray) -> float:
    """Analyze the causal relationship between a potential trigger and symptoms."""
    # Convert data to pandas DataFrame for causal analysis
    symptom_key = list(symptom_data.keys())[0]  # Using first symptom for analysis
    
    data = pd.DataFrame({
        'trigger': trigger_data,
        'symptom': symptom_data[symptom_key],
        'time': np.arange(len(trigger_data))
    })
    
    # Update causal analyzer with current data
    self.causal_analyzer.update_data(data)
    
    # Calculate Granger causality
    max_lag = min(24, len(trigger_data) // 5)  # Maximum lag of 24 hours or 1/5 of data length
    granger_results = self.causal_analyzer.test_granger_causality(
        'trigger', 'symptom', max_lag=max_lag
    )
    
    # Calculate transfer entropy
    transfer_entropy = self.causal_analyzer.calculate_transfer_entropy(
        'trigger', 'symptom'
    )
    
    # Calculate combined causal score
    if granger_results and 'p_values' in granger_results:
        min_p_value = min(granger_results['p_values'])
        f_stat = max(granger_results['f_stats']) if 'f_stats' in granger_results else 0
        
        # Normalize and combine scores
        granger_score = 1 - min(min_p_value / self.causal_threshold, 1.0)
        te_score = min(transfer_entropy * 10, 1.0)  # Scale and cap transfer entropy
        
        return 0.7 * granger_score + 0.3 * te_score
    
    return 0.0
```

This method analyzes the causal relationship between a potential trigger and migraine symptoms by:
1. Applying Granger causality tests to determine if the trigger helps predict future migraine events
2. Calculating transfer entropy to measure information flow from trigger to symptom
3. Combining these metrics into a unified causal score

### Validation and Confidence Scoring

The framework includes methods to evaluate the confidence in identified causal relationships:

```python
def _calculate_trigger_confidence(self,
                                causal_score: float,
                                sensitivity: Dict[str, float],
                                temporal_pattern: Dict[str, Any]) -> float:
    """Calculate confidence score for trigger identification."""
    # Base confidence on causal score
    confidence = causal_score
    
    # Incorporate sensitivity information
    if sensitivity and 'threshold_confidence' in sensitivity:
        confidence = 0.7 * confidence + 0.3 * sensitivity['threshold_confidence']
    
    # Incorporate temporal pattern strength
    if temporal_pattern and 'pattern_strength' in temporal_pattern:
        pattern_factor = min(temporal_pattern['pattern_strength'], 1.0)
        confidence = 0.8 * confidence + 0.2 * pattern_factor
    
    # Incorporate consistency if available
    if temporal_pattern and 'consistency' in temporal_pattern:
        consistency = temporal_pattern['consistency']
        confidence = 0.9 * confidence + 0.1 * consistency
    
    return confidence
```

This confidence scoring mechanism integrates multiple lines of evidence:
1. The strength of the causal relationship
2. The clarity of sensitivity thresholds
3. The strength of temporal patterns
4. The consistency of the relationship over time

## Sensitivity Analysis Implementation

### Threshold Determination

The framework implements sophisticated methods to determine sensitivity thresholds for triggers:

```python
def _analyze_trigger_sensitivity(self,
                               trigger_data: np.ndarray,
                               symptom_data: Dict[str, np.ndarray],
                               timestamps: np.ndarray) -> Dict[str, Any]:
    """Analyze the sensitivity threshold for a trigger."""
    symptom_key = list(symptom_data.keys())[0]
    symptom = symptom_data[symptom_key]
    
    # Find optimal threshold that maximizes correlation with symptoms
    thresholds = np.linspace(np.min(trigger_data), np.max(trigger_data), 20)
    correlations = []
    
    for threshold in thresholds:
        # Create binary variable for trigger exceeding threshold
        binary_trigger = (trigger_data > threshold).astype(float)
        
        # Calculate correlation with symptom severity
        corr = np.corrcoef(binary_trigger, symptom)[0, 1]
        correlations.append((threshold, corr))
    
    # Find threshold with maximum correlation
    if not correlations:
        return {
            'threshold': 0,
            'correlation': 0,
            'threshold_confidence': 0,
            'cluster_analysis': {}
        }
        
    threshold, correlation = max(correlations, key=lambda x: abs(x[1]))
    
    # Perform cluster analysis to find natural groupings
    cluster_results = self._analyze_threshold_levels(trigger_data, symptom_data)
    
    # Calculate confidence in threshold determination
    confidence = min(abs(correlation) * 2, 1.0)
    
    return {
        'threshold': threshold,
        'correlation': correlation,
        'threshold_confidence': confidence,
        'cluster_analysis': cluster_results
    }
```

This method determines the sensitivity threshold by:
1. Testing different threshold levels for the trigger
2. Measuring how well each threshold correlates with migraine symptoms
3. Performing cluster analysis to find natural groupings in the data
4. Calculating a confidence score for the threshold determination

### Temporal Sensitivity Analysis

The framework also analyzes how sensitivity varies over time:

```python
def _analyze_temporal_sensitivity(self,
                                trigger_data: np.ndarray,
                                symptom_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze how trigger sensitivity varies over time."""
    # Extract symptom data
    symptom_key = list(symptom_data.keys())[0]
    symptom = symptom_data[symptom_key]
    
    # Divide data into time segments
    n_segments = min(10, len(trigger_data) // 24)  # At least 24 points per segment
    if n_segments < 2:
        return {'temporal_variation': 0, 'consistent_periods': [], 'variable_periods': []}
        
    segment_size = len(trigger_data) // n_segments
    thresholds = []
    correlations = []
    
    # Calculate optimal threshold for each segment
    for i in range(n_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < n_segments - 1 else len(trigger_data)
        
        segment_trigger = trigger_data[start:end]
        segment_symptom = symptom[start:end]
        
        # Skip segments with no variation
        if np.std(segment_trigger) == 0 or np.std(segment_symptom) == 0:
            thresholds.append(None)
            correlations.append(0)
            continue
        
        # Test different thresholds for this segment
        test_thresholds = np.linspace(np.min(segment_trigger), np.max(segment_trigger), 10)
        segment_correlations = []
        
        for threshold in test_thresholds:
            binary_trigger = (segment_trigger > threshold).astype(float)
            corr = np.corrcoef(binary_trigger, segment_symptom)[0, 1]
            segment_correlations.append((threshold, corr))
        
        if not segment_correlations:
            thresholds.append(None)
            correlations.append(0)
            continue
            
        best_threshold, best_corr = max(segment_correlations, key=lambda x: abs(x[1]))
        thresholds.append(best_threshold)
        correlations.append(best_corr)
    
    # Calculate temporal variation
    valid_thresholds = [t for t in thresholds if t is not None]
    if not valid_thresholds or len(valid_thresholds) < 2:
        return {'temporal_variation': 0, 'consistent_periods': [], 'variable_periods': []}
        
    threshold_variation = np.std(valid_thresholds) / np.mean(valid_thresholds)
    
    # Identify consistent and variable periods
    median_threshold = np.median(valid_thresholds)
    consistent_periods = []
    variable_periods = []
    
    for i, threshold in enumerate(thresholds):
        if threshold is None:
            continue
            
        if abs(threshold - median_threshold) < 0.25 * median_threshold:
            consistent_periods.append(i)
        else:
            variable_periods.append(i)
    
    return {
        'temporal_variation': threshold_variation,
        'consistent_periods': consistent_periods,
        'variable_periods': variable_periods
    }
```

This analysis reveals how trigger sensitivity may vary over time, which is crucial for understanding:
1. Whether the same threshold applies consistently
2. If there are periods of increased or decreased sensitivity
3. How temporal factors influence trigger effectiveness

## Multi-Trigger Interaction Detection

### Pairwise Interaction Analysis

The framework analyzes how pairs of triggers may interact:

```python
def _analyze_trigger_interactions(self,
                                triggers: Dict[str, np.ndarray],
                                symptom_data: Dict[str, np.ndarray],
                                timestamps: np.ndarray) -> Dict[str, float]:
    """Analyze interactions between multiple triggers."""
    interaction_scores = {}
    trigger_names = list(triggers.keys())
    
    # Analyze pairwise interactions
    for i in range(len(trigger_names)):
        for j in range(i+1, len(trigger_names)):
            name1 = trigger_names[i]
            name2 = trigger_names[j]
            
            interaction_key = f"{name1}+{name2}"
            
            # Calculate interaction score
            score = self._calculate_interaction_score(
                triggers[name1],
                triggers[name2],
                symptom_data
            )
            
            interaction_scores[interaction_key] = score
    
    # Analyze higher-order interactions up to max_interaction_order
    for order in range(3, min(len(trigger_names) + 1, self.max_interaction_order + 1)):
        higher_order = self._analyze_higher_order_interactions(
            triggers,
            symptom_data,
            order
        )
        
        # Add higher-order interactions to results
        interaction_scores.update(higher_order)
    
    return interaction_scores
```

This method identifies potential interactions between triggers by:
1. Analyzing how pairs of triggers correlate with migraine symptoms
2. Calculating interaction scores that quantify synergistic effects
3. Extending the analysis to higher-order interactions (3 or more triggers)

### Higher-Order Interaction Analysis

The framework extends the analysis to combinations of multiple triggers:

```python
def _analyze_higher_order_interactions(self,
                                     triggers: Dict[str, np.ndarray],
                                     symptom_data: Dict[str, np.ndarray],
                                     order: int) -> Dict[str, float]:
    """Analyze higher-order interactions between triggers."""
    interaction_scores = {}
    trigger_names = list(triggers.keys())
    
    # Skip if not enough triggers for requested order
    if len(trigger_names) < order:
        return {}
    
    # Generate all combinations of 'order' triggers
    from itertools import combinations
    for combo in combinations(trigger_names, order):
        # Create interaction key
        interaction_key = "+".join(combo)
        
        # Calculate joint effect
        joint_trigger = np.ones_like(triggers[combo[0]])
        for name in combo:
            # Normalize trigger data to 0-1 range
            normalized = (triggers[name] - np.min(triggers[name])) / (np.max(triggers[name]) - np.min(triggers[name]) + 1e-10)
            joint_trigger *= normalized
        
        # Calculate correlation with symptom
        symptom_key = list(symptom_data.keys())[0]
        correlation = np.corrcoef(joint_trigger, symptom_data[symptom_key])[0, 1]
        
        # Calculate interaction score
        individual_effects = [np.corrcoef(triggers[name], symptom_data[symptom_key])[0, 1] for name in combo]
        max_individual = max(abs(effect) for effect in individual_effects)
        
        # Score is positive if joint effect is stronger than individual effects
        interaction_scores[interaction_key] = abs(correlation) - max_individual
    
    return interaction_scores
```

This higher-order analysis helps identify complex interactions where:
1. Multiple triggers together have stronger effects than any single trigger
2. Certain combinations show synergistic or antagonistic relationships
3. Complex threshold effects emerge from trigger combinations

## Personalized Trigger Profile Generation

### Profile Structure

The framework generates comprehensive personalized trigger profiles:

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

These profiles contain:
1. **Trigger Sensitivities**: How responsive the patient is to each trigger
2. **Interaction Effects**: How triggers interact for this specific patient
3. **Temporal Patterns**: Daily, weekly, and seasonal patterns in trigger effectiveness
4. **Threshold Ranges**: Personalized trigger threshold values and ranges
5. **Confidence Scores**: Confidence in each aspect of the profile

### Profile Generation

The framework includes methods to generate personalized profiles from historical data:

```python
def generate_trigger_profile(self,
                           trigger_history: List[TriggerEvent],
                           migraine_events: List[datetime],
                           context_history: Optional[Dict[str, List[Any]]] = None) -> TriggerProfile:
    """Generate a personalized trigger profile based on historical data."""
    # Calculate trigger sensitivities
    sensitivities = self._calculate_trigger_sensitivities(
        trigger_history,
        migraine_events
    )
    
    # Analyze interaction effects
    interactions = self._analyze_historical_interactions(
        trigger_history,
        migraine_events
    )
    
    # Detect temporal patterns
    patterns = self._analyze_temporal_trigger_patterns(
        trigger_history,
        migraine_events
    )
    
    # Calculate threshold ranges
    thresholds = self._calculate_threshold_ranges(
        trigger_history,
        migraine_events
    )
    
    # Calculate confidence scores
    confidence = self._calculate_profile_confidence(
        trigger_history,
        migraine_events,
        patterns
    )
    
    return TriggerProfile(
        trigger_sensitivities=sensitivities,
        interaction_effects=interactions,
        temporal_patterns=patterns,
        threshold_ranges=thresholds,
        confidence_scores=confidence
    )
```

This profile generation process integrates:
1. The patient's history of trigger exposures
2. Their migraine event history
3. Contextual information surrounding triggers and migraines
4. Temporal patterns unique to the individual

### Temporal Pattern Analysis

The framework analyzes various temporal patterns in trigger effectiveness:

```python
def _analyze_temporal_trigger_patterns(self,
                                     trigger_history: List[TriggerEvent],
                                     migraine_events: List[datetime]) -> Dict[str, Dict[str, Any]]:
    """Analyze temporal patterns in historical trigger effectiveness."""
    patterns = {}
    
    # Group trigger events by type
    trigger_types = set(event.trigger_type for event in trigger_history)
    
    for trigger_type in trigger_types:
        type_events = [event for event in trigger_history if event.trigger_type == trigger_type]
        
        if len(type_events) < 5:  # Need sufficient data for pattern analysis
            continue
        
        # Analyze daily patterns
        daily_pattern = self._analyze_historical_daily_pattern(type_events)
        
        # Analyze weekly patterns
        weekly_pattern = self._analyze_historical_weekly_pattern(type_events)
        
        # Analyze seasonal patterns
        seasonal_pattern = self._analyze_historical_seasonal_pattern(type_events)
        
        # Combine pattern results
        patterns[trigger_type] = {
            'daily': daily_pattern,
            'weekly': weekly_pattern,
            'seasonal': seasonal_pattern,
            'consistency': (daily_pattern.get('consistency', 0) + 
                           weekly_pattern.get('consistency', 0) + 
                           seasonal_pattern.get('consistency', 0)) / 3,
            'pattern_strength': max(
                daily_pattern.get('strength', 0),
                weekly_pattern.get('strength', 0),
                seasonal_pattern.get('strength', 0)
            )
        }
    
    return patterns
```

This analysis identifies:
1. **Daily Patterns**: Times of day when triggers are most effective
2. **Weekly Patterns**: Days of the week with heightened sensitivity
3. **Seasonal Patterns**: Seasonal variations in trigger effectiveness
4. **Overall Pattern Strength**: How consistent and strong these patterns are

## Example Usage

### Basic Trigger Identification

```python
# Create trigger identification analyzer
analyzer = TriggerIdentificationAnalyzer(
    causal_threshold=0.05,
    sensitivity_window=48,  # 48 hours
    min_confidence=0.7,
    max_interaction_order=3
)

# Prepare data
symptom_data = {'headache_severity': symptom_array}
potential_triggers = {
    'stress': stress_array,
    'sleep_deficit': sleep_deficit_array,
    'caffeine': caffeine_array,
    'weather_pressure': pressure_array
}

# Identify triggers
results = analyzer.identify_triggers(
    symptom_data=symptom_data,
    potential_triggers=potential_triggers,
    timestamps=timestamps
)

# Print identified triggers
print("Identified triggers:")
for trigger in results['triggers']:
    confidence = results['confidence_scores'][trigger]
    threshold = results['sensitivity_thresholds'][trigger]['threshold']
    print(f"- {trigger} (confidence: {confidence:.2f}, threshold: {threshold:.2f})")

# Print trigger interactions
if 'interaction_effects' in results:
    print("\nTrigger interactions:")
    for interaction, score in results['interaction_effects'].items():
        if score > 0.1:  # Only show meaningful interactions
            print(f"- {interaction}: {score:.2f}")
```

### Generating a Personalized Trigger Profile

```python
# Create trigger history
trigger_history = [
    TriggerEvent(
        trigger_type="stress",
        timestamp=datetime(2023, 5, 10, 14, 30),
        intensity=8.5,
        duration=timedelta(hours=3),
        confidence=0.9
    ),
    TriggerEvent(
        trigger_type="caffeine",
        timestamp=datetime(2023, 5, 11, 9, 15),
        intensity=4.2,
        duration=timedelta(hours=6),
        confidence=0.85
    ),
    # ... more trigger events
]

# Migraine event history
migraine_events = [
    datetime(2023, 5, 10, 18, 45),
    datetime(2023, 5, 12, 7, 30),
    # ... more migraine events
]

# Generate profile
profile = analyzer.generate_trigger_profile(
    trigger_history=trigger_history,
    migraine_events=migraine_events
)

# Print trigger sensitivities
print("Trigger sensitivities:")
for trigger, sensitivity in profile.trigger_sensitivities.items():
    print(f"- {trigger}: {sensitivity:.2f}")

# Print threshold ranges
print("\nTrigger thresholds:")
for trigger, (min_val, max_val) in profile.threshold_ranges.items():
    print(f"- {trigger}: {min_val:.1f} - {max_val:.1f}")

# Print strongest temporal patterns
print("\nKey temporal patterns:")
for trigger, patterns in profile.temporal_patterns.items():
    if patterns['pattern_strength'] > 0.4:
        print(f"- {trigger}:")
        if patterns['daily']['strength'] > 0.4:
            print(f"  Daily: {patterns['daily']['peak_times']} hours")
        if patterns['weekly']['strength'] > 0.4:
            print(f"  Weekly: {patterns['weekly']['peak_days']}")
```

### Sensitivity Analysis

```python
# Analyze trigger sensitivity in detail
sensitivity_results = analyzer.analyze_trigger_sensitivity(
    trigger_data=caffeine_data,
    symptom_data={'headache': headache_data},
    baseline_period=(datetime(2023, 4, 1), datetime(2023, 4, 15))
)

# Print threshold information
print(f"Optimal threshold: {sensitivity_results['optimal_threshold']:.2f}")
print(f"Threshold range: {sensitivity_results['threshold_range'][0]:.2f} - {sensitivity_results['threshold_range'][1]:.2f}")

# Print temporal variations
if 'temporal_variations' in sensitivity_results:
    print("\nTemporal sensitivity variations:")
    for period, variation in sensitivity_results['temporal_variations'].items():
        print(f"- {period}: {variation['sensitivity']:.2f}")

# Print contextual factors
if 'contextual_factors' in sensitivity_results:
    print("\nContextual influences:")
    for factor, impact in sensitivity_results['contextual_factors'].items():
        print(f"- {factor}: {impact:.2f}")
```

## Integration with Digital Twin

The Trigger Identification framework integrates with the Digital Twin system to:

1. **Inform Twin State**: Provide trigger sensitivity profiles for the digital twin's patient state
2. **Guide Simulations**: Enable accurate simulation of trigger responses
3. **Support Prediction**: Provide causal models for migraine prediction
4. **Enable Personalization**: Adapt to individual differences in trigger responses
5. **Facilitate Intervention Planning**: Help design optimal trigger avoidance strategies

```python
# Example of integration with Digital Twin
def update_digital_twin_with_trigger_profile(digital_twin, trigger_profile):
    """Update digital twin with personalized trigger information."""
    # Update sensitivity parameters
    digital_twin.update_sensitivity_parameters(trigger_profile.trigger_sensitivities)
    
    # Update interaction models
    digital_twin.update_interaction_models(trigger_profile.interaction_effects)
    
    # Update temporal sensitivity patterns
    digital_twin.update_temporal_patterns(trigger_profile.temporal_patterns)
    
    # Define thresholds for simulations
    digital_twin.set_trigger_thresholds(trigger_profile.threshold_ranges)
    
    return digital_twin
```

## Conclusion

The Trigger Identification framework provides a sophisticated approach to understanding the complex relationships between potential triggers and migraine events. By leveraging advanced causal inference, sensitivity analysis, and interaction modeling techniques, it enables the creation of highly personalized trigger profiles that can significantly improve migraine prediction and management.

Future enhancements to the framework may include:

1. **Advanced machine learning integration** for more accurate trigger identification
2. **Real-time trigger detection** from continuous monitoring data
3. **Bayesian uncertainty modeling** for more robust confidence estimation
4. **Reinforcement learning approaches** for adaptive trigger threshold determination
5. **Federated learning capabilities** to leverage population data while maintaining privacy

By continuing to refine these techniques, the framework will provide increasingly valuable insights into migraine mechanisms and support more effective personalized interventions. 
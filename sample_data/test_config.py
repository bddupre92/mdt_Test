"""
Test configuration for real data validation
"""

# Feature mapping configuration
FEATURE_MAPPING = {
    'bp_sys': 'bp_sys',
    'bp_dia': 'bp_dia',
    'heart_rate': 'heart_rate',
    'sleep_hours': 'sleep_hours',
    'stress_level': 'stress_level',
    'weather_temp': 'weather_temp',
    'weather_humidity': 'weather_humidity',
    'medication_dose': 'medication_dose',
    'migraine_severity': 'migraine'  # Target column
}

# Expert type mapping
EXPERT_TYPE_MAPPING = {
    'bp_sys': 'physiological',
    'bp_dia': 'physiological',
    'heart_rate': 'physiological',
    'sleep_hours': 'behavioral',
    'stress_level': 'behavioral',
    'weather_temp': 'environmental',
    'weather_humidity': 'environmental',
    'medication_dose': 'medication'
}

# Reference ranges for clinical features
REFERENCE_RANGES = {
    'bp_sys': {'min': 70, 'max': 180, 'unit': 'mmHg'},
    'bp_dia': {'min': 40, 'max': 120, 'unit': 'mmHg'},
    'heart_rate': {'min': 40, 'max': 200, 'unit': 'bpm'},
    'sleep_hours': {'min': 0, 'max': 24, 'unit': 'hours'},
    'stress_level': {'min': 0, 'max': 10, 'unit': 'level'},
    'migraine_severity': {'min': 0, 'max': 10, 'unit': 'severity'}
}

# Missing value handling strategy
MISSING_VALUE_STRATEGY = {
    'default': 'median'  # Default strategy for features not specified
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'missing_data_threshold': 0.2,  # Maximum allowed proportion of missing values
    'outlier_threshold': 3.0,  # Z-score threshold for outlier detection
}

# Comparison configuration
COMPARISON_CONFIG = {
    'statistical_tests': ['ks', 't-test'],
    'visualization_types': ['distribution', 'correlation']
}

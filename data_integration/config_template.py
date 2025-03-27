"""
Real Data Integration Configuration Template

This file serves as a template for configuring the real data integration components.
Copy this file, rename it (e.g., to clinical_data_config.py), and customize it for your specific dataset.
"""

# Feature mapping configuration
# Maps clinical data columns to standardized feature names used by the MoE framework
FEATURE_MAPPING = {
    # Original column name: Standardized feature name
    # Examples:
    'blood_pressure_systolic': 'bp_sys',
    'blood_pressure_diastolic': 'bp_dia',
    'heart_rate': 'hr',
    'sleep_duration_hours': 'sleep_hours',
    'stress_level': 'stress',
    'medication_dose_mg': 'med_dose',
    'headache_severity': 'migraine'  # This is typically your target column
}

# Expert type mapping
# Maps features to different expert types in the MoE framework
EXPERT_TYPE_MAPPING = {
    # Feature: Expert type
    # Examples:
    'bp_sys': 'physiological',
    'bp_dia': 'physiological',
    'hr': 'physiological',
    'sleep_hours': 'behavioral',
    'stress': 'behavioral',
    'weather_temp': 'environmental',
    'weather_humidity': 'environmental',
    'med_dose': 'medication'
}

# Reference ranges for clinical features
# Used for data validation and normalization
REFERENCE_RANGES = {
    # Feature: {'min': minimum value, 'max': maximum value, 'unit': unit of measurement}
    # Examples:
    'bp_sys': {'min': 70, 'max': 180, 'unit': 'mmHg'},
    'bp_dia': {'min': 40, 'max': 120, 'unit': 'mmHg'},
    'hr': {'min': 40, 'max': 200, 'unit': 'bpm'},
    'sleep_hours': {'min': 0, 'max': 24, 'unit': 'hours'},
    'stress': {'min': 0, 'max': 10, 'unit': 'level'},
    'migraine': {'min': 0, 'max': 10, 'unit': 'severity'}
}

# Missing value handling strategy
MISSING_VALUE_STRATEGY = {
    # Feature: Strategy ('mean', 'median', 'mode', 'zero', 'remove_row', 'default')
    # Examples:
    'bp_sys': 'median',
    'bp_dia': 'median',
    'hr': 'median',
    'sleep_hours': 'mean',
    'stress': 'mode',
    'weather_temp': 'mean',
    'weather_humidity': 'mean',
    'med_dose': 'zero',
    'default': 'median'  # Default strategy for features not specified
}

# Default values for missing data (when using 'default' strategy)
DEFAULT_VALUES = {
    # Feature: Default value
    # Examples:
    'sleep_hours': 7,
    'stress': 3
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    'missing_data_threshold': 0.2,  # Maximum allowed proportion of missing values (0.0-1.0)
    'outlier_threshold': 3.0,  # Z-score threshold for outlier detection
    'distribution_similarity_threshold': 0.7,  # Minimum similarity score (0.0-1.0)
    'temporal_consistency_threshold': 0.8  # Minimum temporal consistency score (0.0-1.0)
}

# Anonymization settings
ANONYMIZATION_SETTINGS = {
    'id_columns': ['patient_id', 'record_id'],  # Columns to hash or replace
    'date_columns': ['date_of_birth', 'visit_date'],  # Date columns to modify
    'sensitive_columns': ['name', 'address', 'phone', 'email'],  # Columns to remove
    'date_shift_range': {'min_days': -30, 'max_days': 30}  # Range for random date shifting
}

# Feature importance configuration
FEATURE_IMPORTANCE = {
    'top_n_features': 10,  # Number of top features to include in reports
    'explain_method': 'shap'  # Explainability method to use ('shap', 'permutation', etc.)
}

# Comparison configuration
COMPARISON_CONFIG = {
    'statistical_tests': ['ks', 't-test', 'chi2'],  # Statistical tests to run
    'visualization_types': ['distribution', 'correlation', 'pca', 'feature_importance'],
    'correlation_method': 'pearson'  # Correlation method ('pearson', 'spearman', 'kendall')
}

# Interactive report settings
REPORT_SETTINGS = {
    'title': 'Clinical Data Integration Report',
    'description': 'Validation and comparison of real clinical data with MoE framework',
    'color_theme': 'blue',  # Report color theme
    'include_sections': [
        'validation_summary',
        'data_quality',
        'distribution_comparison',
        'feature_importance',
        'model_performance'
    ]
}

"""
Clinical Data Validator for MoE Validation Framework.

This module provides tools for validating the quality and compatibility of
clinical data with the MoE validation framework.
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from json import JSONEncoder

# Custom JSON Encoder to handle NumPy data types
class NumpyEncoder(JSONEncoder):
    """Custom JSON encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)


# Define standard clinical reference ranges for common measurements
CLINICAL_REFERENCE_RANGES = {
    'heart_rate': {'min': 40, 'max': 180, 'unit': 'bpm'},
    'bp_sys': {'min': 70, 'max': 200, 'unit': 'mmHg'},
    'bp_dia': {'min': 40, 'max': 120, 'unit': 'mmHg'},
    'temperature': {'min': 35, 'max': 40, 'unit': '°C'},
    'respiration_rate': {'min': 8, 'max': 30, 'unit': 'breaths/min'},
    'oxygen_saturation': {'min': 85, 'max': 100, 'unit': '%'},
    'glucose_level': {'min': 3, 'max': 20, 'unit': 'mmol/L'},
    'stress_level': {'min': 0, 'max': 10, 'unit': 'score'},
    'sleep_hours': {'min': 0, 'max': 15, 'unit': 'hours'},
    'pain_intensity': {'min': 0, 'max': 10, 'unit': 'score'},
    'migraine_severity': {'min': 0, 'max': 10, 'unit': 'score'},
    'weather_temp': {'min': -30, 'max': 120, 'unit': '°F'},
    'weather_humidity': {'min': 0, 'max': 100, 'unit': '%'},
    'medication_dose': {'min': 0, 'max': 200, 'unit': 'mg'}
}


class ClinicalDataValidator:
    """Validates clinical data quality and compatibility with MoE requirements."""
    
    def __init__(self, config_path=None, reference_ranges=None):
        """Initialize with optional configuration and reference ranges."""
        self.config = self._load_config(config_path)
        self.reference_ranges = reference_ranges or CLINICAL_REFERENCE_RANGES
        self.required_columns = self.config.get('required_columns', [])
        self.validation_history = []
        
    def _load_config(self, config_path):
        """Load configuration from JSON or Python file or use defaults."""
        if not config_path or not os.path.exists(config_path):
            return {
                'required_columns': [],
                'validation_strictness': 'medium'
            }
            
        # Handle Python module config files
        if config_path.endswith('.py'):
            import importlib.util
            import sys
            
            spec = importlib.util.spec_from_file_location("config_module", config_path)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["config_module"] = config_module
            spec.loader.exec_module(config_module)
            
            config = {}
            for attr in dir(config_module):
                if not attr.startswith('__'):
                    config[attr] = getattr(config_module, attr)
            return config
        
        # Handle JSON config files
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Attempt to load as JSON for backward compatibility
        else:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Config file {config_path} is not a valid JSON or Python file")
        
    def validate_structure(self, data_df, required_columns=None):
        """Ensure data has required columns and structure."""
        # Define a more flexible set of required columns based on the dataset
        # Just need a patient identifier, a date/time column, and at least one measurement
        required_categories = [
            # At least one identifier column
            ['patient_id', 'subject_id', 'participant_id', 'id'],
            # At least one date/time column
            ['date', 'timestamp', 'datetime', 'time', 'visit_date'],
            # At least one clinical measurement
            ['heart_rate', 'bp_sys', 'bp_dia', 'temperature', 'migraine_severity', 'stress_level']
        ]
        
        missing_categories = []
        for category in required_categories:
            if not any(col in data_df.columns for col in category):
                missing_categories.append(category[0])  # Use first name as example
        
        # Also check explicit required columns if specified
        required = required_columns or self.required_columns
        missing_cols = set(required) - set(data_df.columns)
        
        validation_result = {
            'valid': len(missing_categories) == 0 and len(missing_cols) == 0,
            'missing_categories': missing_categories,
            'missing_columns': list(missing_cols),
            'row_count': len(data_df),
            'column_count': len(data_df.columns)
        }
        
        self.validation_history.append({
            'validation_type': 'structure',
            'result': validation_result
        })
        
        if missing_cols:
            validation_result['error'] = f"Missing required columns: {missing_cols}"
            return validation_result
        
        return validation_result
        
    def check_distributions(self, data_df):
        """Compare data distributions with expected ranges for clinical data."""
        distribution_issues = []
        distribution_stats = {}
        
        # Check values against clinical reference ranges
        for col, expected_range in self.reference_ranges.items():
            if col in data_df.columns:
                min_val, max_val = expected_range.get('min'), expected_range.get('max')
                
                if min_val is not None and max_val is not None:
                    out_of_range = ((data_df[col] < min_val) | (data_df[col] > max_val)).sum()
                    out_of_range_pct = 100 * out_of_range / len(data_df)
                    
                    distribution_stats[col] = {
                        'mean': data_df[col].mean(),
                        'median': data_df[col].median(),
                        'min': data_df[col].min(),
                        'max': data_df[col].max(),
                        'out_of_range_count': int(out_of_range),
                        'out_of_range_percent': float(out_of_range_pct),
                        'expected_min': min_val,
                        'expected_max': max_val,
                        'unit': expected_range.get('unit', '')
                    }
                    
                    # Flag columns with significant out-of-range values
                    if out_of_range_pct > 5:  # More than 5% values out of range
                        distribution_issues.append({
                            'column': col,
                            'issue': f"{out_of_range_pct:.1f}% of values outside clinical " 
                                    f"reference range ({min_val}-{max_val})",
                            'severity': 'warning' if out_of_range_pct < 20 else 'error'
                        })
        
        validation_result = {
            'valid': len(distribution_issues) == 0,
            'distribution_issues': distribution_issues,
            'distribution_stats': distribution_stats
        }
        
        self.validation_history.append({
            'validation_type': 'distributions',
            'result': validation_result
        })
        
        return validation_result
        
    def detect_data_quality_issues(self, data_df):
        """Identify potential quality issues in clinical data."""
        quality_issues = []
        quality_stats = {}
        
        # Make a copy to avoid modifying the original
        df_copy = data_df.copy()
        
        # 1. Check for missing values
        missing_counts = df_copy.isnull().sum()
        missing_pct = 100 * missing_counts / len(df_copy)
        
        high_missing_cols = missing_pct[missing_pct > 5].index.tolist()
        
        if high_missing_cols:
            for col in high_missing_cols:
                severity = 'warning' if missing_pct[col] < 20 else 'error'
                quality_issues.append({
                    'column': col,
                    'issue': f"High missing value rate: {missing_pct[col]:.1f}%",
                    'severity': severity
                })
            
        # 2. Check for duplicate timestamps (if time column exists)
        time_cols = [col for col in df_copy.columns if any(
            time_term in col.lower() for time_term in ['time', 'date', 'timestamp'])]
        
        if time_cols:
            time_col = time_cols[0]
            # Handle non-datetime columns safely
            try:
                # Only convert if it's not already a datetime
                if not pd.api.types.is_datetime64_dtype(df_copy[time_col]):
                    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
                
                # Check for duplicates only if conversion was successful
                id_cols = [col for col in df_copy.columns if 'id' in col.lower()]
                
                if id_cols and not df_copy[time_col].isna().any():
                    # Check duplicates within each patient
                    id_col = id_cols[0]
                    duplicates = df_copy.duplicated(subset=[id_col, time_col], keep=False)
            except Exception as e:
                # Log the error but don't fail
                quality_issues.append({
                    'column': time_col,
                    'issue': f"Date/time validation error: {str(e)}",
                    'severity': 'warning'
                })
                dup_count = duplicates.sum()
                
                if dup_count > 0:
                    dup_pct = 100 * dup_count / len(data_df)
                    quality_issues.append({
                        'column': time_col,
                        'issue': f"{dup_count} duplicate timestamps ({dup_pct:.1f}%)",
                        'severity': 'warning' if dup_pct < 5 else 'error'
                    })
        
        # 3. Check for inconsistent time intervals
        if time_cols and id_cols:
            time_col, id_col = time_cols[0], id_cols[0]
            
            # Convert time column to datetime first if it's not already
            try:
                if not pd.api.types.is_datetime64_dtype(data_df[time_col]):
                    # Make a copy to avoid SettingWithCopyWarning
                    temp_df = data_df.copy()
                    temp_df[time_col] = pd.to_datetime(temp_df[time_col], errors='coerce')
                    
                    # Log conversion results
                    null_after = temp_df[time_col].isna().sum()
                    if null_after > 0:
                        self.logger.warning(f"Datetime conversion created {null_after} NaT values in {time_col}")
                    
                    # Get time differences for each patient with properly formatted datetime
                    patient_groups = temp_df.sort_values(time_col).groupby(id_col)
                else:
                    # Use original dataframe if already datetime
                    patient_groups = data_df.sort_values(time_col).groupby(id_col)
                
                # Calculate time differences
                time_diffs = patient_groups[time_col].diff()
            except Exception as e:
                self.logger.warning(f"Error calculating time differences: {e}")
                quality_issues.append({
                    'column': time_col,
                    'issue': f"Could not analyze time intervals: {str(e)}",
                    'severity': 'warning'
                })
                time_diffs = pd.Series(dtype='timedelta64[ns]')  # Empty series
            
            # Convert to seconds for easier comparison
            if pd.api.types.is_timedelta64_dtype(time_diffs):
                time_diffs = time_diffs.dt.total_seconds()
                
                # Calculate variation in sampling intervals
                if not time_diffs.isnull().all():
                    mean_interval = time_diffs.mean()
                    cv_interval = time_diffs.std() / mean_interval if mean_interval > 0 else 0
                    
                    quality_stats['time_intervals'] = {
                        'mean_interval_seconds': mean_interval,
                        'cv_interval': cv_interval
                    }
                    
                    # High coefficient of variation indicates inconsistent sampling
                    if cv_interval > 1.0:  # CV > 100% indicates high variability
                        quality_issues.append({
                            'column': time_col,
                            'issue': f"Highly variable sampling intervals (CV: {cv_interval:.2f})",
                            'severity': 'warning'
                        })
        
        # 4. Check for implausible changes in values
        for col in data_df.select_dtypes(include=[np.number]).columns:
            if col in time_cols or col in id_cols:
                continue
                
            # Check if we have patient IDs and timestamps
            if time_cols and id_cols:
                # Sort by patient and time
                sorted_df = data_df.sort_values([id_cols[0], time_cols[0]])
                
                # Calculate changes by patient
                value_changes = sorted_df.groupby(id_cols[0])[col].diff()
                
                # Skip if all changes are NaN
                if value_changes.isnull().all():
                    continue
                    
                # Look for extreme changes that might indicate errors
                if col in self.reference_ranges:
                    range_width = self.reference_ranges[col]['max'] - self.reference_ranges[col]['min']
                    extreme_changes = (value_changes.abs() > range_width * 0.5).sum()
                    
                    if extreme_changes > 0:
                        extreme_pct = 100 * extreme_changes / len(value_changes.dropna())
                        if extreme_pct > 2:  # More than 2% extreme changes
                            quality_issues.append({
                                'column': col,
                                'issue': f"{extreme_pct:.1f}% of sequential changes " 
                                        f"are physiologically implausible",
                                'severity': 'warning' if extreme_pct < 10 else 'error'
                            })
        
        validation_result = {
            'valid': len(quality_issues) == 0,
            'quality_issues': quality_issues,
            'quality_stats': quality_stats
        }
        
        self.validation_history.append({
            'validation_type': 'data_quality',
            'result': validation_result
        })
        
        return validation_result
    
    def check_temporal_consistency(self, data_df):
        """Check for temporal consistency in clinical data."""
        time_cols = [col for col in data_df.columns if any(
            time_term in col.lower() for time_term in ['time', 'date', 'timestamp'])]
        
        if not time_cols:
            return {'valid': True, 'message': 'No time columns found for temporal validation'}
            
        time_col = time_cols[0]
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_dtype(data_df[time_col]):
            data_df[time_col] = pd.to_datetime(data_df[time_col], errors='coerce')
            
        # Check for invalid dates
        invalid_dates = data_df[time_col].isnull().sum()
        invalid_pct = 100 * invalid_dates / len(data_df)
        
        temporal_issues = []
        
        if invalid_pct > 0:
            temporal_issues.append({
                'column': time_col,
                'issue': f"{invalid_dates} invalid dates/times ({invalid_pct:.1f}%)",
                'severity': 'warning' if invalid_pct < 5 else 'error'
            })
            
        # Check for future dates
        now = pd.Timestamp.now()
        future_dates = (data_df[time_col] > now).sum()
        future_pct = 100 * future_dates / len(data_df)
        
        if future_pct > 0:
            temporal_issues.append({
                'column': time_col,
                'issue': f"{future_dates} future dates ({future_pct:.1f}%)",
                'severity': 'warning' if future_pct < 1 else 'error'
            })
            
        # Check for consistent time ordering by patient
        id_cols = [col for col in data_df.columns if 'id' in col.lower()]
        if id_cols:
            id_col = id_cols[0]
            
            # Group by patient and check for non-increasing timestamps
            out_of_order = 0
            
            for _, group in data_df.groupby(id_col):
                sorted_times = group[time_col].sort_values()
                if not sorted_times.equals(group[time_col]):
                    out_of_order += 1
                    
            if out_of_order > 0:
                pct_patients = 100 * out_of_order / data_df[id_col].nunique()
                temporal_issues.append({
                    'issue': f"{out_of_order} patients ({pct_patients:.1f}%) have non-chronological data",
                    'severity': 'warning'
                })
        
        validation_result = {
            'valid': len(temporal_issues) == 0,
            'temporal_issues': temporal_issues
        }
        
        self.validation_history.append({
            'validation_type': 'temporal_consistency',
            'result': validation_result
        })
        
        return validation_result
        
    def validate_compatibility_with_moe(self, data_df):
        """Check if data is compatible with MoE framework requirements."""
        moe_issues = []
        
        # 1. Check for target variable
        target_vars = [col for col in data_df.columns 
                      if col in ['migraine', 'migraine_event', 'headache', 'pain_event']]
        
        if not target_vars:
            moe_issues.append({
                'issue': "No target variable found for migraine prediction",
                'severity': 'error'
            })
            
        # 2. Check for expert-specific features
        expert_types = {
            'physiological': ['heart_rate', 'temperature', 'blood_pressure', 'resp'],
            'environmental': ['humidity', 'pressure', 'temperature_ambient', 'weather'],
            'behavioral': ['stress', 'sleep', 'activity', 'exercise'],
            'medication': ['medication', 'drug', 'treatment', 'dose']
        }
        
        missing_experts = []
        for expert_type, keywords in expert_types.items():
            # Check if we have any columns matching the expert keywords
            has_features = any(
                any(keyword in col.lower() for keyword in keywords)
                for col in data_df.columns
            )
            
            if not has_features:
                missing_experts.append(expert_type)
                
        if missing_experts:
            moe_issues.append({
                'issue': f"Missing features for experts: {', '.join(missing_experts)}",
                'severity': 'warning'
            })
            
        # 3. Check for minimum data requirements
        if len(data_df) < 100:
            moe_issues.append({
                'issue': f"Insufficient data points for reliable MoE training ({len(data_df)} rows)",
                'severity': 'warning' if len(data_df) >= 50 else 'error'
            })
            
        # 4. Check for minimum time span if temporal data exists
        time_cols = [col for col in data_df.columns if any(
            time_term in col.lower() for time_term in ['time', 'date', 'timestamp'])]
        
        if time_cols:
            time_col = time_cols[0]
            
            # Ensure datetime format
            if not pd.api.types.is_datetime64_dtype(data_df[time_col]):
                data_df[time_col] = pd.to_datetime(data_df[time_col], errors='coerce')
                
            # Calculate time span
            time_span = data_df[time_col].max() - data_df[time_col].min()
            
            if time_span < pd.Timedelta(days=7):
                moe_issues.append({
                    'issue': f"Insufficient time span for drift detection ({time_span.days} days)",
                    'severity': 'warning'
                })
        
        validation_result = {
            'valid': len(moe_issues) == 0,
            'moe_compatibility_issues': moe_issues
        }
        
        self.validation_history.append({
            'validation_type': 'moe_compatibility',
            'result': validation_result
        })
        
        return validation_result
    
    def generate_validation_report(self, save_path=None):
        """Generate a comprehensive validation report from validation history."""
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'validation_summary': {
                'total_validations': len(self.validation_history),
                'passed_validations': sum(1 for v in self.validation_history if v['result'].get('valid', False)),
                'warnings': sum(1 for v in self.validation_history 
                              for issue in v['result'].get('quality_issues', []) + 
                                          v['result'].get('distribution_issues', []) +
                                          v['result'].get('temporal_issues', []) +
                                          v['result'].get('moe_compatibility_issues', [])
                              if issue.get('severity') == 'warning'),
                'errors': sum(1 for v in self.validation_history 
                            for issue in v['result'].get('quality_issues', []) + 
                                        v['result'].get('distribution_issues', []) +
                                        v['result'].get('temporal_issues', []) +
                                        v['result'].get('moe_compatibility_issues', [])
                            if issue.get('severity') == 'error')
            },
            'validation_results': self.validation_history
        }
        
        # Determine overall validity
        report['validation_summary']['overall_valid'] = (
            report['validation_summary']['errors'] == 0 and
            report['validation_summary']['passed_validations'] > 0
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
                
        return report
    
    def validate_all(self, data_df, generate_report=True, save_path=None):
        """Run all validation checks and generate a comprehensive report."""
        # Reset validation history
        self.validation_history = []
        
        # Run all validation checks
        self.validate_structure(data_df)
        self.check_distributions(data_df)
        self.detect_data_quality_issues(data_df)
        self.check_temporal_consistency(data_df)
        self.validate_compatibility_with_moe(data_df)
        
        # Generate and return report
        if generate_report:
            return self.generate_validation_report(save_path)
        
        return self.validation_history

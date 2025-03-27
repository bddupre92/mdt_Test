"""
Clinical Data Adapter for MoE Validation Framework.

This module provides tools for integrating anonymized clinical data into the
MoE validation framework, with specific attention to preprocessing required
for real patient data.
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats


class PatientAnonymizer:
    """Provides anonymization tools for sensitive patient data."""
    
    def __init__(self, salt=None):
        """Initialize the anonymizer with an optional custom salt."""
        self.salt = salt or os.urandom(16).hex()
        
    def anonymize_patient_id(self, patient_id):
        """Create anonymized hash of patient ID."""
        import hashlib
        
        # Create a salted hash of the patient ID
        hash_obj = hashlib.sha256(f"{patient_id}{self.salt}".encode())
        return hash_obj.hexdigest()[:16]  # Return first 16 chars of hash
    
    def anonymize_dataframe(self, df, id_column='patient_id'):
        """Anonymize an entire dataframe of patient data."""
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in dataframe")
            
        # Create a copy to avoid modifying the original
        anon_df = df.copy()
        
        # Replace patient IDs with anonymized versions
        anon_df[id_column] = anon_df[id_column].apply(self.anonymize_patient_id)
        
        # Remove any other potentially identifying columns
        for col in anon_df.columns:
            if any(id_term in col.lower() for id_term in ['name', 'ssn', 'address', 'phone', 'email', 'zip']):
                anon_df.drop(col, axis=1, inplace=True)
                
        return anon_df


class ClinicalDataAdapter:
    """Adapter for processing anonymized clinical data into MoE framework format."""
    
    def __init__(self, config_path=None):
        """Initialize with optional configuration file path."""
        self.config = self._load_config(config_path)
        self.feature_map = self.config.get('feature_mapping', {})
        self.reference_ranges = self.config.get('clinical_reference_ranges', {})
        self.anonymizer = PatientAnonymizer()
        self.drift_metadata = {}
        
    def _load_config(self, config_path):
        """Load configuration from JSON or Python file or use defaults."""
        if not config_path or not os.path.exists(config_path):
            return {
                'feature_mapping': {},
                'clinical_reference_ranges': {},
                'missing_value_strategy': 'clinical_imputation'
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
        
    def load_data(self, data_path, format_type='csv'):
        """Load clinical data from various formats."""
        try:
            if format_type == 'csv':
                data = pd.read_csv(data_path)
            elif format_type == 'json':
                data = pd.read_json(data_path)
            elif format_type == 'excel':
                data = pd.read_excel(data_path)
            elif format_type == 'parquet':
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
            # Initial data quality assessment
            self._assess_data_quality(data)
            return data
            
        except Exception as e:
            print(f"Error loading data from {data_path}: {str(e)}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()
    
    def _assess_data_quality(self, data_df):
        """Perform initial data quality assessment and store metadata"""
        # Store basic statistics about data
        self.drift_metadata['num_records'] = len(data_df)
        self.drift_metadata['num_features'] = len(data_df.columns)
        self.drift_metadata['missing_values'] = data_df.isna().sum().to_dict()
        self.drift_metadata['dtypes'] = {col: str(dtype) for col, dtype in data_df.dtypes.items()}
        
        # Store statistical properties of numeric columns
        numeric_columns = data_df.select_dtypes(include=[np.number]).columns
        self.drift_metadata['feature_stats'] = {}
        
        for col in numeric_columns:
            col_data = data_df[col].dropna()
            if len(col_data) > 0:
                self.drift_metadata['feature_stats'][col] = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()) if len(col_data) > 1 else 0,
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'skew': float(stats.skew(col_data)) if len(col_data) > 2 else 0,
                    'kurtosis': float(stats.kurtosis(col_data)) if len(col_data) > 3 else 0
                }
    
    def _handle_clinical_missing_values(self, data_df):
        """Apply specialized clinical missing value handling."""
        strategy = self.config.get('missing_value_strategy', 'clinical_imputation')
        
        if strategy == 'clinical_imputation':
            # Use domain-specific knowledge for different feature types
            for col in data_df.columns:
                if col.endswith('_rate') or col.endswith('_frequency'):
                    # For rate/frequency features, 0 is often the appropriate default
                    data_df[col].fillna(0, inplace=True)
                elif col.startswith('has_') or col.endswith('_flag'):
                    # For binary flags, False/0 is the appropriate default
                    data_df[col].fillna(0, inplace=True)
                elif 'temperature' in col or 'pressure' in col or 'level' in col:
                    # For physiological measurements, use mean or median
                    data_df[col].fillna(data_df[col].median(), inplace=True)
                else:
                    # For general features, use column median
                    data_df[col].fillna(data_df[col].median(), inplace=True)
        
        elif strategy == 'knn_imputation':
            try:
                from sklearn.impute import KNNImputer
                # More sophisticated KNN imputation for clinical data
                numeric_cols = data_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    imputer = KNNImputer(n_neighbors=5)
                    data_df[numeric_cols] = imputer.fit_transform(data_df[numeric_cols])
            except ImportError:
                print("KNN imputation requested but sklearn not available. Falling back to median imputation.")
                for col in data_df.select_dtypes(include=[np.number]).columns:
                    data_df[col].fillna(data_df[col].median(), inplace=True)
            
        return data_df
    
    def _map_clinical_features(self, data_df):
        """Map clinical feature names to our internal feature space."""
        if not self.feature_map:
            return data_df  # No mapping defined
            
        # Create a new DataFrame with mapped columns
        mapped_df = pd.DataFrame()
        
        # Apply mappings from the config
        for orig_col, mapped_col in self.feature_map.items():
            if orig_col in data_df.columns:
                mapped_df[mapped_col] = data_df[orig_col]
                
        # Include columns that don't have mappings
        for col in data_df.columns:
            if col not in self.feature_map and col not in mapped_df.columns:
                mapped_df[col] = data_df[col]
                
        return mapped_df
    
    def _normalize_clinical_values(self, data_df):
        """Normalize values based on clinical standards."""
        # Apply reference range normalization for clinical features
        for col, range_info in self.reference_ranges.items():
            if col in data_df.columns:
                min_val, max_val = range_info.get('min', None), range_info.get('max', None)
                if min_val is not None and max_val is not None:
                    # Normalize to 0-1 range based on clinical reference ranges
                    data_df[f"{col}_normalized"] = (data_df[col] - min_val) / (max_val - min_val)
                    
                    # Flag out-of-range values
                    data_df[f"{col}_out_of_range"] = (
                        (data_df[col] < min_val) | (data_df[col] > max_val)
                    ).astype(int)
        
        return data_df
    
    def _extract_temporal_features(self, data_df):
        """Extract temporal patterns specific to real patient data."""
        # Check if we have timestamp column
        time_cols = [col for col in data_df.columns if any(
            time_term in col.lower() for time_term in ['time', 'date', 'timestamp'])]
        
        if not time_cols:
            return data_df  # No temporal features to extract
            
        time_col = time_cols[0]
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(data_df[time_col]):
            data_df[time_col] = pd.to_datetime(data_df[time_col], errors='coerce')
            
        # Add time-based features
        data_df['hour_of_day'] = data_df[time_col].dt.hour
        data_df['day_of_week'] = data_df[time_col].dt.dayofweek
        data_df['weekend'] = (data_df['day_of_week'] >= 5).astype(int)
        
        # If we have patient IDs, we can extract patient-specific temporal features
        id_cols = [col for col in data_df.columns if 'id' in col.lower()]
        if id_cols:
            id_col = id_cols[0]
            
            # Group by patient ID and extract sequence-based features
            patient_groups = data_df.sort_values(time_col).groupby(id_col)
            
            # Extract time differences between consecutive measurements
            data_df['time_since_prev'] = patient_groups[time_col].diff().dt.total_seconds() / 3600  # in hours
            
            # Extract measurement frequency features
            measurement_counts = patient_groups.size()
            data_df['measurement_count'] = data_df[id_col].map(measurement_counts)
            
        return data_df
    
    def partition_by_profile(self, data_df):
        """Segment real patient data into similar profile groups for targeted validation."""
        try:
            from sklearn.cluster import KMeans
            
            # Select only numeric columns for clustering
            numeric_df = data_df.select_dtypes(include=[np.number])
            
            # Make sure there's enough data for meaningful clustering
            if len(numeric_df.columns) < 3 or len(numeric_df) < 10:
                # Not enough data for meaningful clustering, return simple dictionary
                return {'all_patients': data_df}
                
            # Normalize features for clustering
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # Determine optimal number of clusters (2-5)
            from sklearn.metrics import silhouette_score
            best_score = -1
            best_k = 2
            
            for k in range(2, min(6, len(data_df) // 5 + 1)):  # Ensure we have enough samples per cluster
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(scaled_data)
                
                # Skip if we have any single-sample clusters
                if min(np.bincount(labels)) < 2:
                    continue
                    
                score = silhouette_score(scaled_data, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            
            # Final clustering with optimal k
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            data_df['cluster'] = kmeans.fit_predict(scaled_data)
            
            # Analyze clusters to determine likely profile types
            cluster_profiles = {}
            
            # For each cluster, analyze key features
            for cluster_id in range(best_k):
                cluster_data = data_df[data_df['cluster'] == cluster_id]
                
                # Identify distinguishing features
                feature_importance = {}
                for col in numeric_df.columns:
                    # Compare cluster mean to overall mean
                    overall_mean = data_df[col].mean()
                    cluster_mean = cluster_data[col].mean()
                    
                    # Calculate standardized difference
                    if data_df[col].std() > 0:
                        std_diff = (cluster_mean - overall_mean) / data_df[col].std()
                        feature_importance[col] = abs(std_diff)
                
                # Get top distinguishing features
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Determine likely profile type based on top features
                profile_type = "mixed"
                for feature, _ in top_features:
                    if any(term in feature.lower() for term in ['temp', 'heart', 'pressure', 'rate']):
                        profile_type = "physiological_sensitive"
                        break
                    elif any(term in feature.lower() for term in ['humid', 'pressure', 'temp_ambient']):
                        profile_type = "environmental_sensitive"
                        break
                    elif any(term in feature.lower() for term in ['stress', 'sleep', 'activity']):
                        profile_type = "behavioral_sensitive"
                        break
                    elif any(term in feature.lower() for term in ['medication', 'treatment', 'dose']):
                        profile_type = "medication_responsive"
                        break
                
                # Use numbered profile types to avoid duplicates
                profile_key = f"{profile_type}_{cluster_id}"
                cluster_profiles[profile_key] = cluster_data
                
            return cluster_profiles
        except ImportError:
            print("Sklearn not available, returning single profile group")
            return {'all_patients': data_df}
    
    def preprocess(self, data_df):
        """Apply full preprocessing pipeline to clinical data."""
        # 1. Handle missing values
        data_df = self._handle_clinical_missing_values(data_df)
        
        # 2. Map clinical features to internal feature space
        data_df = self._map_clinical_features(data_df)
        
        # 3. Normalize values based on clinical standards
        data_df = self._normalize_clinical_values(data_df)
        
        # 4. Extract temporal patterns
        data_df = self._extract_temporal_features(data_df)
        
        return data_df
        
    def analyze_drift(self, data_df, reference_df=None, drift_type='gradual'):
        """
        Analyze data drift patterns in clinical data
        
        Args:
            data_df: The current data to analyze for drift
            reference_df: Optional reference data to compare against (if None, use statistical expectations)
            drift_type: Type of drift to analyze ('sudden', 'gradual', 'recurring', 'none')
            
        Returns:
            Dictionary containing drift analysis results
        """
        # Initialize drift analysis results
        drift_analysis = {
            'drift_detected': False,
            'drift_type': drift_type,
            'drift_metrics': {},
            'affected_features': [],
            'severity': 'none'
        }
        
        # If no temporal column exists, we can't analyze drift patterns
        time_cols = [col for col in data_df.columns if any(
            time_term in col.lower() for time_term in ['time', 'date', 'timestamp'])]
        
        if not time_cols:
            drift_analysis['error'] = "No temporal column found for drift analysis"
            return drift_analysis
            
        time_col = time_cols[0]
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_dtype(data_df[time_col]):
            data_df[time_col] = pd.to_datetime(data_df[time_col], errors='coerce')
            
        # Only analyze numeric columns
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != time_col and not col.endswith('_id')]
        
        if not numeric_cols:
            drift_analysis['error'] = "No numeric features found for drift analysis"
            return drift_analysis
            
        # Sort data by time
        data_df = data_df.sort_values(by=time_col)
        
        # Split data into time windows for comparison
        if len(data_df) < 10:
            drift_analysis['error'] = "Insufficient data points for drift analysis"
            return drift_analysis
        
        # Calculate drift metrics based on drift type
        if drift_type == 'sudden':
            # For sudden drift, compare first half vs second half
            split_idx = len(data_df) // 2
            first_window = data_df.iloc[:split_idx]
            second_window = data_df.iloc[split_idx:]
            
            # Compute distribution changes for each feature
            for col in numeric_cols:
                # Skip columns with insufficient data
                if first_window[col].dropna().empty or second_window[col].dropna().empty:
                    continue
                    
                # Compute statistical tests
                try:
                    t_stat, p_value = stats.ttest_ind(
                        first_window[col].dropna(), 
                        second_window[col].dropna(),
                        equal_var=False  # Use Welch's t-test for unequal variances
                    )
                    
                    # Calculate effect size (Cohen's d)
                    mean1, mean2 = first_window[col].mean(), second_window[col].mean()
                    std1, std2 = first_window[col].std(), second_window[col].std()
                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    effect_size = abs(mean1 - mean2) / (pooled_std if pooled_std > 0 else 1)
                    
                    drift_analysis['drift_metrics'][col] = {
                        'p_value': float(p_value),
                        'effect_size': float(effect_size),
                        'significant_drift': p_value < 0.05 and effect_size > 0.3
                    }
                    
                    # Identify significant drift
                    if p_value < 0.05 and effect_size > 0.3:
                        drift_analysis['affected_features'].append(col)
                        drift_analysis['drift_detected'] = True
                except Exception as e:
                    print(f"Error computing drift for {col}: {str(e)}")
        
        elif drift_type == 'gradual':
            # For gradual drift, analyze trends over time
            for col in numeric_cols:
                # Skip columns with insufficient data
                if data_df[col].dropna().empty:
                    continue
                
                try:
                    # Compute simple linear regression over time
                    from scipy.stats import linregress
                    
                    # Convert timestamps to numeric (seconds since epoch)
                    time_numeric = data_df[time_col].astype(np.int64) // 10**9
                    
                    # Normalize to start from 0 for numerical stability
                    time_normalized = time_numeric - time_numeric.min()
                    
                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = linregress(
                        time_normalized, data_df[col].values
                    )
                    
                    # Calculate effect size as normalized slope
                    effect_size = abs(slope) * (time_normalized.max() / data_df[col].std() if data_df[col].std() > 0 else 1)
                    
                    drift_analysis['drift_metrics'][col] = {
                        'slope': float(slope),
                        'p_value': float(p_value),
                        'effect_size': float(effect_size),
                        'r_squared': float(r_value**2),
                        'significant_drift': p_value < 0.05 and effect_size > 0.3
                    }
                    
                    # Identify significant drift
                    if p_value < 0.05 and effect_size > 0.3:
                        drift_analysis['affected_features'].append(col)
                        drift_analysis['drift_detected'] = True
                except Exception as e:
                    print(f"Error computing gradual drift for {col}: {str(e)}")
                    
        elif drift_type == 'recurring':
            # For recurring drift, analyze cyclical patterns
            for col in numeric_cols:
                # Skip columns with insufficient data
                if data_df[col].dropna().empty:
                    continue
                    
                try:
                    # Extract hour of day and day of week if available
                    if 'hour_of_day' not in data_df.columns:
                        data_df['hour_of_day'] = data_df[time_col].dt.hour
                    if 'day_of_week' not in data_df.columns:
                        data_df['day_of_week'] = data_df[time_col].dt.dayofweek
                        
                    # Check for hourly patterns (group by hour and analyze variance)
                    hourly_means = data_df.groupby('hour_of_day')[col].mean()
                    daily_means = data_df.groupby('day_of_week')[col].mean()
                    
                    # Calculate coefficient of variation for hourly and daily means
                    hourly_cv = hourly_means.std() / hourly_means.mean() if hourly_means.mean() != 0 else 0
                    daily_cv = daily_means.std() / daily_means.mean() if daily_means.mean() != 0 else 0
                    
                    drift_analysis['drift_metrics'][col] = {
                        'hourly_variation': float(hourly_cv),
                        'daily_variation': float(daily_cv),
                        'significant_hourly': hourly_cv > 0.2,
                        'significant_daily': daily_cv > 0.2,
                        'significant_drift': hourly_cv > 0.2 or daily_cv > 0.2
                    }
                    
                    # Identify significant recurring patterns
                    if hourly_cv > 0.2 or daily_cv > 0.2:
                        drift_analysis['affected_features'].append(col)
                        drift_analysis['drift_detected'] = True
                except Exception as e:
                    print(f"Error computing recurring drift for {col}: {str(e)}")
                    
        # Determine overall drift severity
        if drift_analysis['drift_detected']:
            if len(drift_analysis['affected_features']) >= len(numeric_cols) * 0.5:
                drift_analysis['severity'] = 'high'
            elif len(drift_analysis['affected_features']) >= len(numeric_cols) * 0.25:
                drift_analysis['severity'] = 'medium'
            else:
                drift_analysis['severity'] = 'low'
                
        return drift_analysis

"""
Multi-modal data ingestion pipeline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json

@dataclass
class DataSourceConfig:
    """Configuration for data sources."""
    source_type: str  # diary, weather, mri, ehr, wearable
    format: str  # csv, json, dicom, text
    path: str
    required_fields: List[str]
    optional_fields: List[str]
    timestamp_field: str
    patient_id_field: str

class DataSource(ABC):
    """Base class for all data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from source."""
        pass
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess loaded data."""
        pass

class DiaryDataSource(DataSource):
    """Handle patient diary data."""
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.config.path)
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Convert timestamps
        data[self.config.timestamp_field] = pd.to_datetime(data[self.config.timestamp_field])
        
        # Handle missing values
        for field in self.config.required_fields:
            if field in data.columns:
                # Use domain-specific imputation
                if 'stress_level' in field:
                    data[field].fillna(data[field].mean(), inplace=True)
                elif 'sleep_hours' in field:
                    data[field].fillna(7.0, inplace=True)
        
        return data

class WeatherDataSource(DataSource):
    """Handle weather data."""
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.config.path)
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Convert pressure to standard units (hPa)
        if 'pressure_inches' in data.columns:
            data['pressure_hpa'] = data['pressure_inches'] * 33.8639
        
        # Handle missing values with forward fill
        data.fillna(method='ffill', inplace=True)
        
        return data

class WearableDataSource(DataSource):
    """Handle wearable sensor data."""
    
    def load_data(self) -> pd.DataFrame:
        return pd.read_json(self.config.path)
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Resample to hourly averages
        data.set_index(self.config.timestamp_field, inplace=True)
        data = data.resample('1H').mean()
        
        # Handle missing values
        data.interpolate(method='time', inplace=True)
        
        return data

class DataIngestionPipeline:
    """Multi-modal data ingestion pipeline."""
    
    def __init__(self, configs: List[DataSourceConfig]):
        self.sources = []
        for config in configs:
            if config.source_type == 'diary':
                self.sources.append(DiaryDataSource(config))
            elif config.source_type == 'weather':
                self.sources.append(WeatherDataSource(config))
            elif config.source_type == 'wearable':
                self.sources.append(WearableDataSource(config))
    
    def ingest(self) -> pd.DataFrame:
        """Run ingestion pipeline."""
        datasets = []
        
        # Load and preprocess each source
        for source in self.sources:
            data = source.load_data()
            data = source.preprocess(data)
            datasets.append(data)
        
        # Merge all datasets
        merged_data = datasets[0]
        for data in datasets[1:]:
            merged_data = pd.merge(
                merged_data, data,
                on=[source.config.timestamp_field, source.config.patient_id_field],
                how='outer'
            )
        
        return merged_data

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate ingested data."""
        validation_results = {
            'missing_required_fields': [],
            'data_quality_issues': [],
            'coverage_gaps': []
        }
        
        # Check required fields
        for source in self.sources:
            for field in source.config.required_fields:
                if field not in data.columns:
                    validation_results['missing_required_fields'].append(field)
        
        # Check data quality
        for column in data.columns:
            missing_pct = data[column].isnull().mean()
            if missing_pct > 0.2:  # More than 20% missing
                validation_results['data_quality_issues'].append({
                    'field': column,
                    'missing_percentage': missing_pct
                })
        
        # Check temporal coverage
        timestamps = pd.to_datetime(data[self.sources[0].config.timestamp_field])
        gaps = timestamps.diff() > pd.Timedelta(days=1)
        if gaps.any():
            gap_starts = timestamps[gaps].tolist()
            validation_results['coverage_gaps'] = gap_starts
        
        return validation_results

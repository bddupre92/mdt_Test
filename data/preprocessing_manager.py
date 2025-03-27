"""
Preprocessing Manager Module

This module provides a unified interface for the preprocessing pipeline, integrating
all preprocessing components and providing a simple API for the dashboard and command-line tools.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split

from data.preprocessing_pipeline import (
    PreprocessingPipeline, MissingValueHandler, OutlierHandler,
    FeatureScaler, CategoryEncoder, FeatureSelector, TimeSeriesProcessor
)
from data.advanced_feature_engineering import (
    PolynomialFeatureGenerator, DimensionalityReducer,
    StatisticalFeatureGenerator, ClusterFeatureGenerator
)
from data.domain_specific_preprocessing import (
    MedicationNormalizer, SymptomExtractor,
    TemporalPatternExtractor, ComorbidityAnalyzer
)
from data.pipeline_optimizer import PipelineOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingManager:
    """Unified interface for the preprocessing pipeline."""
    
    def __init__(self, config_path: str = None):
        """Initialize the preprocessing manager.
        
        Args:
            config_path: Path to a configuration file
        """
        self.pipeline = None
        self.optimizer = None
        self.config = {}
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            # Set default configuration
            self.config = self._default_config()
            
    def _default_config(self) -> Dict[str, Any]:
        """Create a default configuration."""
        return {
            'pipeline_name': 'default_pipeline',
            'data_split': {
                'test_size': 0.2,
                'validation_size': 0.25,  # 25% of the remaining data (20% of total)
                'random_state': 42,
                'stratify_column': None
            },
            'operations': {
                'missing_value_handler': {
                    'include': True,
                    'params': {
                        'strategy': 'mean',
                        'categorical_strategy': 'most_frequent'
                    }
                },
                'outlier_handler': {
                    'include': True,
                    'params': {
                        'method': 'zscore',
                        'threshold': 3.0,
                        'strategy': 'winsorize'
                    }
                },
                'feature_scaler': {
                    'include': True,
                    'params': {
                        'method': 'standard'
                    }
                },
                'category_encoder': {
                    'include': True,
                    'params': {
                        'method': 'onehot'
                    }
                },
                'feature_selector': {
                    'include': False,
                    'params': {
                        'method': 'variance',
                        'threshold': 0.01
                    }
                }
            },
            'advanced_operations': {
                'polynomial_feature_generator': {
                    'include': False,
                    'params': {
                        'degree': 2,
                        'interaction_only': True
                    }
                },
                'statistical_feature_generator': {
                    'include': False,
                    'params': {
                        'window_sizes': [5, 10, 20],
                        'stats': ['mean', 'std']
                    }
                },
                'dimensionality_reducer': {
                    'include': False,
                    'params': {
                        'method': 'pca',
                        'n_components': 5
                    }
                },
                'cluster_feature_generator': {
                    'include': False,
                    'params': {
                        'n_clusters': 3,
                        'method': 'kmeans'
                    }
                }
            },
            'domain_operations': {
                'medication_normalizer': {
                    'include': False,
                    'params': {
                        'medication_cols': [],
                        'dosage_cols': []
                    }
                },
                'symptom_extractor': {
                    'include': False,
                    'params': {
                        'text_cols': [],
                        'extract_severity': True
                    }
                },
                'temporal_pattern_extractor': {
                    'include': False,
                    'params': {
                        'timestamp_col': '',
                        'event_cols': [],
                        'extract_frequency': True,
                        'extract_periodicity': True
                    }
                },
                'comorbidity_analyzer': {
                    'include': False,
                    'params': {
                        'condition_cols': [],
                        'calculate_indices': True
                    }
                }
            },
            'optimization': {
                'enabled': False,
                'params': {
                    'target_col': None,
                    'task_type': 'classification',
                    'scoring': None,
                    'cv': 5,
                    'population_size': 20,
                    'generations': 10
                }
            }
        }
        
    def load_config(self, config_path: str) -> None:
        """Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._default_config()
            
    def save_config(self, config_path: str) -> None:
        """Save configuration to a file.
        
        Args:
            config_path: Path to save the configuration file
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
    def create_pipeline(self) -> PreprocessingPipeline:
        """Create a preprocessing pipeline based on the current configuration.
        
        Returns:
            The created preprocessing pipeline
        """
        pipeline = PreprocessingPipeline(name=self.config.get('pipeline_name', 'default_pipeline'))
        
        # Add basic operations
        operations_config = self.config.get('operations', {})
        
        if operations_config.get('missing_value_handler', {}).get('include', False):
            params = operations_config.get('missing_value_handler', {}).get('params', {})
            pipeline.add_operation(MissingValueHandler(**params))
            
        if operations_config.get('outlier_handler', {}).get('include', False):
            params = operations_config.get('outlier_handler', {}).get('params', {})
            pipeline.add_operation(OutlierHandler(**params))
            
        if operations_config.get('feature_scaler', {}).get('include', False):
            params = operations_config.get('feature_scaler', {}).get('params', {})
            pipeline.add_operation(FeatureScaler(**params))
            
        if operations_config.get('category_encoder', {}).get('include', False):
            params = operations_config.get('category_encoder', {}).get('params', {})
            pipeline.add_operation(CategoryEncoder(**params))
            
        # Add advanced operations
        advanced_config = self.config.get('advanced_operations', {})
        
        if advanced_config.get('polynomial_feature_generator', {}).get('include', False):
            params = advanced_config.get('polynomial_feature_generator', {}).get('params', {})
            pipeline.add_operation(PolynomialFeatureGenerator(**params))
            
        if advanced_config.get('statistical_feature_generator', {}).get('include', False):
            params = advanced_config.get('statistical_feature_generator', {}).get('params', {})
            pipeline.add_operation(StatisticalFeatureGenerator(**params))
            
        if advanced_config.get('dimensionality_reducer', {}).get('include', False):
            params = advanced_config.get('dimensionality_reducer', {}).get('params', {})
            pipeline.add_operation(DimensionalityReducer(**params))
            
        if advanced_config.get('cluster_feature_generator', {}).get('include', False):
            params = advanced_config.get('cluster_feature_generator', {}).get('params', {})
            pipeline.add_operation(ClusterFeatureGenerator(**params))
            
        # Add domain-specific operations
        domain_config = self.config.get('domain_operations', {})
        
        if domain_config.get('medication_normalizer', {}).get('include', False):
            params = domain_config.get('medication_normalizer', {}).get('params', {})
            pipeline.add_operation(MedicationNormalizer(**params))
            
        if domain_config.get('symptom_extractor', {}).get('include', False):
            params = domain_config.get('symptom_extractor', {}).get('params', {})
            pipeline.add_operation(SymptomExtractor(**params))
            
        if domain_config.get('temporal_pattern_extractor', {}).get('include', False):
            params = domain_config.get('temporal_pattern_extractor', {}).get('params', {})
            pipeline.add_operation(TemporalPatternExtractor(**params))
            
        if domain_config.get('comorbidity_analyzer', {}).get('include', False):
            params = domain_config.get('comorbidity_analyzer', {}).get('params', {})
            pipeline.add_operation(ComorbidityAnalyzer(**params))
            
        # Add feature selector last (if included)
        if operations_config.get('feature_selector', {}).get('include', False):
            params = operations_config.get('feature_selector', {}).get('params', {})
            pipeline.add_operation(FeatureSelector(**params))
            
        self.pipeline = pipeline
        return pipeline
        
    def optimize_pipeline(self, data: pd.DataFrame) -> PreprocessingPipeline:
        """Optimize the preprocessing pipeline for the given data.
        
        Args:
            data: The data to optimize the pipeline for
            
        Returns:
            The optimized preprocessing pipeline
        """
        opt_config = self.config.get('optimization', {})
        
        if not opt_config.get('enabled', False):
            logger.info("Pipeline optimization is disabled in the configuration")
            return self.create_pipeline()
            
        # Create optimizer
        params = opt_config.get('params', {})
        self.optimizer = PipelineOptimizer(**params)
        
        # Run optimization
        logger.info("Starting pipeline optimization...")
        optimized_pipeline = self.optimizer.optimize(data)
        logger.info("Pipeline optimization complete")
        
        # Update pipeline
        self.pipeline = optimized_pipeline
        
        # Update configuration with optimized parameters
        self._update_config_from_pipeline(optimized_pipeline)
        
        return optimized_pipeline
        
    def _update_config_from_pipeline(self, pipeline: PreprocessingPipeline) -> None:
        """Update configuration with parameters from an optimized pipeline.
        
        Args:
            pipeline: The optimized pipeline
        """
        # Map operation types to configuration sections
        type_to_section = {
            MissingValueHandler: ('operations', 'missing_value_handler'),
            OutlierHandler: ('operations', 'outlier_handler'),
            FeatureScaler: ('operations', 'feature_scaler'),
            CategoryEncoder: ('operations', 'category_encoder'),
            FeatureSelector: ('operations', 'feature_selector'),
            PolynomialFeatureGenerator: ('advanced_operations', 'polynomial_feature_generator'),
            StatisticalFeatureGenerator: ('advanced_operations', 'statistical_feature_generator'),
            DimensionalityReducer: ('advanced_operations', 'dimensionality_reducer'),
            ClusterFeatureGenerator: ('advanced_operations', 'cluster_feature_generator'),
            MedicationNormalizer: ('domain_operations', 'medication_normalizer'),
            SymptomExtractor: ('domain_operations', 'symptom_extractor'),
            TemporalPatternExtractor: ('domain_operations', 'temporal_pattern_extractor'),
            ComorbidityAnalyzer: ('domain_operations', 'comorbidity_analyzer')
        }
        
        # Update configuration for each operation in the pipeline
        for operation in pipeline.operations:
            for op_type, section_info in type_to_section.items():
                if isinstance(operation, op_type):
                    section, key = section_info
                    self.config[section][key] = {
                        'include': True,
                        'params': operation.get_params()
                    }
                    break
                    
    def preprocess_data(self, data: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
        """Preprocess data using the current pipeline.
        
        Args:
            data: The data to preprocess
            target_col: The target column (if any)
            
        Returns:
            A dictionary containing the preprocessed data and related information
        """
        # Create pipeline if not already created
        if self.pipeline is None:
            self.create_pipeline()
            
        # Split data if target column is provided
        if target_col is not None and target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Get split configuration
            split_config = self.config.get('data_split', {})
            test_size = split_config.get('test_size', 0.2)
            val_size = split_config.get('validation_size', 0.25)
            random_state = split_config.get('random_state', 42)
            stratify_col = split_config.get('stratify_column')
            
            # Determine stratify parameter
            stratify = y if stratify_col is None else data[stratify_col] if stratify_col in data.columns else None
            
            # Split into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
            
            # Split train into train and validation
            if stratify is not None:
                # Recalculate stratify for the training set
                stratify_train = y_train if stratify_col is None else X_train[stratify_col]
            else:
                stratify_train = None
                
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state, stratify=stratify_train
            )
            
            # Fit pipeline on training data
            X_train_transformed = self.pipeline.fit_transform(X_train)
            
            # Transform validation and test data
            X_val_transformed = self.pipeline.transform(X_val)
            X_test_transformed = self.pipeline.transform(X_test)
            
            # Return preprocessed data and related information
            return {
                'X_train': X_train_transformed,
                'y_train': y_train,
                'X_val': X_val_transformed,
                'y_val': y_val,
                'X_test': X_test_transformed,
                'y_test': y_test,
                'pipeline': self.pipeline,
                'quality_metrics': self.pipeline.get_quality_metrics()
            }
        else:
            # No target column, just preprocess all data
            transformed_data = self.pipeline.fit_transform(data)
            
            return {
                'transformed_data': transformed_data,
                'pipeline': self.pipeline,
                'quality_metrics': self.pipeline.get_quality_metrics()
            }
            
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the current pipeline.
        
        Returns:
            A dictionary containing pipeline summary information
        """
        if self.pipeline is None:
            self.create_pipeline()
            
        operations = []
        for i, op in enumerate(self.pipeline.operations):
            op_type = type(op).__name__
            operations.append({
                'index': i,
                'type': op_type,
                'params': op.get_params()
            })
            
        return {
            'name': self.pipeline.name,
            'operations_count': len(self.pipeline.operations),
            'operations': operations,
            'quality_metrics': self.pipeline.get_quality_metrics()
        }
        
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get results of pipeline optimization.
        
        Returns:
            A dictionary containing optimization results
        """
        if self.optimizer is None:
            return {
                'status': 'not_run',
                'message': 'Pipeline optimization has not been run'
            }
            
        history = self.optimizer.get_optimization_history()
        
        return {
            'status': 'complete',
            'best_score': self.optimizer.best_score,
            'generations': len(history),
            'history': history.to_dict(orient='records'),
            'best_pipeline': self.get_pipeline_summary()
        }
        
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the configuration with new values.
        
        Args:
            new_config: New configuration values
        """
        # Helper function to recursively update nested dictionaries
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v
                    
        update_nested_dict(self.config, new_config)
        
        # Reset pipeline since configuration has changed
        self.pipeline = None
        
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            The current configuration dictionary
        """
        return self.config

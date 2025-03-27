"""
Data Quality Assessment for MoE Framework

This module provides tools for assessing the quality of data in the MoE framework,
with specific attention to quality metrics that can feed into the Meta-Optimizer's
algorithm selection process.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats

logger = logging.getLogger(__name__)

class DataQualityAssessment:
    """
    Assesses data quality and provides metrics for EC algorithm selection.
    
    This class analyzes data quality aspects that are particularly relevant for
    evolutionary computation algorithms, providing metrics that can inform the
    Meta-Optimizer's algorithm selection process.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, verbose: bool = False):
        """
        Initialize the data quality assessment.
        
        Args:
            config: Optional configuration dictionary with quality thresholds
            verbose: Whether to display detailed logs during processing
        """
        self.config = config or {}
        self.verbose = verbose
        self.quality_metrics = {}
        self.quality_score = 0.0
        
        # Default quality thresholds
        self.thresholds = {
            'missing_data': {
                'good': 0.05,  # Less than 5% missing values is good
                'acceptable': 0.20,  # Less than 20% missing values is acceptable
                'poor': 0.50  # Less than 50% missing values is poor but usable
            },
            'outliers': {
                'good': 0.01,  # Less than 1% outliers is good
                'acceptable': 0.05,  # Less than 5% outliers is acceptable
                'poor': 0.10  # Less than 10% outliers is poor but usable
            },
            'class_imbalance': {
                'good': 0.20,  # Less than 20% difference between classes is good
                'acceptable': 0.40,  # Less than 40% difference is acceptable
                'poor': 0.80  # Less than 80% difference is poor but usable
            },
            'feature_correlation': {
                'good': 0.70,  # Less than 70% correlation is good
                'acceptable': 0.85,  # Less than 85% correlation is acceptable
                'poor': 0.95  # Less than 95% correlation is poor but usable
            }
        }
        
        # Override defaults with config values
        if 'thresholds' in self.config:
            for category, values in self.config['thresholds'].items():
                if category in self.thresholds:
                    self.thresholds[category].update(values)
    
    def assess_quality(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess the quality of the data.
        
        Args:
            data: DataFrame to assess
            target_column: Name of the target column (if applicable)
            
        Returns:
            Dictionary with quality assessment results
        """
        if data.empty:
            logger.warning("Empty DataFrame provided to assess_quality")
            return {'quality_score': 0.0, 'quality_level': 'unusable'}
        
        # Reset quality metrics
        self.quality_metrics = {}
        
        # Perform quality assessments
        self._assess_missing_data(data)
        self._assess_outliers(data)
        self._assess_feature_correlation(data)
        self._assess_data_distribution(data)
        
        # Assess class imbalance if target column is provided
        if target_column and target_column in data.columns:
            self._assess_class_imbalance(data, target_column)
        
        # Calculate overall quality score
        self._calculate_quality_score()
        
        # Add EC algorithm selection recommendations
        self._add_ec_algorithm_recommendations()
        
        if self.verbose:
            logger.info(f"Data quality assessment complete. Quality score: {self.quality_score:.2f}")
            logger.info(f"Quality level: {self.quality_metrics['quality_level']}")
        
        return self.quality_metrics
    
    def _assess_missing_data(self, data: pd.DataFrame) -> None:
        """
        Assess missing data in the DataFrame.
        
        Args:
            data: DataFrame to assess
        """
        # Calculate missing value percentages
        missing_counts = data.isnull().sum()
        missing_pct = missing_counts / len(data)
        overall_missing_pct = missing_pct.mean()
        
        # Identify columns with high missing values
        high_missing_cols = missing_pct[missing_pct > self.thresholds['missing_data']['acceptable']].index.tolist()
        
        # Determine quality level for missing data
        if overall_missing_pct <= self.thresholds['missing_data']['good']:
            missing_quality = 'good'
        elif overall_missing_pct <= self.thresholds['missing_data']['acceptable']:
            missing_quality = 'acceptable'
        elif overall_missing_pct <= self.thresholds['missing_data']['poor']:
            missing_quality = 'poor'
        else:
            missing_quality = 'unusable'
        
        # Store results
        self.quality_metrics['missing_data'] = {
            'overall_missing_percentage': float(overall_missing_pct * 100),
            'columns_with_high_missing': high_missing_cols,
            'column_missing_percentages': {col: float(pct * 100) for col, pct in missing_pct.items()},
            'quality_level': missing_quality
        }
    
    def _assess_outliers(self, data: pd.DataFrame) -> None:
        """
        Assess outliers in the DataFrame.
        
        Args:
            data: DataFrame to assess
        """
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            self.quality_metrics['outliers'] = {
                'overall_outlier_percentage': 0.0,
                'columns_with_outliers': [],
                'quality_level': 'not_applicable'
            }
            return
        
        # Calculate Z-scores for each numeric column
        z_scores = pd.DataFrame()
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                z_scores[col] = np.abs(stats.zscore(col_data, nan_policy='omit'))
        
        # Identify outliers (Z-score > 3)
        outliers = (z_scores > 3).sum()
        outlier_pct = outliers / z_scores.count()
        overall_outlier_pct = outlier_pct.mean()
        
        # Identify columns with high outliers
        high_outlier_cols = outlier_pct[outlier_pct > self.thresholds['outliers']['acceptable']].index.tolist()
        
        # Determine quality level for outliers
        if overall_outlier_pct <= self.thresholds['outliers']['good']:
            outlier_quality = 'good'
        elif overall_outlier_pct <= self.thresholds['outliers']['acceptable']:
            outlier_quality = 'acceptable'
        elif overall_outlier_pct <= self.thresholds['outliers']['poor']:
            outlier_quality = 'poor'
        else:
            outlier_quality = 'unusable'
        
        # Store results
        self.quality_metrics['outliers'] = {
            'overall_outlier_percentage': float(overall_outlier_pct * 100),
            'columns_with_outliers': high_outlier_cols,
            'column_outlier_percentages': {col: float(pct * 100) for col, pct in outlier_pct.items()},
            'quality_level': outlier_quality
        }
    
    def _assess_class_imbalance(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Assess class imbalance in the target column.
        
        Args:
            data: DataFrame to assess
            target_column: Name of the target column
        """
        if target_column not in data.columns:
            self.quality_metrics['class_imbalance'] = {
                'imbalance_ratio': 0.0,
                'class_distribution': {},
                'quality_level': 'not_applicable'
            }
            return
        
        # Calculate class distribution
        class_counts = data[target_column].value_counts()
        class_pct = class_counts / len(data)
        
        # Calculate imbalance ratio (difference between largest and smallest class)
        if len(class_pct) > 1:
            imbalance_ratio = class_pct.max() - class_pct.min()
        else:
            imbalance_ratio = 0.0
        
        # Determine quality level for class imbalance
        if imbalance_ratio <= self.thresholds['class_imbalance']['good']:
            imbalance_quality = 'good'
        elif imbalance_ratio <= self.thresholds['class_imbalance']['acceptable']:
            imbalance_quality = 'acceptable'
        elif imbalance_ratio <= self.thresholds['class_imbalance']['poor']:
            imbalance_quality = 'poor'
        else:
            imbalance_quality = 'unusable'
        
        # Store results
        self.quality_metrics['class_imbalance'] = {
            'imbalance_ratio': float(imbalance_ratio),
            'class_distribution': {str(cls): float(pct) for cls, pct in class_pct.items()},
            'quality_level': imbalance_quality
        }
    
    def _assess_feature_correlation(self, data: pd.DataFrame) -> None:
        """
        Assess feature correlation in the DataFrame.
        
        Args:
            data: DataFrame to assess
        """
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty or numeric_data.shape[1] < 2:
            self.quality_metrics['feature_correlation'] = {
                'max_correlation': 0.0,
                'highly_correlated_pairs': [],
                'quality_level': 'not_applicable'
            }
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()
        
        # Get upper triangle of correlation matrix (excluding diagonal)
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find highly correlated feature pairs
        high_corr_threshold = self.thresholds['feature_correlation']['acceptable']
        high_corr_pairs = [(i, j, corr_matrix.loc[i, j]) 
                          for i in corr_matrix.columns 
                          for j in corr_matrix.columns 
                          if i < j and corr_matrix.loc[i, j] > high_corr_threshold]
        
        # Calculate maximum correlation
        max_corr = upper_tri.max().max() if not upper_tri.empty else 0.0
        
        # Determine quality level for feature correlation
        if max_corr <= self.thresholds['feature_correlation']['good']:
            corr_quality = 'good'
        elif max_corr <= self.thresholds['feature_correlation']['acceptable']:
            corr_quality = 'acceptable'
        elif max_corr <= self.thresholds['feature_correlation']['poor']:
            corr_quality = 'poor'
        else:
            corr_quality = 'unusable'
        
        # Store results
        self.quality_metrics['feature_correlation'] = {
            'max_correlation': float(max_corr),
            'highly_correlated_pairs': [{'feature1': i, 'feature2': j, 'correlation': float(c)} 
                                       for i, j, c in high_corr_pairs],
            'quality_level': corr_quality
        }
    
    def _assess_data_distribution(self, data: pd.DataFrame) -> None:
        """
        Assess data distribution characteristics in the DataFrame.
        
        Args:
            data: DataFrame to assess
        """
        numeric_data = data.select_dtypes(include=np.number)
        if numeric_data.empty:
            self.quality_metrics['data_distribution'] = {
                'distribution_metrics': {},
                'quality_level': 'not_applicable'
            }
            return
        
        # Calculate distribution metrics for each numeric column
        distribution_metrics = {}
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                # Calculate basic statistics
                mean = col_data.mean()
                median = col_data.median()
                std = col_data.std()
                skewness = col_data.skew()
                kurtosis = col_data.kurtosis()
                
                # Determine if distribution is normal-like
                # (skewness close to 0 and kurtosis close to 3)
                is_normal = abs(skewness) < 0.5 and abs(kurtosis - 3) < 1.0
                
                # Determine if distribution is multimodal
                # (simplified check using histogram)
                hist, bin_edges = np.histogram(col_data, bins='auto')
                peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
                is_multimodal = len(peaks) > 1
                
                distribution_metrics[col] = {
                    'mean': float(mean),
                    'median': float(median),
                    'std': float(std),
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'is_normal': bool(is_normal),
                    'is_multimodal': bool(is_multimodal),
                    'unique_ratio': float(col_data.nunique() / len(col_data))
                }
        
        # Store results
        self.quality_metrics['data_distribution'] = {
            'distribution_metrics': distribution_metrics
        }
    
    def _calculate_quality_score(self) -> None:
        """Calculate overall quality score based on individual metrics."""
        # Define weights for each quality aspect
        weights = {
            'missing_data': 0.3,
            'outliers': 0.2,
            'class_imbalance': 0.3,
            'feature_correlation': 0.2
        }
        
        # Define scores for each quality level
        level_scores = {
            'good': 1.0,
            'acceptable': 0.7,
            'poor': 0.4,
            'unusable': 0.0,
            'not_applicable': 0.5  # Neutral score for N/A metrics
        }
        
        # Calculate weighted score
        total_weight = 0.0
        weighted_score = 0.0
        
        for aspect, weight in weights.items():
            if aspect in self.quality_metrics:
                quality_level = self.quality_metrics[aspect].get('quality_level', 'not_applicable')
                score = level_scores[quality_level]
                weighted_score += weight * score
                total_weight += weight
        
        # Normalize score if we have any applicable weights
        if total_weight > 0:
            self.quality_score = weighted_score / total_weight
        else:
            self.quality_score = 0.5  # Default to neutral if no applicable metrics
        
        # Determine overall quality level
        if self.quality_score >= 0.8:
            quality_level = 'good'
        elif self.quality_score >= 0.6:
            quality_level = 'acceptable'
        elif self.quality_score >= 0.3:
            quality_level = 'poor'
        else:
            quality_level = 'unusable'
        
        # Store overall results
        self.quality_metrics['quality_score'] = float(self.quality_score)
        self.quality_metrics['quality_level'] = quality_level
    
    def _add_ec_algorithm_recommendations(self) -> None:
        """Add EC algorithm recommendations based on quality assessment."""
        # Define algorithm recommendations based on data characteristics
        recommendations = []
        
        # Check missing data
        missing_quality = self.quality_metrics.get('missing_data', {}).get('quality_level', 'not_applicable')
        if missing_quality in ['poor', 'unusable']:
            recommendations.append({
                'algorithm': 'Differential Evolution (DE)',
                'reason': 'Robust to missing data through its mutation and crossover operations',
                'confidence': 'high' if missing_quality == 'poor' else 'medium'
            })
        
        # Check outliers
        outlier_quality = self.quality_metrics.get('outliers', {}).get('quality_level', 'not_applicable')
        if outlier_quality in ['poor', 'unusable']:
            recommendations.append({
                'algorithm': 'Evolution Strategy (ES)',
                'reason': 'Self-adaptive mutation rates can handle datasets with outliers',
                'confidence': 'high' if outlier_quality == 'poor' else 'medium'
            })
        
        # Check class imbalance
        imbalance_quality = self.quality_metrics.get('class_imbalance', {}).get('quality_level', 'not_applicable')
        if imbalance_quality in ['poor', 'unusable']:
            recommendations.append({
                'algorithm': 'Grey Wolf Optimizer (GWO)',
                'reason': 'Hierarchical leadership structure helps navigate imbalanced solution spaces',
                'confidence': 'high' if imbalance_quality == 'poor' else 'medium'
            })
        
        # Check feature correlation
        corr_quality = self.quality_metrics.get('feature_correlation', {}).get('quality_level', 'not_applicable')
        if corr_quality in ['poor', 'unusable']:
            recommendations.append({
                'algorithm': 'Ant Colony Optimization (ACO)',
                'reason': 'Effective for feature selection in datasets with high feature correlation',
                'confidence': 'high' if corr_quality == 'poor' else 'medium'
            })
        
        # Add general recommendation based on overall quality
        overall_quality = self.quality_metrics.get('quality_level', 'not_applicable')
        if overall_quality == 'good':
            recommendations.append({
                'algorithm': 'Particle Swarm Optimization (PSO)',
                'reason': 'Efficient for high-quality datasets with well-defined fitness landscapes',
                'confidence': 'high'
            })
        elif overall_quality == 'acceptable':
            recommendations.append({
                'algorithm': 'Differential Evolution (DE)',
                'reason': 'Good balance of exploration and exploitation for datasets with moderate quality issues',
                'confidence': 'high'
            })
        elif overall_quality == 'poor':
            recommendations.append({
                'algorithm': 'Artificial Bee Colony (ABC)',
                'reason': 'Robust search capabilities for datasets with significant quality issues',
                'confidence': 'medium'
            })
        
        # Store recommendations
        self.quality_metrics['ec_algorithm_recommendations'] = recommendations

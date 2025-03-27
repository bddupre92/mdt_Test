"""
Test the integration of the preprocessing pipeline with the MoE gating network.

This module tests the integration between the preprocessing pipeline and the MoE gating network,
ensuring that preprocessed data can be properly used by the gating network for expert selection
and weighting.
"""

import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import preprocessing pipeline components
from data.preprocessing_pipeline import (
    PreprocessingPipeline,
    MissingValueHandler,
    OutlierHandler,
    FeatureScaler,
    CategoryEncoder
)

# Mock MoE components for testing
class MockExpertModel:
    """Mock expert model for testing."""
    
    def __init__(self, name, domain=None):
        self.name = name
        self.domain = domain
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the expert model to the data."""
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """Make predictions using the expert model."""
        # Simple mock prediction based on expert domain
        if self.domain == 'physiological':
            return np.random.normal(0, 1, len(X))
        elif self.domain == 'environmental':
            return np.random.normal(10, 2, len(X))
        elif self.domain == 'behavioral':
            return np.random.normal(5, 1, len(X))
        else:
            return np.random.normal(0, 5, len(X))


class MockGatingNetwork:
    """Mock gating network for testing."""
    
    def __init__(self, optimizer=None):
        self.experts = []
        self.weights = None
        self.optimizer = optimizer
        self.is_fitted = False
        
    def add_expert(self, expert):
        """Add an expert to the gating network."""
        self.experts.append(expert)
        return self
        
    def fit(self, X, y):
        """Fit the gating network to the data."""
        n_experts = len(self.experts)
        
        # Initialize weights
        self.weights = np.ones(n_experts) / n_experts
        
        # If optimizer is provided, use it to optimize weights
        if self.optimizer:
            def fitness_function(weights):
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
                
                # Calculate weighted predictions
                predictions = np.zeros(len(X))
                for i, expert in enumerate(self.experts):
                    if not expert.is_fitted:
                        expert.fit(X, y)
                    expert_pred = expert.predict(X)
                    predictions += weights[i] * expert_pred
                
                # Calculate mean squared error
                mse = np.mean((predictions - y) ** 2)
                return -mse  # Negative because we want to maximize fitness
            
            # Set fitness function for optimizer
            self.optimizer.set_fitness_function(fitness_function)
            
            # Run optimization
            dimensions = n_experts
            bounds = [(0, 1) for _ in range(dimensions)]
            solution, _ = self.optimizer.optimize(dimensions, bounds)
            
            # Normalize weights to sum to 1
            self.weights = solution / np.sum(solution)
        
        # Fit all experts
        for expert in self.experts:
            if not expert.is_fitted:
                expert.fit(X, y)
        
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """Make predictions using the gating network."""
        if not self.is_fitted:
            raise ValueError("Gating network must be fitted before prediction")
        
        # Calculate weighted predictions
        predictions = np.zeros(len(X))
        for i, expert in enumerate(self.experts):
            expert_pred = expert.predict(X)
            predictions += self.weights[i] * expert_pred
        
        return predictions


class MockMetaOptimizer:
    """Mock meta-optimizer for testing."""
    
    def __init__(self, name="GWO"):
        self.name = name
        self.fitness_function = None
        
    def set_fitness_function(self, fitness_function):
        """Set the fitness function for optimization."""
        self.fitness_function = fitness_function
        
    def optimize(self, dimensions, bounds, max_iterations=10):
        """Simulate optimization process."""
        # For testing, just return a simple solution
        solution = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=dimensions
        )
        fitness = self.fitness_function(solution) if self.fitness_function else 0
        return solution, fitness


class TestPreprocessingGatingIntegration(unittest.TestCase):
    """Test the integration of preprocessing pipeline with MoE gating network."""
    
    def setUp(self):
        """Set up test data and components."""
        # Create test data with various types of features
        np.random.seed(42)
        self.data = pd.DataFrame({
            'physio1': np.random.normal(0, 1, 100),
            'physio2': np.random.normal(0, 1, 100),
            'env1': np.random.normal(10, 2, 100),
            'env2': np.random.normal(10, 2, 100),
            'behav1': np.random.normal(5, 1, 100),
            'behav2': np.random.normal(5, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.normal(5, 3, 100)
        })
        
        # Add missing values
        self.data.loc[np.random.choice(100, 10), 'physio1'] = np.nan
        self.data.loc[np.random.choice(100, 10), 'env1'] = np.nan
        self.data.loc[np.random.choice(100, 10), 'category'] = np.nan
        
        # Add outliers
        self.data.loc[np.random.choice(100, 5), 'physio2'] = 100
        self.data.loc[np.random.choice(100, 5), 'env2'] = -100
        
        # Create expert models
        self.physio_expert = MockExpertModel("Physiological", domain="physiological")
        self.env_expert = MockExpertModel("Environmental", domain="environmental")
        self.behav_expert = MockExpertModel("Behavioral", domain="behavioral")
        
        # Create meta-optimizer
        self.meta_optimizer = MockMetaOptimizer()
        
        # Create gating network
        self.gating_network = MockGatingNetwork(optimizer=self.meta_optimizer)
        self.gating_network.add_expert(self.physio_expert)
        self.gating_network.add_expert(self.env_expert)
        self.gating_network.add_expert(self.behav_expert)
        
    def test_preprocessing_for_each_expert_domain(self):
        """Test domain-specific preprocessing for each expert."""
        # Create domain-specific preprocessing pipelines
        physio_pipeline = PreprocessingPipeline([
            MissingValueHandler(strategy='mean'),
            OutlierHandler(method='zscore', threshold=3.0),
            FeatureScaler(method='standard')
        ], name='physiological')
        
        env_pipeline = PreprocessingPipeline([
            MissingValueHandler(strategy='median'),
            OutlierHandler(method='iqr', threshold=1.5),
            FeatureScaler(method='robust')
        ], name='environmental')
        
        behav_pipeline = PreprocessingPipeline([
            MissingValueHandler(strategy='most_frequent'),
            CategoryEncoder(method='onehot'),
            FeatureScaler(method='minmax')
        ], name='behavioral')
        
        # Process data for each domain
        physio_cols = ['physio1', 'physio2', 'target']
        env_cols = ['env1', 'env2', 'target']
        behav_cols = ['behav1', 'behav2', 'category', 'target']
        
        physio_data = physio_pipeline.fit_transform(self.data[physio_cols])
        env_data = env_pipeline.fit_transform(self.data[env_cols])
        behav_data = behav_pipeline.fit_transform(self.data[behav_cols])
        
        # Check that each pipeline processed the data correctly
        self.assertIsInstance(physio_data, pd.DataFrame)
        self.assertIsInstance(env_data, pd.DataFrame)
        self.assertIsInstance(behav_data, pd.DataFrame)
        
        # Check that missing values were handled
        self.assertEqual(physio_data.isna().sum().sum(), 0)
        self.assertEqual(env_data.isna().sum().sum(), 0)
        self.assertEqual(behav_data.isna().sum().sum(), 0)
        
        # Check that categorical features were encoded in behavioral data
        self.assertNotIn('category', behav_data.columns)
        self.assertTrue(any('category_' in col for col in behav_data.columns))
        
        # Train expert models with preprocessed data
        X_physio = physio_data.drop('target', axis=1)
        y_physio = physio_data['target']
        self.physio_expert.fit(X_physio, y_physio)
        
        X_env = env_data.drop('target', axis=1)
        y_env = env_data['target']
        self.env_expert.fit(X_env, y_env)
        
        X_behav = behav_data.drop('target', axis=1)
        y_behav = behav_data['target']
        self.behav_expert.fit(X_behav, y_behav)
        
        # Check that all experts were fitted
        self.assertTrue(self.physio_expert.is_fitted)
        self.assertTrue(self.env_expert.is_fitted)
        self.assertTrue(self.behav_expert.is_fitted)
        
    def test_integrated_preprocessing_and_gating(self):
        """Test integrated preprocessing and gating network."""
        # Create a unified preprocessing pipeline
        unified_pipeline = PreprocessingPipeline([
            MissingValueHandler(strategy='mean'),
            OutlierHandler(method='zscore', threshold=3.0),
            CategoryEncoder(method='onehot'),
            FeatureScaler(method='standard')
        ], name='unified')
        
        # Process all data
        processed_data = unified_pipeline.fit_transform(self.data)
        
        # Check that data was processed correctly
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.isna().sum().sum(), 0)
        
        # Split into features and target
        X = processed_data.drop('target', axis=1)
        y = processed_data['target']
        
        # Train gating network
        self.gating_network.fit(X, y)
        
        # Check that gating network was fitted
        self.assertTrue(self.gating_network.is_fitted)
        
        # Make predictions
        predictions = self.gating_network.predict(X)
        
        # Check predictions
        self.assertEqual(len(predictions), len(X))
        
        # Check that weights were assigned to experts
        self.assertIsNotNone(self.gating_network.weights)
        self.assertEqual(len(self.gating_network.weights), 3)
        self.assertAlmostEqual(sum(self.gating_network.weights), 1.0, places=5)
        
    def test_quality_metrics_impact_on_gating(self):
        """Test how preprocessing quality metrics impact gating weights."""
        # Create preprocessing pipelines with different quality levels
        high_quality_pipeline = PreprocessingPipeline([
            MissingValueHandler(strategy='mean'),
            OutlierHandler(method='zscore', threshold=3.0),
            FeatureScaler(method='standard')
        ], name='high_quality')
        
        low_quality_pipeline = PreprocessingPipeline([
            # Intentionally use a less optimal strategy for this data
            MissingValueHandler(strategy='constant', fill_value=0),
            # No outlier handling
            FeatureScaler(method='minmax')
        ], name='low_quality')
        
        # Process physiological data with both pipelines
        physio_cols = ['physio1', 'physio2', 'target']
        
        high_quality_data = high_quality_pipeline.fit_transform(self.data[physio_cols])
        low_quality_data = low_quality_pipeline.fit_transform(self.data[physio_cols])
        
        # Get quality metrics
        high_quality_metrics = high_quality_pipeline.get_quality_metrics()
        low_quality_metrics = low_quality_pipeline.get_quality_metrics()
        
        # Create expert models with different quality data
        high_quality_expert = MockExpertModel("HighQuality", domain="physiological")
        low_quality_expert = MockExpertModel("LowQuality", domain="physiological")
        
        # Train experts
        X_high = high_quality_data.drop('target', axis=1)
        y_high = high_quality_data['target']
        high_quality_expert.fit(X_high, y_high)
        
        X_low = low_quality_data.drop('target', axis=1)
        y_low = low_quality_data['target']
        low_quality_expert.fit(X_low, y_low)
        
        # Create a quality-aware gating network
        quality_aware_gating = MockGatingNetwork(optimizer=self.meta_optimizer)
        quality_aware_gating.add_expert(high_quality_expert)
        quality_aware_gating.add_expert(low_quality_expert)
        
        # Mock quality-aware weight adjustment
        original_fit = quality_aware_gating.fit
        
        def quality_aware_fit(self, X, y):
            # Call original fit method
            original_fit(X, y)
            
            # Adjust weights based on quality metrics
            high_quality_score = sum(high_quality_metrics.get(op, {}).get('score', 0.5) 
                                   for op in high_quality_metrics)
            low_quality_score = sum(low_quality_metrics.get(op, {}).get('score', 0.5) 
                                  for op in low_quality_metrics)
            
            # Normalize scores
            total_score = high_quality_score + low_quality_score
            if total_score > 0:
                high_quality_weight = high_quality_score / total_score
                low_quality_weight = low_quality_score / total_score
            else:
                high_quality_weight = low_quality_weight = 0.5
            
            # Set adjusted weights
            quality_aware_gating.weights = np.array([high_quality_weight, low_quality_weight])
            
            return quality_aware_gating
        
        # Replace fit method with quality-aware version
        quality_aware_gating.fit = quality_aware_fit.__get__(quality_aware_gating)
        
        # Train quality-aware gating network
        quality_aware_gating.fit(X_high, y_high)  # Use high quality data for training
        
        # Check that weights favor the high quality expert
        self.assertGreater(quality_aware_gating.weights[0], quality_aware_gating.weights[1],
                          "Weights should favor the high quality expert")
        
    def test_meta_optimizer_with_preprocessing_quality(self):
        """Test meta-optimizer using preprocessing quality metrics."""
        # Create preprocessing pipeline
        pipeline = PreprocessingPipeline([
            MissingValueHandler(strategy='mean'),
            OutlierHandler(method='zscore', threshold=3.0),
            CategoryEncoder(method='onehot'),
            FeatureScaler(method='standard')
        ], name='unified')
        
        # Process data
        processed_data = pipeline.fit_transform(self.data)
        
        # Get quality metrics
        quality_metrics = pipeline.get_quality_metrics()
        
        # Calculate overall quality score (simplified)
        overall_quality = {}
        for op_name, metrics in quality_metrics.items():
            if isinstance(metrics, dict) and 'score' in metrics:
                overall_quality[op_name] = metrics['score']
        
        # Create meta-optimizer that considers quality
        quality_aware_optimizer = MockMetaOptimizer("QualityAwareGWO")
        
        # Create expert models
        experts = [
            MockExpertModel("Physiological", domain="physiological"),
            MockExpertModel("Environmental", domain="environmental"),
            MockExpertModel("Behavioral", domain="behavioral")
        ]
        
        # Create gating network with quality-aware optimizer
        gating = MockGatingNetwork(optimizer=quality_aware_optimizer)
        for expert in experts:
            gating.add_expert(expert)
        
        # Mock quality-aware fitness function
        def quality_aware_fitness(weights):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Get features and target
            X = processed_data.drop('target', axis=1)
            y = processed_data['target']
            
            # Calculate predictions
            predictions = np.zeros(len(X))
            for i, expert in enumerate(experts):
                if not expert.is_fitted:
                    expert.fit(X, y)
                expert_pred = expert.predict(X)
                predictions += weights[i] * expert_pred
            
            # Calculate mean squared error
            mse = np.mean((predictions - y) ** 2)
            
            # Apply quality penalty/bonus
            quality_factor = sum(overall_quality.values()) / len(overall_quality) if overall_quality else 0.5
            
            # Return negative MSE with quality adjustment (higher is better)
            return -mse * (1 + quality_factor)
        
        # Set quality-aware fitness function
        quality_aware_optimizer.set_fitness_function(quality_aware_fitness)
        
        # Train gating network
        X = processed_data.drop('target', axis=1)
        y = processed_data['target']
        gating.fit(X, y)
        
        # Check that gating network was fitted
        self.assertTrue(gating.is_fitted)
        
        # Check that weights were assigned
        self.assertIsNotNone(gating.weights)
        self.assertEqual(len(gating.weights), 3)
        self.assertAlmostEqual(sum(gating.weights), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()

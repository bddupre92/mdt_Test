"""
Test the integration of the preprocessing pipeline with MoE and EC algorithms.

This module tests the integration of the preprocessing pipeline with the Mixture of Experts (MoE)
framework and evolutionary computation (EC) algorithms. It verifies that the preprocessing
operations can be properly configured, optimized, and integrated with the MoE framework.
"""

import os
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
    CategoryEncoder,
    FeatureSelector,
    TimeSeriesProcessor
)

# Mock EC algorithm for testing
class MockECAlgorithm:
    """Mock evolutionary computation algorithm for testing."""
    
    def __init__(self, name="mock_ec", fitness_function=None):
        self.name = name
        self.fitness_function = fitness_function or (lambda x: sum(x))
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    def optimize(self, dimensions, bounds, max_iterations=10):
        """Simulate optimization process."""
        # For testing, just return a simple solution
        self.best_solution = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=dimensions
        )
        self.best_fitness = self.fitness_function(self.best_solution)
        return self.best_solution, self.best_fitness


class TestPreprocessingMoEIntegration(unittest.TestCase):
    """Test the integration of preprocessing pipeline with MoE and EC algorithms."""
    
    def setUp(self):
        """Set up test data and components."""
        # Create test data with various types of features
        np.random.seed(42)
        self.data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'numeric1': np.random.normal(0, 1, 100),
            'numeric2': np.random.normal(10, 2, 100),
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y', 'Z'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Add missing values
        self.data.loc[np.random.choice(100, 10), 'numeric1'] = np.nan
        self.data.loc[np.random.choice(100, 10), 'categorical1'] = np.nan
        
        # Add outliers
        self.data.loc[np.random.choice(100, 5), 'numeric2'] = 100
        
        # Create a temporary directory for saving pipeline configurations
        self.temp_dir = Path("./temp_test_dir")
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary files
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()
        
    def test_feature_selector_with_ec_algorithm(self):
        """Test FeatureSelector with EC algorithm integration."""
        # Create a feature selector with evolutionary computation
        selector = FeatureSelector(
            method='evolutionary',
            use_evolutionary=True,
            ec_algorithm='aco',  # This would be replaced with a real EC algorithm in production
            target_col='target'
        )
        
        # Mock the EC algorithm integration
        # In a real implementation, this would be handled by the Meta-Optimizer
        original_fit = selector.fit
        
        def mock_fit(self, data, **kwargs):
            # Call original fit method
            original_fit(data, **kwargs)
            
            # Mock EC algorithm selection and execution
            ec_algo = MockECAlgorithm(name="ACO")
            
            # Get numeric features for selection
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'target']
            
            # Mock feature selection using EC algorithm
            dimensions = len(numeric_cols)
            bounds = [(0, 1) for _ in range(dimensions)]
            
            # Optimize (in a real implementation, this would use a fitness function
            # that evaluates model performance with different feature subsets)
            solution, _ = ec_algo.optimize(dimensions, bounds)
            
            # Select features based on solution (threshold > 0.5)
            selected_indices = [i for i, val in enumerate(solution) if val > 0.5]
            self.selected_features = [numeric_cols[i] for i in selected_indices]
            if not self.selected_features:  # Ensure at least one feature is selected
                self.selected_features = [numeric_cols[0]] if numeric_cols else []
            if not self.selected_features:  # Ensure at least one feature is selected
                self.selected_features = [numeric_cols[0]] if numeric_cols else []
            
            return self
            
        # Replace fit method with mock
        selector.fit = mock_fit.__get__(selector)
        
        # Fit the selector to the data
        selector.fit(self.data)
        
        # Check that features were selected
        self.assertTrue(len(selector.selected_features) > 0)
        
        # Transform the data
        result = selector.transform(self.data)
        
        # Check that selected features and target are in the result
        expected_columns = set(selector.selected_features + ['target'])
        self.assertTrue(expected_columns.issubset(set(result.columns)),
                      "Expected columns not found in result")
        
    def test_preprocessing_pipeline_with_ec_integration(self):
        """Test full preprocessing pipeline with EC integration."""
        # Create a pipeline with all preprocessing operations
        pipeline = PreprocessingPipeline([
            MissingValueHandler(strategy='mean'),
            OutlierHandler(method='zscore', threshold=3.0),
            FeatureScaler(method='standard'),
            CategoryEncoder(method='onehot'),
            FeatureSelector(
                method='evolutionary',
                use_evolutionary=True,
                ec_algorithm='aco',
                target_col='target'
            ),
            TimeSeriesProcessor(time_col='timestamp', lag_features=[1, 2])
        ])
        
        # Mock the EC integration for feature selection
        feature_selector = pipeline.get_operation_by_type('FeatureSelector')
        original_fit = feature_selector.fit
        
        def mock_fit(self, data, **kwargs):
            # Call original fit method
            original_fit(data, **kwargs)
            
            # Mock EC algorithm selection and execution
            ec_algo = MockECAlgorithm(name="ACO")
            
            # Get numeric features for selection
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'target']
            
            # Mock feature selection using EC algorithm
            dimensions = len(numeric_cols)
            bounds = [(0, 1) for _ in range(dimensions)]
            
            # Optimize
            solution, _ = ec_algo.optimize(dimensions, bounds)
            
            # Select features based on solution (threshold > 0.5)
            selected_indices = [i for i, val in enumerate(solution) if val > 0.5]
            self.selected_features = [numeric_cols[i] for i in selected_indices]
            if not self.selected_features:  # Ensure at least one feature is selected
                self.selected_features = [numeric_cols[0]] if numeric_cols else []
            
            return self
            
        # Replace fit method with mock
        feature_selector.fit = mock_fit.__get__(feature_selector)
        
        # Fit and transform the data
        result = pipeline.fit_transform(self.data)
        
        # Check that the pipeline processed the data
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
        # Check that quality metrics were recorded
        metrics = pipeline.get_quality_metrics()
        self.assertGreater(len(metrics), 0)
        
        # Save and load the pipeline configuration
        config_path = self.temp_dir / "pipeline_config.json"
        pipeline.save_config(str(config_path))
        
        # Load the configuration into a new pipeline
        loaded_pipeline = PreprocessingPipeline.load_config(str(config_path))
        
        # Check that the loaded pipeline has the same operations
        self.assertEqual(len(loaded_pipeline.operations), len(pipeline.operations))
        
    def test_pipeline_optimization_with_ec(self):
        """Test optimization of pipeline parameters using EC algorithms."""
        # Create a basic pipeline
        pipeline = PreprocessingPipeline([
            MissingValueHandler(),
            OutlierHandler(threshold=3.0),  # Ensure threshold is a float
            FeatureScaler()
        ])
        
        # Define parameter space for optimization
        param_space = {
            'MissingValueHandler': {
                'strategy': ['mean', 'median', 'most_frequent'],
            },
            'OutlierHandler': {
                'method': ['zscore', 'iqr'],
                'threshold': (1.0, 5.0),  # Range for continuous parameter
                'strategy': ['winsorize', 'remove']
            },
            'FeatureScaler': {
                'method': ['minmax', 'standard', 'robust']
            }
        }
        
        # Mock EC algorithm for pipeline optimization
        ec_algo = MockECAlgorithm(name="DE")
        
        # Mock fitness function that evaluates pipeline performance
        def fitness_function(params_vector):
            # Convert parameter vector to pipeline configuration
            config = self._vector_to_config(params_vector, param_space, pipeline)
            
            # Apply configuration to pipeline
            for op_name, op_params in config.items():
                op = pipeline.get_operation_by_type(op_name)
                if op:
                    op.set_params(op_params)
            
            # Fit and transform data
            try:
                result = pipeline.fit_transform(self.data)
                
                # Simple fitness: number of columns in result (more is better)
                # In a real implementation, this would be a more meaningful metric
                return len(result.columns)
            except Exception:
                # Penalize configurations that cause errors
                return 0
        
        # Set fitness function
        ec_algo.fitness_function = fitness_function
        
        # Define dimensions and bounds for optimization
        # This is a simplified representation; in reality, this would be more complex
        dimensions = 5  # Total number of parameters to optimize
        bounds = [(0, 1) for _ in range(dimensions)]  # Normalized bounds
        
        # Run optimization
        solution, fitness = ec_algo.optimize(dimensions, bounds)
        
        # Apply optimized parameters to pipeline
        optimized_config = self._vector_to_config(solution, param_space, pipeline)
        for op_name, op_params in optimized_config.items():
            op = pipeline.get_operation_by_type(op_name)
            if op:
                op.set_params(op_params)
        
        # Verify that the pipeline works with optimized parameters
        result = pipeline.fit_transform(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
    def _vector_to_config(self, vector, param_space, pipeline):
        """Convert a parameter vector to a pipeline configuration.
        
        This is a simplified implementation for testing purposes.
        In a real implementation, this would be more sophisticated.
        """
        config = {}
        vector_index = 0
        
        for op_name, params in param_space.items():
            op_config = {}
            
            for param_name, param_values in params.items():
                if isinstance(param_values, list):
                    # Categorical parameter
                    index = int(vector[vector_index] * len(param_values)) % len(param_values)
                    op_config[param_name] = param_values[index]
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Continuous parameter
                    min_val, max_val = param_values
                    op_config[param_name] = float(min_val) + float(vector[vector_index]) * (float(max_val) - float(min_val))
                
                vector_index += 1
                if vector_index >= len(vector):
                    break
            
            config[op_name] = op_config
            if vector_index >= len(vector):
                break
        
        return config


if __name__ == '__main__':
    unittest.main()

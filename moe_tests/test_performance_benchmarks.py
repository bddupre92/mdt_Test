"""
Tests for the performance benchmarking framework.

This module tests the performance benchmarking capabilities, including
measurement of execution time, memory usage, and accuracy across different
MoE configurations.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from moe_framework.benchmark.performance_benchmarks import BenchmarkRunner
from moe_framework.workflow.moe_pipeline import MoEPipeline


class TestPerformanceBenchmarks:
    """Tests for the performance benchmarking framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary output directory
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_benchmark_results')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X = np.random.rand(n_samples, n_features)
        y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, n_samples)
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.data = pd.DataFrame(X, columns=feature_names)
        self.data['target'] = y
        
        # Define base configuration
        self.base_config = {
            'experts': {
                'expert1': {
                    'type': 'simple',
                    'model': 'linear'
                },
                'expert2': {
                    'type': 'simple',
                    'model': 'gradient_boosting'
                }
            },
            'gating_network': {
                'type': 'fixed',
                'weights': {
                    'expert1': 0.6,
                    'expert2': 0.4
                }
            },
            'integration': {
                'strategy': 'weighted_average'
            }
        }
        
        # Create benchmark runner
        self.runner = BenchmarkRunner(
            output_dir=self.temp_dir,
            create_visualizations=False  # Disable for testing
        )
    
    def teardown_method(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('moe_framework.workflow.moe_pipeline.MoEPipeline')
    def test_benchmark_pipeline_configuration(self, mock_pipeline):
        """Test benchmarking a single pipeline configuration."""
        # Configure the mock
        mock_instance = Mock()
        mock_pipeline.return_value = mock_instance
        mock_instance.train.return_value = None
        mock_instance.predict.return_value = np.random.rand(len(self.data))
        
        # Run benchmark with minimal iterations
        features = [col for col in self.data.columns if col != 'target']
        result = self.runner.benchmark_pipeline_configuration(
            config=self.base_config,
            data=self.data,
            features=features,
            target='target',
            name='test_config',
            iterations=1
        )
        
        # Verify result structure
        assert 'name' in result
        assert 'config' in result
        assert 'performance' in result
        assert 'average' in result['performance']
        
        # Verify metrics
        assert 'accuracy' in result['performance']['average']
        assert 'rmse' in result['performance']['average']['accuracy']
        assert 'r2' in result['performance']['average']['accuracy']
        
        assert 'execution_time' in result['performance']['average']
        assert 'total' in result['performance']['average']['execution_time']
        
        assert 'memory_usage' in result['performance']['average']
        assert 'delta' in result['performance']['average']['memory_usage']
        
        # Verify file was saved
        result_files = os.listdir(self.temp_dir)
        assert len(result_files) > 0
    
    @patch('moe_framework.workflow.moe_pipeline.MoEPipeline')
    def test_benchmark_integration_strategies(self, mock_pipeline):
        """Test benchmarking different integration strategies."""
        # Configure the mock
        mock_instance = Mock()
        mock_pipeline.return_value = mock_instance
        mock_instance.train.return_value = None
        mock_instance.predict.return_value = np.random.rand(len(self.data))
        
        # Run benchmark with minimal iterations
        features = [col for col in self.data.columns if col != 'target']
        
        with patch.object(self.runner, 'benchmark_pipeline_configuration') as mock_benchmark:
            # Configure mock to return a properly structured result
            mock_benchmark.return_value = {
                'name': 'test',
                'performance': {
                    'average': {
                        'accuracy': {'rmse': 0.1, 'r2': 0.8},
                        'execution_time': {'total': 1.0, 'training': 0.8, 'prediction': 0.2},
                        'memory_usage': {'delta': 10.0}
                    }
                }
            }
            
            # Run benchmark
            results = self.runner.benchmark_integration_strategies(
                data=self.data,
                features=features,
                target='target',
                base_config=self.base_config
            )
            
            # Should call benchmark_pipeline_configuration for each strategy
            assert mock_benchmark.call_count == 4  # four strategies
            
            # Verify results structure
            assert 'weighted_average' in results
            assert 'confidence_based' in results
            assert 'quality_aware' in results
            assert 'advanced_fusion' in results
    
    @patch('moe_framework.workflow.moe_pipeline.MoEPipeline')
    def test_compare_all_results(self, mock_pipeline):
        """Test comparing benchmark results."""
        # Add some test results
        self.runner.results = [
            {
                'name': 'config1',
                'performance': {
                    'average': {
                        'accuracy': {'rmse': 0.1, 'r2': 0.8},
                        'execution_time': {'total': 1.0, 'training': 0.8, 'prediction': 0.2},
                        'memory_usage': {'delta': 10.0}
                    }
                }
            },
            {
                'name': 'config2',
                'performance': {
                    'average': {
                        'accuracy': {'rmse': 0.2, 'r2': 0.7},
                        'execution_time': {'total': 1.5, 'training': 1.2, 'prediction': 0.3},
                        'memory_usage': {'delta': 15.0}
                    }
                }
            }
        ]
        
        # Run comparison
        comparison = self.runner.compare_all_results()
        
        # Verify comparison structure
        assert 'configurations' in comparison
        assert len(comparison['configurations']) == 2
        assert 'accuracy' in comparison
        assert 'rmse' in comparison['accuracy']
        assert 'execution_time' in comparison
        assert 'memory_usage' in comparison
        
        # Verify values
        assert comparison['accuracy']['rmse'] == [0.1, 0.2]
        assert comparison['accuracy']['r2'] == [0.8, 0.7]
        
        # Verify file was saved
        assert os.path.exists(os.path.join(self.temp_dir, "comparison_summary.json"))
    
    def test_data_generator_function(self):
        """Test the data generator function used for scaling benchmarks."""
        # Define a simple data generator function
        def data_generator(size):
            np.random.seed(42)
            n_features = 5
            X = np.random.rand(size, n_features)
            y = np.sum(X, axis=1) + np.random.normal(0, 0.1, size)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
            df['target'] = y
            return df
        
        # Test with different sizes
        for size in [10, 50, 100]:
            df = data_generator(size)
            assert len(df) == size
            assert 'target' in df.columns
            assert len(df.columns) == 6  # 5 features + target


if __name__ == "__main__":
    pytest.main(["-v", __file__])

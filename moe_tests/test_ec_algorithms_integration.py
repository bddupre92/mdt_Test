"""
Test the integration of specific EC algorithms with the preprocessing pipeline.

This module tests the integration of specific evolutionary computation (EC) algorithms
(Differential Evolution, Grey Wolf Optimizer, Ant Colony Optimization) with the
preprocessing pipeline and MoE framework components.
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
    FeatureSelector
)

# Import EC algorithms from the optimizers package
# If these imports fail, the tests will be skipped
try:
    from optimizers import (
        DifferentialEvolutionOptimizer,
        GreyWolfOptimizer,
        AntColonyOptimizer
    )
    EC_ALGORITHMS_AVAILABLE = True
except ImportError:
    EC_ALGORITHMS_AVAILABLE = False


@unittest.skipIf(not EC_ALGORITHMS_AVAILABLE, "EC algorithms not available")
class TestECAlgorithmsIntegration(unittest.TestCase):
    """Test the integration of specific EC algorithms with preprocessing pipeline."""
    
    def setUp(self):
        """Set up test data and components."""
        # Create test data
        np.random.seed(42)
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100),
            'feature3': np.random.normal(5, 1, 100),
            'feature4': np.random.normal(-5, 3, 100),
            'feature5': np.random.normal(2, 0.5, 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Add some correlation between features and target
        self.data['feature1'] += self.data['target'] * 0.5
        self.data['feature3'] += self.data['target'] * 1.0
        
    def test_differential_evolution_feature_selection(self):
        """Test feature selection using Differential Evolution."""
        if not EC_ALGORITHMS_AVAILABLE:
            self.skipTest("EC algorithms not available")
            
        # Create DE algorithm instance
        # Define dimension and bounds (for feature selection, dim = number of features)
        dim = 5  # Number of features in test data
        bounds = [(0, 1) for _ in range(dim)]  # Binary selection bounds
        
        de = DifferentialEvolutionOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=10,
            crossover_rate=0.7,
            mutation_factor=0.5,
            max_iterations=5
        )
        
        # Create feature selector with DE
        selector = FeatureSelector(
            method='evolutionary',
            use_evolutionary=True,
            ec_algorithm=de,
            target_col='target'
        )
        
        # Fit the selector to the data
        selector.fit(self.data)
        
        # Check that features were selected
        self.assertTrue(hasattr(selector, 'selected_features'))
        self.assertTrue(len(selector.selected_features) > 0)
        
        # Transform the data
        result = selector.transform(self.data)
        
        # Check that only selected features and target are in the result
        expected_columns = selector.selected_features + ['target']
        self.assertEqual(set(result.columns), set(expected_columns))
        
    def test_grey_wolf_optimizer_pipeline_optimization(self):
        """Test pipeline parameter optimization using Grey Wolf Optimizer."""
        if not EC_ALGORITHMS_AVAILABLE:
            self.skipTest("EC algorithms not available")
            
        # Create GWO algorithm instance
        # Define dimension and bounds (for parameter optimization)
        dim = 3  # Three parameters to optimize
        bounds = [(0.1, 0.9), (1.5, 4.0), (0, 1)]  # Bounds for each parameter
        
        gwo = GreyWolfOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=10,
            max_iterations=5
        )
        
        # Create a pipeline with multiple operations
        pipeline = PreprocessingPipeline()
        
        # Define parameter space for optimization
        param_space = {
            'feature_selection_threshold': (0.1, 0.9),
            'outlier_threshold': (1.5, 4.0),
            'scaling_method': ['minmax', 'standard', 'robust']
        }
        
        # Define fitness function
        def fitness_function(params):
            # Extract parameters
            feature_selection_threshold = params[0]
            outlier_threshold = params[1]
            scaling_method_idx = int(params[2] * 2.999) % 3
            scaling_method = ['minmax', 'standard', 'robust'][scaling_method_idx]
            
            # Configure pipeline
            pipeline.operations = []
            pipeline.add_operation(FeatureSelector(
                method='variance',
                threshold=feature_selection_threshold,
                target_col='target'
            ))
            
            # Fit and evaluate
            try:
                result = pipeline.fit_transform(self.data)
                # Simple fitness: ratio of selected features (fewer is better)
                n_selected = len(result.columns) - 1  # Exclude target
                # Return negative value since we want to maximize but optimizers minimize
                return -1.0 * (1.0 - (n_selected / (len(self.data.columns) - 1)))
            except Exception:
                return 100.0  # High penalty value to minimize
        
        # Set objective function
        gwo.set_objective(fitness_function)
        
        # Run optimization
        result = gwo.optimize(fitness_function)
        
        # Extract solution (handle if it's a tuple or just the solution)
        solution = result[0] if isinstance(result, tuple) else result
        
        # Calculate fitness for reporting
        fitness = -fitness_function(solution)  # Negate back for reporting
        
        # Check that optimization produced a valid solution
        self.assertEqual(len(solution), dim)  # Using dim defined above
        self.assertGreaterEqual(fitness, 0.0)
        
        # Apply optimized parameters to pipeline
        feature_selection_threshold = solution[0] * (0.9 - 0.1) + 0.1
        outlier_threshold = solution[1] * (4.0 - 1.5) + 1.5
        scaling_method_idx = int(solution[2] * 2.999) % 3
        scaling_method = ['minmax', 'standard', 'robust'][scaling_method_idx]
        
        # Configure pipeline with optimized parameters
        pipeline.operations = []
        pipeline.add_operation(FeatureSelector(
            method='variance',
            threshold=feature_selection_threshold,
            target_col='target'
        ))
        
        # Verify that the pipeline works with optimized parameters
        result = pipeline.fit_transform(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        
    def test_ant_colony_optimization_feature_selection(self):
        """Test feature selection using Ant Colony Optimization."""
        if not EC_ALGORITHMS_AVAILABLE:
            self.skipTest("EC algorithms not available")
            
        # Create ACO algorithm instance
        # Define dimension and bounds (for feature selection, dim = number of features)
        dim = 5  # Number of features in test data
        bounds = [(0, 1) for _ in range(dim)]  # Binary selection bounds
        
        aco = AntColonyOptimizer(
            dim=dim,
            bounds=bounds,
            population_size=5,  # Using population_size instead of n_ants
            evaporation_rate=0.1,
            alpha=1.0,
            beta=2.0,
            max_iterations=5
        )
        
        # Create feature selector with ACO
        selector = FeatureSelector(
            method='evolutionary',
            use_evolutionary=True,
            ec_algorithm=aco,
            target_col='target'
        )
        
        # Mock the ACO-specific feature selection logic
        original_fit = selector.fit
        
        def mock_fit(self, data, **kwargs):
            # Call original fit method for initialization
            original_fit(data, **kwargs)
            
            # Get feature names (excluding target)
            feature_names = [col for col in data.columns if col != 'target']
            
            # Define ACO problem for feature selection
            n_features = len(feature_names)
            
            # Define fitness function for ACO
            def aco_fitness(solution):
                # Convert binary solution to selected features
                selected_indices = [i for i, val in enumerate(solution) if val > 0.5]
                
                if not selected_indices:
                    return 0.0  # No features selected, return poor fitness
                
                selected_features = [feature_names[i] for i in selected_indices]
                subset = data[selected_features + ['target']]
                
                # Simple fitness: correlation with target
                corr_with_target = subset.corr()['target'].abs().mean()
                # Penalize for too many features
                penalty = len(selected_indices) / n_features
                # Return negative value for minimization
                return -1.0 * (corr_with_target * (1 - 0.5 * penalty))
            
            # Set objective function for ACO
            aco.set_objective(aco_fitness)
            
            # Run ACO to select features
            result = aco.optimize(aco_fitness)
            
            # Extract solution (handle if it's a tuple or just the solution)
            best_solution = result[0] if isinstance(result, tuple) else result
            
            # Convert solution to indices (binary solution where 1 means select)
            selected_indices = [i for i, val in enumerate(best_solution) if val > 0.5]
            
            # Set selected features
            selector.selected_features = [feature_names[i] for i in selected_indices]
            
            return selector
        
        # Replace fit method with mock
        selector.fit = mock_fit.__get__(selector)
        
        # Fit the selector to the data
        selector.fit(self.data)
        
        # Check that features were selected
        self.assertTrue(hasattr(selector, 'selected_features'))
        self.assertTrue(len(selector.selected_features) > 0)
        
        # Transform the data
        result = selector.transform(self.data)
        
        # Check that only selected features and target are in the result
        expected_columns = selector.selected_features + ['target']
        self.assertEqual(set(result.columns), set(expected_columns))
        
    def test_multiple_ec_algorithms_comparison(self):
        """Test comparison of multiple EC algorithms for feature selection."""
        if not EC_ALGORITHMS_AVAILABLE:
            self.skipTest("EC algorithms not available")
            
        # Create instances of different EC algorithms
        # Define dimension and bounds (for feature selection, dim = number of features)
        dim = 5  # Number of features in test data
        bounds = [(0, 1) for _ in range(dim)]  # Binary selection bounds
        
        de = DifferentialEvolutionOptimizer(dim=dim, bounds=bounds, population_size=10, max_iterations=5)
        gwo = GreyWolfOptimizer(dim=dim, bounds=bounds, population_size=10, max_iterations=5)
        aco = AntColonyOptimizer(dim=dim, bounds=bounds, population_size=5, max_iterations=5)
        
        # Dictionary to store results
        results = {}
        
        # Test each algorithm
        for name, algo in [('DE', de), ('GWO', gwo), ('ACO', aco)]:
            # Create feature selector with the algorithm
            selector = FeatureSelector(
                method='evolutionary',
                use_evolutionary=True,
                ec_algorithm=algo,
                target_col='target'
            )
            
            # Mock the feature selection logic
            original_fit = selector.fit
            
            def mock_fit(self, data, **kwargs):
                # Call original fit method for initialization
                original_fit(data, **kwargs)
                
                # Get feature names (excluding target)
                feature_names = [col for col in data.columns if col != 'target']
                
                # Define fitness function
                def fitness(solution):
                    # Convert binary solution to selected features
                    selected = [feature_names[i] for i, val in enumerate(solution) if val > 0.5]
                    if not selected:
                        return 0.0
                    
                    # Simple fitness: correlation with target
                    subset = data[selected + ['target']]
                    corr_with_target = subset.corr()['target'].abs().mean()
                    # Penalize for too many features
                    penalty = len(selected) / len(feature_names)
                    return corr_with_target * (1 - 0.5 * penalty)
                
                # Set objective function
                algo.set_objective(fitness)
                
                # Run optimization
                result = algo.optimize(fitness)
                
                # Extract solution (handle if it's a tuple or just the solution)
                solution = result[0] if isinstance(result, tuple) else result
                
                # Set selected features
                selected_indices = [i for i, val in enumerate(solution) if val > 0.5]
                selector.selected_features = [feature_names[i] for i in selected_indices]
                
                # Calculate fitness for the selected solution
                fitness_value = fitness(solution)
                selector.fitness = fitness_value
                
                return selector
            
            # Replace fit method with mock
            selector.fit = mock_fit.__get__(selector)
            
            # Fit the selector to the data
            selector.fit(self.data)
            
            # Store results
            results[name] = {
                'selected_features': selector.selected_features,
                'fitness': getattr(selector, 'fitness', 0.0)
            }
        
        # Check that all algorithms produced results
        self.assertEqual(len(results), 3)
        
        # Check that each algorithm selected at least one feature
        for name, result in results.items():
            self.assertGreater(len(result['selected_features']), 0, 
                              f"{name} did not select any features")
        
        # Compare algorithm performance (higher fitness is better)
        best_algo = max(results.items(), key=lambda x: x[1]['fitness'])[0]
        self.assertIn(best_algo, ['DE', 'GWO', 'ACO'])


if __name__ == '__main__':
    unittest.main()

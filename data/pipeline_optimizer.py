"""
Pipeline Optimizer Module

This module provides automated pipeline configuration optimization using evolutionary algorithms.
It integrates with the MoE framework's evolutionary computation components to find optimal
preprocessing pipelines for specific datasets and tasks.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score, f1_score, roc_auc_score
import copy
import logging

from data.preprocessing_pipeline import (
    PreprocessingPipeline, MissingValueHandler, OutlierHandler,
    FeatureScaler, CategoryEncoder, FeatureSelector, TimeSeriesProcessor
)
from data.advanced_feature_engineering import (
    PolynomialFeatureGenerator, DimensionalityReducer,
    StatisticalFeatureGenerator, ClusterFeatureGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOptimizer:
    """Optimize preprocessing pipeline configurations using evolutionary algorithms."""
    
    def __init__(self, 
                 target_col: str = None,
                 task_type: str = 'classification',
                 scoring: str = None,
                 cv: int = 5,
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 algorithm: str = 'genetic',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """Initialize the pipeline optimizer.
        
        Args:
            target_col: Target column for supervised optimization
            task_type: Type of task. Options: 'classification', 'regression'
            scoring: Scoring metric. If None, uses accuracy for classification, neg_mean_squared_error for regression
            cv: Number of cross-validation folds
            population_size: Size of the population in the evolutionary algorithm
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            algorithm: Evolutionary algorithm to use. Options: 'genetic', 'de', 'gwo'
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for cross-validation
        """
        self.target_col = target_col
        self.task_type = task_type
        self.scoring = scoring
        self.cv = cv
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Set default scoring if not provided
        if self.scoring is None:
            if self.task_type == 'classification':
                self.scoring = 'accuracy'
            else:
                self.scoring = 'neg_mean_squared_error'
                
        # Initialize evolutionary algorithm components
        self.best_pipeline = None
        self.best_score = float('-inf')
        self.evolution_history = []
        
        # Define operation space
        self.operation_space = self._define_operation_space()
        
    def _define_operation_space(self) -> Dict[str, Dict[str, Any]]:
        """Define the space of possible preprocessing operations and their parameters."""
        return {
            'missing_value_handler': {
                'include': [True, False],
                'params': {
                    'strategy': ['mean', 'median', 'most_frequent', 'constant'],
                    'categorical_strategy': ['most_frequent', 'constant']
                }
            },
            'outlier_handler': {
                'include': [True, False],
                'params': {
                    'method': ['zscore', 'iqr'],
                    'threshold': [2.0, 2.5, 3.0, 3.5],
                    'strategy': ['winsorize', 'remove']
                }
            },
            'feature_scaler': {
                'include': [True, False],
                'params': {
                    'method': ['minmax', 'standard', 'robust']
                }
            },
            'category_encoder': {
                'include': [True, False],
                'params': {
                    'method': ['label', 'onehot']
                }
            },
            'feature_selector': {
                'include': [True, False],
                'params': {
                    'method': ['variance', 'kbest', 'evolutionary'],
                    'threshold': [0.0, 0.01, 0.05, 0.1],
                    'k': [5, 10, 15, 'auto'],
                    'use_evolutionary': [True, False]
                }
            },
            'polynomial_feature_generator': {
                'include': [True, False],
                'params': {
                    'degree': [2, 3],
                    'interaction_only': [True, False]
                }
            },
            'dimensionality_reducer': {
                'include': [True, False],
                'params': {
                    'method': ['pca', 'kernel_pca'],
                    'n_components': [2, 3, 5, 10, 'auto']
                }
            },
            'statistical_feature_generator': {
                'include': [True, False],
                'params': {
                    'window_sizes': [[5, 10], [10, 20], [5, 10, 20]],
                    'stats': [['mean', 'std'], ['mean', 'std', 'min', 'max'], ['mean', 'std', 'skew']]
                }
            },
            'cluster_feature_generator': {
                'include': [True, False],
                'params': {
                    'n_clusters': [2, 3, 5, 8],
                    'method': ['kmeans']
                }
            }
        }
        
    def _create_pipeline_from_config(self, config: Dict[str, Any]) -> PreprocessingPipeline:
        """Create a preprocessing pipeline from a configuration dictionary."""
        pipeline = PreprocessingPipeline(name=f"optimized_{int(time.time())}")
        
        # Add operations based on configuration
        if config.get('missing_value_handler', {}).get('include', False):
            params = config.get('missing_value_handler', {}).get('params', {})
            pipeline.add_operation(MissingValueHandler(**params))
            
        if config.get('outlier_handler', {}).get('include', False):
            params = config.get('outlier_handler', {}).get('params', {})
            pipeline.add_operation(OutlierHandler(**params))
            
        if config.get('feature_scaler', {}).get('include', False):
            params = config.get('feature_scaler', {}).get('params', {})
            pipeline.add_operation(FeatureScaler(**params))
            
        if config.get('category_encoder', {}).get('include', False):
            params = config.get('category_encoder', {}).get('params', {})
            pipeline.add_operation(CategoryEncoder(**params))
            
        if config.get('polynomial_feature_generator', {}).get('include', False):
            params = config.get('polynomial_feature_generator', {}).get('params', {})
            pipeline.add_operation(PolynomialFeatureGenerator(**params))
            
        if config.get('statistical_feature_generator', {}).get('include', False):
            params = config.get('statistical_feature_generator', {}).get('params', {})
            pipeline.add_operation(StatisticalFeatureGenerator(**params))
            
        if config.get('dimensionality_reducer', {}).get('include', False):
            params = config.get('dimensionality_reducer', {}).get('params', {})
            # Handle 'auto' n_components
            if params.get('n_components') == 'auto':
                params['n_components'] = min(5, max(2, len(pipeline.operations) * 2))
            pipeline.add_operation(DimensionalityReducer(**params))
            
        if config.get('cluster_feature_generator', {}).get('include', False):
            params = config.get('cluster_feature_generator', {}).get('params', {})
            pipeline.add_operation(ClusterFeatureGenerator(**params))
            
        if config.get('feature_selector', {}).get('include', False):
            params = config.get('feature_selector', {}).get('params', {})
            # Set target column for supervised feature selection
            params['target_col'] = self.target_col
            # Handle 'auto' k
            if params.get('k') == 'auto':
                params['k'] = min(10, max(5, len(pipeline.operations) * 3))
            pipeline.add_operation(FeatureSelector(**params))
            
        return pipeline
        
    def _evaluate_pipeline(self, pipeline: PreprocessingPipeline, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate a preprocessing pipeline using cross-validation."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        try:
            # Apply preprocessing
            X_transformed = pipeline.fit_transform(X)
            
            # Skip if no features remain after preprocessing
            if X_transformed.shape[1] == 0:
                return float('-inf')
                
            # Create a simple model based on task type
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
                
            # Evaluate using cross-validation
            scores = cross_val_score(
                model, X_transformed, y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs
            )
            
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Error evaluating pipeline: {e}")
            return float('-inf')
            
    def _generate_random_config(self) -> Dict[str, Any]:
        """Generate a random pipeline configuration."""
        config = {}
        
        for op_name, op_space in self.operation_space.items():
            include = np.random.choice(op_space['include'])
            
            if include:
                params = {}
                for param_name, param_values in op_space['params'].items():
                    params[param_name] = np.random.choice(param_values)
                    
                config[op_name] = {
                    'include': True,
                    'params': params
                }
            else:
                config[op_name] = {
                    'include': False,
                    'params': {}
                }
                
        return config
        
    def _mutate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a pipeline configuration."""
        mutated_config = copy.deepcopy(config)
        
        # Randomly select an operation to mutate
        op_name = np.random.choice(list(self.operation_space.keys()))
        op_space = self.operation_space[op_name]
        
        # Flip inclusion with probability mutation_rate
        if np.random.random() < self.mutation_rate:
            include = not mutated_config.get(op_name, {}).get('include', False)
            
            if include:
                params = {}
                for param_name, param_values in op_space['params'].items():
                    params[param_name] = np.random.choice(param_values)
                    
                mutated_config[op_name] = {
                    'include': True,
                    'params': params
                }
            else:
                mutated_config[op_name] = {
                    'include': False,
                    'params': {}
                }
        # Otherwise, mutate parameters if operation is included
        elif mutated_config.get(op_name, {}).get('include', False):
            params = mutated_config[op_name]['params']
            
            # Randomly select a parameter to mutate
            param_name = np.random.choice(list(op_space['params'].keys()))
            param_values = op_space['params'][param_name]
            
            # Select a new value different from the current one
            current_value = params.get(param_name)
            new_values = [v for v in param_values if v != current_value]
            
            if new_values:
                params[param_name] = np.random.choice(new_values)
                
        return mutated_config
        
    def _crossover_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two pipeline configurations."""
        child1 = copy.deepcopy(config1)
        child2 = copy.deepcopy(config2)
        
        # Randomly select operations for crossover
        op_names = list(self.operation_space.keys())
        np.random.shuffle(op_names)
        crossover_point = np.random.randint(1, len(op_names))
        
        # Perform crossover
        for i, op_name in enumerate(op_names):
            if i >= crossover_point:
                child1[op_name] = copy.deepcopy(config2.get(op_name, {'include': False, 'params': {}}))
                child2[op_name] = copy.deepcopy(config1.get(op_name, {'include': False, 'params': {}}))
                
        return child1, child2
        
    def optimize(self, data: pd.DataFrame) -> PreprocessingPipeline:
        """Optimize the preprocessing pipeline for the given data.
        
        Args:
            data: The data to optimize the pipeline for
            
        Returns:
            The optimized preprocessing pipeline
        """
        if self.target_col is None or self.target_col not in data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
            
        # Split data into features and target
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]
        
        # Initialize population with random configurations
        population = [self._generate_random_config() for _ in range(self.population_size)]
        
        # Evolutionary optimization
        for generation in range(self.generations):
            logger.info(f"Generation {generation+1}/{self.generations}")
            
            # Evaluate all configurations
            scores = []
            pipelines = []
            
            for config in population:
                pipeline = self._create_pipeline_from_config(config)
                score = self._evaluate_pipeline(pipeline, X, y)
                scores.append(score)
                pipelines.append(pipeline)
                
                # Update best pipeline if needed
                if score > self.best_score:
                    self.best_score = score
                    self.best_pipeline = pipeline
                    logger.info(f"New best score: {self.best_score:.4f}")
                    
            # Record history
            self.evolution_history.append({
                'generation': generation,
                'best_score': self.best_score,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores)
            })
            
            # Create next generation
            next_population = []
            
            # Elitism: keep the best configuration
            best_idx = np.argmax(scores)
            next_population.append(population[best_idx])
            
            # Generate the rest of the population
            while len(next_population) < self.population_size:
                # Selection
                idx1, idx2 = np.random.choice(len(population), size=2, replace=False, 
                                             p=np.array(scores) / np.sum(scores))
                parent1 = population[idx1]
                parent2 = population[idx2]
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover_configs(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                    
                # Mutation
                child1 = self._mutate_config(child1)
                child2 = self._mutate_config(child2)
                
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
                    
            # Update population
            population = next_population
            
        logger.info(f"Optimization complete. Best score: {self.best_score:.4f}")
        
        return self.best_pipeline
        
    def get_optimization_history(self) -> pd.DataFrame:
        """Get the optimization history as a DataFrame."""
        return pd.DataFrame(self.evolution_history)
        
    def plot_optimization_history(self, save_path: str = None):
        """Plot the optimization history."""
        import matplotlib.pyplot as plt
        
        history = self.get_optimization_history()
        
        plt.figure(figsize=(10, 6))
        plt.plot(history['generation'], history['best_score'], 'b-', label='Best Score')
        plt.plot(history['generation'], history['mean_score'], 'r--', label='Mean Score')
        plt.fill_between(
            history['generation'],
            history['mean_score'] - history['std_score'],
            history['mean_score'] + history['std_score'],
            alpha=0.2, color='r'
        )
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('Pipeline Optimization History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
        
    def save_best_pipeline(self, filepath: str):
        """Save the best pipeline configuration to a file."""
        if self.best_pipeline is None:
            raise ValueError("No best pipeline available. Run optimize() first.")
            
        self.best_pipeline.save_config(filepath)
        
    def load_best_pipeline(self, filepath: str) -> PreprocessingPipeline:
        """Load a pipeline configuration from a file."""
        self.best_pipeline = PreprocessingPipeline.load_config(filepath)
        return self.best_pipeline

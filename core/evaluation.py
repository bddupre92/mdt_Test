import os
import numpy as np
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
import argparse
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import warnings

from utils.json_utils import save_json
from utils.plotting import save_plot, setup_plot_style

def create_synthetic_data(n_samples=100, n_features=5, random_state=None):
    """
    Create synthetic data for testing evaluation functions
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate
    n_features : int, optional
        Number of features to generate
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X (features) and y (targets)
    """
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=0.1, 
        random_state=random_state
    )
    
    return X, y

def create_default_model(model_type='rf', random_state=None):
    """
    Create a default model for evaluation
    
    Parameters:
    -----------
    model_type : str, optional
        Type of model to create ('rf', 'gb', 'lr', 'svm')
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    object
        Scikit-learn compatible model
    """
    if model_type == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_type == 'gb':
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif model_type == 'lr':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_type == 'svm':
        from sklearn.svm import SVR
        model = SVR()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def evaluate_model(model, X_test, y_test, metrics=None):
    """
    Evaluate a model on test data
    
    Parameters:
    -----------
    model : object
        Trained scikit-learn compatible model
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test targets
    metrics : list, optional
        List of metrics to compute
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with evaluation results
    """
    if metrics is None:
        metrics = ['mse', 'mae', 'r2']
    
    results = {}
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    if 'mse' in metrics:
        results['mse'] = float(mean_squared_error(y_test, y_pred))
    
    if 'rmse' in metrics:
        results['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    if 'mae' in metrics:
        results['mae'] = float(mean_absolute_error(y_test, y_pred))
    
    if 'r2' in metrics:
        results['r2'] = float(r2_score(y_test, y_pred))
    
    if 'explained_variance' in metrics:
        from sklearn.metrics import explained_variance_score
        results['explained_variance'] = float(explained_variance_score(y_test, y_pred))
    
    if 'max_error' in metrics:
        from sklearn.metrics import max_error
        results['max_error'] = float(max_error(y_test, y_pred))
    
    return results

def evaluate_with_cross_validation(model, X, y, cv=5, metrics=None, random_state=None):
    """
    Evaluate a model using cross-validation
    
    Parameters:
    -----------
    model : object
        Scikit-learn compatible model
    X : np.ndarray or pd.DataFrame
        Features
    y : np.ndarray or pd.Series
        Targets
    cv : int, optional
        Number of cross-validation folds
    metrics : list, optional
        List of metrics to compute
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Dictionary with cross-validation results
    """
    if metrics is None:
        metrics = ['mse', 'mae', 'r2']
    
    results = {}
    
    for metric in metrics:
        if metric == 'mse':
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            results['mse'] = {
                'mean': float(-np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(-np.min(scores)),
                'max': float(-np.max(scores))
            }
        
        elif metric == 'rmse':
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            scores = np.sqrt(-scores)
            results['rmse'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
            
        elif metric == 'mae':
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
            results['mae'] = {
                'mean': float(-np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(-np.min(scores)),
                'max': float(-np.max(scores))
            }
            
        elif metric == 'r2':
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            results['r2'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
        
        elif metric == 'explained_variance':
            scores = cross_val_score(model, X, y, cv=cv, scoring='explained_variance')
            results['explained_variance'] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
    
    return results

def run_evaluation(args=None, model=None, X_test=None, y_test=None):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    args : argparse.Namespace, optional
        Command-line arguments (used when called from main)
    model : object, optional
        Trained model to evaluate (if None, creates a default model)
    X_test : np.ndarray or pd.DataFrame, optional
        Test features (if None, creates synthetic data)
    y_test : np.ndarray or pd.Series, optional
        Test targets (if None, creates synthetic data)
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with evaluation results
    """
    logging.info("Running model evaluation")
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # If called from main with args
    if args is not None and model is None:
        # Create a simple model for demonstration
        model_type = getattr(args, 'model', 'rf')
        random_seed = getattr(args, 'random_seed', 42)
        
        model = create_default_model(model_type=model_type, random_state=random_seed)
        
        # Load or create data
        if hasattr(args, 'data_path') and args.data_path:
            try:
                data = pd.read_csv(args.data_path)
                # Assume last column is target
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
                
                # Split data if needed
                if X_test is None or y_test is None:
                    test_size = getattr(args, 'test_size', 0.2)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_seed
                    )
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
            except Exception as e:
                logging.error(f"Error loading data: {str(e)}")
                logging.info("Falling back to synthetic data")
                
                # Create synthetic data
                X, y = create_synthetic_data(
                    n_samples=getattr(args, 'n_samples', 100),
                    n_features=getattr(args, 'n_features', 5),
                    random_state=random_seed
                )
                
                # Split data
                test_size = getattr(args, 'test_size', 0.2)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_seed
                )
                
                # Train model
                model.fit(X_train, y_train)
        else:
            # Create synthetic data
            X, y = create_synthetic_data(
                n_samples=getattr(args, 'n_samples', 100),
                n_features=getattr(args, 'n_features', 5),
                random_state=random_seed
            )
            
            # Split data
            test_size = getattr(args, 'test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed
            )
            
            # Train model
            model.fit(X_train, y_train)
    
    # If model is provided but no test data
    elif model is not None and (X_test is None or y_test is None):
        # Create synthetic data
        X, y = create_synthetic_data(n_samples=100, n_features=5)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model if needed
        if not hasattr(model, 'predict'):
            model.fit(X_train, y_train)
    
    # If no model is provided but test data is provided
    elif model is None and X_test is not None and y_test is not None:
        # Create a default model
        model = create_default_model()
        
        # Create training data
        X_train, y_train = create_synthetic_data(n_samples=100, n_features=X_test.shape[1])
        
        # Train model
        model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = getattr(args, 'metrics', ['mse', 'mae', 'r2']) if args is not None else ['mse', 'mae', 'r2']
    
    # Get evaluation results
    evaluation_results = evaluate_model(model, X_test, y_test, metrics=metrics)
    
    # Run cross-validation if requested
    if args is not None and getattr(args, 'cross_validation', False):
        n_folds = getattr(args, 'n_folds', 5)
        
        # Get cross-validation results
        cv_results = evaluate_with_cross_validation(
            model, 
            X_test, 
            y_test, 
            cv=n_folds, 
            metrics=metrics
        )
        
        # Add cross-validation results to evaluation results
        evaluation_results['cross_validation'] = cv_results
    
    # Create visualizations if requested
    if args is not None and getattr(args, 'visualize', False):
        try:
            setup_plot_style()
            
            # Scatter plot of actual vs predicted values
            y_pred = model.predict(X_test)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            
            # Add diagonal line (perfect predictions)
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Actual vs Predicted Values')
            
            # Add metrics as text
            metrics_text = '\n'.join([f"{k}: {v:.4f}" for k, v in evaluation_results.items() if not isinstance(v, dict)])
            ax.text(
                0.05, 
                0.95, 
                metrics_text, 
                transform=ax.transAxes, 
                fontsize=12, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Save plot
            save_plot(fig, 'actual_vs_predicted', plot_type='evaluation')
            
            # If model has feature importances, create a feature importance plot
            if hasattr(model, 'feature_importances_'):
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get feature names
                if hasattr(X_test, 'columns'):
                    feature_names = X_test.columns
                else:
                    feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
                
                # Sort importances
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot importances
                ax.bar(range(len(importances)), importances[indices])
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance')
                
                # Save plot
                save_plot(fig, 'feature_importance', plot_type='evaluation')
        
        except Exception as e:
            logging.error(f"Error creating visualizations: {str(e)}")
    
    # Save evaluation results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'evaluation_results_{timestamp}.json'
    
    save_json(evaluation_results, results_file)
    logging.info(f"Evaluation results saved to {results_file}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print("===================")
    for metric, value in evaluation_results.items():
        if isinstance(value, dict):
            print(f"\n{metric}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{metric}: {value:.4f}")
    
    return evaluation_results

def run_optimization_and_evaluation(args=None, 
                                   data_path=None, 
                                   save_dir='results',
                                   n_runs=30,
                                   max_evals=1000):
    """
    Run complete optimization and evaluation pipeline with visualizations.
    
    Parameters:
    -----------
    args : argparse.Namespace, optional
        Command-line arguments
    data_path : str, optional
        Path to data file
    save_dir : str, optional
        Directory to save results
    n_runs : int, optional
        Number of optimization runs
    max_evals : int, optional
        Maximum number of function evaluations per run
        
    Returns:
    --------
    Dict[str, Any]
        Results of optimization and evaluation
    """
    logging.info("Running optimization and evaluation pipeline")
    
    # Parse arguments if provided
    if args is not None:
        data_path = getattr(args, 'data_path', data_path)
        save_dir = getattr(args, 'save_dir', save_dir)
        n_runs = getattr(args, 'n_runs', n_runs)
        max_evals = getattr(args, 'max_evals', max_evals)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'optimization': {},
        'evaluation': {},
        'metadata': {
            'data_path': data_path,
            'n_runs': n_runs,
            'max_evals': max_evals
        }
    }
    
    # Load data if provided, otherwise create synthetic data
    if data_path:
        try:
            data = pd.read_csv(data_path)
            # Assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.info("Falling back to synthetic data")
            X, y = create_synthetic_data(n_samples=100, n_features=5)
    else:
        X, y = create_synthetic_data(n_samples=100, n_features=5)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    logging.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples, {X_train.shape[1]} features")
    
    # Load optimizers
    try:
        from core.optimization import create_optimizer, create_objective_function
        
        # Define optimizers to use
        optimizer_types = ['DE', 'ES', 'ACO', 'GWO']
        
        # Define bounds for hyperparameters
        bounds = [
            (10, 200),    # n_estimators
            (1, 10),      # max_depth
            (1, 20),      # min_samples_split
            (1, 20),      # min_samples_leaf
            (0.01, 1.0)   # max_features (percentage)
        ]
        
        # Define objective function for hyperparameter optimization
        def objective_function(x):
            """Objective function for hyperparameter optimization"""
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            
            # Convert hyperparameters
            n_estimators = int(x[0])
            max_depth = int(x[1])
            min_samples_split = int(x[2])
            min_samples_leaf = int(x[3])
            max_features = x[4]  # Percentage of features
            
            # Create model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
            
            # Perform cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            
            # Return negative mean score (minimize)
            return -np.mean(scores)
        
        # Run optimization for each optimizer type
        optimization_results = {}
        
        for opt_type in optimizer_types:
            logging.info(f"Running optimization with {opt_type}")
            
            # Create optimizer
            optimizer = create_optimizer(
                optimizer_type=opt_type,
                dim=len(bounds),
                bounds=bounds,
                population_size=30
            )
            
            # Run optimization
            try:
                best_solution, best_score = optimizer.optimize(
                    objective_function,
                    max_evals=max_evals,
                    verbose=True
                )
                
                # Convert hyperparameters
                n_estimators = int(best_solution[0])
                max_depth = int(best_solution[1])
                min_samples_split = int(best_solution[2])
                min_samples_leaf = int(best_solution[3])
                max_features = best_solution[4]  # Percentage of features
                
                # Create model with optimized hyperparameters
                model = create_default_model(model_type='rf')
                model.set_params(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                evaluation_result = evaluate_model(model, X_test, y_test)
                
                # Store results
                optimization_results[opt_type] = {
                    'hyperparameters': {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'max_features': max_features
                    },
                    'optimization_score': float(-best_score),
                    'evaluation': evaluation_result
                }
                
                logging.info(f"Optimization completed for {opt_type}")
                
            except Exception as e:
                logging.error(f"Error optimizing with {opt_type}: {str(e)}")
                optimization_results[opt_type] = {
                    'error': str(e)
                }
        
        # Determine best optimizer based on test performance
        best_optimizer = None
        best_r2 = -float('inf')
        
        for opt_type, result in optimization_results.items():
            if 'evaluation' in result and 'r2' in result['evaluation']:
                r2 = result['evaluation']['r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_optimizer = opt_type
        
        results['optimization'] = optimization_results
        results['best_optimizer'] = best_optimizer
        
        # Create comparative visualization
        if len(optimization_results) > 0:
            # Prepare data for visualization
            opt_types = []
            mse_values = []
            r2_values = []
            
            for opt_type, result in optimization_results.items():
                if 'evaluation' in result and 'mse' in result['evaluation'] and 'r2' in result['evaluation']:
                    opt_types.append(opt_type)
                    mse_values.append(result['evaluation']['mse'])
                    r2_values.append(result['evaluation']['r2'])
            
            if len(opt_types) > 0:
                setup_plot_style()
                
                # Create MSE plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(opt_types, mse_values)
                ax.set_xlabel('Optimizer')
                ax.set_ylabel('Mean Squared Error (MSE)')
                ax.set_title('Model Performance by Optimizer (MSE)')
                
                save_plot(fig, 'optimizer_comparison_mse', plot_type='evaluation')
                
                # Create R2 plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(opt_types, r2_values)
                ax.set_xlabel('Optimizer')
                ax.set_ylabel('R² Score')
                ax.set_title('Model Performance by Optimizer (R²)')
                
                save_plot(fig, 'optimizer_comparison_r2', plot_type='evaluation')
        
    except Exception as e:
        logging.error(f"Error in optimization and evaluation pipeline: {str(e)}")
        results['error'] = str(e)
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(save_dir, f'optimization_evaluation_results_{timestamp}.json')
    
    save_json(results, results_file)
    logging.info(f"Optimization and evaluation results saved to {results_file}")
    
    # Print summary
    print("\nOptimization and Evaluation Summary:")
    print("====================================")
    
    if 'best_optimizer' in results and results['best_optimizer']:
        print(f"Best optimizer: {results['best_optimizer']}")
        
        best_opt = results['best_optimizer']
        if best_opt in results['optimization'] and 'evaluation' in results['optimization'][best_opt]:
            best_result = results['optimization'][best_opt]
            
            print("\nBest Hyperparameters:")
            for param, value in best_result['hyperparameters'].items():
                print(f"  {param}: {value}")
            
            print("\nPerformance Metrics:")
            for metric, value in best_result['evaluation'].items():
                print(f"  {metric}: {value:.4f}")
    
    return results

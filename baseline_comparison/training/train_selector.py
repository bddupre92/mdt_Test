"""
Training functions for the SATzilla-inspired algorithm selector

This module provides functions for training the SATzilla-inspired algorithm selector,
including problem generation, feature extraction, and model training.
"""

import time
import logging
import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import copy
import json
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_problem_variations(
    base_problems: List,
    num_problems: int,
    dimensions: int,
    random_seed: int = 42
) -> List:
    """
    Generate problem variations for training
    
    Args:
        base_problems: List of base benchmark problems
        num_problems: Total number of problems to generate
        dimensions: Number of dimensions for the problems
        random_seed: Random seed for reproducibility
        
    Returns:
        List of problem variations
    """
    np.random.seed(random_seed)
    
    # If we already have enough problems, return them
    if len(base_problems) >= num_problems:
        return base_problems[:num_problems]
    
    logger.info(f"Generating {num_problems} problem variations from {len(base_problems)} base problems")
    
    # Create variations of the base problems
    variations = []
    
    # Add all base problems
    variations.extend(base_problems)
    
    # Number of variations to create
    num_variations = num_problems - len(base_problems)
    
    # Create problem variations
    for i in range(num_variations):
        # Select a random base problem
        base_idx = np.random.randint(0, len(base_problems))
        base_problem = base_problems[base_idx]
        
        # Create a variation by modifying the problem
        try:
            # Try to deep copy the problem
            variation = copy.deepcopy(base_problem)
            
            # Modify the variation (e.g., add noise, shift, etc.)
            # This will depend on the specific problem implementation
            # For now, we'll just use a simple approach
            
            # Apply a transformation to the problem if supported
            if hasattr(variation, 'apply_transformation'):
                # Apply random transformation
                transformation_type = np.random.choice(['shift', 'rotate', 'noise', 'scale'])
                variation.apply_transformation(transformation_type)
                variation.name = f"{base_problem.name}_variation_{i+1}_{transformation_type}"
            else:
                # If transformation not supported, use a wrapper
                variation = ProblemVariation(
                    base_problem, 
                    shift=np.random.uniform(-2.0, 2.0, dimensions),
                    scale=np.random.uniform(0.5, 2.0),
                    noise_level=np.random.uniform(0.0, 0.2),
                    name=f"{base_problem.name}_variation_{i+1}"
                )
            
            variations.append(variation)
            
        except Exception as e:
            logger.warning(f"Error creating problem variation: {e}")
            # If deep copy fails, add the original problem again
            variations.append(base_problems[base_idx])
    
    logger.info(f"Generated {len(variations)} problem variations")
    
    return variations[:num_problems]

class ProblemVariation:
    """Wrapper class for creating variations of benchmark problems"""
    
    def __init__(
        self,
        base_problem,
        shift: Optional[np.ndarray] = None,
        scale: float = 1.0,
        noise_level: float = 0.0,
        name: Optional[str] = None
    ):
        """
        Create a problem variation
        
        Args:
            base_problem: The base problem to create a variation of
            shift: Vector to shift the problem (if None, no shift)
            scale: Scale factor for the problem
            noise_level: Amount of noise to add
            name: Name for the variation
        """
        self.base_problem = base_problem
        self.shift = shift
        self.scale = scale
        self.noise_level = noise_level
        self.name = name or f"{base_problem.name}_variation"
        
        # Copy problem properties
        self.dims = base_problem.dims
        if hasattr(base_problem, 'bounds'):
            self.bounds = base_problem.bounds
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the problem at point x
        
        Args:
            x: Point to evaluate
            
        Returns:
            Function value at x
        """
        # Apply transformations
        x_transformed = x.copy()
        
        # Apply shift if provided
        if self.shift is not None:
            x_transformed = x_transformed - self.shift
        
        # Apply scale
        x_transformed = x_transformed / self.scale
        
        # Evaluate base problem
        value = self.base_problem.evaluate(x_transformed)
        
        # Add noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * abs(value))
            value += noise
        
        return value

def train_satzilla_selector(
    selector,
    problems: List,
    max_evaluations: int = 1000,
    export_features: bool = True,
    export_dir: Optional[Path] = None
) -> Any:
    """
    Train the SATzilla-inspired selector
    
    Args:
        selector: The SATzilla-inspired selector to train
        problems: List of problems to train on
        max_evaluations: Maximum evaluations per algorithm
        export_features: Whether to export extracted features
        export_dir: Directory to export data (if None, don't export)
        
    Returns:
        Trained selector
    """
    logger.info(f"Training SATzilla-inspired selector with {len(problems)} problems")
    start_time = time.time()
    
    # Train the selector
    selector.train(problems, max_evaluations=max_evaluations)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Export features if requested
    if export_features and export_dir:
        export_training_data(selector, export_dir)
        
        # Generate feature importance visualization if SHAP available
        try:
            import shap
            logger.info("Generating feature importance visualization using SHAP")
            
            # For each algorithm model
            for alg in selector.algorithms:
                if selector.models[alg] is not None:
                    # Convert features to DataFrame for better visualization
                    feature_names = list(selector.X_train[0].keys())
                    X_train_df = pd.DataFrame([list(f.values()) for f in selector.X_train], 
                                              columns=feature_names)
                    
                    # Get SHAP values
                    explainer = shap.TreeExplainer(selector.models[alg])
                    shap_values = explainer.shap_values(X_train_df)
                    
                    # Create SHAP summary plot
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_train_df, plot_type="bar",
                                     title=f"Feature Importance for {alg}")
                    plt.tight_layout()
                    plt.savefig(export_dir / f"feature_importance_{alg}.png", dpi=300, bbox_inches='tight')
                    plt.close()
        except ImportError:
            logger.warning("SHAP not available. Skipping feature importance visualization.")
            # Create basic feature importance plot using built-in methods
            create_basic_feature_importance(selector, export_dir)
    
    # Return the trained selector
    return selector

def export_training_data(selector, export_dir: Path) -> None:
    """
    Export training data and features
    
    Args:
        selector: The trained SATzilla-inspired selector
        export_dir: Directory to export data
    """
    logger.info(f"Exporting training data to {export_dir}")
    
    # Create data directory
    data_dir = export_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Check what format X_train is in
    if hasattr(selector, 'X_train'):
        if isinstance(selector.X_train, np.ndarray):
            # Get feature names from the selector
            if hasattr(selector, 'feature_names') and selector.feature_names:
                feature_names = selector.feature_names
                # Create DataFrame from numpy array
                features_df = pd.DataFrame(selector.X_train, columns=feature_names)
                features_df.to_csv(data_dir / "problem_features.csv", index=False)
            else:
                logger.warning("No feature names found in selector. Using generic column names.")
                # Create DataFrame with generic column names
                features_df = pd.DataFrame(selector.X_train)
                features_df.to_csv(data_dir / "problem_features.csv", index=False)
        elif isinstance(selector.X_train, list) and len(selector.X_train) > 0:
            # Check if elements are dictionaries
            if isinstance(selector.X_train[0], dict):
                # Old format - list of dictionaries
                feature_names = list(selector.X_train[0].keys())
                features_df = pd.DataFrame([list(f.values()) for f in selector.X_train], 
                                        columns=feature_names)
                features_df.to_csv(data_dir / "problem_features.csv", index=False)
            else:
                # List of arrays or other format
                logger.warning("Unexpected X_train format. Using generic export.")
                features_df = pd.DataFrame(np.array(selector.X_train))
                features_df.to_csv(data_dir / "problem_features.csv", index=False)
        else:
            logger.warning(f"Unexpected X_train type: {type(selector.X_train)}. Skipping feature export.")
            features_df = None
    else:
        logger.warning("No X_train attribute found in selector. Skipping feature export.")
        features_df = None
    
    # Export performance data for each algorithm
    if hasattr(selector, 'y_train') and isinstance(selector.y_train, dict):
        performance_data = {}
        for alg in selector.algorithms:
            if alg in selector.y_train:
                performance_data[alg] = selector.y_train[alg]
        
        # Convert to DataFrame
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_csv(data_dir / "algorithm_performance.csv", index=False)
        else:
            logger.warning("No performance data found. Skipping performance export.")
    else:
        logger.warning("No y_train dictionary found in selector. Skipping performance export.")
    
    # Export model evaluation metrics
    try:
        model_metrics = evaluate_selector_models(selector)
        with open(data_dir / "model_metrics.json", "w") as f:
            json.dump(model_metrics, f, indent=2)
        logger.info(f"Model metrics exported to {data_dir / 'model_metrics.json'}")
    except Exception as e:
        logger.warning(f"Error exporting model metrics: {e}")
    
    logger.info(f"Training data exported to {data_dir}")

def evaluate_selector_models(selector) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the trained selector models
    
    Args:
        selector: Trained SATzilla-inspired selector
        
    Returns:
        Dictionary of evaluation metrics for each algorithm
    """
    logger.info("Evaluating trained models")
    
    try:
        evaluation_results = {}
        
        # Check if selector has trained models
        if not hasattr(selector, 'models') or not selector.models:
            logger.warning("No models found in selector")
            return {}
            
        # Check if we have feature names
        if not hasattr(selector, 'feature_names') or not selector.feature_names:
            logger.warning("No feature names found in selector")
            return {}
            
        # Check if we have training data
        if not hasattr(selector, 'X_train') or selector.X_train is None or len(selector.X_train) == 0:
            logger.warning("No training data found in selector")
            return {}
        
        # Prepare X data in DataFrame format
        if isinstance(selector.X_train, np.ndarray):
            X_train_df = pd.DataFrame(selector.X_train, columns=selector.feature_names)
        elif isinstance(selector.X_train, list) and isinstance(selector.X_train[0], dict):
            # Handle old format where X_train is a list of dictionaries
            X_train_df = pd.DataFrame(selector.X_train)
        else:
            logger.warning(f"Unexpected X_train format: {type(selector.X_train)}")
            return {}
            
        # For each algorithm, evaluate the model
        for alg, model in selector.models.items():
            if model is None:
                logger.warning(f"No model found for algorithm {alg}")
                continue
                
            # Check if we have performance data
            if not hasattr(selector, 'y_train') or not isinstance(selector.y_train, dict) or alg not in selector.y_train:
                logger.warning(f"No performance data found for algorithm {alg}")
                continue
                
            # Get the performance data
            y_train = np.array(selector.y_train[alg])
            
            # Calculate metrics
            metrics = {}
            
            try:
                # Use model's score method
                score = model.score(X_train_df, y_train)
                metrics["r2_score"] = float(score)
                
                # Make predictions
                y_pred = model.predict(X_train_df)
                
                # Calculate mean squared error
                mse = np.mean((y_train - y_pred) ** 2)
                metrics["mse"] = float(mse)
                
                # Calculate mean absolute error
                mae = np.mean(np.abs(y_train - y_pred))
                metrics["mae"] = float(mae)
                
                evaluation_results[alg] = metrics
            except Exception as e:
                logger.warning(f"Error evaluating model for {alg}: {e}")
        
        return evaluation_results
    except Exception as e:
        logger.warning(f"Error evaluating models: {e}")
        return {}

def create_basic_feature_importance(selector, export_dir: Path) -> None:
    """
    Create basic feature importance visualizations
    
    Args:
        selector: Trained selector
        export_dir: Directory to save visualizations
    """
    logger.info("Creating basic feature importance visualizations")
    
    try:
        # Ensure visualization directory exists
        viz_dir = export_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if selector has trained models
        if not hasattr(selector, 'models') or not selector.models:
            logger.warning("No models found in selector. Skipping feature importance visualization.")
            return
            
        # Check if we have feature names
        if not hasattr(selector, 'feature_names') or not selector.feature_names:
            logger.warning("No feature names found in selector. Skipping feature importance visualization.")
            return
        
        # Get feature names from selector
        feature_names = selector.feature_names
        
        # For each algorithm with a trained model
        for alg, model in selector.models.items():
            if model is None:
                continue
                
            # Check if model has feature importances
            if not hasattr(model, 'feature_importances_'):
                logger.warning(f"Model for {alg} does not have feature_importances_ attribute")
                continue
                
            # Extract feature importances
            importances = model.feature_importances_
            
            # Create plot
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importances)[::-1]
            
            # Limit to top 20 features or all if less
            n_features = min(len(feature_names), 20)
            plt.barh(range(n_features), importances[indices[:n_features]], align='center')
            plt.yticks(range(n_features), [feature_names[i] for i in indices[:n_features]])
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance for {alg}')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(viz_dir / f"feature_importance_{alg}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        logger.info(f"Basic feature importance visualizations saved to {viz_dir}")
    except Exception as e:
        logger.warning(f"Error creating feature importance visualizations: {e}")
        logger.warning(traceback.format_exc())

def save_trained_selector(selector, file_path: Path) -> None:
    """
    Save the trained selector to a file
    
    Args:
        selector: The trained selector
        file_path: Path to save the selector
    """
    logger.info(f"Saving trained selector to {file_path}")
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Ensure the file has .joblib extension
    if not str(file_path).endswith('.joblib'):
        file_path = Path(f"{str(file_path)}.joblib")
    
    # Save using the selector's save_model method for consistency with BaselineComparisonCommand
    if hasattr(selector, 'save_model'):
        selector.save_model(str(file_path))
    else:
        # Fallback to direct joblib dump if save_model not available
        joblib.dump(selector, file_path)
    
    logger.info(f"Selector saved to {file_path}")

def load_trained_selector(file_path: Path):
    """
    Load a trained selector from a file
    
    Args:
        file_path: Path to the saved selector
        
    Returns:
        Loaded selector
    """
    logger.info(f"Loading trained selector from {file_path}")
    
    # Handle different file extensions
    if not file_path.exists() and not str(file_path).endswith('.joblib'):
        # Try with .joblib extension
        joblib_path = Path(f"{str(file_path)}.joblib")
        if joblib_path.exists():
            file_path = joblib_path
            logger.info(f"Using file with .joblib extension: {file_path}")
    
    try:
        # First try to load with joblib
        selector = joblib.load(file_path)
        
        # If we loaded model data (dictionary) instead of a selector instance
        if isinstance(selector, dict) and 'models' in selector and 'is_trained' in selector:
            # Create a new selector and load the model data
            from baseline_comparison.baseline_algorithms.satzilla_inspired import SatzillaInspiredSelector
            new_selector = SatzillaInspiredSelector()
            new_selector.load_model(str(file_path))
            selector = new_selector
    except Exception as e:
        logger.warning(f"Failed to load with joblib: {e}. Trying pickle as fallback.")
        # Fallback to pickle for backward compatibility
        with open(file_path, 'rb') as f:
            selector = pickle.load(f)
    
    logger.info(f"Selector loaded from {file_path}")
    
    return selector 
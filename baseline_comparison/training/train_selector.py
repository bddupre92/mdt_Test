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
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import copy
import json
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

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
    
    # Export features as CSV
    feature_names = list(selector.X_train[0].keys())
    features_df = pd.DataFrame([list(f.values()) for f in selector.X_train], 
                               columns=feature_names)
    features_df.to_csv(data_dir / "problem_features.csv", index=False)
    
    # Export performance data for each algorithm
    performance_data = {}
    for alg in selector.algorithms:
        performance_data[alg] = selector.y_train[alg]
    
    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv(data_dir / "algorithm_performance.csv", index=False)
    
    # Export model evaluation metrics
    model_metrics = evaluate_selector_models(selector)
    with open(data_dir / "model_metrics.json", "w") as f:
        json.dump(model_metrics, f, indent=2)
    
    logger.info(f"Training data exported to {data_dir}")

def evaluate_selector_models(selector) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the performance of the trained models
    
    Args:
        selector: The trained selector
        
    Returns:
        Dictionary of evaluation metrics for each algorithm model
    """
    logger.info("Evaluating trained models")
    
    metrics = {}
    
    # For each algorithm model
    for alg in selector.algorithms:
        if selector.models[alg] is not None:
            # Get training data
            X = np.array([list(f.values()) for f in selector.X_train])
            X_scaled = selector.scaler.transform(X)
            y = np.array(selector.y_train[alg])
            
            # Get model predictions
            y_pred = selector.models[alg].predict(X_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                selector.models[alg], 
                X_scaled, 
                y, 
                cv=min(5, len(X)), 
                scoring='neg_mean_squared_error'
            )
            cv_mse = -cv_scores.mean()
            
            # Store metrics
            metrics[alg] = {
                "mse": float(mse),
                "r2": float(r2),
                "cv_mse": float(cv_mse),
                "feature_importance": [float(imp) for imp in selector.models[alg].feature_importances_]
            }
    
    return metrics

def create_basic_feature_importance(selector, export_dir: Path) -> None:
    """
    Create basic feature importance visualization using built-in methods
    
    Args:
        selector: The trained selector
        export_dir: Directory to export visualizations
    """
    logger.info("Creating basic feature importance visualizations")
    
    # Create visualizations directory
    viz_dir = export_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # For each algorithm model
    for alg in selector.algorithms:
        if selector.models[alg] is not None and hasattr(selector.models[alg], 'feature_importances_'):
            # Get feature names and importance scores
            feature_names = list(selector.X_train[0].keys())
            importance = selector.models[alg].feature_importances_
            
            # Create DataFrame for better plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'Feature Importance for {alg}')
            plt.tight_layout()
            plt.savefig(viz_dir / f"feature_importance_{alg}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create correlation heatmap of features
    try:
        # Convert features to DataFrame
        feature_names = list(selector.X_train[0].keys())
        features_df = pd.DataFrame([list(f.values()) for f in selector.X_train], 
                                  columns=feature_names)
        
        # Calculate correlation
        corr = features_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", 
                   xticklabels=feature_names, yticklabels=feature_names)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(viz_dir / "feature_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Error creating feature correlation heatmap: {e}")

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
    
    # Save the selector
    with open(file_path, 'wb') as f:
        pickle.dump(selector, f)
    
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
    
    # Load the selector
    with open(file_path, 'rb') as f:
        selector = pickle.load(f)
    
    logger.info(f"Selector loaded from {file_path}")
    
    return selector 
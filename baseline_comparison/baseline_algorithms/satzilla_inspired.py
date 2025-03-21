"""
SATzilla-inspired algorithm selector for optimization problems

This module implements a SATzilla-inspired algorithm selector that uses
problem features to select optimization algorithms for specific problems.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import pickle
import traceback
from analysis.problem_analyzer import extract_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SatzillaInspiredSelector:
    """
    SATzilla-inspired algorithm selector that uses problem features
    to select optimization algorithms for specific problems
    """
    
    def __init__(
        self,
        algorithms: Optional[List[str]] = None,
        random_seed: int = 42
    ):
        """
        Initialize the SATzilla-inspired selector
        
        Args:
            algorithms: List of available optimization algorithms
            random_seed: Random seed for reproducibility
        """
        # Set random seed
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Available optimization algorithms
        self.algorithms = algorithms or [
            "differential_evolution",
            "particle_swarm",
            "genetic_algorithm",
            "simulated_annealing",
            "cma_es"
        ]
        
        # Initialize models for predicting algorithm performance
        self.models = {alg: None for alg in self.algorithms}
        
        # Feature standardization
        self.scaler = StandardScaler()
        
        # Training data
        self.X_train = []  # Features
        self.y_train = {alg: [] for alg in self.algorithms}  # Performance
        
        # Feature names
        self.feature_names = [
            "dimensions",
            "mean",
            "std",
            "min",
            "max",
            "range",
            "skewness",
            "kurtosis",
            "ruggedness",
            "gradient_variation",
            "evaluation_time"
        ]
        
        # Flag to indicate if the model has been trained
        self._is_trained = False
        
        # Track the last selected algorithm
        self.last_selected_algorithm = None
        
        logger.info(f"Initialized SatzillaInspiredSelector with {len(self.algorithms)} algorithms")
    
    @property
    def is_trained(self) -> bool:
        """Return True if the selector is trained"""
        return self._is_trained
        
    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        """Set the trained status of the selector"""
        self._is_trained = value
    
    def get_available_algorithms(self) -> List[str]:
        """
        Get the list of available optimization algorithms
        
        Returns:
            List of algorithm names
        """
        return self.algorithms
    
    def extract_features(self, problem) -> Dict[str, float]:
        """
        Extract features from a problem
        
        Args:
            problem: The optimization problem to analyze
            
        Returns:
            Dictionary of problem features
        """
        try:
            features = extract_features(problem)
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            logger.error(traceback.format_exc())
            
            # Return default features
            return {
                'dimensions': problem.dimensions,
                'mean': 0.0,
                'std': 1.0,
                'min': -1.0,
                'max': 1.0,
                'range': 2.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'gradient_variation': 0.5,
                'ruggedness': 0.5,
                'evaluation_time': 0.001
            }
    
    def _calculate_ruggedness(self, problem, dims, lb, ub, num_walks=10, steps_per_walk=20) -> float:
        """
        Calculate a ruggedness metric based on random walks
        
        Args:
            problem: The optimization problem
            dims: Number of dimensions
            lb: Lower bounds
            ub: Upper bounds
            num_walks: Number of random walks
            steps_per_walk: Steps per random walk
            
        Returns:
            Ruggedness metric
        """
        # Initialize ruggedness measure
        ruggedness = 0.0
        
        try:
            # Convert lb and ub to numpy arrays for element-wise operations
            lb_array = np.array(lb)
            ub_array = np.array(ub)
            
            for _ in range(num_walks):
                # Start at a random point
                point = np.random.uniform(lb_array, ub_array, dims)
                
                # Perform random walk
                differences = []
                prev_value = problem.evaluate(point)
                
                for _ in range(steps_per_walk):
                    # Take a small step in a random direction
                    step_size = 0.01 * (ub_array - lb_array)  # Element-wise with numpy arrays
                    direction = np.random.randn(dims)
                    direction = direction / np.linalg.norm(direction)
                    
                    # Ensure we stay within bounds
                    new_point = np.clip(point + step_size * direction, lb_array, ub_array)
                    
                    # Evaluate new point
                    new_value = problem.evaluate(new_point)
                    
                    # Calculate absolute difference
                    differences.append(abs(new_value - prev_value))
                    
                    # Update for next step
                    point = new_point
                    prev_value = new_value
                
                # Ruggedness is the average absolute difference
                if differences:
                    ruggedness += np.mean(differences)
            
            # Average over all walks
            ruggedness /= num_walks
            
        except Exception as e:
            logger.warning(f"Error calculating ruggedness: {e}. Using default value.")
            ruggedness = 0.1
        
        return ruggedness
    
    def _calculate_gradient_variation(self, problem, dims, lb, ub, num_samples=10) -> float:
        """
        Calculate gradient variation as a measure of problem difficulty
        
        Args:
            problem: The optimization problem
            dims: Number of dimensions
            lb: Lower bounds
            ub: Upper bounds
            num_samples: Number of samples
            
        Returns:
            Gradient variation metric
        """
        # Initialize gradient variation
        gradient_variations = []
        
        try:
            # Convert lb and ub to numpy arrays for element-wise operations
            lb_array = np.array(lb)
            ub_array = np.array(ub)
            
            for _ in range(num_samples):
                # Random point
                point = np.random.uniform(lb_array, ub_array, dims)
                
                # Approximate gradient
                gradient = np.zeros(dims)
                base_value = problem.evaluate(point)
                
                # Finite difference approximation
                h = 1e-6 * (ub_array - lb_array)  # Element-wise with numpy arrays
                
                for i in range(dims):
                    # Forward difference
                    point_h = point.copy()
                    point_h[i] += h[i]  # Use indexed h
                    
                    # Calculate derivative
                    value_h = problem.evaluate(point_h)
                    gradient[i] = (value_h - base_value) / h[i]  # Use indexed h
                
                # Calculate gradient norm
                gradient_norm = np.linalg.norm(gradient)
                gradient_variations.append(gradient_norm)
            
            # Gradient variation is the std dev of gradient norms
            if gradient_variations:
                gradient_variation = np.std(gradient_variations)
            else:
                gradient_variation = 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating gradient variation: {e}. Using default value.")
            gradient_variation = 0.1
        
        return gradient_variation
    
    def train(self, problems: List['OptimizationProblem'], max_evaluations: int = 1000) -> None:
        """
        Train the selector on a set of problems
        
        Args:
            problems: List of optimization problems for training
            max_evaluations: Maximum function evaluations per algorithm
        """
        logger.info("Training SatzillaInspiredSelector...")
        
        # Extract features from each problem
        feature_dicts = []
        for problem in problems:
            features = self.extract_features(problem)
            feature_dicts.append(features)
            
        # Set feature names based on extracted features
        if feature_dicts:
            # Get feature names from first problem's features
            feature_names = list(feature_dicts[0].keys())
            
            # Log if feature names have changed
            if self.feature_names != feature_names:
                logger.info(f"Updated feature_names from training data: {feature_names}")
                logger.warning(f"Feature names changed during training. Original: {self.feature_names}, New: {feature_names}")
                self.feature_names = feature_names
        
        # Create training data: X (features) and y (performance for each algorithm)
        X = []
        y_dict = {alg: [] for alg in self.algorithms}
        
        # Collect data for each problem and algorithm
        for problem, features in zip(problems, feature_dicts):
            # Extract feature values
            feature_values = [features[feat] for feat in self.feature_names]
            X.append(feature_values)
            
            # Run each algorithm and record performance
            for alg in self.algorithms:
                perf = self._run_algorithm(problem, alg, max_evaluations)
                y_dict[alg].append(perf)
        
        # Convert to numpy arrays
        X = np.array(X)
        self.X_train = X  # Store for future reference
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train a regression model for each algorithm
        for alg in self.algorithms:
            y = np.array(y_dict[alg])
            
            # Skip if all values are identical or no valid performance data
            if len(np.unique(y)) < 2 or np.all(np.isnan(y)) or np.all(np.isinf(y)):
                logger.warning(f"Skipping model training for {alg}: insufficient variation in performance data")
                self.models[alg] = None
                continue
                
            # Replace any inf or nan values with large finite values
            y[np.isnan(y) | np.isinf(y)] = 1e10
            
            # Train random forest regression model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            try:
                model.fit(X_scaled, y)
                self.models[alg] = model
                logger.info(f"Successfully trained model for algorithm: {alg}")
            except Exception as e:
                logger.error(f"Failed to train model for {alg}: {e}")
                self.models[alg] = None
        
        # Validate the trained models
        model_valid = self._validate_model()
        
        # Update trained flag
        self.is_trained = model_valid
        
        if model_valid:
            logger.info("Training completed successfully. Model is ready for use.")
        else:
            logger.warning("Training completed with issues. Model may not be fully functional.")
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to a file
        
        Args:
            model_path: Path to save the model
        """
        import joblib
        import pickle
        from pathlib import Path
        
        model_path = Path(model_path)
        logger.info(f"Saving model to {model_path}")
        
        # Ensure parent directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a dictionary with all necessary components
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'algorithms': self.algorithms,
            'is_trained': self.is_trained
        }
        
        # Save model in multiple formats to ensure compatibility
        # Primary format: .joblib (recommended)
        joblib_path = model_path
        if not str(joblib_path).endswith('.joblib'):
            joblib_path = Path(f"{str(model_path)}.joblib")
        
        try:
            # Save using joblib (primary format)
            joblib.dump(model_data, joblib_path)
            logger.info(f"Model saved to {joblib_path} (primary format)")
            
            # Save a copy with .pkl extension for backward compatibility
            if not str(model_path).endswith('.pkl'):
                pickle_path = Path(f"{str(model_path)}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"Model also saved to {pickle_path} for backward compatibility")
                
            # If file doesn't have an extension, save it directly too
            if '.' not in model_path.name:
                joblib.dump(model_data, model_path)
                logger.info(f"Model also saved to {model_path} (no extension)")
                
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}")
            raise RuntimeError(f"Failed to save model to {model_path}: {e}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained selector model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # Check if path exists
            if not os.path.exists(model_path):
                logger.warning(f"Model path {model_path} does not exist")
                return False
                
            # Determine model format
            is_joblib = model_path.endswith('.joblib')
            is_pickle = model_path.endswith('.pkl')
            
            # Try loading the model based on its format
            if is_joblib:
                try:
                    import joblib
                    model_data = joblib.load(model_path)
                    logger.info(f"Loaded joblib model from {model_path}")
                except (ImportError, Exception) as e:
                    logger.warning(f"Error loading joblib model: {e}. Falling back to pickle.")
                    is_joblib = False
                    is_pickle = True
            
            if is_pickle or not is_joblib:
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        logger.info(f"Loaded pickle model from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading pickle model: {e}")
                    return False
            
            # Extract model components
            if isinstance(model_data, dict):
                # Modern format - dictionary with all components
                if 'models' in model_data:
                    self.models = model_data['models']
                if 'scaler' in model_data:
                    self.scaler = model_data['scaler']
                    if 'feature_names' in model_data:
                        self.feature_names = model_data['feature_names']
                if 'algorithms' in model_data:
                    self.algorithms = model_data['algorithms']
                if 'X_train' in model_data:
                    self.X_train = model_data['X_train']
                if 'is_trained' in model_data:
                    self.is_trained = model_data['is_trained']
                # Set is_trained flag if key components are present
                if self.models and self.scaler and self.feature_names:
                    self._is_trained = True
            elif hasattr(model_data, 'models') and hasattr(model_data, 'scaler'):
                # Legacy format - instance of SatzillaInspiredSelector
                self.models = model_data.models
                self.scaler = model_data.scaler
                self.feature_names = getattr(model_data, 'feature_names', [])
                self.algorithms = getattr(model_data, 'algorithms', self.algorithms)
                self._is_trained = getattr(model_data, 'is_trained', True)
            else:
                logger.error("Invalid model format")
                return False
            
            # Validate loaded model
            success = self._validate_model()
            if success:
                logger.info(f"Model loaded and validated successfully from {model_path}")
                self.is_trained = True
                return True
            else:
                logger.error("Model validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
            
    def _validate_model(self) -> bool:
        """
        Validate that the loaded model components are consistent
        
        Returns:
            True if model is valid, False otherwise
        """
        # Check that models dictionary exists and has entries
        if not hasattr(self, 'models') or not self.models:
            logger.warning("No models found in loaded data")
            return False
            
        # Check that at least some algorithms have models
        valid_models = []
        for alg, model in self.models.items():
            if model is not None:
                valid_models.append(alg)
                
        if not valid_models:
            logger.warning("No valid models found")
            return False
        
        # Check that scaler exists
        if not hasattr(self, 'scaler') or self.scaler is None:
            logger.warning("No scaler found in loaded data")
            # Can continue without scaler, but with warning
            
        # Check that feature names exist
        if not hasattr(self, 'feature_names') or not self.feature_names:
            logger.warning("No feature names found in loaded data")
            # Can proceed with default features
            
        logger.info(f"Valid models found for algorithms: {valid_models}")
        return True
    
    def _run_algorithm(self, problem, algorithm: str, max_evaluations: int) -> float:
        """
        Run an optimization algorithm on a problem
        
        Args:
            problem: The optimization problem
            algorithm: The algorithm to run
            max_evaluations: Maximum number of function evaluations
            
        Returns:
            Best fitness found
        """
        # This is a placeholder for running the actual algorithm
        # In a real implementation, you would run the actual algorithm
        
        # Simulate different performance for different algorithms
        if algorithm == "differential_evolution":
            best_fitness = np.random.uniform(0, 0.5)
        elif algorithm == "particle_swarm":
            best_fitness = np.random.uniform(0.2, 0.7)
        elif algorithm == "genetic_algorithm":
            best_fitness = np.random.uniform(0.3, 0.8)
        elif algorithm == "simulated_annealing":
            best_fitness = np.random.uniform(0.4, 0.9)
        elif algorithm == "cma_es":
            best_fitness = np.random.uniform(0.1, 0.6)
        else:
            best_fitness = np.random.uniform(0.5, 1.0)
        
        return best_fitness
    
    def select_algorithm(self, problem: 'OptimizationProblem', max_evals: int) -> str:
        """
        Select the best algorithm for the given problem
        
        Args:
            problem: The optimization problem to solve
            max_evals: Maximum number of function evaluations
            
        Returns:
            The name of the selected algorithm
        """
        # Extract problem features
        try:
            features = self.extract_features(problem)
            logger.debug(f"Extracted features: {features}")
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            logger.error(traceback.format_exc())
            # Default to random algorithm selection
            return self._select_random()
            
        # Select algorithm based on model predictions if trained
        if self.is_trained and self.models and len(self.models) > 0:
            logger.debug("Using trained model for algorithm selection")
            try:
                # Prepare feature vector
                feature_vector = self._prepare_feature_vector(features)
                
                # No prediction if feature vector is None
                if feature_vector is None:
                    logger.warning("Feature vector is None, defaulting to random selection")
                    return self._select_random()
                
                # Normalize features if scaler is available
                if self.scaler is not None:
                    try:
                        # Reshape to match training data format
                        feature_vector_reshaped = feature_vector.reshape(1, -1)
                        feature_vector_scaled = self.scaler.transform(feature_vector_reshaped)
                        feature_vector = feature_vector_scaled.reshape(-1)
                    except Exception as e:
                        logger.warning(f"Error scaling features: {e}")
                        # Continue with unscaled features rather than failing
                
                # Make predictions for each algorithm
                predictions = {}
                valid_predictions = False
                
                for alg, model in self.models.items():
                    if model is not None:
                        try:
                            # Reshape for prediction
                            X = feature_vector.reshape(1, -1)
                            # Predict expected performance
                            score = float(model.predict(X)[0])
                            predictions[alg] = score
                            valid_predictions = True
                            logger.debug(f"Predicted score for {alg}: {score}")
                        except Exception as e:
                            logger.warning(f"Failed to predict for {alg}: {e}")
            
                # If we have valid predictions, select the best algorithm
                if valid_predictions:
                    # Select algorithm with the lowest predicted runtime/best predicted performance
                    best_alg = min(predictions.items(), key=lambda x: x[1])[0]
                    logger.info(f"Selected algorithm: {best_alg} based on predictions")
                    return best_alg
                else:
                    logger.warning("No valid predictions, defaulting to random selection")
            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                logger.error(traceback.format_exc())
        else:
            # Not trained or no models, use random selection
            if not self.is_trained:
                logger.warning("Model not trained. Selecting random algorithm.")
            elif not self.models:
                logger.warning("No models available. Selecting random algorithm.")
            else:
                logger.warning(f"Models dictionary empty or invalid: {self.models}")
                
        # Default to random selection if model prediction fails
        return self._select_random()
    
    def optimize(self, problem, max_evaluations: int) -> Tuple[np.ndarray, float]:
        """
        Optimize a problem using the selected algorithm
        
        Args:
            problem: The optimization problem
            max_evaluations: Maximum number of function evaluations
            
        Returns:
            Tuple of (best solution, best fitness)
        """
        # Extract features from the problem
        features = self.extract_features(problem)
        
        # Select the best algorithm based on features
        selected_algorithm = self.select_algorithm(problem, max_evaluations)
        self.last_selected_algorithm = selected_algorithm
        
        # Run the selected algorithm
        # This part would need to be implemented based on your optimization algorithms
        # For now, we'll just return a placeholder result
        return np.zeros(problem.dims), 0.0
    
    def get_selected_algorithm(self) -> str:
        """
        Get the last selected algorithm
        
        Returns:
            Name of the last selected algorithm
        """
        return self.last_selected_algorithm or "unknown"

    def _prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare a feature vector for prediction
        
        Args:
            features: Dictionary of problem features
            
        Returns:
            Numpy array of feature values
        """
        if not self.feature_names:
            logger.warning("No feature names defined. Cannot prepare feature vector.")
            return None
            
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(features.keys())
        if missing_features:
            logger.warning(f"Missing features for prediction: {missing_features}. Using default values (0.0).")
            # Add missing features with default values
            for f in missing_features:
                features[f] = 0.0
                
        # Prepare feature vector with only the expected features
        feature_values = np.array([features.get(f, 0.0) for f in self.feature_names])
        
        return feature_values
        
    def _select_random(self) -> str:
        """
        Select a random algorithm
        
        Returns:
            Name of a randomly selected algorithm
        """
        alg = np.random.choice(self.algorithms)
        logger.info(f"Randomly selected algorithm: {alg}")
        return alg

    def set_available_algorithms(self, algorithms: List[str]) -> None:
        """
        Set the list of available optimization algorithms
        
        Args:
            algorithms: List of algorithm names
        """
        if not algorithms:
            logger.warning("Empty algorithm list provided, keeping current algorithms")
            return
            
        # Only keep algorithms that were in the original list
        valid_algorithms = [alg for alg in algorithms if alg in self.models]
        
        if not valid_algorithms:
            logger.warning(f"None of the provided algorithms {algorithms} are in the original set {list(self.models.keys())}")
            return
            
        if len(valid_algorithms) != len(algorithms):
            logger.warning(f"Some algorithms were not in the original set and were ignored: {set(algorithms) - set(valid_algorithms)}")
            
        self.algorithms = valid_algorithms
        logger.info(f"Updated available algorithms: {self.algorithms}")
    
    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility
        
        Args:
            seed: Random seed
        """
        self.random_seed = seed
        np.random.seed(seed)
        logger.info(f"Set random seed to {seed}")


# Utility functions for statistical calculations

def skewness(x):
    """Calculate the skewness of a distribution"""
    n = len(x)
    if n < 3:
        return 0.0
    
    m3 = np.sum((x - np.mean(x))**3) / n
    s3 = np.std(x)**3
    
    if s3 == 0:
        return 0.0
        
    return m3 / s3

def kurtosis(x):
    """Calculate the kurtosis of a distribution"""
    n = len(x)
    if n < 4:
        return 0.0
    
    m4 = np.sum((x - np.mean(x))**4) / n
    s4 = np.std(x)**4
    
    if s4 == 0:
        return 0.0
        
    return m4 / s4 
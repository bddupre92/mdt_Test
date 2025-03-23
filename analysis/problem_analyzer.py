import numpy as np
import time
import traceback
from scipy.stats import skew, kurtosis
import logging

# Configure logging
logger = logging.getLogger(__name__)

def estimate_gradient_variation(problem, samples):
    """
    Estimate the variation in gradient across the search space.
    
    Args:
        problem: The optimization problem
        samples: Sample points in the search space
        
    Returns:
        Gradient variation metric
    """
    try:
        # Calculate numerical gradients at each sample point
        gradients = []
        h = 1e-6  # Small step size for numerical differentiation
        
        for x in samples[:min(20, len(samples))]:  # Limit to 20 samples for efficiency
            dims = len(x)
            grad = np.zeros(dims)
            f_x = problem.evaluate(x)
            
            for i in range(dims):
                # Forward difference approximation
                x_h = x.copy()
                x_h[i] += h
                f_x_h = problem.evaluate(x_h)
                grad[i] = (f_x_h - f_x) / h
                
            gradients.append(grad)
            
        if not gradients:
            return 0.0
            
        # Calculate the Frobenius norm of gradients
        grad_norms = [np.linalg.norm(g) for g in gradients]
        
        # Return the coefficient of variation (std/mean) of gradient norms
        if np.mean(grad_norms) > 0:
            return np.std(grad_norms) / np.mean(grad_norms)
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Error calculating gradient variation: {e}")
        return 0.0

def estimate_ruggedness(problem, samples, step_size=0.1):
    """
    Estimate the ruggedness of the objective function.
    
    Args:
        problem: The optimization problem
        samples: Sample points in the search space
        step_size: Step size for numerical differentiation
        
    Returns:
        Ruggedness metric
    """
    try:
        # Limit to fewer samples for efficiency
        limited_samples = samples[:min(10, len(samples))]
        
        # Calculate the Hessian at each sample point
        ruggedness_values = []
        
        for x in limited_samples:
            dims = len(x)
            
            # Calculate function value at sample point
            f_x = problem.evaluate(x)
            
            # Calculate second derivatives
            second_derivatives = []
            
            for i in range(dims):
                # Forward and backward steps
                x_forward = x.copy()
                x_forward[i] += step_size
                x_backward = x.copy()
                x_backward[i] -= step_size
                
                # Function values
                f_forward = problem.evaluate(x_forward)
                f_backward = problem.evaluate(x_backward)
                
                # Centered second derivative
                second_deriv = (f_forward - 2*f_x + f_backward) / (step_size**2)
                second_derivatives.append(abs(second_deriv))
            
            if second_derivatives:
                # Use max second derivative as ruggedness indicator
                ruggedness_values.append(np.max(second_derivatives))
        
        if ruggedness_values:
            # Normalize ruggedness to [0, 1] using sigmoid
            avg_ruggedness = np.mean(ruggedness_values)
            return 1.0 / (1.0 + np.exp(-0.1 * avg_ruggedness))
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"Error calculating ruggedness: {e}")
        return 0.0

def estimate_modality(problem, samples, values=None, n_bins=20):
    """
    Estimate the modality (number of local optima) of the objective function.
    
    Args:
        problem: The optimization problem
        samples: Sample points in the search space
        values: Pre-computed function values (optional)
        n_bins: Number of bins for histogram analysis
        
    Returns:
        Modality estimate
    """
    try:
        if values is None:
            values = np.array([problem.evaluate(x) for x in samples])
        
        # Simple peak counting in the histogram
        hist, bin_edges = np.histogram(values, bins=n_bins)
        
        # Count peaks in the histogram
        peaks = 0
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks += 1
        
        # Ensure at least one modality
        return max(1, peaks)
    except Exception as e:
        logger.warning(f"Error estimating modality: {e}")
        return 1  # Default to unimodal

def extract_features(problem, num_samples=100, **kwargs):
    """
    Extract features from an optimization problem.
    
    Args:
        problem: The optimization problem to analyze
        num_samples: Number of samples to use for feature calculation
        **kwargs: Additional parameters for specific feature extractors
        
    Returns:
        Dictionary containing extracted features
    """
    logger.debug(f"Extracting features from problem: {problem.name}")
    
    # Get dimension information
    dims = getattr(problem, 'dimensions', getattr(problem, 'dims', getattr(problem, 'dim', 2)))
    
    # Initialize features with dimension
    features = {
        'dimensions': dims
    }

    try:
        # Create sampling grid based on problem bounds
        if hasattr(problem, 'bounds'):
            bounds = problem.bounds
            if bounds is not None:
                # Handle different bounds formats
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and not hasattr(bounds[0], '__iter__'):
                    # Scalar bounds [min, max] - expand to all dimensions
                    lower_bounds = np.array([bounds[0]] * dims)
                    upper_bounds = np.array([bounds[1]] * dims)
                else:
                    # Parse bounds as array of [min, max] pairs
                    bounds_array = np.array(bounds)
                    if bounds_array.ndim == 1 and len(bounds_array) == 2:
                        # Single pair for all dimensions
                        lower_bounds = np.array([bounds_array[0]] * dims)
                        upper_bounds = np.array([bounds_array[1]] * dims)
                    elif bounds_array.ndim == 2 and bounds_array.shape[0] == dims:
                        # Dimension-specific bounds
                        lower_bounds = bounds_array[:, 0]
                        upper_bounds = bounds_array[:, 1]
                    else:
                        # Default when bounds format is unexpected
                        logger.warning(f"Unexpected bounds format: {bounds}, using defaults")
                        lower_bounds = np.array([-5.0] * dims)
                        upper_bounds = np.array([5.0] * dims)
            else:
                # Default bounds when None is provided
                logger.warning("No bounds provided, using defaults")
                lower_bounds = np.array([-5.0] * dims)
                upper_bounds = np.array([5.0] * dims)
        else:
            # Default bounds when attribute not present
            logger.warning("Problem does not have bounds attribute, using defaults")
            lower_bounds = np.array([-5.0] * dims)
            upper_bounds = np.array([5.0] * dims)

        # Generate uniform random samples within the bounds
        samples = np.random.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(num_samples, dims)
        )
        
        # Evaluate function at each sample point
        start_time = time.time()
        values = np.zeros(num_samples)
        for i, sample in enumerate(samples):
            values[i] = problem.evaluate(sample)
        evaluation_time = time.time() - start_time
        
        # Store evaluation time
        features['evaluation_time'] = evaluation_time / num_samples
        
        # Basic statistical features
        features['mean'] = np.mean(values)
        features['std'] = np.std(values)
        features['min'] = np.min(values)
        features['max'] = np.max(values)
        features['range'] = features['max'] - features['min']
        
        # Higher-order statistical moments
        features['skewness'] = skew(values) if not np.isnan(skew(values)) else 0.0
        features['kurtosis'] = kurtosis(values) if not np.isnan(kurtosis(values)) else 0.0
        
        # Extract landscape features
        gradient_variation = estimate_gradient_variation(problem, samples)
        features['gradient_variation'] = gradient_variation if not np.isnan(gradient_variation) and not np.isinf(gradient_variation) else 0.0
        
        ruggedness = estimate_ruggedness(problem, samples)
        features['ruggedness'] = ruggedness if not np.isnan(ruggedness) and not np.isinf(ruggedness) else 0.0
        
        # Modality detection is complex and can fail, so wrap in try-except
        try:
            modality = estimate_modality(problem, samples, values)
            features['modality'] = modality
        except Exception as e:
            logger.warning(f"Failed to extract modality: {str(e)}")
            features['modality'] = 1  # Default to unimodal
        
        logger.debug(f"Extracted features: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return basic features with defaults for failed extractions
        return {
            'dimensions': dims,
            'evaluation_time': 0.001,
            'mean': 0.0,
            'std': 1.0,
            'min': -1.0,
            'max': 1.0,
            'range': 2.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'gradient_variation': 0.5,
            'ruggedness': 0.5,
            'modality': 1
        } 
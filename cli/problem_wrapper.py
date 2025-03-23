"""
Problem wrapper class for optimization problems.
"""

class ProblemWrapper:
    def __init__(self, function, dimensions, bounds=None):
        self.function = function
        
        # Handle dimensions from function if available
        if hasattr(function, 'dims'):
            self.dimensions = function.dims
        elif hasattr(function, 'dimension'):
            self.dimensions = function.dimension
        else:
            self.dimensions = dimensions
        
        # Extract name from the function if available
        if hasattr(function, 'name'):
            self.name = function.name.lower()
        else:
            self.name = function.__class__.__name__.lower()
        
        # Extract bounds from function if available and not provided
        if bounds is None and hasattr(function, 'bounds'):
            function_bounds = function.bounds
            # Handle different bounds formats
            if isinstance(function_bounds, tuple) and len(function_bounds) == 2:
                # Global bounds format (min, max)
                low, high = function_bounds
                self._bounds_list = [(low, high)] * self.dimensions
                self.bounds = [(low, high)] * self.dimensions
            elif isinstance(function_bounds, list) and len(function_bounds) == self.dimensions:
                # Per-dimension bounds
                self._bounds_list = function_bounds
                self.bounds = function_bounds
            else:
                # Default bounds
                self._bounds_list = [(0, 1)] * self.dimensions
                self.bounds = [(0, 1)] * self.dimensions
        else:
            # Initialize with provided or default bounds
            if bounds is None:
                self._bounds_list = [(0, 1)] * self.dimensions
                self.bounds = [(0, 1)] * self.dimensions  # Make bounds a list of tuples for better compatibility
            else:
                self._bounds_list = bounds
                self.bounds = bounds  # Use provided bounds
            
        # Extract additional properties if available
        self.global_optimum = None
        if hasattr(function, 'global_optimum'):
            self.global_optimum = function.global_optimum
            
        self.global_optimum_value = float('inf')
        if hasattr(function, 'global_optimum_value'):
            self.global_optimum_value = function.global_optimum_value
            
        # Initialize evaluation counter
        self.evaluations = 0
        self.tracking_objective = None

    @property
    def dims(self):
        """Return the number of dimensions for compatibility with baseline algorithms."""
        return self.dimensions

    def evaluate(self, x):
        """Evaluate the function at point x."""
        self.evaluations += 1
        result = self.function.evaluate(x)
        if self.tracking_objective is not None:
            self.tracking_objective(result)  # Pass the result to tracking_objective
        return result

    def __call__(self, x):
        """Call method for backward compatibility."""
        return self.evaluate(x)

    def reset(self):
        self.evaluations = 0
        self.tracking_objective = None 
"""
Module for storing optimization results.
"""

class OptimizationResult:
    """
    Class for storing optimization results including best solution, score,
    convergence curve, execution time, and number of function evaluations.
    
    This class can be extended with additional attributes specific to
    certain optimizers, such as algorithm selections for meta-optimizers.
    """
    def __init__(self, best_solution, best_score, convergence_curve, execution_time, evaluations):
        """
        Initialize the OptimizationResult with the optimization outputs.
        
        Args:
            best_solution: The best solution found
            best_score: The score of the best solution
            convergence_curve: List of best scores over iterations
            execution_time: Total execution time in seconds
            evaluations: Number of function evaluations
        """
        self.best_solution = best_solution
        self.best_score = best_score
        self.convergence_curve = convergence_curve
        self.execution_time = execution_time
        self.evaluations = evaluations
        
        # The following attributes are optional and can be set after initialization
        # Mainly used by meta-optimizers
        self.algorithm_selections = None  # List of selected algorithms over iterations
        self.parameter_history = None     # List of parameter dictionaries over iterations

"""
Convergence Analysis for Optimization Algorithms.

This module provides theoretical analysis of convergence properties for
various optimization algorithms, including formal proofs, convergence rates,
and comparative analysis across different problem types.
"""

import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from core.theory.base import AlgorithmProperty


class ConvergenceAnalyzer(AlgorithmProperty):
    """
    Analyzer for theoretical convergence properties of optimization algorithms.
    
    This class provides methods for analyzing and comparing the convergence
    properties of different optimization algorithms, including convergence rates,
    conditions for convergence, and formal guarantees.
    """
    
    CONVERGENCE_TYPES = {
        'global': 'Guaranteed convergence to global optimum',
        'local': 'Guaranteed convergence to local optimum',
        'probabilistic': 'Convergence with probability 1 as iterations → ∞',
        'asymptotic': 'Asymptotic convergence properties',
        'none': 'No formal convergence guarantees'
    }
    
    def __init__(self, algorithm_type: str, description: str = ""):
        """
        Initialize a convergence analyzer for a specific algorithm type.
        
        Args:
            algorithm_type: The type of algorithm to analyze (e.g., "DE", "PSO", "ACO")
            description: A description of the analyzer
        """
        super().__init__(f"ConvergenceAnalyzer_{algorithm_type}", algorithm_type, description)
        self.convergence_type = self._determine_convergence_type(algorithm_type)
        self.convergence_rate = self._determine_convergence_rate(algorithm_type)
        self.convergence_conditions = self._determine_convergence_conditions(algorithm_type)
    
    def _determine_convergence_type(self, algorithm_type: str) -> str:
        """
        Determine the type of convergence guarantee for the algorithm.
        
        Args:
            algorithm_type: The type of algorithm
            
        Returns:
            The type of convergence guarantee
        """
        # Map of algorithm types to convergence types
        convergence_map = {
            'DE': 'probabilistic',
            'PSO': 'probabilistic',
            'ACO': 'probabilistic',
            'ES': 'probabilistic',
            'GWO': 'probabilistic',
            'GD': 'local',
            'NEWTON': 'local',
            'SIMPLEX': 'local',
            'RANDOM': 'none',
            'GRID': 'global'  # Only for discrete, bounded domains
        }
        
        return convergence_map.get(algorithm_type.upper(), 'probabilistic')
    
    def _determine_convergence_rate(self, algorithm_type: str) -> Dict[str, Any]:
        """
        Determine the convergence rate properties for the algorithm.
        
        Args:
            algorithm_type: The type of algorithm
            
        Returns:
            A dictionary containing convergence rate properties
        """
        # Default convergence rate properties
        rate_properties = {
            'order': None,  # Order of convergence (e.g., linear, quadratic)
            'rate_constant': None,  # Rate constant for the algorithm
            'asymptotic_complexity': None,  # Asymptotic complexity
            'expected_iterations': None,  # Expected number of iterations to ε-convergence
            'dimension_dependency': None  # How convergence scales with dimension
        }
        
        # Populate based on algorithm type
        if algorithm_type.upper() == 'DE':
            rate_properties['order'] = 'linear'
            rate_properties['asymptotic_complexity'] = 'O(d * NP)'  # d: dimension, NP: population size
            rate_properties['dimension_dependency'] = 'linear'
        elif algorithm_type.upper() == 'PSO':
            rate_properties['order'] = 'linear'
            rate_properties['asymptotic_complexity'] = 'O(d * S)'  # d: dimension, S: swarm size
            rate_properties['dimension_dependency'] = 'linear'
        elif algorithm_type.upper() == 'GD':
            rate_properties['order'] = 'linear'
            rate_properties['rate_constant'] = 'α'  # Learning rate
            rate_properties['asymptotic_complexity'] = 'O(1/ε)'  # ε: target accuracy
            rate_properties['dimension_dependency'] = 'independent'
        elif algorithm_type.upper() == 'NEWTON':
            rate_properties['order'] = 'quadratic'
            rate_properties['asymptotic_complexity'] = 'O(log(1/ε))'  # ε: target accuracy
            rate_properties['dimension_dependency'] = 'quadratic'  # Hessian computation
        
        return rate_properties
    
    def _determine_convergence_conditions(self, algorithm_type: str) -> List[str]:
        """
        Determine the conditions necessary for convergence.
        
        Args:
            algorithm_type: The type of algorithm
            
        Returns:
            A list of conditions necessary for convergence
        """
        # Default conditions
        conditions = []
        
        # Common conditions
        conditions.append("Sufficient iteration budget")
        
        # Algorithm-specific conditions
        if algorithm_type.upper() == 'DE':
            conditions.extend([
                "Proper choice of crossover rate (CR)",
                "Appropriate scaling factor (F)",
                "Sufficient population size",
                "Selection preserves best solution"
            ])
        elif algorithm_type.upper() == 'PSO':
            conditions.extend([
                "Proper balance between cognitive and social parameters",
                "Appropriate inertia weight",
                "Velocity constraints",
                "Sufficient swarm size"
            ])
        elif algorithm_type.upper() == 'GD':
            conditions.extend([
                "Appropriate learning rate",
                "Function smoothness",
                "Lipschitz continuous gradient"
            ])
        elif algorithm_type.upper() == 'NEWTON':
            conditions.extend([
                "Non-singular Hessian",
                "Twice differentiable objective function",
                "Starting point sufficiently close to optimum"
            ])
        
        return conditions
    
    def analyze(self, algorithm_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the theoretical convergence properties of an algorithm.
        
        Args:
            algorithm_parameters: A dictionary of algorithm parameters
            
        Returns:
            A dictionary containing the analysis results, including convergence
            type, rate, and conditions
        """
        # Extract algorithm parameters
        algorithm_type = self.algorithm_type
        
        # Start with basic convergence properties
        analysis_results = {
            'algorithm_type': algorithm_type,
            'convergence_type': self.convergence_type,
            'convergence_type_description': self.CONVERGENCE_TYPES[self.convergence_type],
            'convergence_rate': self.convergence_rate,
            'convergence_conditions': self.convergence_conditions,
        }
        
        # Analyze parameter-specific impacts on convergence
        parameter_impacts = self._analyze_parameter_impacts(algorithm_parameters)
        analysis_results['parameter_impacts'] = parameter_impacts
        
        # Overall convergence assessment
        analysis_results['overall_assessment'] = self._assess_overall_convergence(algorithm_parameters)
        
        return analysis_results
    
    def _analyze_parameter_impacts(self, algorithm_parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze the impact of specific parameters on convergence.
        
        Args:
            algorithm_parameters: A dictionary of algorithm parameters
            
        Returns:
            A dictionary mapping parameters to their impact on convergence
        """
        impacts = {}
        
        if self.algorithm_type.upper() == 'DE':
            # Analyze DE parameters
            if 'crossover_rate' in algorithm_parameters:
                cr = algorithm_parameters['crossover_rate']
                if cr < 0.1:
                    impacts['crossover_rate'] = 'Too low, may slow convergence'
                elif cr > 0.9:
                    impacts['crossover_rate'] = 'Too high, may lead to premature convergence'
                else:
                    impacts['crossover_rate'] = 'Appropriate for balanced exploration/exploitation'
                    
            if 'scaling_factor' in algorithm_parameters:
                f = algorithm_parameters['scaling_factor']
                if f < 0.4:
                    impacts['scaling_factor'] = 'Too low, may lead to premature convergence'
                elif f > 1.0:
                    impacts['scaling_factor'] = 'Too high, may slow convergence'
                else:
                    impacts['scaling_factor'] = 'Appropriate for balanced search'
        
        elif self.algorithm_type.upper() == 'PSO':
            # Analyze PSO parameters
            if 'inertia_weight' in algorithm_parameters:
                w = algorithm_parameters['inertia_weight']
                if w < 0.4:
                    impacts['inertia_weight'] = 'Too low, may lead to premature convergence'
                elif w > 1.0:
                    impacts['inertia_weight'] = 'Too high, may lead to divergence'
                else:
                    impacts['inertia_weight'] = 'Appropriate for balanced search'
                    
            if 'cognitive_param' in algorithm_parameters and 'social_param' in algorithm_parameters:
                c1 = algorithm_parameters['cognitive_param']
                c2 = algorithm_parameters['social_param']
                if c1 + c2 > 4.0:
                    impacts['cognitive_social_sum'] = 'Sum too high, may lead to divergence'
                else:
                    impacts['cognitive_social_sum'] = 'Appropriate for convergent behavior'
        
        return impacts
    
    def _assess_overall_convergence(self, algorithm_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide an overall assessment of convergence properties.
        
        Args:
            algorithm_parameters: A dictionary of algorithm parameters
            
        Returns:
            A dictionary containing the overall convergence assessment
        """
        assessment = {
            'expected_convergence': None,
            'reliability': None,
            'efficiency': None,
            'robustness': None,
            'limitations': []
        }
        
        # Set defaults based on algorithm type
        if self.convergence_type == 'global':
            assessment['expected_convergence'] = 'Global optimum'
            assessment['reliability'] = 'High'
        elif self.convergence_type == 'local':
            assessment['expected_convergence'] = 'Local optimum'
            assessment['reliability'] = 'Moderate'
            assessment['limitations'].append('May converge to local optimum instead of global')
        elif self.convergence_type == 'probabilistic':
            assessment['expected_convergence'] = 'Global optimum with high probability'
            assessment['reliability'] = 'Moderate to High'
            assessment['limitations'].append('No deterministic guarantee of global convergence')
        else:
            assessment['expected_convergence'] = 'No guarantee'
            assessment['reliability'] = 'Low'
            assessment['limitations'].append('No formal convergence guarantee')
        
        # Algorithm-specific assessments
        if self.algorithm_type.upper() == 'DE':
            assessment['efficiency'] = 'Moderate to High'
            assessment['robustness'] = 'High'
            assessment['limitations'].append('Performance depends on appropriate parameter tuning')
        elif self.algorithm_type.upper() == 'PSO':
            assessment['efficiency'] = 'High'
            assessment['robustness'] = 'Moderate'
            assessment['limitations'].append('May suffer from premature convergence')
            assessment['limitations'].append('Sensitive to parameter settings')
        elif self.algorithm_type.upper() == 'GD':
            assessment['efficiency'] = 'Moderate'
            assessment['robustness'] = 'Low to Moderate'
            assessment['limitations'].append('Only guarantees local convergence')
            assessment['limitations'].append('Sensitive to starting point')
            assessment['limitations'].append('May struggle with ill-conditioned problems')
        
        return assessment
    
    def compare_algorithms(self, 
                           algorithms: List[Dict[str, Any]], 
                           problem_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare the theoretical convergence properties of multiple algorithms.
        
        Args:
            algorithms: A list of dictionaries containing algorithm parameters
            problem_characteristics: A dictionary of problem characteristics
            
        Returns:
            A dictionary containing the comparison results
        """
        comparison_results = {
            'problem_characteristics': problem_characteristics,
            'algorithm_comparisons': [],
            'ranking': {},
            'recommended_algorithm': None,
            'recommendation_reason': None
        }
        
        # Analyze each algorithm
        for algorithm in algorithms:
            algorithm_type = algorithm.get('type', 'unknown')
            
            # Create a convergence analyzer for this algorithm type
            if algorithm_type != self.algorithm_type:
                analyzer = ConvergenceAnalyzer(algorithm_type)
                analysis = analyzer.analyze(algorithm)
            else:
                analysis = self.analyze(algorithm)
            
            # Calculate theoretical convergence score based on problem characteristics
            score = self._calculate_convergence_score(analysis, problem_characteristics)
            
            # Add to comparisons
            comparison_results['algorithm_comparisons'].append({
                'algorithm': algorithm_type,
                'analysis': analysis,
                'theoretical_score': score
            })
        
        # Rank algorithms by theoretical convergence score
        ranked_algorithms = sorted(
            comparison_results['algorithm_comparisons'],
            key=lambda x: x['theoretical_score'],
            reverse=True
        )
        
        # Create ranking
        for i, algo in enumerate(ranked_algorithms):
            comparison_results['ranking'][algo['algorithm']] = i + 1
        
        # Recommend best algorithm
        if ranked_algorithms:
            best_algorithm = ranked_algorithms[0]['algorithm']
            comparison_results['recommended_algorithm'] = best_algorithm
            comparison_results['recommendation_reason'] = self._get_recommendation_reason(
                best_algorithm, problem_characteristics
            )
        
        return comparison_results
    
    def _calculate_convergence_score(self, 
                                    analysis: Dict[str, Any], 
                                    problem_characteristics: Dict[str, Any]) -> float:
        """
        Calculate a theoretical convergence score for an algorithm.
        
        Args:
            analysis: The algorithm analysis results
            problem_characteristics: A dictionary of problem characteristics
            
        Returns:
            A score representing the theoretical convergence quality
        """
        score = 0.0
        
        # Base score based on convergence type
        convergence_type_scores = {
            'global': 10.0,
            'local': 7.0,
            'probabilistic': 8.0,
            'asymptotic': 6.0,
            'none': 2.0
        }
        score += convergence_type_scores.get(analysis['convergence_type'], 5.0)
        
        # Adjust based on problem characteristics
        if 'modality' in problem_characteristics:
            modality = problem_characteristics['modality']
            if modality == 'unimodal':
                # For unimodal problems, local convergence is sufficient
                if analysis['convergence_type'] in ['local', 'global']:
                    score += 3.0
            elif modality == 'multimodal':
                # For multimodal problems, global or probabilistic convergence is better
                if analysis['convergence_type'] in ['global', 'probabilistic']:
                    score += 5.0
                else:
                    score -= 2.0
        
        if 'dimension' in problem_characteristics:
            dimension = problem_characteristics['dimension']
            # Check dimension dependency in convergence rate
            dim_dependency = analysis['convergence_rate'].get('dimension_dependency')
            if dim_dependency == 'independent' and dimension > 10:
                score += 3.0
            elif dim_dependency == 'linear' and dimension > 10:
                score += 1.0
            elif dim_dependency == 'quadratic' and dimension > 10:
                score -= 2.0
        
        if 'landscape_smoothness' in problem_characteristics:
            smoothness = problem_characteristics['landscape_smoothness']
            algorithm_type = analysis['algorithm_type'].upper()
            if smoothness == 'smooth':
                if algorithm_type in ['GD', 'NEWTON']:
                    score += 3.0
            elif smoothness == 'rugged':
                if algorithm_type in ['DE', 'PSO', 'ACO']:
                    score += 3.0
                else:
                    score -= 2.0
        
        # Cap the score
        return min(max(score, 0.0), 10.0)
    
    def _get_recommendation_reason(self, 
                                  algorithm: str, 
                                  problem_characteristics: Dict[str, Any]) -> str:
        """
        Get a reason for recommending an algorithm.
        
        Args:
            algorithm: The recommended algorithm
            problem_characteristics: A dictionary of problem characteristics
            
        Returns:
            A string explaining why the algorithm is recommended
        """
        reasons = []
        
        # General reason based on algorithm
        if algorithm.upper() == 'DE':
            reasons.append("Differential Evolution provides a good balance between exploration and exploitation")
        elif algorithm.upper() == 'PSO':
            reasons.append("Particle Swarm Optimization offers fast convergence with good global search capability")
        elif algorithm.upper() == 'ACO':
            reasons.append("Ant Colony Optimization is effective for discrete optimization problems")
        elif algorithm.upper() == 'GD':
            reasons.append("Gradient Descent is efficient for smooth, convex problems")
        elif algorithm.upper() == 'NEWTON':
            reasons.append("Newton's method offers quadratic convergence rate for twice-differentiable functions")
        
        # Problem-specific reasons
        if 'modality' in problem_characteristics:
            modality = problem_characteristics['modality']
            if modality == 'unimodal' and algorithm.upper() in ['GD', 'NEWTON']:
                reasons.append("Well-suited for unimodal problems with guaranteed local convergence")
            elif modality == 'multimodal' and algorithm.upper() in ['DE', 'PSO', 'ACO']:
                reasons.append("Effective for multimodal problems due to population-based global search")
        
        if 'dimension' in problem_characteristics:
            dimension = problem_characteristics['dimension']
            if dimension > 20 and algorithm.upper() in ['DE', 'PSO']:
                reasons.append(f"Scales relatively well to high-dimensional problems ({dimension} dimensions)")
        
        if 'landscape_smoothness' in problem_characteristics:
            smoothness = problem_characteristics['landscape_smoothness']
            if smoothness == 'smooth' and algorithm.upper() in ['GD', 'NEWTON']:
                reasons.append("Takes advantage of smoothness in the objective function")
            elif smoothness == 'rugged' and algorithm.upper() in ['DE', 'PSO', 'ACO']:
                reasons.append("Robust to non-smooth or rugged objective functions")
        
        # Combine reasons
        if not reasons:
            return "Best overall theoretical convergence properties for the given problem"
        else:
            return ". ".join(reasons) + "."
    
    def get_formal_definition(self) -> str:
        """
        Get the formal mathematical definition of convergence properties.
        
        Returns:
            A string containing the formal mathematical definition
        """
        if self.algorithm_type.upper() == 'DE':
            return """
            For Differential Evolution, convergence is defined probabilistically:
            
            P(lim_{t→∞} f(x_t^*) = f(x^*)) = 1
            
            where x_t^* is the best solution at iteration t, and x^* is the global optimum.
            The convergence rate is typically O(1/t) for well-tuned parameters.
            """
        elif self.algorithm_type.upper() == 'GD':
            return """
            For Gradient Descent with step size α, assuming f is μ-strongly convex with L-Lipschitz gradients:
            
            f(x_t) - f(x^*) ≤ (1 - α·μ)^t · (f(x_0) - f(x^*))
            
            where x_t is the solution at iteration t, and x^* is the optimum.
            This gives a linear convergence rate when α is in (0, 2/L).
            """
        else:
            return f"Formal convergence definition for {self.algorithm_type} follows general principles of stochastic optimization algorithms." 
"""
Stochastic Guarantees for Optimization Algorithms.

This module provides theoretical analysis of probabilistic performance bounds
and stochastic guarantees for optimization algorithms, focusing on expected
performance, confidence intervals, and probability of success.
"""

import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from scipy.stats import norm

from core.theory.base import AlgorithmProperty


class StochasticGuaranteeAnalyzer(AlgorithmProperty):
    """
    Analyzer for stochastic guarantees of optimization algorithms.
    
    This class provides methods for analyzing the probabilistic performance bounds,
    convergence guarantees, and stochastic properties of optimization algorithms,
    including expected performance, confidence intervals, and success probabilities.
    """
    
    def __init__(self, algorithm_type: str, description: str = ""):
        """
        Initialize a stochastic guarantee analyzer for a specific algorithm type.
        
        Args:
            algorithm_type: Type of algorithm to analyze (e.g., "DE", "PSO", "ACO")
            description: Description of the analyzer
        """
        super().__init__(f"StochasticAnalyzer_{algorithm_type}", algorithm_type, description)
        self.stochastic_properties = self._determine_stochastic_properties(algorithm_type)
    
    def _determine_stochastic_properties(self, algorithm_type: str) -> Dict[str, Any]:
        """
        Determine the stochastic properties of an algorithm.
        
        Args:
            algorithm_type: Type of algorithm
            
        Returns:
            Dictionary of stochastic properties
        """
        properties = {
            'stochastic_nature': None,
            'convergence_probability': None,
            'performance_distribution': None,
            'expected_iterations': None,
            'confidence_bounds': None,
            'theoretical_notes': []
        }
        
        # Algorithm-specific properties
        if algorithm_type.upper() == 'DE':
            properties['stochastic_nature'] = 'highly stochastic'
            properties['convergence_probability'] = {
                'expression': 'P(|f(x_t) - f(x*)| < ε) → 1 as t → ∞',
                'asymptotic_rate': '~O(1/t)',
                'notes': 'Probabilistic convergence with asymptotic guarantees'
            }
            properties['performance_distribution'] = {
                'type': 'approximately normal after sufficient iterations',
                'parameters': {
                    'mean': 'dependent on problem difficulty and dimensionality',
                    'variance': 'decreases with population size and iteration count'
                }
            }
            properties['expected_iterations'] = {
                'expression': 'E[T(ε)] = O(d·log(1/ε))',
                'notes': 'Expected iterations to reach ε-proximity to optimum'
            }
            properties['confidence_bounds'] = {
                'expression': 'f(x_t) ± z_{α/2}·σ_t',
                'confidence_level': '95%',
                'notes': 'Confidence interval narrows with iterations'
            }
            properties['theoretical_notes'] = [
                "DE's stochastic nature comes from its random parent selection and differential mutation",
                "Performance distribution approximates normality due to central limit theorem effects",
                "Convergence probability approaches 1 asymptotically but without finite-time guarantees"
            ]
        
        elif algorithm_type.upper() == 'PSO':
            properties['stochastic_nature'] = 'highly stochastic'
            properties['convergence_probability'] = {
                'expression': 'P(|f(x_t) - f(x*)| < ε) → 1 as t → ∞ (under constraints)',
                'asymptotic_rate': '~O(1/t)',
                'notes': 'Probabilistic convergence requiring parameter constraints'
            }
            properties['performance_distribution'] = {
                'type': 'approximately normal for global best after sufficient iterations',
                'parameters': {
                    'mean': 'dependent on problem landscape and swarm topology',
                    'variance': 'decreases with swarm size and iteration count'
                }
            }
            properties['expected_iterations'] = {
                'expression': 'E[T(ε)] = O(d·log(1/ε))',
                'notes': 'Expected iterations with proper parameter selection'
            }
            properties['confidence_bounds'] = {
                'expression': 'f(x_t) ± z_{α/2}·σ_t',
                'confidence_level': '90%',
                'notes': 'Wider confidence intervals than some alternatives'
            }
            properties['theoretical_notes'] = [
                "PSO's stochastic guarantees depend heavily on parameter settings like inertia weight",
                "Theoretical convergence requires velocity clamping or constriction coefficient",
                "Social influence can lead to premature convergence, affecting probability guarantees"
            ]
        
        elif algorithm_type.upper() == 'GD':
            properties['stochastic_nature'] = 'deterministic (stochastic for SGD variant)'
            properties['convergence_probability'] = {
                'expression': 'P(|f(x_t) - f(x*)| < ε) = 1 for t > log(1/ε)/α',
                'asymptotic_rate': '~O(e^{-αt})',
                'notes': 'Deterministic convergence for convex problems with appropriate step size'
            }
            properties['performance_distribution'] = {
                'type': 'deterministic (normal distribution for SGD)',
                'parameters': {
                    'mean': 'follows deterministic trajectory',
                    'variance': 'zero (non-zero for SGD, decreasing with iteration count)'
                }
            }
            properties['expected_iterations'] = {
                'expression': 'T(ε) = O(log(1/ε)) for strongly convex functions',
                'notes': 'Deterministic iteration count for specified precision'
            }
            properties['confidence_bounds'] = {
                'expression': 'f(x_t) ± 0 (deterministic)',
                'confidence_level': '100%',
                'notes': 'No uncertainty in deterministic setting'
            }
            properties['theoretical_notes'] = [
                "Classical GD has deterministic guarantees for convex, smooth functions",
                "Stochastic variant (SGD) has probabilistic guarantees with variance dependent on batch size",
                "Convergence rate depends on condition number of the Hessian for convex problems"
            ]
        
        return properties
    
    def analyze(self, algorithm_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the stochastic guarantees of an algorithm.
        
        Args:
            algorithm_parameters: Dictionary of algorithm parameters
            
        Returns:
            Dictionary containing the analysis results
        """
        # Extract algorithm parameters
        algorithm_type = self.algorithm_type
        
        # Basic stochastic property analysis
        analysis_results = {
            'algorithm_type': algorithm_type,
            'stochastic_properties': self.stochastic_properties,
            'parameter_effects': self._analyze_parameter_effects(algorithm_parameters)
        }
        
        # Add problem-independent theoretical guarantees
        analysis_results['theoretical_guarantees'] = self._get_theoretical_guarantees(algorithm_type)
        
        return analysis_results
    
    def _analyze_parameter_effects(self, algorithm_parameters: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Analyze how algorithm parameters affect stochastic guarantees.
        
        Args:
            algorithm_parameters: Dictionary of algorithm parameters
            
        Returns:
            Dictionary mapping parameters to their effects on stochastic guarantees
        """
        effects = {}
        
        # Algorithm-specific parameter effects
        if self.algorithm_type.upper() == 'DE':
            # Population size effects
            if 'population_size' in algorithm_parameters:
                np_value = algorithm_parameters['population_size']
                effects['population_size'] = {
                    'convergence_probability': f"Larger population size ({np_value}) increases convergence probability",
                    'confidence_bounds': f"Variance scales approximately with 1/{np_value}",
                    'expected_iterations': "Larger population may reduce required iterations but increase total evaluations"
                }
            
            # Crossover rate effects
            if 'crossover_rate' in algorithm_parameters:
                cr_value = algorithm_parameters['crossover_rate']
                effects['crossover_rate'] = {
                    'convergence_probability': f"Balanced CR ({cr_value}) optimizes convergence probability",
                    'performance_distribution': "Affects variance in performance distribution",
                    'expected_iterations': "Extreme values may increase expected iterations to convergence"
                }
            
            # Scaling factor effects
            if 'scaling_factor' in algorithm_parameters:
                f_value = algorithm_parameters['scaling_factor']
                effects['scaling_factor'] = {
                    'convergence_probability': f"F value ({f_value}) affects exploration/exploitation balance",
                    'performance_distribution': "Larger F increases variance initially but may improve final precision",
                    'expected_iterations': "Balanced value minimizes expected iterations"
                }
        
        elif self.algorithm_type.upper() == 'PSO':
            # Swarm size effects
            if 'swarm_size' in algorithm_parameters:
                s_value = algorithm_parameters['swarm_size']
                effects['swarm_size'] = {
                    'convergence_probability': f"Larger swarm ({s_value}) generally increases convergence probability",
                    'confidence_bounds': f"Variance reduces approximately with 1/{s_value}",
                    'expected_iterations': "Larger swarm may require fewer iterations but more evaluations"
                }
            
            # Inertia weight effects
            if 'inertia_weight' in algorithm_parameters:
                w_value = algorithm_parameters['inertia_weight']
                effects['inertia_weight'] = {
                    'convergence_probability': f"Inertia weight ({w_value}) must be in appropriate range for convergence",
                    'performance_distribution': "Affects trajectory stability and variance",
                    'expected_iterations': "Optimal value minimizes expected iterations"
                }
            
            # Cognitive and social parameter effects
            if 'cognitive_param' in algorithm_parameters and 'social_param' in algorithm_parameters:
                c1_value = algorithm_parameters['cognitive_param']
                c2_value = algorithm_parameters['social_param']
                sum_value = c1_value + c2_value
                effects['cognitive_social_sum'] = {
                    'convergence_probability': f"Sum of parameters ({sum_value}) must be less than 4 for convergence",
                    'performance_distribution': "Balance affects exploitation vs exploration tradeoff",
                    'expected_iterations': "Improper balance can significantly increase iterations needed"
                }
        
        elif self.algorithm_type.upper() == 'GD':
            # Learning rate effects
            if 'learning_rate' in algorithm_parameters:
                lr_value = algorithm_parameters['learning_rate']
                effects['learning_rate'] = {
                    'convergence_probability': f"Learning rate ({lr_value}) must be within stability bounds",
                    'performance_distribution': "Deterministic trajectory determined by learning rate",
                    'expected_iterations': f"Expected iterations scale with 1/{lr_value} for appropriate range"
                }
            
            # Momentum effects
            if 'momentum' in algorithm_parameters:
                m_value = algorithm_parameters['momentum']
                effects['momentum'] = {
                    'convergence_probability': f"Momentum ({m_value}) accelerates convergence when properly tuned",
                    'performance_distribution': "Affects trajectory oscillation",
                    'expected_iterations': "Can reduce expected iterations by factor of sqrt(condition_number)"
                }
        
        return effects
    
    def _get_theoretical_guarantees(self, algorithm_type: str) -> Dict[str, Any]:
        """
        Get theoretical guarantees for an algorithm.
        
        Args:
            algorithm_type: Type of algorithm
            
        Returns:
            Dictionary of theoretical guarantees
        """
        guarantees = {
            'global_convergence': {
                'guaranteed': False,
                'conditions': [],
                'probability': None
            },
            'local_convergence': {
                'guaranteed': False,
                'conditions': [],
                'probability': None
            },
            'iteration_bounds': {
                'expression': None,
                'notes': None
            },
            'progress_rate': {
                'expression': None,
                'notes': None
            },
            'failure_probability': {
                'expression': None,
                'notes': None
            }
        }
        
        # Algorithm-specific guarantees
        if algorithm_type.upper() == 'DE':
            guarantees['global_convergence']['guaranteed'] = False
            guarantees['global_convergence']['probability'] = "P → 1 as t → ∞ under appropriate conditions"
            guarantees['global_convergence']['conditions'] = [
                "Sufficient population diversity maintained",
                "Proper balance of crossover rate and scaling factor",
                "Elitist selection preserving best solution",
                "Unbounded computing resources"
            ]
            
            guarantees['local_convergence']['guaranteed'] = True
            guarantees['local_convergence']['probability'] = "P = 1 (in finite time with appropriate conditions)"
            guarantees['local_convergence']['conditions'] = [
                "Elitist selection preserving best solution",
                "Non-zero mutation probability"
            ]
            
            guarantees['iteration_bounds']['expression'] = "E[T(ε)] = O(d·log(1/ε))"
            guarantees['iteration_bounds']['notes'] = "Expected number of iterations to reach ε-approximate solution"
            
            guarantees['progress_rate']['expression'] = "E[f(x_{t+1}) - f(x_t)] ≤ -c·σ_t²"
            guarantees['progress_rate']['notes'] = "Expected progress per iteration where σ_t is population variance"
            
            guarantees['failure_probability']['expression'] = "P(failure) ≤ exp(-population_size·t/d)"
            guarantees['failure_probability']['notes'] = "Upper bound on probability of not finding global optimum"
        
        elif algorithm_type.upper() == 'PSO':
            guarantees['global_convergence']['guaranteed'] = False
            guarantees['global_convergence']['probability'] = "P → 1 as t → ∞ under restrictive conditions"
            guarantees['global_convergence']['conditions'] = [
                "Velocity bounds or constriction coefficient",
                "Inertia weight in appropriate range",
                "Cognitive and social parameters properly balanced",
                "Unbounded computing resources"
            ]
            
            guarantees['local_convergence']['guaranteed'] = True
            guarantees['local_convergence']['probability'] = "P = 1 (in finite time with appropriate conditions)"
            guarantees['local_convergence']['conditions'] = [
                "Appropriate parameter selection",
                "Local memory (personal best) retention"
            ]
            
            guarantees['iteration_bounds']['expression'] = "E[T(ε)] = O(d·log(1/ε))"
            guarantees['iteration_bounds']['notes'] = "Expected number of iterations with properly tuned parameters"
            
            guarantees['progress_rate']['expression'] = "E[f(g_{t+1}) - f(g_t)] ≤ -c·σ_t"
            guarantees['progress_rate']['notes'] = "Expected progress of global best per iteration"
            
            guarantees['failure_probability']['expression'] = "P(failure) ≤ exp(-swarm_size·t/d)"
            guarantees['failure_probability']['notes'] = "Upper bound on probability of not finding global optimum"
        
        elif algorithm_type.upper() == 'GD':
            guarantees['global_convergence']['guaranteed'] = True
            guarantees['global_convergence']['conditions'] = [
                "Function is convex",
                "Function is smooth (Lipschitz continuous gradient)",
                "Appropriate learning rate (< 2/L where L is Lipschitz constant)"
            ]
            guarantees['global_convergence']['probability'] = "P = 1 (deterministic)"
            
            guarantees['local_convergence']['guaranteed'] = True
            guarantees['local_convergence']['probability'] = "P = 1 (deterministic)"
            guarantees['local_convergence']['conditions'] = [
                "Function is smooth (Lipschitz continuous gradient)",
                "Appropriate learning rate"
            ]
            
            guarantees['iteration_bounds']['expression'] = "T(ε) = O(κ·log(1/ε))"
            guarantees['iteration_bounds']['notes'] = "Where κ is the condition number of the Hessian for convex functions"
            
            guarantees['progress_rate']['expression'] = "f(x_{t+1}) - f(x_t) ≤ -η·||∇f(x_t)||²"
            guarantees['progress_rate']['notes'] = "Deterministic progress rate for appropriate learning rate η"
            
            guarantees['failure_probability']['expression'] = "P(failure) = 0 for convex functions"
            guarantees['failure_probability']['notes'] = "Deterministic convergence to global optimum for convex functions"
        
        return guarantees
    
    def compare_algorithms(self, 
                          algorithms: List[Dict[str, Any]], 
                          problem_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare algorithms based on stochastic guarantees.
        
        Args:
            algorithms: List of dictionaries containing algorithm parameters
            problem_characteristics: Dictionary of problem characteristics
            
        Returns:
            Dictionary containing the comparison results
        """
        comparison_results = {
            'problem_characteristics': problem_characteristics,
            'algorithm_comparisons': [],
            'stochastic_ranking': {},
            'confidence_levels': {},
            'expected_iterations_comparison': {},
            'robustness_comparison': {}
        }
        
        # Extract relevant problem properties
        modality = problem_characteristics.get('modality', 'unknown')
        smoothness = problem_characteristics.get('landscape_smoothness', 'unknown')
        dimension = problem_characteristics.get('dimension', 10)
        target_precision = problem_characteristics.get('target_precision', 1e-4)
        
        # Analyze each algorithm
        for algorithm in algorithms:
            algorithm_type = algorithm.get('type', 'unknown')
            
            # Create a stochastic analyzer for this algorithm type if needed
            if algorithm_type != self.algorithm_type:
                analyzer = StochasticGuaranteeAnalyzer(algorithm_type)
                stochastic_analysis = analyzer.analyze(algorithm)
            else:
                stochastic_analysis = self.analyze(algorithm)
            
            # Calculate stochastic performance metrics
            metrics = self._calculate_stochastic_metrics(stochastic_analysis, 
                                                     modality, smoothness, dimension, target_precision)
            
            # Add to comparisons
            comparison_results['algorithm_comparisons'].append({
                'algorithm': algorithm_type,
                'stochastic_analysis': stochastic_analysis,
                'performance_metrics': metrics
            })
            
            # Store individual metrics for easy comparison
            comparison_results['confidence_levels'][algorithm_type] = metrics['confidence_level']
            comparison_results['expected_iterations_comparison'][algorithm_type] = metrics['expected_iterations']
            comparison_results['robustness_comparison'][algorithm_type] = metrics['robustness_score']
        
        # Rank algorithms by overall stochastic performance
        ranked_algorithms = sorted(
            comparison_results['algorithm_comparisons'],
            key=lambda x: x['performance_metrics']['overall_score'],
            reverse=True
        )
        
        # Create ranking
        for i, algo in enumerate(ranked_algorithms):
            comparison_results['stochastic_ranking'][algo['algorithm']] = i + 1
        
        # Add qualitative comparison summary
        comparison_results['summary'] = self._generate_comparison_summary(
            comparison_results['algorithm_comparisons'],
            modality, smoothness, dimension
        )
        
        return comparison_results
    
    def _calculate_stochastic_metrics(self, 
                                    stochastic_analysis: Dict[str, Any],
                                    modality: str,
                                    smoothness: str,
                                    dimension: int,
                                    target_precision: float = 1e-4) -> Dict[str, Any]:
        """
        Calculate stochastic performance metrics for an algorithm.
        
        Args:
            stochastic_analysis: Analysis of algorithm's stochastic properties
            modality: Problem modality
            smoothness: Problem smoothness
            dimension: Problem dimension
            target_precision: Target precision level
            
        Returns:
            Dictionary of stochastic performance metrics
        """
        metrics = {
            'confidence_level': None,
            'expected_iterations': None,
            'probability_of_success': None,
            'robustness_score': None,
            'overall_score': None
        }
        
        algorithm_type = stochastic_analysis['algorithm_type']
        properties = stochastic_analysis['stochastic_properties']
        guarantees = stochastic_analysis['theoretical_guarantees']
        
        # Calculate confidence level (normalized to 0-10 scale)
        if properties['confidence_bounds'] and 'confidence_level' in properties['confidence_bounds']:
            confidence_str = properties['confidence_bounds']['confidence_level']
            try:
                # Extract percentage and convert to 0-10 scale
                confidence_pct = float(confidence_str.strip('%'))
                metrics['confidence_level'] = min(confidence_pct / 10.0, 10.0)
            except ValueError:
                # Default if parsing fails
                metrics['confidence_level'] = 5.0
        else:
            metrics['confidence_level'] = 5.0
        
        # Estimate expected iterations based on dimension and algorithm properties
        # This is a theoretical approximation
        if algorithm_type.upper() == 'GD' and smoothness == 'smooth' and modality == 'unimodal':
            # GD has logarithmic convergence for nice functions
            metrics['expected_iterations'] = int(dimension * math.log(1.0 / target_precision))
        elif algorithm_type.upper() in ['DE', 'PSO']:
            # Population methods need more iterations for higher dimensions
            # but handle multimodality better
            base_multiplier = 10 if modality == 'unimodal' else 20
            if modality == 'highly multimodal':
                base_multiplier = 40
            
            metrics['expected_iterations'] = int(base_multiplier * dimension * math.log(1.0 / target_precision))
        else:
            # Default estimate
            metrics['expected_iterations'] = int(30 * dimension * math.log(1.0 / target_precision))
        
        # Calculate probability of success based on problem characteristics
        if guarantees['global_convergence']['guaranteed']:
            # Guaranteed convergence to global optimum
            # Check for algorithm type and modality exceptions
            if algorithm_type.upper() == 'GD' and 'multimodal' in modality:
                # GD struggles with multimodality even if convergence is guaranteed
                metrics['probability_of_success'] = 3.0
            else:
                metrics['probability_of_success'] = 10.0
        elif modality == 'unimodal' and smoothness == 'smooth' and algorithm_type.upper() == 'GD':
            # GD works well on nice problems
            metrics['probability_of_success'] = 9.5
        elif modality == 'unimodal' and algorithm_type.upper() in ['DE', 'PSO']:
            # Population methods are good but not optimal for unimodal
            metrics['probability_of_success'] = 8.0
        elif 'multimodal' in modality and algorithm_type.upper() in ['DE', 'PSO']:
            # Population methods handle multimodality well
            metrics['probability_of_success'] = 7.5
        elif 'multimodal' in modality and algorithm_type.upper() == 'GD':
            # GD struggles with multimodality
            metrics['probability_of_success'] = 3.0
        else:
            # Default moderate probability
            metrics['probability_of_success'] = 5.0
        
        # Calculate robustness score based on stochastic nature
        if properties['stochastic_nature'] == 'deterministic':
            # Deterministic methods are very reliable but may be trapped
            if modality == 'unimodal':
                metrics['robustness_score'] = 9.0
            else:
                metrics['robustness_score'] = 4.0
        elif 'highly stochastic' in properties['stochastic_nature']:
            # Highly stochastic methods are more robust to varying problems
            if 'multimodal' in modality:
                metrics['robustness_score'] = 8.0
            else:
                metrics['robustness_score'] = 7.0
        else:
            # Default moderate robustness
            metrics['robustness_score'] = 6.0
        
        # Calculate overall score as weighted average
        # Weight based on problem characteristics
        if 'multimodal' in modality:
            # For multimodal problems, success probability and robustness matter more
            weights = {
                'confidence_level': 0.2,
                'expected_iterations': 0.1,
                'probability_of_success': 0.4,
                'robustness_score': 0.3
            }
        else:
            # For unimodal problems, expected iterations and confidence matter more
            weights = {
                'confidence_level': 0.3,
                'expected_iterations': 0.3,
                'probability_of_success': 0.2,
                'robustness_score': 0.2
            }
        
        # Convert expected iterations to a 0-10 scale (lower is better)
        # Assuming reasonable range is 10d to 1000d
        iter_score = max(0, 10 - metrics['expected_iterations'] / (100 * dimension))
        
        # Calculate weighted score
        metrics['overall_score'] = (
            weights['confidence_level'] * metrics['confidence_level'] +
            weights['expected_iterations'] * iter_score +
            weights['probability_of_success'] * metrics['probability_of_success'] +
            weights['robustness_score'] * metrics['robustness_score']
        )
        
        return metrics
    
    def _generate_comparison_summary(self, 
                                   algorithm_comparisons: List[Dict[str, Any]],
                                   modality: str,
                                   smoothness: str,
                                   dimension: int) -> str:
        """
        Generate a qualitative summary of the stochastic comparison.
        
        Args:
            algorithm_comparisons: List of algorithm comparison results
            modality: Problem modality
            smoothness: Problem smoothness
            dimension: Problem dimension
            
        Returns:
            Qualitative summary string
        """
        # Sort algorithms by overall score
        sorted_algorithms = sorted(
            algorithm_comparisons,
            key=lambda x: x['performance_metrics']['overall_score'],
            reverse=True
        )
        
        if not sorted_algorithms:
            return "No algorithms to compare."
        
        best_algorithm = sorted_algorithms[0]['algorithm']
        best_metrics = sorted_algorithms[0]['performance_metrics']
        
        # Construct appropriate summary based on problem characteristics
        if modality == 'unimodal' and smoothness == 'smooth':
            if best_algorithm.upper() == 'GD':
                summary = (
                    f"For this unimodal, smooth problem, {best_algorithm} provides the strongest stochastic guarantees "
                    f"with a confidence level of {best_metrics['confidence_level']:.1f}/10 and deterministic convergence. "
                    f"It requires approximately {best_metrics['expected_iterations']} iterations to reach the target precision."
                )
            else:
                summary = (
                    f"For this unimodal, smooth problem, {best_algorithm} provides good stochastic guarantees "
                    f"with a confidence level of {best_metrics['confidence_level']:.1f}/10, though a deterministic method "
                    f"like Gradient Descent might provide stronger theoretical guarantees. "
                    f"It requires approximately {best_metrics['expected_iterations']} iterations to reach the target precision."
                )
        
        elif 'multimodal' in modality:
            if best_algorithm.upper() in ['DE', 'PSO']:
                summary = (
                    f"For this {modality} problem, {best_algorithm} provides the strongest stochastic guarantees "
                    f"with a probability of success of {best_metrics['probability_of_success']:.1f}/10 and robustness of "
                    f"{best_metrics['robustness_score']:.1f}/10. It requires approximately {best_metrics['expected_iterations']} "
                    f"iterations to reach the target precision. Population-based methods like this are theoretically "
                    f"better suited for multimodal problems due to their global exploration capabilities."
                )
            else:
                summary = (
                    f"For this {modality} problem, {best_algorithm} provides the strongest overall stochastic guarantees "
                    f"based on the weighted criteria, but population-based methods may offer better global exploration. "
                    f"It has a probability of success of {best_metrics['probability_of_success']:.1f}/10 and robustness of "
                    f"{best_metrics['robustness_score']:.1f}/10, requiring approximately {best_metrics['expected_iterations']} "
                    f"iterations to reach the target precision."
                )
        
        elif smoothness == 'rugged':
            if best_algorithm.upper() in ['DE', 'PSO']:
                summary = (
                    f"For this rugged landscape, {best_algorithm} provides the strongest stochastic guarantees "
                    f"with a robustness score of {best_metrics['robustness_score']:.1f}/10. Its stochastic nature "
                    f"helps it navigate rough landscapes more effectively than deterministic methods. "
                    f"It requires approximately {best_metrics['expected_iterations']} iterations to reach the target precision."
                )
            else:
                summary = (
                    f"For this rugged landscape, {best_algorithm} provides the strongest overall stochastic guarantees "
                    f"based on the weighted criteria, though stochastic methods are typically better suited for rough landscapes. "
                    f"It has a robustness score of {best_metrics['robustness_score']:.1f}/10 and requires approximately "
                    f"{best_metrics['expected_iterations']} iterations to reach the target precision."
                )
        
        else:
            # General case
            summary = (
                f"Based on stochastic guarantee analysis, {best_algorithm} provides the strongest overall guarantees "
                f"for this problem with an overall score of {best_metrics['overall_score']:.1f}/10. "
                f"It offers a confidence level of {best_metrics['confidence_level']:.1f}/10, "
                f"probability of success of {best_metrics['probability_of_success']:.1f}/10, "
                f"and robustness of {best_metrics['robustness_score']:.1f}/10. "
                f"It requires approximately {best_metrics['expected_iterations']} iterations to reach the target precision."
            )
        
        # Add comparison to second-best if available
        if len(sorted_algorithms) > 1:
            second_best = sorted_algorithms[1]['algorithm']
            second_metrics = sorted_algorithms[1]['performance_metrics']
            score_diff = best_metrics['overall_score'] - second_metrics['overall_score']
            
            if score_diff < 0.5:
                summary += (
                    f" However, {second_best} is very competitive with an overall score of {second_metrics['overall_score']:.1f}/10, "
                    f"just {score_diff:.1f} points behind. The choice between these algorithms may depend on specific implementation "
                    f"considerations or additional problem characteristics."
                )
            elif score_diff < 2.0:
                summary += (
                    f" {second_best} is also a reasonable alternative with an overall score of {second_metrics['overall_score']:.1f}/10."
                )
        
        return summary
    
    def get_formal_definition(self) -> str:
        """
        Get the formal definition of stochastic guarantees.
        
        Returns:
            Formal definition as a string
        """
        return """
        Stochastic Guarantee Theory for Optimization Algorithms
        
        For an algorithm A applied to an optimization problem f:
        
        1. Probabilistic Convergence Guarantee:
           P(lim_{t→∞} |f(x_t) - f(x*)| < ε) = 1
           
           where x_t is the solution at iteration t, x* is the global optimum,
           and ε > 0 is an arbitrarily small constant.
        
        2. Expected Convergence Time:
           E[T(ε)] = expected number of iterations to reach ε-approximation
        
        3. Confidence Bounds:
           With probability (1-α), the true optimum f(x*) lies in the interval:
           [f(x_t) - z_{α/2}·σ_t, f(x_t) + z_{α/2}·σ_t]
           
           where z_{α/2} is the critical value of the normal distribution
           and σ_t is the standard deviation at iteration t.
        
        4. Failure Probability Bound:
           P(|f(x_t) - f(x*)| > ε) ≤ C·exp(-rₑ·t)
           
           where C is a constant and rₑ is the effective convergence rate.
        
        These stochastic guarantees formalize the probabilistic performance
        bounds of optimization algorithms under uncertainty.
        """ 
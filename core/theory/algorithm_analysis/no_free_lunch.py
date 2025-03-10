"""
No Free Lunch Theorem Analysis for Optimization Algorithms.

This module provides theoretical analysis based on the No Free Lunch (NFL) theorems,
which establish fundamental limitations on algorithm performance across all possible
problems and provide a theoretical foundation for meta-optimization approaches.
"""

import numpy as np
import math
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from core.theory.base import AlgorithmProperty


class NoFreeLunchAnalyzer(AlgorithmProperty):
    """
    Analyzer for No Free Lunch theorem implications on optimization algorithms.
    
    This class provides methods for analyzing the theoretical implications of
    the No Free Lunch theorems on algorithm performance, specialization, and
    the theoretical foundations of meta-optimization and algorithm selection.
    """
    
    NFL_PRINCIPLES = {
        'performance_equality': 'All algorithms perform equally when averaged over all possible problems',
        'specialization_necessity': 'Superior performance on one class of problems implies inferior performance on others',
        'problem_knowledge': 'Algorithm performance depends on matching algorithm bias to problem structure',
        'meta_optimization': 'Meta-optimization is theoretically justified by the NFL theorems',
        'free_lunches': 'Restricted problem classes can provide "free lunches" where some algorithms consistently outperform others'
    }
    
    def __init__(self, algorithm_type: str, description: str = ""):
        """
        Initialize a No Free Lunch analyzer for a specific algorithm type.
        
        Args:
            algorithm_type: Type of algorithm to analyze (e.g., "DE", "PSO", "ACO")
            description: Description of the analyzer
        """
        super().__init__(f"NFLAnalyzer_{algorithm_type}", algorithm_type, description)
        self.algorithm_bias = self._determine_algorithm_bias(algorithm_type)
    
    def _determine_algorithm_bias(self, algorithm_type: str) -> Dict[str, Any]:
        """
        Determine the implicit bias of an algorithm, which determines its
        effectiveness on different problem classes according to NFL theorems.
        
        Args:
            algorithm_type: Type of algorithm
            
        Returns:
            Dictionary describing the algorithm's bias
        """
        # Default bias structure
        bias = {
            'bias_type': None,
            'favored_problem_classes': [],
            'disfavored_problem_classes': [],
            'bias_strength': None,
            'bias_description': None
        }
        
        # Algorithm-specific biases
        if algorithm_type.upper() == 'DE':
            bias['bias_type'] = 'evolutionary'
            bias['favored_problem_classes'] = [
                'multimodal', 
                'nonseparable', 
                'rugged',
                'black-box'
            ]
            bias['disfavored_problem_classes'] = [
                'simple unimodal',
                'linear',
                'highly constrained'
            ]
            bias['bias_strength'] = 'moderate'
            bias['bias_description'] = (
                "Differential Evolution has a bias toward exploration through its "
                "population-based approach and differential mutation operator, making "
                "it effective for complex multimodal problems but potentially inefficient "
                "for simple unimodal problems."
            )
        
        elif algorithm_type.upper() == 'PSO':
            bias['bias_type'] = 'swarm intelligence'
            bias['favored_problem_classes'] = [
                'multimodal with structure',
                'continuous',
                'moderately rugged'
            ]
            bias['disfavored_problem_classes'] = [
                'highly deceptive',
                'discrete',
                'highly constrained'
            ]
            bias['bias_strength'] = 'moderate to strong'
            bias['bias_description'] = (
                "Particle Swarm Optimization has a bias toward social information "
                "sharing and momentum-based search, making it effective for problems "
                "with some global structure but potentially vulnerable to deception."
            )
        
        elif algorithm_type.upper() == 'GD':
            bias['bias_type'] = 'gradient-based'
            bias['favored_problem_classes'] = [
                'unimodal',
                'smooth',
                'convex',
                'differentiable'
            ]
            bias['disfavored_problem_classes'] = [
                'multimodal',
                'rugged',
                'non-differentiable',
                'discrete'
            ]
            bias['bias_strength'] = 'strong'
            bias['bias_description'] = (
                "Gradient Descent has a strong bias toward following local gradient "
                "information, making it highly effective for smooth, convex problems "
                "but ineffective for multimodal or non-differentiable problems."
            )
        
        elif algorithm_type.upper() == 'RANDOM':
            bias['bias_type'] = 'none'
            bias['favored_problem_classes'] = []
            bias['disfavored_problem_classes'] = [
                'all non-trivial problems'
            ]
            bias['bias_strength'] = 'none'
            bias['bias_description'] = (
                "Random search has no algorithmic bias, which according to NFL theorems "
                "means it has no advantages on any problem class but also no disadvantages "
                "when averaged across all possible functions."
            )
        
        return bias
    
    def analyze(self, algorithm_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the NFL theorem implications for an algorithm.
        
        Args:
            algorithm_parameters: Dictionary of algorithm parameters
            
        Returns:
            Dictionary containing the analysis results
        """
        # Extract algorithm parameters
        algorithm_type = self.algorithm_type
        
        # Basic NFL analysis
        analysis_results = {
            'algorithm_type': algorithm_type,
            'algorithm_bias': self.algorithm_bias,
            'nfl_principles': self.NFL_PRINCIPLES,
            'nfl_implications': self._get_nfl_implications(algorithm_type)
        }
        
        # Parameter-specific analysis if relevant
        if algorithm_parameters:
            analysis_results['parameter_influence'] = self._analyze_parameter_influence(algorithm_parameters)
        
        return analysis_results
    
    def _get_nfl_implications(self, algorithm_type: str) -> List[Dict[str, str]]:
        """
        Get the NFL theorem implications for a specific algorithm.
        
        Args:
            algorithm_type: Type of algorithm
            
        Returns:
            List of NFL implications for the algorithm
        """
        implications = []
        
        # Common implications for all algorithms
        implications.append({
            'principle': 'performance_equality',
            'implication': f"According to NFL, {algorithm_type} cannot outperform other algorithms when averaged across all possible functions, suggesting its observed advantages must be specific to certain problem classes."
        })
        
        implications.append({
            'principle': 'specialization_necessity',
            'implication': f"Any superior performance of {algorithm_type} on its favored problem classes implies inferior performance on some other classes of problems."
        })
        
        # Algorithm-specific implications
        if self.algorithm_bias['bias_type'] == 'evolutionary':
            implications.append({
                'principle': 'problem_knowledge',
                'implication': f"The evolutionary bias of {algorithm_type} implicitly encodes knowledge about problems with complex landscapes, multiple optima, and interdependent variables."
            })
        
        elif self.algorithm_bias['bias_type'] == 'gradient-based':
            implications.append({
                'principle': 'problem_knowledge',
                'implication': f"The gradient-based bias of {algorithm_type} implicitly encodes knowledge about problems with smooth, continuous landscapes where local information points toward optima."
            })
        
        elif self.algorithm_bias['bias_type'] == 'swarm intelligence':
            implications.append({
                'principle': 'problem_knowledge',
                'implication': f"The swarm intelligence bias of {algorithm_type} implicitly encodes knowledge about problems where social information sharing and memory of good solutions are beneficial."
            })
        
        # Meta-optimization implications
        implications.append({
            'principle': 'meta_optimization',
            'implication': f"The bias of {algorithm_type} creates a theoretical justification for meta-optimization approaches that select between this and other algorithms based on problem characteristics."
        })
        
        # Free lunch implications if algorithm has strong bias
        if self.algorithm_bias['bias_strength'] in ['strong', 'moderate to strong']:
            implications.append({
                'principle': 'free_lunches',
                'implication': f"The strong bias of {algorithm_type} may provide 'free lunches' on restricted problem classes that align with its bias, allowing it to consistently outperform less specialized algorithms on these problems."
            })
        
        return implications
    
    def _analyze_parameter_influence(self, algorithm_parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze how algorithm parameters influence its bias under NFL theorems.
        
        Args:
            algorithm_parameters: Dictionary of algorithm parameters
            
        Returns:
            Dictionary mapping parameters to their influence on bias
        """
        influences = {}
        
        if self.algorithm_type.upper() == 'DE':
            if 'crossover_rate' in algorithm_parameters:
                cr = algorithm_parameters['crossover_rate']
                if cr > 0.7:
                    influences['crossover_rate'] = "High CR increases exploitation bias, narrowing effective problem classes"
                elif cr < 0.3:
                    influences['crossover_rate'] = "Low CR increases exploration bias, broadening effective problem classes"
            
            if 'population_size' in algorithm_parameters:
                pop = algorithm_parameters['population_size']
                if pop > 50:
                    influences['population_size'] = "Large population reduces algorithm bias, broadening effective problem classes"
                elif pop < 20:
                    influences['population_size'] = "Small population increases exploitation bias, narrowing effective problem classes"
        
        elif self.algorithm_type.upper() == 'PSO':
            if 'inertia_weight' in algorithm_parameters:
                w = algorithm_parameters['inertia_weight']
                if w > 0.8:
                    influences['inertia_weight'] = "High inertia increases exploration bias, broadening effective problem classes"
                elif w < 0.4:
                    influences['inertia_weight'] = "Low inertia increases exploitation bias, narrowing effective problem classes"
        
        elif self.algorithm_type.upper() == 'GD':
            if 'learning_rate' in algorithm_parameters:
                lr = algorithm_parameters['learning_rate']
                if lr > 0.1:
                    influences['learning_rate'] = "High learning rate reduces gradient-following bias, potentially broadening effective problem classes"
                elif lr < 0.01:
                    influences['learning_rate'] = "Low learning rate strengthens gradient-following bias, narrowing effective problem classes"
        
        return influences
    
    def compare_algorithms(self, 
                          algorithms: List[Dict[str, Any]], 
                          problem_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare algorithms based on NFL theorem implications.
        
        Args:
            algorithms: List of dictionaries containing algorithm parameters
            problem_characteristics: Dictionary of problem characteristics
            
        Returns:
            Dictionary containing the comparison results
        """
        comparison_results = {
            'problem_characteristics': problem_characteristics,
            'problem_class': self._determine_problem_class(problem_characteristics),
            'algorithm_comparisons': [],
            'theoretical_free_lunch': None,
            'nfl_implied_ranking': {},
            'meta_optimization_potential': None
        }
        
        # Analyze each algorithm
        for algorithm in algorithms:
            algorithm_type = algorithm.get('type', 'unknown')
            
            # Create a NFL analyzer for this algorithm type if needed
            if algorithm_type != self.algorithm_type:
                analyzer = NoFreeLunchAnalyzer(algorithm_type)
                nfl_analysis = analyzer.analyze(algorithm)
            else:
                nfl_analysis = self.analyze(algorithm)
            
            # Calculate theoretical NFL alignment score
            alignment_score = self._calculate_nfl_alignment(nfl_analysis, 
                                                        comparison_results['problem_class'])
            
            # Add to comparisons
            comparison_results['algorithm_comparisons'].append({
                'algorithm': algorithm_type,
                'nfl_analysis': nfl_analysis,
                'alignment_score': alignment_score
            })
        
        # Determine if there's a theoretical "free lunch"
        free_lunch = self._check_for_free_lunch(comparison_results['algorithm_comparisons'],
                                             comparison_results['problem_class'])
        comparison_results['theoretical_free_lunch'] = free_lunch
        
        # Rank algorithms by NFL alignment
        ranked_algorithms = sorted(
            comparison_results['algorithm_comparisons'],
            key=lambda x: x['alignment_score'],
            reverse=True
        )
        
        # Create ranking
        for i, algo in enumerate(ranked_algorithms):
            comparison_results['nfl_implied_ranking'][algo['algorithm']] = i + 1
        
        # Assess meta-optimization potential
        score_differences = [a['alignment_score'] for a in comparison_results['algorithm_comparisons']]
        if score_differences and len(score_differences) > 1:
            score_range = max(score_differences) - min(score_differences)
            if score_range > 5.0:
                potential = "High"
            elif score_range > 2.0:
                potential = "Moderate"
            else:
                potential = "Low"
        else:
            potential = "Undetermined"
        
        comparison_results['meta_optimization_potential'] = {
            'level': potential,
            'explanation': self._get_meta_optimization_explanation(potential, comparison_results['problem_class'])
        }
        
        return comparison_results
    
    def _determine_problem_class(self, problem_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the problem class based on characteristics.
        
        Args:
            problem_characteristics: Dictionary of problem characteristics
            
        Returns:
            Dictionary describing the problem class
        """
        problem_class = {
            'modality': 'unknown',
            'smoothness': 'unknown',
            'separability': 'unknown',
            'constraints': 'unknown',
            'dimensionality': 'unknown',
            'description': 'General problem with unknown characteristics'
        }
        
        # Determine modality
        if 'modality' in problem_characteristics:
            problem_class['modality'] = problem_characteristics['modality']
        
        # Determine smoothness
        if 'landscape_smoothness' in problem_characteristics:
            smoothness = problem_characteristics['landscape_smoothness']
            problem_class['smoothness'] = smoothness
        
        # Determine separability
        if 'separability' in problem_characteristics:
            problem_class['separability'] = problem_characteristics['separability']
        
        # Determine dimensionality
        if 'dimension' in problem_characteristics:
            dim = problem_characteristics['dimension']
            if dim <= 2:
                problem_class['dimensionality'] = 'very low'
            elif dim <= 10:
                problem_class['dimensionality'] = 'low'
            elif dim <= 50:
                problem_class['dimensionality'] = 'medium'
            elif dim <= 100:
                problem_class['dimensionality'] = 'high'
            else:
                problem_class['dimensionality'] = 'very high'
        
        # Create a description
        description_parts = []
        if problem_class['modality'] != 'unknown':
            description_parts.append(problem_class['modality'])
        if problem_class['smoothness'] != 'unknown':
            description_parts.append(problem_class['smoothness'])
        if problem_class['separability'] != 'unknown':
            description_parts.append(problem_class['separability'])
        if problem_class['dimensionality'] != 'unknown':
            description_parts.append(f"{problem_class['dimensionality']} dimensional")
        
        if description_parts:
            problem_class['description'] = " ".join(description_parts) + " optimization problem"
        
        return problem_class
    
    def _calculate_nfl_alignment(self, 
                               nfl_analysis: Dict[str, Any], 
                               problem_class: Dict[str, Any]) -> float:
        """
        Calculate how well an algorithm's bias aligns with a problem class under NFL.
        
        Args:
            nfl_analysis: NFL analysis for an algorithm
            problem_class: Problem class characteristics
            
        Returns:
            Alignment score (0-10, higher is better alignment)
        """
        bias = nfl_analysis['algorithm_bias']
        score = 5.0  # Neutral starting point
        
        # Check if problem class matches favored classes
        for favored_class in bias['favored_problem_classes']:
            favored_terms = favored_class.lower().split()
            problem_desc = problem_class['description'].lower()
            
            # Check for partial matches
            match_count = sum(1 for term in favored_terms if term in problem_desc)
            if match_count == len(favored_terms):
                # Complete match
                score += 2.0
            elif match_count > 0:
                # Partial match
                score += match_count * 0.5
        
        # Check if problem class matches disfavored classes
        for disfavored_class in bias['disfavored_problem_classes']:
            disfavored_terms = disfavored_class.lower().split()
            problem_desc = problem_class['description'].lower()
            
            # Check for partial matches
            match_count = sum(1 for term in disfavored_terms if term in problem_desc)
            if match_count == len(disfavored_terms):
                # Complete match
                score -= 2.0
            elif match_count > 0:
                # Partial match
                score -= match_count * 0.5
        
        # Specific checks for known problem characteristics
        if problem_class['modality'] == 'unimodal' and 'gradient-based' in bias['bias_type']:
            score += 2.0
        elif 'multimodal' in problem_class['modality'] and 'evolutionary' in bias['bias_type']:
            score += 2.0
        
        if problem_class['smoothness'] == 'smooth' and 'gradient-based' in bias['bias_type']:
            score += 1.5
        elif problem_class['smoothness'] == 'rugged' and 'evolutionary' in bias['bias_type']:
            score += 1.5
        
        # Cap the score
        return min(max(score, 0.0), 10.0)
    
    def _check_for_free_lunch(self, 
                            algorithm_comparisons: List[Dict[str, Any]], 
                            problem_class: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the problem class provides a "free lunch" for any algorithm.
        
        Args:
            algorithm_comparisons: List of algorithm comparisons
            problem_class: Problem class characteristics
            
        Returns:
            Dictionary describing any free lunch found
        """
        if not algorithm_comparisons:
            return {
                'exists': False,
                'explanation': "No algorithms to compare"
            }
        
        # Find algorithm with highest alignment score
        best_algorithm = max(algorithm_comparisons, key=lambda x: x['alignment_score'])
        
        # Check if the best algorithm has a significantly higher score
        other_scores = [a['alignment_score'] for a in algorithm_comparisons 
                      if a['algorithm'] != best_algorithm['algorithm']]
        
        if not other_scores:
            return {
                'exists': False,
                'explanation': "Only one algorithm to evaluate"
            }
        
        score_difference = best_algorithm['alignment_score'] - max(other_scores)
        
        if score_difference > 3.0:
            return {
                'exists': True,
                'algorithm': best_algorithm['algorithm'],
                'alignment_score': best_algorithm['alignment_score'],
                'score_difference': score_difference,
                'explanation': f"The problem class strongly aligns with the bias of {best_algorithm['algorithm']}, creating a theoretical 'free lunch' according to NFL theorems."
            }
        elif score_difference > 1.0:
            return {
                'exists': True,
                'algorithm': best_algorithm['algorithm'],
                'alignment_score': best_algorithm['alignment_score'],
                'score_difference': score_difference,
                'explanation': f"The problem class moderately aligns with the bias of {best_algorithm['algorithm']}, suggesting a potential 'free lunch' according to NFL theorems."
            }
        else:
            return {
                'exists': False,
                'explanation': "No significant alignment advantage for any algorithm"
            }
    
    def _get_meta_optimization_explanation(self, potential: str, problem_class: Dict[str, Any]) -> str:
        """
        Get an explanation of meta-optimization potential based on NFL.
        
        Args:
            potential: Meta-optimization potential level
            problem_class: Problem class characteristics
            
        Returns:
            Explanation of meta-optimization potential
        """
        if potential == "High":
            return (
                f"For the {problem_class['description']}, NFL theorems suggest high potential "
                f"for meta-optimization due to significant differences in algorithm bias alignment. "
                f"A meta-optimizer would likely provide substantial benefits by selecting "
                f"the most appropriate algorithm."
            )
        elif potential == "Moderate":
            return (
                f"For the {problem_class['description']}, NFL theorems suggest moderate potential "
                f"for meta-optimization. While some algorithms have better bias alignment, "
                f"the differences are not extreme. A meta-optimizer would provide benefits "
                f"but may need to consider additional factors."
            )
        elif potential == "Low":
            return (
                f"For the {problem_class['description']}, NFL theorems suggest low potential "
                f"for meta-optimization. The algorithms have similar bias alignment with this "
                f"problem class, indicating that algorithm selection may offer limited benefits."
            )
        else:
            return "Meta-optimization potential could not be determined from the available information."
    
    def get_formal_definition(self) -> str:
        """
        Get the formal definition of the No Free Lunch theorems.
        
        Returns:
            Formal definition as a string
        """
        return """
        The No Free Lunch (NFL) Theorems for Optimization formally state:
        
        For any two algorithms a and b:
        
        ∑_f P(dᵧ_m|f,m,a) = ∑_f P(dᵧ_m|f,m,b)
        
        Where:
        - f ranges over all possible functions from search space X to codomain Y
        - dᵧ_m is a time-ordered sequence of m distinct points in Y
        - P(dᵧ_m|f,m,a) is the probability of obtaining sequence dᵧ_m when
          running algorithm a on function f
        
        This theorem implies that when performance is averaged across all possible 
        functions, all algorithms perform exactly the same. Superior performance 
        on one class of functions mathematically necessitates inferior performance 
        on another class.
        
        This provides the theoretical foundation for meta-optimization, as it 
        establishes that algorithm selection must be based on matching algorithm 
        bias to problem structure.
        """ 
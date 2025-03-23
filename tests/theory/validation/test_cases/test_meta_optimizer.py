"""
Test Cases for Meta-Optimizer Components.

This module provides test cases for validating the theoretical correctness
of meta-optimizer components using synthetic data generators.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from tests.theory.validation.test_harness import TestCase, CoreLayerHarness
from tests.theory.validation.synthetic_generators.patient_generators import (
    PatientGenerator, LongitudinalDataGenerator
)
from tests.theory.validation.synthetic_generators.algorithm_generators import (
    AlgorithmPerformanceGenerator, DriftScenarioGenerator
)
from core.theory.algorithm_analysis.convergence_analysis import ConvergenceAnalyzer
from core.theory.algorithm_analysis.landscape_theory import LandscapeAnalyzer
from core.theory.algorithm_analysis.no_free_lunch import NFLAnalyzer
from core.meta_optimizer.algorithm_selection import AlgorithmSelector
from core.meta_optimizer.drift_detection import DriftDetector
from core.meta_optimizer.explainability import ExplainabilityEngine

class TestAlgorithmSelection:
    """Test cases for algorithm selection in the meta-optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()
        self.convergence = ConvergenceAnalyzer()
        self.landscape = LandscapeAnalyzer()
        self.nfl = NFLAnalyzer()
        self.alg_gen = AlgorithmPerformanceGenerator()
    
    def test_algorithm_ranking(self):
        """Test ranking of algorithms based on theoretical properties."""
        # Generate synthetic algorithm performance data
        num_algorithms = 5
        num_datasets = 10
        
        algorithm_properties = self.alg_gen.generate_algorithm_properties(
            num_algorithms=num_algorithms,
            property_types=[
                "convergence_rate", 
                "stability", 
                "computational_complexity",
                "sample_efficiency",
                "uncertainty_handling"
            ]
        )
        
        performance_data = self.alg_gen.generate_performance_matrix(
            num_algorithms=num_algorithms,
            num_datasets=num_datasets,
            metrics=["accuracy", "f1_score", "auc", "training_time"]
        )
        
        # Create test case
        test_case = TestCase(
            name="algorithm_ranking",
            inputs={
                "algorithm_properties": algorithm_properties,
                "performance_data": performance_data,
                "optimization_criteria": {
                    "accuracy": 0.5,
                    "speed": 0.3,
                    "stability": 0.2
                }
            },
            expected_outputs={
                "ranking_quality": 0.8,  # Expected ranking quality
                "top_algorithm_score": 0.85,  # Expected score for top algorithm
                "theoretical_alignment": 0.8,  # Alignment with theoretical expectations
                "ranking_stability": {
                    "bootstrap_consistency": 0.85,  # Consistency across bootstrap samples
                    "criteria_sensitivity": 0.2   # Sensitivity to criteria changes
                }
            },
            tolerance={
                "quality": 0.1,      # ±10% tolerance
                "alignment": 0.1,    # ±10% tolerance
                "stability": 0.1     # ±10% tolerance
            },
            metadata={
                "description": "Validate algorithm ranking functionality",
                "true_ranking": self.alg_gen.get_true_ranking()
            }
        )
        
        # Create validation function
        def validate_ranking(
            algorithm_properties: Dict[str, Dict[str, float]],
            performance_data: Dict[str, np.ndarray],
            optimization_criteria: Dict[str, float]
        ) -> Dict[str, Any]:
            # Perform algorithm ranking
            results = self.selector.rank_algorithms(
                properties=algorithm_properties,
                performance=performance_data,
                criteria=optimization_criteria
            )
            
            # Validate theoretical alignment
            theoretical_ranking = self.convergence.predict_ranking(
                algorithm_properties,
                criteria=optimization_criteria
            )
            
            alignment_score = self.selector.compute_ranking_similarity(
                results['ranking'],
                theoretical_ranking
            )
            
            # Compute bootstrap stability
            bootstrap_results = self.selector.bootstrap_ranking(
                properties=algorithm_properties,
                performance=performance_data,
                criteria=optimization_criteria,
                n_samples=100
            )
            
            # Compute criteria sensitivity
            sensitivity_results = self.selector.criteria_sensitivity_analysis(
                properties=algorithm_properties,
                performance=performance_data,
                base_criteria=optimization_criteria,
                perturbation=0.1
            )
            
            return {
                "ranking_quality": results['quality_score'],
                "top_algorithm_score": results['algorithm_scores'][results['ranking'][0]],
                "theoretical_alignment": alignment_score,
                "ranking_stability": {
                    "bootstrap_consistency": bootstrap_results['consistency'],
                    "criteria_sensitivity": sensitivity_results['sensitivity']
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("algorithm_ranking", validate_ranking)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_algorithm_selection_for_patient(self):
        """Test selection of optimal algorithm for specific patient characteristics."""
        # Generate synthetic patient data
        patient_gen = PatientGenerator(num_patients=50)
        profiles = [
            patient_gen.generate_profile(complexity='low'),
            patient_gen.generate_profile(complexity='medium'),
            patient_gen.generate_profile(complexity='high')
        ]
        
        # Generate algorithm performance for different patient types
        algorithm_performance = self.alg_gen.generate_conditional_performance(
            num_algorithms=5,
            condition_types=["patient_complexity", "data_density", "symptom_variability"],
            condition_values={
                "patient_complexity": ["low", "medium", "high"],
                "data_density": ["sparse", "moderate", "dense"],
                "symptom_variability": ["stable", "variable", "highly_variable"]
            }
        )
        
        # Create test case
        test_case = TestCase(
            name="patient_specific_selection",
            inputs={
                "patient_profiles": profiles,
                "algorithm_performance": algorithm_performance,
                "selection_criteria": {
                    "prediction_accuracy": 0.6,
                    "computational_efficiency": 0.2,
                    "interpretability": 0.2
                }
            },
            expected_outputs={
                "selection_quality": 0.85,  # Expected selection quality
                "profile_differentiation": 0.8,  # Different algorithms for different profiles
                "theoretical_justification": 0.8,  # Theoretical justification score
                "nfl_compliance": True  # Compliance with No Free Lunch theorem
            },
            tolerance={
                "quality": 0.1,      # ±10% tolerance
                "differentiation": 0.1,  # ±10% tolerance
                "justification": 0.1  # ±10% tolerance
            },
            metadata={
                "description": "Validate patient-specific algorithm selection",
                "profile_complexities": ["low", "medium", "high"]
            }
        )
        
        # Create validation function
        def validate_patient_selection(
            patient_profiles: List[Dict[str, Any]],
            algorithm_performance: Dict[str, Dict[str, Dict[str, float]]],
            selection_criteria: Dict[str, float]
        ) -> Dict[str, Any]:
            # Select algorithms for each patient profile
            selections = {}
            for i, profile in enumerate(patient_profiles):
                result = self.selector.select_for_patient(
                    patient_profile=profile,
                    algorithm_performance=algorithm_performance,
                    criteria=selection_criteria
                )
                selections[f"profile_{i}"] = result
            
            # Check if different profiles get different algorithms
            algorithms = [s['selected_algorithm'] for s in selections.values()]
            differentiation = len(set(algorithms)) / len(algorithms)
            
            # Validate against NFL theorem
            nfl_analysis = self.nfl.validate_selection(
                selections=selections,
                performance=algorithm_performance
            )
            
            # Get theoretical justification
            justification_scores = []
            for profile_id, selection in selections.items():
                justification = self.selector.explain_selection(
                    algorithm=selection['selected_algorithm'],
                    patient_profile=patient_profiles[int(profile_id.split('_')[1])],
                    performance=algorithm_performance
                )
                justification_scores.append(justification['justification_score'])
            
            return {
                "selection_quality": np.mean([s['quality_score'] for s in selections.values()]),
                "profile_differentiation": differentiation,
                "theoretical_justification": np.mean(justification_scores),
                "nfl_compliance": nfl_analysis['compliant']
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("patient_selection", validate_patient_selection)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestDriftDetection:
    """Test cases for drift detection in the meta-optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DriftDetector()
        self.drift_gen = DriftScenarioGenerator()
    
    def test_concept_drift_detection(self):
        """Test detection of concept drift in patient data."""
        # Generate synthetic data with known drift points
        duration_days = 90  # 3 months
        drift_points = [30, 60]  # Drift at 1 month and 2 months
        drift_types = ["gradual", "sudden"]
        
        drift_data = self.drift_gen.generate_drift_scenario(
            duration_days=duration_days,
            drift_points=drift_points,
            drift_types=drift_types,
            metrics=["prediction_error", "feature_importance", "data_distribution"]
        )
        
        # Create test case
        test_case = TestCase(
            name="concept_drift_detection",
            inputs={
                "time_series_data": drift_data['time_series'],
                "prediction_errors": drift_data['prediction_error'],
                "feature_importances": drift_data['feature_importance'],
                "detection_sensitivity": 0.8
            },
            expected_outputs={
                "detection_quality": {
                    "precision": 0.9,  # Few false positives
                    "recall": 0.9,     # Few false negatives
                    "f1_score": 0.9    # Overall quality
                },
                "drift_characterization": {
                    "type_accuracy": 0.8,  # Accuracy of drift type identification
                    "magnitude_error": 0.2  # Error in magnitude estimation
                },
                "detection_latency": {
                    "gradual": 5,  # Days to detect gradual drift
                    "sudden": 2    # Days to detect sudden drift
                }
            },
            tolerance={
                "quality": 0.1,      # ±10% tolerance
                "characterization": 0.1,  # ±10% tolerance
                "latency": 1.0       # ±1 day tolerance
            },
            metadata={
                "description": "Validate concept drift detection",
                "true_drift_points": drift_points,
                "true_drift_types": drift_types
            }
        )
        
        # Create validation function
        def validate_drift_detection(
            time_series_data: Dict[str, np.ndarray],
            prediction_errors: np.ndarray,
            feature_importances: Dict[str, np.ndarray],
            detection_sensitivity: float
        ) -> Dict[str, Any]:
            # Detect drift
            results = self.detector.detect_drift(
                time_series=time_series_data,
                prediction_errors=prediction_errors,
                feature_importances=feature_importances,
                sensitivity=detection_sensitivity
            )
            
            # Calculate detection quality
            true_drift_days = test_case.metadata['true_drift_points']
            detected_drift_days = results['drift_points']
            
            # A detection is considered correct if within ±3 days of true drift
            true_positives = sum(any(abs(d - t) <= 3 for t in true_drift_days) for d in detected_drift_days)
            false_positives = len(detected_drift_days) - true_positives
            false_negatives = sum(not any(abs(d - t) <= 3 for d in detected_drift_days) for t in true_drift_days)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate drift characterization accuracy
            type_matches = 0
            for i, true_type in enumerate(test_case.metadata['true_drift_types']):
                if i < len(results['drift_types']):
                    if results['drift_types'][i] == true_type:
                        type_matches += 1
            
            type_accuracy = type_matches / len(test_case.metadata['true_drift_types'])
            
            # Calculate detection latency
            latency = {}
            for i, (true_day, true_type) in enumerate(zip(true_drift_days, test_case.metadata['true_drift_types'])):
                if i < len(detected_drift_days):
                    latency[true_type] = abs(detected_drift_days[i] - true_day)
            
            return {
                "detection_quality": {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                },
                "drift_characterization": {
                    "type_accuracy": type_accuracy,
                    "magnitude_error": results['magnitude_error']
                },
                "detection_latency": latency
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("drift_detection", validate_drift_detection)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_adaptation_to_drift(self):
        """Test adaptation strategies in response to detected drift."""
        # Generate synthetic data with drift and adaptation
        duration_days = 120  # 4 months
        drift_points = [30, 60, 90]  # Monthly drifts
        adaptation_strategies = ["retraining", "ensemble_update", "feature_reweighting"]
        
        adaptation_data = self.drift_gen.generate_adaptation_scenario(
            duration_days=duration_days,
            drift_points=drift_points,
            adaptation_strategies=adaptation_strategies,
            metrics=["prediction_error", "adaptation_cost", "recovery_time"]
        )
        
        # Create test case
        test_case = TestCase(
            name="drift_adaptation",
            inputs={
                "drift_scenario": adaptation_data['scenario'],
                "available_strategies": adaptation_strategies,
                "resource_constraints": {
                    "computation_budget": 0.7,  # 70% of max computation
                    "memory_budget": 0.8,      # 80% of max memory
                    "latency_requirement": 0.5  # 50% of max latency
                }
            },
            expected_outputs={
                "adaptation_effectiveness": {
                    "error_reduction": 0.7,  # Error reduction after adaptation
                    "recovery_speed": 0.8,   # Speed of recovery
                    "stability": 0.75       # Stability after adaptation
                },
                "strategy_selection": {
                    "optimality": 0.8,      # Optimality of strategy selection
                    "resource_efficiency": 0.85  # Efficient use of resources
                },
                "theoretical_guarantees": {
                    "convergence_maintained": True,  # Convergence properties maintained
                    "uncertainty_quantified": True   # Uncertainty properly quantified
                }
            },
            tolerance={
                "effectiveness": 0.1,  # ±10% tolerance
                "selection": 0.1,      # ±10% tolerance
                "guarantees": 0.0      # No tolerance for theoretical guarantees
            },
            metadata={
                "description": "Validate adaptation to concept drift",
                "true_drift_points": drift_points,
                "true_strategies": adaptation_strategies
            }
        )
        
        # Create validation function
        def validate_adaptation(
            drift_scenario: Dict[str, Any],
            available_strategies: List[str],
            resource_constraints: Dict[str, float]
        ) -> Dict[str, Any]:
            # Perform adaptation
            results = self.detector.adapt_to_drift(
                scenario=drift_scenario,
                strategies=available_strategies,
                constraints=resource_constraints
            )
            
            # Analyze convergence properties
            convergence_analysis = self.detector.analyze_convergence(
                before_drift=drift_scenario['before_drift'],
                after_adaptation=results['after_adaptation']
            )
            
            # Analyze uncertainty quantification
            uncertainty_analysis = self.detector.analyze_uncertainty(
                before_drift=drift_scenario['before_drift'],
                after_adaptation=results['after_adaptation']
            )
            
            return {
                "adaptation_effectiveness": {
                    "error_reduction": results['error_reduction'],
                    "recovery_speed": results['recovery_speed'],
                    "stability": results['stability']
                },
                "strategy_selection": {
                    "optimality": results['strategy_optimality'],
                    "resource_efficiency": results['resource_efficiency']
                },
                "theoretical_guarantees": {
                    "convergence_maintained": convergence_analysis['maintained'],
                    "uncertainty_quantified": uncertainty_analysis['properly_quantified']
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("drift_adaptation", validate_adaptation)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results

class TestExplainability:
    """Test cases for explainability in the meta-optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.explainer = ExplainabilityEngine()
        self.patient_gen = PatientGenerator(num_patients=10)
    
    def test_algorithm_decision_explanation(self):
        """Test explanation of algorithm selection decisions."""
        # Generate synthetic algorithm selection scenario
        algorithm_properties = {
            "algorithm_1": {
                "convergence_rate": 0.8,
                "stability": 0.7,
                "interpretability": 0.9,
                "computational_complexity": 0.6
            },
            "algorithm_2": {
                "convergence_rate": 0.9,
                "stability": 0.6,
                "interpretability": 0.5,
                "computational_complexity": 0.8
            },
            "algorithm_3": {
                "convergence_rate": 0.7,
                "stability": 0.9,
                "interpretability": 0.7,
                "computational_complexity": 0.7
            }
        }
        
        selection_decision = {
            "selected_algorithm": "algorithm_3",
            "scores": {
                "algorithm_1": 0.75,
                "algorithm_2": 0.72,
                "algorithm_3": 0.78
            },
            "criteria_weights": {
                "convergence_rate": 0.3,
                "stability": 0.4,
                "interpretability": 0.2,
                "computational_complexity": 0.1
            }
        }
        
        # Create test case
        test_case = TestCase(
            name="algorithm_explanation",
            inputs={
                "algorithm_properties": algorithm_properties,
                "selection_decision": selection_decision,
                "explanation_level": "detailed"
            },
            expected_outputs={
                "explanation_quality": {
                    "completeness": 0.9,  # Explanation covers all factors
                    "correctness": 0.95,  # Explanation is factually correct
                    "coherence": 0.9     # Explanation is logically coherent
                },
                "theoretical_grounding": 0.85,  # Explanation references theory
                "counterfactual_quality": 0.8,  # Quality of counterfactual explanations
                "feature_attribution": {
                    "accuracy": 0.9,     # Attribution accuracy
                    "consistency": 0.85  # Attribution consistency
                }
            },
            tolerance={
                "quality": 0.1,      # ±10% tolerance
                "grounding": 0.1,    # ±10% tolerance
                "attribution": 0.1   # ±10% tolerance
            },
            metadata={
                "description": "Validate algorithm selection explanation",
                "expected_key_factors": ["stability", "convergence_rate"]
            }
        )
        
        # Create validation function
        def validate_explanation(
            algorithm_properties: Dict[str, Dict[str, float]],
            selection_decision: Dict[str, Any],
            explanation_level: str
        ) -> Dict[str, Any]:
            # Generate explanation
            explanation = self.explainer.explain_algorithm_selection(
                properties=algorithm_properties,
                decision=selection_decision,
                level=explanation_level
            )
            
            # Validate explanation contains key factors
            key_factors = test_case.metadata['expected_key_factors']
            factor_coverage = sum(1 for f in key_factors if f in explanation['key_factors']) / len(key_factors)
            
            # Generate counterfactual explanations
            counterfactuals = self.explainer.generate_counterfactuals(
                properties=algorithm_properties,
                decision=selection_decision,
                num_counterfactuals=2
            )
            
            # Validate feature attribution
            attribution = self.explainer.attribute_features(
                properties=algorithm_properties,
                decision=selection_decision
            )
            
            # Check theoretical references
            theory_references = self.explainer.get_theoretical_references(
                properties=algorithm_properties,
                decision=selection_decision
            )
            
            return {
                "explanation_quality": {
                    "completeness": explanation['completeness_score'],
                    "correctness": explanation['correctness_score'],
                    "coherence": explanation['coherence_score']
                },
                "theoretical_grounding": len(theory_references) / 5.0,  # Normalize by expected max references
                "counterfactual_quality": counterfactuals['quality_score'],
                "feature_attribution": {
                    "accuracy": attribution['accuracy_score'],
                    "consistency": attribution['consistency_score']
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("algorithm_explanation", validate_explanation)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results
    
    def test_clinical_insight_generation(self):
        """Test generation of clinical insights from model decisions."""
        # Generate synthetic patient data
        patient_data = self.patient_gen.generate_clinical_cases(
            num_patients=10,
            with_explanations=True
        )
        
        model_decisions = {
            "patient_1": {
                "prediction": "high_risk",
                "confidence": 0.85,
                "contributing_factors": [
                    {"factor": "stress_level", "importance": 0.4},
                    {"factor": "sleep_quality", "importance": 0.3},
                    {"factor": "weather_change", "importance": 0.2},
                    {"factor": "caffeine_intake", "importance": 0.1}
                ]
            },
            "patient_2": {
                "prediction": "low_risk",
                "confidence": 0.75,
                "contributing_factors": [
                    {"factor": "regular_sleep", "importance": 0.5},
                    {"factor": "stress_management", "importance": 0.3},
                    {"factor": "medication_adherence", "importance": 0.2}
                ]
            }
        }
        
        # Create test case
        test_case = TestCase(
            name="clinical_insights",
            inputs={
                "patient_data": patient_data,
                "model_decisions": model_decisions,
                "insight_types": ["trigger_identification", "intervention_suggestion", "risk_explanation"]
            },
            expected_outputs={
                "insight_quality": {
                    "clinical_relevance": 0.85,  # Relevance to clinical practice
                    "actionability": 0.8,       # Actionable insights
                    "personalization": 0.75     # Personalized to patient
                },
                "explanation_characteristics": {
                    "clarity": 0.9,         # Clear explanations
                    "consistency": 0.85,    # Consistent with model
                    "simplicity": 0.8       # Simple enough for users
                },
                "theoretical_foundation": {
                    "causal_validity": 0.8,  # Valid causal relationships
                    "evidence_based": 0.85   # Based on evidence
                }
            },
            tolerance={
                "quality": 0.1,        # ±10% tolerance
                "characteristics": 0.1,  # ±10% tolerance
                "foundation": 0.1      # ±10% tolerance
            },
            metadata={
                "description": "Validate clinical insight generation",
                "expected_insight_count": 3  # Expected insights per patient
            }
        )
        
        # Create validation function
        def validate_insights(
            patient_data: Dict[str, Dict[str, Any]],
            model_decisions: Dict[str, Dict[str, Any]],
            insight_types: List[str]
        ) -> Dict[str, Any]:
            # Generate clinical insights
            insights = self.explainer.generate_clinical_insights(
                patient_data=patient_data,
                model_decisions=model_decisions,
                insight_types=insight_types
            )
            
            # Validate with clinical knowledge base
            clinical_validation = self.explainer.validate_clinical_relevance(
                insights=insights,
                knowledge_base="migraine_clinical_guidelines"
            )
            
            # Validate causal relationships
            causal_validation = self.explainer.validate_causal_relationships(
                insights=insights,
                patient_data=patient_data
            )
            
            # Evaluate explanation characteristics
            explanation_eval = self.explainer.evaluate_explanations(
                insights=insights,
                criteria=["clarity", "consistency", "simplicity"]
            )
            
            return {
                "insight_quality": {
                    "clinical_relevance": clinical_validation['relevance_score'],
                    "actionability": clinical_validation['actionability_score'],
                    "personalization": clinical_validation['personalization_score']
                },
                "explanation_characteristics": {
                    "clarity": explanation_eval['clarity_score'],
                    "consistency": explanation_eval['consistency_score'],
                    "simplicity": explanation_eval['simplicity_score']
                },
                "theoretical_foundation": {
                    "causal_validity": causal_validation['validity_score'],
                    "evidence_based": causal_validation['evidence_score']
                }
            }
        
        # Create harness and validate
        harness = CoreLayerHarness("clinical_insights", validate_insights)
        harness.add_test_case(test_case)
        results = harness.run_all()
        
        return results 
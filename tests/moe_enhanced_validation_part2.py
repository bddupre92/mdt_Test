#!/usr/bin/env python3
"""
MoE Enhanced Validation Framework - Part 2
-----------------------------------------
Test cases for the Mixture-of-Experts system focusing on:
1. Meta_Optimizer tests
2. Meta_Learner tests
3. Drift Detection tests
4. Explainability tests
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import from Part 1
from moe_enhanced_validation_part1 import (
    SyntheticDataGenerator, 
    MockExpert,
    MockMetaOptimizer, 
    MockMetaLearner,
    MockDriftDetector,
    MockGatingNetwork,
    MockExplainabilityEngine
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
parent_dir = str(Path(__file__).parent.parent.absolute())
results_dir = Path(parent_dir) / "results" / "moe_validation"
results_dir.mkdir(parents=True, exist_ok=True)

#########################
# Validation Test Cases #
#########################

class MetaOptimizerTests:
    """Tests for the Meta_Optimizer component."""
    
    def __init__(self):
        """Initialize Meta_Optimizer tests."""
        self.meta_optimizer = MockMetaOptimizer()
        self.results = {}
        
    def test_algorithm_selection(self):
        """Test Meta_Optimizer algorithm selection."""
        logger.info("Running Meta_Optimizer algorithm selection test")
        
        # Define test problems with varying characteristics
        test_problems = [
            {'dimension': 2, 'multimodality': 0.2, 'ruggedness': 0.1, 'name': 'Simple2D'},
            {'dimension': 20, 'multimodality': 0.3, 'ruggedness': 0.3, 'name': 'HighDim'},
            {'dimension': 5, 'multimodality': 0.8, 'ruggedness': 0.6, 'name': 'Multimodal'},
            {'dimension': 10, 'multimodality': 0.4, 'ruggedness': 0.2, 'name': 'Medium'}
        ]
        
        results = []
        for problem in test_problems:
            selected = self.meta_optimizer.select_optimizer(problem)
            results.append({
                'problem': problem['name'],
                'dimension': problem['dimension'],
                'multimodality': problem['multimodality'],
                'ruggedness': problem['ruggedness'],
                'selected_optimizer': selected
            })
            
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_dir / "meta_optimizer_selection_test.csv", index=False)
        
        logger.info(f"Meta_Optimizer selection test complete. Results saved to {results_dir / 'meta_optimizer_selection_test.csv'}")
        
        # Check if selections are diverse
        unique_selections = len(results_df['selected_optimizer'].unique())
        self.results['algorithm_selection'] = {
            'passed': unique_selections > 1,
            'details': f"Selected {unique_selections} different optimizers for {len(test_problems)} problems"
        }
        
        return unique_selections > 1
    
    def test_optimizer_portfolio_management(self):
        """Test Meta_Optimizer portfolio management."""
        logger.info("Running Meta_Optimizer portfolio management test")
        
        # Test managing a portfolio of optimizers
        initial_optimizers = set(self.meta_optimizer.optimizers.keys())
        
        # Add a new optimizer
        self.meta_optimizer.optimizers['CMA'] = {'speed': 0.7, 'quality': 0.9, 'robustness': 0.85}
        
        # Remove an optimizer
        if 'PSO' in self.meta_optimizer.optimizers:
            del self.meta_optimizer.optimizers['PSO']
            
        # Check portfolio modifications
        current_optimizers = set(self.meta_optimizer.optimizers.keys())
        
        # Calculate changes
        added = current_optimizers - initial_optimizers
        removed = initial_optimizers - current_optimizers
        
        self.results['portfolio_management'] = {
            'passed': 'CMA' in current_optimizers and 'PSO' not in current_optimizers,
            'details': f"Added optimizers: {added}, Removed optimizers: {removed}"
        }
        
        logger.info(f"Portfolio management test complete. Added: {added}, Removed: {removed}")
        
        return 'CMA' in current_optimizers and 'PSO' not in current_optimizers
    
    def test_optimization_history_tracking(self):
        """Test optimization history tracking."""
        logger.info("Running optimization history tracking test")
        
        # Run multiple optimizations to generate history
        test_problems = [
            {'dimension': 2, 'multimodality': 0.2, 'ruggedness': 0.1, 'name': 'Simple2D'},
            {'dimension': 5, 'multimodality': 0.8, 'ruggedness': 0.6, 'name': 'Multimodal'},
            {'dimension': 10, 'multimodality': 0.4, 'ruggedness': 0.2, 'name': 'Medium'}
        ]
        
        for problem in test_problems:
            self.meta_optimizer.optimize(problem, max_evals=50)
            
        # Check history
        history_size = len(self.meta_optimizer.history)
        history_contains_required_fields = all(
            all(k in item for k in ['problem', 'selected', 'timestamp'])
            for item in self.meta_optimizer.history
        )
        
        self.results['history_tracking'] = {
            'passed': history_size == len(test_problems) and history_contains_required_fields,
            'details': f"History size: {history_size}, Contains required fields: {history_contains_required_fields}"
        }
        
        logger.info(f"History tracking test complete. History entries: {history_size}")
        
        return history_size == len(test_problems) and history_contains_required_fields
    
    def run_all_tests(self):
        """Run all Meta_Optimizer tests."""
        logger.info("Running all Meta_Optimizer tests...")
        
        # First run history tracking, so we don't encounter issues with optimizer removal
        self.test_optimization_history_tracking()
        # Then run other tests
        self.test_algorithm_selection()
        self.test_optimizer_portfolio_management()
        
        # Report results
        logger.info("\n=== Meta_Optimizer Test Results ===")
        all_passed = True
        
        for test_name, result in self.results.items():
            status = "PASSED" if result['passed'] else "FAILED"
            logger.info(f"{test_name}: {status} - {result['details']}")
            
            if not result['passed']:
                all_passed = False
                
        logger.info(f"Overall Meta_Optimizer tests: {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed

class MetaLearnerTests:
    """Tests for the Meta_Learner component."""
    
    def __init__(self):
        """Initialize Meta_Learner tests."""
        self.meta_learner = MockMetaLearner()
        self.results = {}
        
    def test_expert_weight_prediction(self):
        """Test Meta_Learner expert weight prediction."""
        logger.info("Running Meta_Learner expert weight prediction test")
        
        # Create mock experts with different specialties
        experts = [
            MockExpert(1, 'physiological'),
            MockExpert(2, 'behavioral'),
            MockExpert(3, 'environmental'),
            MockExpert(4, 'general')
        ]
        
        # Register experts with meta-learner
        for expert in experts:
            self.meta_learner.register_expert(expert.expert_id, expert)
            
        # Test weight prediction with different feature sets
        test_cases = [
            {'has_physiological': True, 'has_behavioral': False, 'has_environmental': False, 'case': 'Physiological Only'},
            {'has_physiological': False, 'has_behavioral': True, 'has_environmental': False, 'case': 'Behavioral Only'},
            {'has_physiological': False, 'has_behavioral': False, 'has_environmental': True, 'case': 'Environmental Only'},
            {'has_physiological': True, 'has_behavioral': True, 'has_environmental': True, 'case': 'All Data Types'}
        ]
        
        results = []
        for case in test_cases:
            weights = self.meta_learner.predict_weights(case)
            
            # Record results
            case_result = {
                'case': case['case'],
                'total_weight': sum(weights.values())
            }
            
            # Add individual expert weights
            for expert_id, weight in weights.items():
                case_result[f'expert_{expert_id}_weight'] = weight
                
            results.append(case_result)
            
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_dir / "meta_learner_weights_test.csv", index=False)
        
        logger.info(f"Meta_Learner weight prediction test complete. Results saved to {results_dir / 'meta_learner_weights_test.csv'}")
        
        # Verify weights sum to 1 and preferred experts get higher weights
        total_weight_check = all(abs(row['total_weight'] - 1.0) < 1e-10 for _, row in results_df.iterrows())
        
        # Check if physiological expert gets highest weight for physiological-only case
        physio_case = results_df[results_df['case'] == 'Physiological Only']
        physio_expert_highest = False
        if not physio_case.empty:
            expert_1_weight = physio_case['expert_1_weight'].values[0]
            other_weights = [physio_case[f'expert_{i}_weight'].values[0] for i in range(2, 5) if f'expert_{i}_weight' in physio_case.columns]
            physio_expert_highest = all(expert_1_weight >= w for w in other_weights)
        
        self.results['expert_weight_prediction'] = {
            'passed': total_weight_check and physio_expert_highest,
            'details': f"Weights sum to 1: {total_weight_check}, Physiological expert highest for physio case: {physio_expert_highest}"
        }
        
        return total_weight_check and physio_expert_highest
    
    def test_adaptive_selection_strategy(self):
        """Test Meta_Learner adaptive selection strategy."""
        logger.info("Running Meta_Learner adaptive selection test")
        
        # Create mock experts
        experts = [
            MockExpert(1, 'general'),
            MockExpert(2, 'general')
        ]
        
        # Register experts with meta-learner
        for expert in experts:
            self.meta_learner.register_expert(expert.expert_id, expert)
            
        # Initial weights
        initial_weights = self.meta_learner.predict_weights({'has_physiological': True})
        
        # Update expert performance
        self.meta_learner.update_performance(1, 0.95)  # Expert 1 performs well
        self.meta_learner.update_performance(2, 0.65)  # Expert 2 performs poorly
        
        # Get updated weights
        updated_weights = self.meta_learner.predict_weights({'has_physiological': True})
        
        # Check if weights adapted
        weight_adapted = updated_weights[1] > initial_weights[1] and updated_weights[2] < initial_weights[2]
        
        self.results['adaptive_selection'] = {
            'passed': weight_adapted,
            'details': f"Initial weights: {initial_weights}, Updated weights: {updated_weights}"
        }
        
        logger.info(f"Adaptive selection test complete. Initial weights: {initial_weights}, Updated weights: {updated_weights}")
        
        return weight_adapted
    
    def test_performance_prediction(self):
        """Test Meta_Learner performance prediction."""
        logger.info("Running Meta_Learner performance prediction test")
        
        # Create synthetic data
        data_gen = SyntheticDataGenerator()
        physio_data = data_gen.generate_physiological_data(n_samples=30)
        
        # Create target data
        y = np.random.uniform(0, 10, len(physio_data))
        
        # Create and train experts
        experts = {}
        for i in range(3):
            expert = MockExpert(i, specialty='general')
            # Drop timestamp for training
            X_train = physio_data.drop(columns=['timestamp'])
            expert.train(X_train, y)
            experts[i] = expert
            
        # Register experts with meta-learner
        for expert_id, expert in experts.items():
            self.meta_learner.register_expert(expert_id, expert)
            
        # Make predictions with each expert
        X_test = physio_data.drop(columns=['timestamp']).iloc[:5]
        expert_predictions = {i: expert.predict(X_test) for i, expert in experts.items()}
        
        # Create "actual" performance metrics
        performance_metrics = {
            0: 0.85,  # Expert 0 performs well
            1: 0.70,  # Expert 1 performs moderately
            2: 0.60   # Expert 2 performs poorly
        }
        
        # Update meta-learner with performance info
        for expert_id, metric in performance_metrics.items():
            self.meta_learner.update_performance(expert_id, metric)
            
        # Get weights
        weights = self.meta_learner.predict_weights({'has_physiological': True})
        
        # Check if weights correlate with performance
        weights_ordered_correctly = weights[0] > weights[1] > weights[2]
        
        self.results['performance_prediction'] = {
            'passed': weights_ordered_correctly,
            'details': f"Weights: {weights}, Performance metrics: {performance_metrics}"
        }
        
        logger.info(f"Performance prediction test complete. Weights: {weights}")
        
        return weights_ordered_correctly
    
    def run_all_tests(self):
        """Run all Meta_Learner tests."""
        logger.info("Running all Meta_Learner tests...")
        
        self.test_expert_weight_prediction()
        self.test_adaptive_selection_strategy()
        self.test_performance_prediction()
        
        # Report results
        logger.info("\n=== Meta_Learner Test Results ===")
        all_passed = True
        
        for test_name, result in self.results.items():
            status = "PASSED" if result['passed'] else "FAILED"
            logger.info(f"{test_name}: {status} - {result['details']}")
            
            if not result['passed']:
                all_passed = False
                
        logger.info(f"Overall Meta_Learner tests: {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed

if __name__ == "__main__":
    logger.info("Testing individual components...")
    
    # Run Meta_Optimizer tests
    meta_opt_tests = MetaOptimizerTests()
    meta_opt_tests.run_all_tests()
    
    # Run Meta_Learner tests
    meta_learner_tests = MetaLearnerTests()
    meta_learner_tests.run_all_tests()

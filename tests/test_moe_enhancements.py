#!/usr/bin/env python
"""
Test script for demonstrating the MoE validation enhancements working together.
This script showcases:
1. Enhanced Drift Notifications with visual explanations
2. Selective Expert Retraining based on drift impact
3. Continuous Explainability monitoring
4. Confidence Metrics that incorporate drift severity

Author: Blair Dupre
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.enhanced_drift_notifications import EnhancedDriftNotifier
from core.selective_expert_retraining import SelectiveExpertRetrainer
from core.continuous_explainability import ContinuousExplainabilityPipeline
from core.confidence_metrics import ConfidenceMetricsCalculator
from core.moe_validation_enhancements import MoEValidationEnhancer

# Import sklearn components
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MoEEnhancementDemo:
    """Class to demonstrate the MoE validation enhancements."""
    
    def __init__(self, results_dir="results/moe_enhancements_demo"):
        """Initialize the demo with specified results directory."""
        self.results_dir = results_dir
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectories
        self.plots_dir = os.path.join(results_dir, "plots")
        self.data_dir = os.path.join(results_dir, "data")
        Path(self.plots_dir).mkdir(exist_ok=True)
        Path(self.data_dir).mkdir(exist_ok=True)
        
        # Initialize enhancement components
        self.drift_notifier = EnhancedDriftNotifier(
            notify_threshold=0.3,
            results_dir=self.plots_dir
        )
        
        self.expert_retrainer = SelectiveExpertRetrainer(
            impact_threshold=0.3,
            results_dir=self.results_dir
        )
        
        self.continuous_explainer = ContinuousExplainabilityPipeline(
            update_interval=10,  # Shorter interval for demo purposes
            explainer_types=["shap", "feature_importance"],
            results_dir=self.plots_dir
        )
        
        self.confidence_calculator = ConfidenceMetricsCalculator(
            drift_weight=0.5,
            results_dir=self.results_dir
        )
        
        # Initialize the integrated MoE enhanced validation
        self.moe_validation = MoEValidationEnhancer(
            results_dir=self.results_dir
        )
    
    def generate_data(self, n_samples=500, drift_point=250, n_features=5):
        """Generate synthetic data with concept drift for demonstration."""
        logger.info(f"Generating {n_samples} samples with drift at sample {drift_point}")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target variable (pre-drift)
        weights_pre = np.random.uniform(-1, 1, n_features)
        y_pre = np.dot(X[:drift_point], weights_pre) + np.random.normal(0, 0.5, drift_point)
        
        # Generate target variable (post-drift) with different feature importance
        weights_post = np.random.uniform(-1, 1, n_features)
        # Make one feature much more important after drift (to simulate concept drift)
        weights_post[0] = weights_post[0] * 3.0
        y_post = np.dot(X[drift_point:], weights_post) + np.random.normal(0, 0.5, n_samples - drift_point)
        
        # Combine pre and post drift data
        y = np.concatenate([y_pre, y_post])
        
        # Add time-based features
        day_of_month = np.ones(n_samples)
        hour = np.ones(n_samples) * 12
        X = np.column_stack([X, day_of_month, hour])
        
        # Save the generated data
        feature_cols = [f"feature_{i}" for i in range(n_features)] + ['day_of_month', 'hour']
        data_df = pd.DataFrame(X, columns=feature_cols)
        data_df['target'] = y
        data_df['is_drift'] = 0
        data_df.loc[drift_point:, 'is_drift'] = 1
        data_df.to_csv(os.path.join(self.data_dir, f"synthetic_data_{self.timestamp}.csv"), index=False)
        
        return X, y, drift_point
    
    def train_moe_model(self, X, y):
        """Train the MoE model with multiple experts."""
        logger.info("Training MoE model with multiple experts")
        
        # Create different experts focused on different feature subsets
        experts = []
        
        # Expert 1: Time-based features only
        time_expert = RandomForestRegressor(n_estimators=10, random_state=42)
        # Use only day_of_month and hour features (last two columns)
        time_features = X[:, -2:]
        time_expert.fit(time_features, y)
        experts.append(time_expert)
        logger.info("Trained Expert 0 (time_based)")
        
        # Expert 2: First subset of numeric features
        features_0_2_expert = RandomForestRegressor(n_estimators=10, random_state=42)
        # Use features 0 and 2
        subset1 = np.column_stack([X[:, 0], X[:, 2]])
        features_0_2_expert.fit(subset1, y)
        experts.append(features_0_2_expert)
        logger.info("Trained Expert 1 (feature_0_2)")
        
        # Expert 3: Second subset of numeric features
        features_1_3_expert = RandomForestRegressor(n_estimators=10, random_state=42)
        # Use features 1 and 3
        subset2 = np.column_stack([X[:, 1], X[:, 3]])
        features_1_3_expert.fit(subset2, y)
        experts.append(features_1_3_expert)
        logger.info("Trained Expert 2 (feature_1_3)")
        
        # Expert 4: All features
        all_features_expert = RandomForestRegressor(n_estimators=10, random_state=42)
        all_features_expert.fit(X, y)
        experts.append(all_features_expert)
        logger.info("Trained Expert 3 (all_features)")
        
        return experts
    
    def run_demo(self):
        """Run the full enhancement demonstration."""
        logger.info("Starting MoE Enhancement Demonstration")
        
        # Step 1: Generate data with concept drift
        X, y, drift_idx = self.generate_data(n_samples=500, drift_point=250)
        n_samples = len(y)
        
        # Step 2: Train the initial MoE model
        experts = self.train_moe_model(X[:drift_idx], y[:drift_idx])
        
        # Step 3: Simulate streaming data and apply enhancements
        logger.info("Simulating streaming data with all enhancements enabled")
        
        # Track predictions, confidence, and notifications
        predictions = []
        confidences = []
        notifications = []
        experts_retrained = set()
        
        # Setup continuous explainer with the initial experts
        self.continuous_explainer.start_monitoring(
            model=experts[3],  # Use the all-features expert as the main model to monitor
            data_source=X[:drift_idx],
            feature_names=[f'feature_{i}' for i in range(X.shape[1])]
        )
        
        # Process each data point to simulate streaming
        for i in range(drift_idx, n_samples):
            # Current data point
            x_i = X[i:i+1]
            y_i = y[i:i+1]
            
            # Get experts' predictions with proper feature slicing
            # We need to handle both original and retrained experts
            expert_preds = []
            
            # Define the feature indices used by each expert
            expert_features = [
                [-2, -1],              # Expert 0: Time-based features (last 2 columns)
                [0, 2],                # Expert 1: Features 0 and 2
                [1, 3],                # Expert 2: Features 1 and 3
                list(range(x_i.shape[1]))  # Expert 3: All features
            ]
            
            for idx, expert in enumerate(experts):
                try:
                    # First try with the original expert's feature subset
                    if idx == 0:  # Time-based expert
                        pred = expert.predict(x_i[:, -2:])[0]
                    elif idx == 1:  # Features 0, 2
                        pred = expert.predict(np.column_stack([x_i[:, 0], x_i[:, 2]]))[0]
                    elif idx == 2:  # Features 1, 3
                        pred = expert.predict(np.column_stack([x_i[:, 1], x_i[:, 3]]))[0]
                    else:  # All features expert
                        pred = expert.predict(x_i)[0]
                    expert_preds.append(pred)
                except ValueError as e:
                    if 'features as input' in str(e):
                        # If the expert was retrained, it may now expect all features
                        # This is a simplification for the demo
                        logger.warning(f"Expert {idx} was retrained and expects different features. Using all features.")
                        pred = expert.predict(x_i)[0]
                        expert_preds.append(pred)
                    else:
                        # Re-raise if it's not a feature dimension issue
                        raise
            
            # Simple equal weighting for this demo
            weights = np.ones(len(experts)) / len(experts)
            ensemble_pred = np.sum(weights * np.array(expert_preds))
            predictions.append(ensemble_pred)
            
            # Check for drift
            if i > drift_idx + 10:  # Allow some data to accumulate
                # Calculate drift using a simple window
                pre_drift_data = X[drift_idx-20:drift_idx]
                post_drift_window = X[i-20:i]
                
                # Create drift data dictionary
                drift_data = {
                    'drift_score': 0.7,  # Simulated high drift score after drift point
                    'model': experts[3],  # Using the all-features expert
                    'data': post_drift_window,
                    'feature_importance': {
                        f'feature_{j}': np.random.uniform(-0.5, 0.5) for j in range(5)
                    }
                }
                
                # Calculate expert-specific drift impacts
                expert_impacts = {
                    f'Expert_{j}': np.random.uniform(0.2, 0.8) for j in range(len(experts))
                }
                
                # Generate notification with the drift data
                notification = self.drift_notifier.generate_notification(drift_data, expert_impacts)
                
                if notification:  # If notification threshold was exceeded
                    notifications.append((i, notification))
                    logger.info(f"Drift detected at sample {i}: {notification['message']}")
                    
                    # Visualize the notification
                    viz_path = self.drift_notifier.visualize_notification(notification)
                    logger.info(f"Visualization saved to: {viz_path}")
                    
                    # Create a selection map for experts to retrain
                    # Here we'll retrain experts with high drift impact values
                    experts_to_retrain = {}
                    for idx, impact in enumerate(expert_impacts.values()):
                        if impact > 0.5:  # Only retrain if impact is high enough
                            experts_to_retrain[idx] = True
                    
                    if experts_to_retrain:
                        # Create a simple MoE model mock for retraining
                        class MoEModelMock:
                            def __init__(self, experts_dict):
                                self.experts = experts_dict
                                
                            def get_expert(self, name):
                                return self.experts.get(name)
                                
                            def replace_expert(self, name, expert):
                                self.experts[name] = expert
                                return True
                        
                        # Create a dictionary of experts with names as keys
                        experts_dict = {f'Expert_{idx}': experts[idx] for idx in experts_to_retrain.keys()}
                        moe_model = MoEModelMock(experts_dict)
                        
                        # Convert numeric indices to string expert names for the API
                        experts_to_retrain_names = [f'Expert_{idx}' for idx in experts_to_retrain.keys()]
                        
                        # Create training data in the expected format
                        new_data = {'X': X[i-30:i], 'y': y[i-30:i]}
                        
                        # Retrain selected experts with the proper API
                        moe_model, retraining_metrics = self.expert_retrainer.retrain_selected_experts(
                            moe_model=moe_model,
                            experts_to_retrain=experts_to_retrain_names,
                            new_data=new_data
                        )
                        
                        # Update the experts list with retrained experts
                        if retraining_metrics.get('retraining_performed', False):
                            for expert_name in experts_to_retrain_names:
                                # Extract index from expert name
                                idx = int(expert_name.split('_')[1])
                                # Update the expert in our list
                                experts[idx] = moe_model.experts[expert_name]
                                experts_retrained.add(idx)
                            
                            logger.info(f"Retrained experts: {experts_to_retrain_names}")
            
            # Update continuous explainability
            if i % 50 == 0:  # Update less frequently for demo
                # We don't need to explicitly call generate_explanations since it's handled
                # internally by the monitoring thread. Log the action instead:
                logger.info(f"Continuous explainability running at sample {i}")
                # If we need explanations at this point, we could access them from the pipeline
                # through its history, but we'll focus on the monitoring for this demo
            
            # Calculate confidence score that accounts for drift
            # First, simulate prediction probabilities for confidence calculation
            # (assuming binary classification with 2 classes)
            # Create mock prediction probabilities for the demo
            prediction_probs = np.array([[0.3, 0.7]]) if i > drift_idx else np.array([[0.15, 0.85]])
            
            # Calculate expert impacts based on drift state
            expert_impacts = {
                f'expert_{j}': 0.8 if i > drift_idx and j == 0 else 0.2
                for j in range(len(experts))
            }
            
            # Calculate expert weights (equal weighting for demo)
            expert_weights = {f'expert_{j}': 1.0/len(experts) for j in range(len(experts))}
            
            # Calculate confidence
            confidence = self.confidence_calculator.calculate_confidence(
                prediction_probabilities=prediction_probs,
                drift_score=0.7 if i > drift_idx else 0.1,  # Simulated drift score
                expert_impacts=expert_impacts,
                expert_weights=expert_weights
            )[0]  # Take first confidence score since we're processing one sample
            confidences.append(confidence)
        
        # Save results
        results_df = pd.DataFrame({
            'true_value': y[drift_idx:n_samples],
            'prediction': predictions,
            'confidence': confidences
        })
        results_df.to_csv(os.path.join(self.data_dir, f"results_{self.timestamp}.csv"), index=False)
        
        # Plot results
        self.plot_results(y[drift_idx:n_samples], predictions, confidences, notifications)
        
        # Generate summary report
        self.generate_report(experts_retrained, notifications, confidences)
        
        # Demonstrate the integrated MoE validation enhancer
        logger.info("Demonstrating the MoE validation enhancer integration...")
        
        # Create a synthetic drift detection result to showcase the enhancer
        drift_detection_data = {
            'drift_score': 0.7,  # High drift score
            'model': experts[3],  # Using the all-features expert
            'data': X[drift_idx:drift_idx+50],  # Data after drift
            'before_drift': {'X': X[:drift_idx], 'y': y[:drift_idx]},
            'after_drift': {'X': X[drift_idx:], 'y': y[drift_idx:]},
            'feature_importance': {
                f'feature_{j}': np.random.uniform(-0.5, 0.5) for j in range(5)
            },
            'drift_metrics': {
                'jensen_shannon_distance': 0.32,
                'kullback_leibler_divergence': 0.45,
                'wasserstein_distance': 0.28
            }
        }
        
        # Process the drift results with the MoE validation enhancer
        enhancer_results = self.moe_validation.process_drift_results(
            drift_data=drift_detection_data,
            notify=True,
            notify_threshold=0.3
        )
        
        logger.info(f"MoE enhancer generated notification: {enhancer_results.get('notification', {}).get('message', 'No notification')}")
        if 'notification' in enhancer_results and 'visualization' in enhancer_results['notification']:
            logger.info(f"Visualization saved to: {enhancer_results['notification']['visualization']}")
        
        # Add enhancer results to the demo output
        logger.info(f"Demo completed. Results saved to {self.results_dir}")
        return {
            'success': True,
            'experts_retrained': experts_retrained,
            'notifications': len(notifications),
            'average_confidence': np.mean(confidences),
            'moe_enhancer_results': enhancer_results,
            'results_dir': self.results_dir
        }
    
    def plot_results(self, y_true, predictions, confidences, notifications):
        """Plot the results of the demonstration."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: True values vs Predictions
        plt.subplot(3, 1, 1)
        plt.plot(y_true, label='True Values', color='blue')
        plt.plot(predictions, label='Predictions', color='red', linestyle='--')
        
        # Mark drift notifications
        for notification in notifications:
            idx, _ = notification
            plt.axvline(x=idx-250, color='green', linestyle='-', alpha=0.5)
        
        plt.title('True Values vs Predictions with Drift Notifications')
        plt.legend()
        
        # Plot 2: Confidence over time
        plt.subplot(3, 1, 2)
        plt.plot(confidences, color='purple')
        plt.title('Confidence Scores Over Time')
        plt.ylim(0, 1)
        
        # Plot 3: Error over time
        plt.subplot(3, 1, 3)
        errors = np.abs(np.array(y_true) - np.array(predictions))
        plt.plot(errors, color='orange')
        plt.title('Absolute Error Over Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"results_summary_{self.timestamp}.png"))
        plt.close()
    
    def generate_report(self, experts_retrained, notifications, confidences):
        """Generate a summary report of the enhancement demonstration."""
        with open(os.path.join(self.results_dir, f"enhancement_report_{self.timestamp}.txt"), 'w') as f:
            f.write("=== MoE Enhancement Demonstration Report ===\n\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write("=== Drift Notifications ===\n")
            f.write(f"Number of notifications: {len(notifications)}\n")
            for idx, notification in notifications:
                f.write(f"- At sample {idx}: {notification.get('message', '')}\n")
            
            f.write("\n=== Expert Retraining ===\n")
            f.write(f"Number of experts retrained: {len(experts_retrained)}\n")
            f.write(f"Experts retrained: {experts_retrained}\n")
            
            f.write("\n=== Confidence Metrics ===\n")
            f.write(f"Average confidence: {np.mean(confidences):.4f}\n")
            f.write(f"Min confidence: {np.min(confidences):.4f}\n")
            f.write(f"Max confidence: {np.max(confidences):.4f}\n")
            
            f.write("\n=== Continuous Explainability ===\n")
            explainer_files = [f for f in os.listdir(self.plots_dir) if "explainer" in f]
            f.write(f"Generated {len(explainer_files)} explainability visualizations\n")
            for file in explainer_files:
                f.write(f"- {file}\n")

def main():
    """Main function to run the demo."""
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/moe_enhancements_demo/{timestamp}"
    
    # Run the demo
    demo = MoEEnhancementDemo(results_dir=results_dir)
    results = demo.run_demo()
    
    # Display summary
    logger.info("=== Enhancement Demo Summary ===")
    logger.info(f"Experts retrained: {len(results['experts_retrained'])}")
    logger.info(f"Number of drift notifications: {results['notifications']}")
    logger.info(f"Average confidence score: {results['average_confidence']:.4f}")
    logger.info(f"Results saved to: {results['results_dir']}")

if __name__ == "__main__":
    main()

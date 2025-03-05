"""
Command Line Interface for the Meta-Optimizer package.
This module serves as the main entry point when running the package from the command line.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Meta-Optimizer Framework: A comprehensive framework for optimization, '
                    'meta-learning, explainability, and drift detection'
    )
    
    # Core functionality
    parser.add_argument('--optimize', action='store_true',
                        help='Run optimization with multiple optimizers on benchmark functions')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate a trained model on test data')
    parser.add_argument('--meta', action='store_true',
                        help='Run meta-learning to find the best optimizer for different problems')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary of results for any operation')
    
    # Drift detection
    parser.add_argument('--drift', action='store_true',
                        help='Run drift detection on synthetic data')
    parser.add_argument('--drift-window', type=int, default=50,
                        help='Window size for drift detection')
    parser.add_argument('--drift-threshold', type=float, default=0.5,
                        help='Threshold for drift detection')
    parser.add_argument('--drift-significance', type=float, default=0.05,
                        help='Significance level for drift detection')
    parser.add_argument('--run-meta-learner-with-drift', action='store_true',
                        help='Run meta-learner with drift detection')
    parser.add_argument('--explain-drift', action='store_true',
                        help='Explain drift when detected')
    
    # Explainability
    parser.add_argument('--explain', action='store_true',
                        help='Run explainability analysis on a trained model')
    parser.add_argument('--explainer', type=str, default='shap',
                       choices=['shap', 'lime', 'feature_importance', 'optimizer'],
                       help='Explainer type to use')
    parser.add_argument('--explain-plots', action='store_true',
                        help='Generate and save explainability plots')
    parser.add_argument('--explain-plot-types', type=str, nargs='+',
                        help='Specific plot types to generate (e.g., summary waterfall force dependence)')
    parser.add_argument('--explain-samples', type=int, default=50,
                        help='Number of samples to use for explainability')
    
    # Meta-learning parameters
    parser.add_argument('--method', type=str, default='bayesian',
                        help='Method for meta-learner')
    parser.add_argument('--surrogate', type=str, default=None,
                        help='Surrogate model for meta-learner')
    parser.add_argument('--selection', type=str, default=None,
                        help='Selection strategy for meta-learner')
    parser.add_argument('--exploration', type=float, default=0.2,
                        help='Exploration factor for meta-learner')
    parser.add_argument('--history', type=float, default=0.7,
                        help='History weight for meta-learner')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    
    return parser.parse_args()

def generate_synthetic_data_with_drift(n_samples=1000, n_features=10, 
                                      drift_points=None, noise_level=0.1,
                                      random_state=42):
    """
    Generate synthetic regression data with concept drift at specified points.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features to generate
    drift_points : list of int
        List of indices where drift should occur
    noise_level : float
        Level of noise to add to the target variable
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X : numpy array
        Features
    y : numpy array
        Target values
    coef : numpy array
        Coefficients used to generate the data
    """
    if drift_points is None:
        drift_points = []
    
    # Set random seed
    np.random.seed(random_state)
    
    # Initialize data arrays
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Generate initial feature values
    X = np.random.randn(n_samples, n_features)
    
    # Sort drift points and add first point at 0
    all_points = [0] + sorted(drift_points) + [n_samples]
    
    # Track coefficients used
    all_coefs = []
    
    # For each segment between drift points
    for i in range(len(all_points) - 1):
        start_idx = all_points[i]
        end_idx = all_points[i+1]
        
        # Generate new coefficients after each drift point
        if i == 0:
            coef = np.random.randn(n_features)
        else:
            # Create drift by modifying coefficients
            prev_coef = all_coefs[-1]
            coef = prev_coef + np.random.randn(n_features) * 2  # Significant change
            
            # Introduce new relationships after drift
            shift_mask = np.random.choice([0, 1], size=n_features, p=[0.7, 0.3])
            coef = coef * (1 - shift_mask) + np.random.randn(n_features) * 3 * shift_mask
        
        all_coefs.append(coef)
        
        # Calculate target values for this segment
        segment_y = np.dot(X[start_idx:end_idx], coef)
        segment_y += np.random.randn(end_idx - start_idx) * noise_level
        
        # Store values
        y[start_idx:end_idx] = segment_y
    
    return X, y, np.array(all_coefs)

def main():
    """
    Main entry point for the CLI when used as meta-optimizer command.
    This provides a minimal working example of the package functionality.
    """
    args = parse_args()
    print("Meta-Optimizer CLI")
    print("-----------------")
    
    if not any([args.optimize, args.evaluate, args.meta, args.drift, 
                args.explain, args.run_meta_learner_with_drift, args.explain_drift]):
        print("No action specified. Use --help to see available options.")
        return
    
    if args.optimize:
        print("Running optimization...")
        # Import here to avoid circular imports
        from meta_optimizer.optimizers import OptimizerFactory
        from meta_optimizer.benchmark.test_functions import create_test_suite
        
        # Create a simple optimization example
        dim = 10
        bounds = [(-5, 5)] * dim
        factory = OptimizerFactory()
        optimizers = factory.create_all(dim, bounds)
        
        test_suite = create_test_suite()
        sphere = test_suite['sphere'](dim, bounds)
        
        # Run optimization
        best_values = {}
        for name, opt in optimizers.items():
            opt.optimize(sphere.evaluate, max_evals=1000)
            best_values[name] = opt.best_score  # Changed from best_value to best_score
        
        # Print results
        print("\nOptimization Results:")
        for name, value in best_values.items():
            print(f"{name}: {value:.6f}")
    
    if args.evaluate:
        print("Running evaluation...")
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Generate synthetic data
        X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nEvaluation Results:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
    
    if args.meta:
        print("Running meta-learning...")
        # This would be a simplified version
        print("Meta-learning would analyze different optimizers and select the best one for your problem.")
    
    if args.drift:
        print("Running drift detection...")
        from meta_optimizer.drift_detection.drift_detector import DriftDetector
        
        # Parameters
        n_samples = 500
        n_features = 10
        
        # Import numpy if not already imported
        import numpy as np
        
        # Generate two random datasets with same distribution
        X1 = np.random.normal(0, 1, (n_samples, n_features))
        X2 = np.random.normal(0, 1, (n_samples, n_features))  # No drift for simple example
        
        # Set up detector
        detector = DriftDetector(
            window_size=args.drift_window, 
            drift_threshold=args.drift_threshold,
            significance_level=args.drift_significance,
            method='ks'
        )
        
        # Check for drift
        is_drift, score, p_value = detector.detect_drift(X1, X2)
        
        print(f"\nDrift Detection Results:")
        print(f"Drift detected: {is_drift}")
        print(f"Drift score: {score:.4f}")
        print(f"P-value: {p_value:.4f}")
    
    if args.explain:
        print("Running explainability analysis...")
        print("This would generate explanations for model predictions.")
    
    if args.run_meta_learner_with_drift:
        print("Running meta-learner with drift detection...")
        
        # Import the main function that implements this
        import os
        import sys
        # Get the absolute path to the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.insert(0, project_root)  # Add to beginning of path
        
        try:
            # Explicitly import from the correct main.py file
            import main
            
            # Run meta-learner with drift detection
            results = main.run_meta_learner_with_drift_detection(args)
            
            # Print summary of results if requested
            if args.summary and results:
                print("\nMeta-Learner with Drift Detection Results:")
                print(f"True drift points: {results['true_drift_points']}")
                print(f"Detected drift points: {results['detected_drift_points']}")
                print(f"Number of batches processed: {len(results['mse_history'])}")
        except Exception as e:
            print(f"Error running meta-learner with drift detection: {e}")
            print("Using fallback implementation instead...")
            
            # Fallback implementation
            from meta_optimizer.drift_detection.drift_detector import DriftDetector
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Generate synthetic data with drift
            n_samples = 1000
            n_features = 10
            drift_points = [300, 600]  # Drift at these points
            
            X, y, _ = generate_synthetic_data_with_drift(
                n_samples=n_samples, 
                n_features=n_features, 
                drift_points=drift_points,
                noise_level=0.1
            )
            
            # Set up the drift detector
            detector = DriftDetector(
                window_size=args.drift_window,
                drift_threshold=args.drift_threshold,
                significance_level=args.drift_significance,
                method='ks'
            )
            
            # Train a simple model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X[:200], y[:200])  # Initial training
            
            # Setup metrics tracking
            mse_history = []
            drift_detected_points = []
            
            # Process data in batches
            batch_size = 50
            for i in range(200, len(X) - batch_size, batch_size):
                # Get current batch
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Check for drift
                is_drift, drift_score, p_value = detector.detect_drift(X_batch)
                
                if is_drift:
                    print(f"Drift detected at point {i}, retraining model...")
                    drift_detected_points.append(i)
                    # Retrain model on recent data
                    model.fit(X[i-100:i+batch_size], y[i-100:i+batch_size])
                
                # Make predictions and evaluate
                y_pred = model.predict(X_batch)
                mse = mean_squared_error(y_batch, y_pred)
                mse_history.append(mse)
            
            # Visualize if requested
            if args.visualize:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot the data and drift points
                for feature in range(min(3, n_features)):  # Show only first 3 features
                    ax1.plot(range(n_samples), X[:, feature], alpha=0.5, label=f'Feature {feature}')
                
                # Mark true drift points
                for point in drift_points:
                    ax1.axvline(x=point, color='r', linestyle='--', alpha=0.7, 
                              label='True drift' if point == drift_points[0] else None)
                
                # Mark detected drift points
                for point in drift_detected_points:
                    ax1.axvline(x=point, color='g', linestyle='-', alpha=0.7,
                              label='Detected drift' if point == drift_detected_points[0] else None)
                
                ax1.set_title('Data with Drift Points')
                ax1.set_ylabel('Feature Value')
                ax1.legend()
                
                # Plot MSE history
                ax2.plot(range(200, len(X) - batch_size, batch_size)[:len(mse_history)], mse_history, label='MSE')
                ax2.set_title('Model Performance with Drift Adaptation')
                ax2.set_xlabel('Sample Index')
                ax2.set_ylabel('Mean Squared Error')
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig('meta_learner_drift_adaptation.png')
                print("Figure saved to 'meta_learner_drift_adaptation.png'")
                plt.show()
            
            # Print summary if requested
            if args.summary:
                print("\nMeta-Learner with Drift Detection Results:")
                print(f"True drift points: {drift_points}")
                print(f"Detected drift points: {drift_detected_points}")
                print(f"Number of batches processed: {len(mse_history)}")
                print(f"Final MSE: {mse_history[-1] if mse_history else 'N/A'}")
    
    if args.explain_drift:
        print("Running drift explanation...")
        from meta_optimizer.drift_detection.drift_detector import DriftDetector
        
        # Generate synthetic data with drift
        n_samples = 1000
        n_features = 5
        drift_points = [300, 600]  # Drift at these points
        
        X, y, drift_types = generate_synthetic_data_with_drift(
            n_samples=n_samples, 
            n_features=n_features, 
            drift_points=drift_points,
            noise_level=0.01  # Reduce noise to make drift more detectable
        )
        
        # Set up the drift detector
        detector = DriftDetector(
            window_size=50,
            drift_threshold=0.5,
            significance_level=0.05,
            method='ks'  # Explicitly set KS test as the method
        )
        
        # Monitor for drift
        drift_detected_points = []
        drift_scores = []
        p_values = []
        feature_contributions = []
        
        # First pass to get reference window
        detector.detect_drift(X[:50])
        
        # Second pass to detect drift
        for i in range(100, len(X), 25):
            X_window = X[i-50:i]
            
            is_drift, drift_score, p_value = detector.detect_drift(X_window)
            drift_scores.append(drift_score)
            p_values.append(p_value)
            
            if is_drift:
                drift_detected_points.append(i)
                
                # Calculate feature contributions to drift
                before_drift = X[i-50:i-25]
                after_drift = X[i-25:i]
                
                # Calculate mean shift for each feature
                mean_before = np.mean(before_drift, axis=0)
                mean_after = np.mean(after_drift, axis=0)
                mean_shift = np.abs(mean_after - mean_before)
                
                # Calculate variance shift for each feature
                var_before = np.var(before_drift, axis=0)
                var_after = np.var(after_drift, axis=0)
                var_shift = np.abs(var_after - var_before)
                
                # Calculate distribution shift using KS test
                ks_pvalues = []
                for f in range(n_features):
                    _, ks_pvalue = stats.ks_2samp(before_drift[:, f], after_drift[:, f])
                    ks_pvalues.append(1 - ks_pvalue)  # Convert p-value to a "contribution" score
                
                # Combine the metrics (mean shift, variance shift, KS test)
                if np.max(mean_shift) > 0 and np.max(var_shift) > 0:
                    feature_contrib = (mean_shift / np.max(mean_shift) + 
                                      var_shift / np.max(var_shift) + 
                                      np.array(ks_pvalues)) / 3
                else:
                    feature_contrib = np.array(ks_pvalues)
                
                feature_contributions.append((i, feature_contrib))
        
        # Print drift explanation
        print("\nDrift Explanation:")
        print("-----------------")
        print(f"True drift points: {drift_points}")
        print(f"Detected drift points: {drift_detected_points}")
        
        if feature_contributions:
            print("\nFeature Contributions to Drift:")
            for i, (point, contributions) in enumerate(feature_contributions):
                print(f"\nDrift Point {i+1} (at index {point}):")
                
                # Sort features by contribution
                feature_indices = np.argsort(contributions)[::-1]
                
                for rank, idx in enumerate(feature_indices):
                    print(f"  Rank {rank+1}: Feature {idx} (contribution: {contributions[idx]:.4f})")
                
                # Determine the type of drift if available
                if drift_types and point in drift_types:
                    print(f"  Drift type: {drift_types[point]}")
        
        # Visualization part
        if args.visualize:
            import matplotlib.pyplot as plt
            
            fig, axs = plt.subplots(3, 1, figsize=(12, 15))
            
            # Plot 1: Original data with drift points marked
            for feature in range(min(3, n_features)):  # Show at most 3 features for clarity
                axs[0].plot(range(n_samples), X[:, feature], label=f'Feature {feature}')
            
            # Mark true drift points
            for drift_point in drift_points:
                axs[0].axvline(x=drift_point, color='r', linestyle='--', alpha=0.5)
            
            # Mark detected drift points
            for drift_point in drift_detected_points:
                axs[0].axvline(x=drift_point, color='g', linestyle='-', alpha=0.5)
            
            axs[0].set_title('Feature Values with Drift Points')
            axs[0].set_xlabel('Sample Index')
            axs[0].set_ylabel('Feature Value')
            axs[0].legend()
            
            # Plot 2: Drift scores and p-values
            x_indices = list(range(100, len(X), 25))[:len(drift_scores)]
            
            axs[1].plot(x_indices, drift_scores, label='Drift score')
            axs[1].plot(x_indices, p_values, label='P-value')
            axs[1].axhline(y=0.05, color='r', linestyle='--', label='Significance level')
            axs[1].set_title('Drift Scores and P-values')
            axs[1].set_xlabel('Sample Index')
            axs[1].set_ylabel('Score / P-value')
            axs[1].legend()
            
            # Plot 3: Feature contributions to drift
            if drift_detected_points and feature_contributions:
                # Show for the first drift point as example
                drift_point, contrib = feature_contributions[0]
                
                # Sort features by contribution
                sorted_indices = np.argsort(contrib)[::-1]
                
                axs[2].bar(range(n_features), contrib[sorted_indices])
                axs[2].set_title(f'Feature Contributions to Drift at point {drift_point}')
                axs[2].set_xlabel('Feature Index')
                axs[2].set_ylabel('Contribution')
                axs[2].set_xticks(range(n_features))
                axs[2].set_xticklabels([f'Feature {i}' for i in sorted_indices])
            else:
                axs[2].text(0.5, 0.5, 'No drift detected or not enough feature contribution data', 
                         horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)
            
            plt.tight_layout()
            plt.savefig('drift_detection_results.png')
            print("Figure saved to 'drift_detection_results.png'")
            plt.show()
    
    print("\nAll operations completed successfully.")


if __name__ == "__main__":
    main()

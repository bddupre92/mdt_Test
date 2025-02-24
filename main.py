"""
main.py
-------
Orchestrates data generation, preprocessing, model training,
drift detection, and optimization. Demonstrates a simplified 
workflow for demonstration.
"""

import numpy as np
from data.generate_synthetic import generate_synthetic_data
from data.preprocessing import preprocess_data
from data.domain_knowledge import add_migraine_features

from optimizers.aco import AntColonyOptimizer
from optimizers.gwo import GreyWolfOptimizer
from optimizers.es import EvolutionStrategy
from optimizers.de import DifferentialEvolutionOptimizer

from meta.meta_learner import MetaLearner
from models.sklearn_model import SklearnModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from drift_detection.statistical import ks_drift_test
from drift_detection.performance_monitor import DDM

def evaluate_model(model, X, y):
    """Comprehensive model evaluation"""
    preds = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0),
        'f1': f1_score(y, preds, zero_division=0)
    }
    return metrics

def cross_validate_model(model, X, y, n_folds=5):
    """Perform cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model.sk_model, X, y, cv=kf, scoring='f1')
    return scores.mean(), scores.std()

def main():
    print("Starting migraine prediction pipeline...")
    
    # 1. Generate data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(num_days=200)  # Increased dataset size
    print(f"Generated data shape: {df.shape}")
    
    # 2. Preprocess
    print("\n2. Preprocessing data...")
    df_clean = preprocess_data(df, strategy_numeric='mean', scale_method='minmax',
                               exclude_cols=['migraine_occurred','severity'])
    df_feat = add_migraine_features(df_clean)
    print(f"Data shape after preprocessing: {df_feat.shape}")
    
    # Prepare X,y
    features = [c for c in df_feat.columns if c not in ['migraine_occurred','severity']]
    df_feat = df_feat.dropna(subset=['migraine_occurred'])
    X = df_feat[features].values
    y = df_feat['migraine_occurred'].values.astype(int)
    print(f"Features used: {features}")
    
    # Split into train/test
    split_idx = int(0.7*len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # 3. Use optimizers with expanded parameter space
    print("\n3. Setting up optimization...")
    aco = AntColonyOptimizer()
    gwo = GreyWolfOptimizer(dim=4, bounds=(0,1))  # 4 parameters to optimize
    es = EvolutionStrategy(dim=4)
    de = DifferentialEvolutionOptimizer(bounds=[(0,1)]*4)
    
    ml = MetaLearner(method='bayesian')
    ml.set_algorithms([aco, gwo, es, de])
    chosen_opt = ml.select_algorithm()
    print(f"Selected optimizer: {chosen_opt.__class__.__name__}")
    
    # Define objective function with cross-validation
    def objective_func(x):
        # Map x[0-3] to actual hyperparameters
        n_trees = int(x[0] * 190 + 10)  # 10-200 trees
        max_depth = int(x[1] * 30 + 3)  # 3-33 max_depth
        min_samples_split = int(x[2] * 18 + 2)  # 2-20 min_samples_split
        max_features = min(max(x[3], 0.1), 1.0)  # Ensure between 0.1 and 1.0
        
        # Create and evaluate model with cross-validation
        rf = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42
        )
        model = SklearnModel(rf)
        try:
            cv_score, cv_std = cross_validate_model(model, X_train, y_train)
            return 1.0 - cv_score  # minimize error
        except Exception as e:
            print(f"Warning: Cross-validation failed with error: {str(e)}")
            return 1.0  # Return worst possible score on failure
    
    # Run optimization
    print("\nOptimizing model parameters...")
    best_solution, best_score = chosen_opt.optimize(objective_func)
    print(f"Optimization complete => best CV score: {1.0 - best_score:.3f}")
    
    # Use optimized parameters
    n_trees = int(best_solution[0] * 190 + 10)
    max_depth = int(best_solution[1] * 30 + 3)
    min_samples_split = int(best_solution[2] * 18 + 2)
    max_features = min(max(best_solution[3], 0.1), 1.0)
    
    print(f"\nBest parameters found:")
    print(f"- n_estimators: {n_trees}")
    print(f"- max_depth: {max_depth}")
    print(f"- min_samples_split: {min_samples_split}")
    print(f"- max_features: {max_features:.2f}")
    
    # 4. Train final model with early stopping
    print("\n4. Training final model...")
    final_rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        oob_score=True  # Enable out-of-bag score for early stopping
    )
    model = SklearnModel(final_rf)
    model.train(X_train, y_train)
    
    # Evaluate on both sets
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)
    
    print("\nModel Performance:")
    print("Training Set:")
    for metric, value in train_metrics.items():
        print(f"- {metric}: {value:.3f}")
    print("\nTest Set:")
    for metric, value in test_metrics.items():
        print(f"- {metric}: {value:.3f}")
    
    if hasattr(final_rf, 'oob_score_'):
        print(f"\nOut-of-bag score: {final_rf.oob_score_:.3f}")
    
    # 5. Drift detection with multiple features
    print("\n5. Checking for drift...")
    drift_features = ['weather_pressure', 'sleep_hours', 'stress_level']
    for feature in drift_features:
        old_data = df_feat.iloc[:split_idx]
        new_data = df_feat.iloc[split_idx:]
        p_value = ks_drift_test(old_data, new_data, feature)
        print(f"KS drift test p-value for {feature}: {p_value:.5f}")
    
    # 6. Performance-based drift detection with window
    print("\n6. Running performance-based drift detection...")
    ddm = DDM()
    drift_detected = False
    window_size = 5
    for i in range(0, len(X_test), window_size):
        window = slice(i, min(i + window_size, len(X_test)))
        preds = model.predict(X_test[window])
        correct = (preds == y_test[window]).mean()
        if ddm.update(correct):
            drift_detected = True
            print(f"DDM drift triggered at test window #{i//window_size}")
            break
    
    if drift_detected:
        print("Drift detected => retraining or adaptation would be needed.")
    else:
        print("No drift flagged by DDM on test set.")
    
    print("\nPipeline complete!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

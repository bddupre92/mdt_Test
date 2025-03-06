#!/usr/bin/env python
"""
Universal Data Adapter Example

This script demonstrates how to use the universal data adapter to process
any migraine dataset, regardless of its schema. It covers:
1. Loading data from various formats
2. Automatic schema detection
3. Feature selection using both standard and meta-optimization methods
4. Training and evaluating a migraine prediction model
5. Making predictions on new data

The script can work with real datasets or generate synthetic data for testing.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import logging
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add parent directory to path so we can import from the project
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import project modules
from migraine_prediction_project.src.migraine_model.universal_data_adapter import UniversalDataAdapter
from migraine_prediction_project.src.migraine_model.new_data_migraine_predictor import MigrainePredictorV2
from migraine_prediction_project.src.migraine_model.data_handler import DataHandler

# Try to import meta-feature selector
try:
    from migraine_prediction_project.src.meta.meta_feature_selector import MetaFeatureSelector
    META_FEATURE_SELECTOR_AVAILABLE = True
except ImportError:
    import logging
    logging.warning("Meta-optimizer package not available. Using standard feature selection.")
    META_FEATURE_SELECTOR_AVAILABLE = False

# Import explainability modules if available
try:
    from migraine_prediction_project.src.explainability.explainer_factory import ExplainerFactory
    from migraine_prediction_project.src.explainability.base_explainer import BaseExplainer
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        from explainability.explainer_factory import ExplainerFactory
        from explainability.base_explainer import BaseExplainer
        EXPLAINABILITY_AVAILABLE = True
    except ImportError:
        logging.warning("Explainability components not available. Creating minimal versions.")
        EXPLAINABILITY_AVAILABLE = False
        
        # Create minimal explainability classes if not available
        class BaseExplainer:
            def __init__(self, model, X_train=None, feature_names=None):
                self.model = model
                self.X_train = X_train
                self.feature_names = feature_names
                
            def explain(self, X):
                return {"status": "Not available"}
                
            def get_feature_importance(self):
                if hasattr(self.model, 'feature_importances_'):
                    return dict(zip(self.feature_names, self.model.feature_importances_))
                return {}
                
            def plot(self, plot_type="bar"):
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_importance = self.get_feature_importance()
                if feature_importance:
                    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    features, values = zip(*items[:15])
                    plt.barh(features, values)
                    plt.xlabel('Importance')
                    plt.ylabel('Feature')
                    plt.title('Feature Importance')
                    plt.tight_layout()
                    return fig
                return None
        
        class ExplainerFactory:
            def create_explainer(self, explainer_type, model, X_train=None, feature_names=None):
                return BaseExplainer(model, X_train, feature_names)
                
        def plot_feature_importance(feature_importance, title="Feature Importance"):
            fig, ax = plt.subplots(figsize=(10, 6))
            features = list(feature_importance.keys())
            values = list(feature_importance.values())
            plt.barh(features, values)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(title)
            plt.tight_layout()
            return fig

def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix from sklearn
        class_names: Names for the classes (default: ["No Migraine", "Migraine"])
        figsize: Figure size as (width, height)
    """
    if class_names is None:
        class_names = ["No Migraine", "Migraine"]
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    return plt.gcf()


def plot_roc_curve(y_true, y_prob, figsize=(8, 6)):
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        figsize: Figure size as (width, height)
    """
    # Calculate ROC curve
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal
    
    return plt.gcf()


def create_example_dataset(output_path=None):
    """
    Create a synthetic dataset for the example
    
    Args:
        output_path: Path to save the dataset (optional)
        
    Returns:
        pandas.DataFrame: The generated dataset
    """
    print("Generating synthetic migraine data...")
    
    try:
        # Import synthetic data generation
        try:
            from migraine_prediction_project.examples.clinical_sythetic_data import generate_synthetic_migraine_data as generate_synthetic_data
        except ImportError:
            # Create a simple synthetic data generator if the real one is not available
            def generate_synthetic_data(n_patients=50, n_days=90, **kwargs):
                """
                Generate a simplified synthetic dataset when the real generator is not available.
                """
                import numpy as np
                import pandas as pd
                
                # Total number of observations
                n_observations = n_patients * n_days
                
                # Create a dataframe with basic features
                data = pd.DataFrame({
                    'subject_id': np.repeat(range(1, n_patients + 1), n_days),
                    'observation_date': np.tile(pd.date_range('2023-01-01', periods=n_days), n_patients),
                    'age': np.random.randint(18, 70, n_observations),
                    'gender': np.random.choice(['male', 'female'], n_observations, p=[0.3, 0.7]),
                    'pulse_bpm': np.random.normal(75, 10, n_observations),
                    'body_temp_celsius': np.random.normal(36.8, 0.3, n_observations),
                    'sleep_hours': np.random.normal(7, 1.5, n_observations),
                    'stress_level': np.random.randint(1, 11, n_observations),
                    'atmospheric_pressure_hpa': np.random.normal(1013, 10, n_observations),
                    'humidity': np.random.normal(60, 15, n_observations),
                    'caffeine': np.random.randint(0, 5, n_observations),
                    'alcohol': np.random.randint(0, 3, n_observations),
                    'chocolate': np.random.randint(0, 2, n_observations),
                    'cheese': np.random.randint(0, 2, n_observations),
                    'hormonal_phase': np.random.choice(['none', 'follicular', 'ovulation', 'luteal'], n_observations),
                    'headache_occurred': np.random.choice([0, 1], n_observations, p=[0.85, 0.15]),
                    'data_quality': np.random.choice(['high', 'medium', 'low'], n_observations, p=[0.7, 0.2, 0.1]),
                    'recording_device': np.random.choice(['mobile', 'web', 'device'], n_observations, p=[0.5, 0.3, 0.2])
                })
                
                return data
        
        # Generate synthetic data
        data = generate_synthetic_data(
            n_patients=50,
            n_days=90,
            pct_female=0.6,
            missing_rate=0.15,
            anomaly_rate=0.02
        )
    except Exception as e:
        print(f"Error generating synthetic data: {str(e)}")
        print("Creating simplified synthetic data instead...")
        
        # Create a simplified synthetic dataset
        np.random.seed(42)
        n_patients = 50
        n_days = 90
        
        # Create date range
        start_date = pd.Timestamp('2023-01-01')
        dates = [start_date + pd.Timedelta(days=i) for i in range(n_days)]
        
        # Create patient IDs
        patient_ids = [f"P{i:03d}" for i in range(n_patients)]
        
        # Create data records
        records = []
        for patient_id in patient_ids:
            is_female = np.random.random() < 0.6
            age = np.random.randint(18, 65)
            
            for date in dates:
                # Basic features with some randomization
                heart_rate = np.random.normal(75, 10)
                temperature = np.random.normal(98.6, 0.5)
                sleep_hours = np.random.normal(7, 1.5)
                stress_level = np.random.randint(1, 11)
                pressure = np.random.normal(1013, 20)
                humidity = np.random.uniform(30, 90)
                
                # Diet features
                caffeine = np.random.randint(0, 4)
                alcohol = np.random.randint(0, 3)
                chocolate = np.random.randint(0, 2)
                cheese = np.random.randint(0, 2)
                
                # Menstrual cycle for women
                hormonal_phase = np.nan
                if is_female:
                    cycle_day = date.day % 28
                    if cycle_day < 5:
                        hormonal_phase = 1  # menstruation
                    elif cycle_day < 14:
                        hormonal_phase = 2  # follicular
                    elif cycle_day < 16:
                        hormonal_phase = 3  # ovulation
                    else:
                        hormonal_phase = 4  # luteal
                
                # Target variable (migraine)
                # Higher probability if stress high, sleep low, or certain diet triggers
                p_migraine = 0.05
                p_migraine += 0.01 * stress_level
                p_migraine -= 0.01 * sleep_hours
                p_migraine += 0.03 * caffeine
                p_migraine += 0.05 * alcohol
                p_migraine += 0.02 * chocolate
                p_migraine += 0.02 * cheese
                
                # Weather impact
                if abs(pressure - 1013) > 15:
                    p_migraine += 0.03
                
                # Hormonal impact for women
                if not np.isnan(hormonal_phase) and hormonal_phase in [1, 4]:
                    p_migraine += 0.05
                
                # Cap probability
                p_migraine = min(max(p_migraine, 0.01), 0.3)
                
                # Determine migraine occurrence
                migraine = 1 if np.random.random() < p_migraine else 0
                
                records.append({
                    'patient_id': patient_id,
                    'date': date,
                    'age': age,
                    'gender': 'F' if is_female else 'M',
                    'heart_rate': heart_rate,
                    'temperature': temperature,
                    'sleep_hours': sleep_hours,
                    'stress_level': stress_level,
                    'barometric_pressure': pressure,
                    'humidity': humidity,
                    'caffeine': caffeine,
                    'alcohol': alcohol,
                    'chocolate': chocolate,
                    'cheese': cheese,
                    'hormonal_phase': hormonal_phase,
                    'migraine': migraine
                })
        
        # Create DataFrame
        data = pd.DataFrame(records)
        
        # Add some missing values
        missing_mask = np.random.random(data.shape) < 0.1
        for col in data.columns:
            if col not in ['patient_id', 'date', 'migraine']:
                data.loc[missing_mask[:, data.columns.get_loc(col)], col] = np.nan
        
        # Rename some columns to simulate a different schema
        rename_map = {
            'heart_rate': 'pulse_bpm',
            'temperature': 'body_temp_celsius',
            'barometric_pressure': 'atmospheric_pressure_hpa',
            'migraine': 'headache_occurred',
            'patient_id': 'subject_id',
            'date': 'observation_date'
        }
        
        # Apply renames to columns that exist
        columns_to_rename = {k: v for k, v in rename_map.items() if k in data.columns}
        data = data.rename(columns=columns_to_rename)
        
        # Add some additional columns
        data['data_quality'] = np.random.choice(['good', 'fair', 'poor'], size=len(data))
        data['recording_device'] = np.random.choice(['mobile', 'wearable', 'manual'], size=len(data))
    
    # Save to file if output path is provided
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"Saved synthetic data to {output_path}")
    
    return data


def select_features(data, schema, max_features=15):
    """
    Select most relevant features for model training.
    
    Args:
        data: DataFrame with data
        schema: Schema dictionary from the adapter
        max_features: Maximum number of features to select
    
    Returns:
        List of selected feature names
    """
    # Exclude non-feature columns
    exclude_columns = ['subject_id', 'patient_id', 'observation_date', 'date']
    if schema['target_column']:
        exclude_columns.append(schema['target_column'])
    if schema['date_column']:
        exclude_columns.append(schema['date_column'])
    
    # Get available features
    feature_columns = [col for col in data.columns if col not in exclude_columns and pd.api.types.is_numeric_dtype(data[col])]
    
    # If we have mapped core features, prioritize them
    selected_features = []
    if 'feature_map' in schema and schema['feature_map']:
        for mapped_feature in schema['feature_map'].values():
            if mapped_feature in feature_columns:
                selected_features.append(mapped_feature)
                
    # Add remaining features until we reach max_features
    remaining_features = [f for f in feature_columns if f not in selected_features]
    
    # Try to use meta-feature selection if available
    if META_FEATURE_SELECTOR_AVAILABLE and len(remaining_features) > 0:
        try:
            print("Attempting meta-feature selection...")
            # Prepare data for feature selection
            X = data[remaining_features].copy()
            y = data[schema['target_column']]
            
            # Convert any categorical columns to numeric
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = pd.Categorical(X[col]).codes
                
            # Fill any missing values for feature selection
            X = X.fillna(X.mean())
            
            # Create meta-feature selector
            from migraine_prediction_project.src.meta.meta_feature_selector import MetaFeatureSelector
            n_additional = max_features - len(selected_features)
            meta_selector = MetaFeatureSelector(
                n_features=min(n_additional, len(remaining_features)),
                meta_method='de',
                surrogate='rf',
                verbose=True
            )
            
            # Run meta-optimization
            meta_selector.fit(X, y, feature_names=X.columns.tolist())
            additional_features = meta_selector.selected_features_
            
            print(f"Meta-selector chose {len(additional_features)} features.")
            selected_features.extend(additional_features)
        except Exception as e:
            print(f"Meta-feature selection failed: {str(e)}")
            print("Falling back to simple feature selection.")
            
    # If meta-selection didn't work or isn't available, use correlation-based selection
    if len(selected_features) < max_features and len(remaining_features) > 0:
        print("Using correlation-based feature selection...")
        try:
            # Calculate correlation with target
            correlations = {}
            target = data[schema['target_column']]
            
            for feature in remaining_features:
                if feature not in selected_features:
                    try:
                        # Skip non-numeric and constant features
                        if not pd.api.types.is_numeric_dtype(data[feature]) or data[feature].nunique() <= 1:
                            continue
                        
                        corr = data[feature].corr(target)
                        if not np.isnan(corr):
                            correlations[feature] = abs(corr)
                    except Exception:
                        pass
            
            # Sort by absolute correlation
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Add features until we reach max_features
            for feature, corr in sorted_features:
                if len(selected_features) < max_features:
                    selected_features.append(feature)
                else:
                    break
                    
            print(f"Correlation-based selection added {len(sorted_features)} features.")
        except Exception as e:
            print(f"Correlation-based selection failed: {str(e)}")

    # If we still don't have enough features, add remaining ones up to max_features
    if len(selected_features) < max_features and len(remaining_features) > 0:
        print("Adding remaining features to reach target count...")
        for feature in remaining_features:
            if len(selected_features) < max_features and feature not in selected_features:
                selected_features.append(feature)
            if len(selected_features) >= max_features:
                break
    
    print(f"Selected {len(selected_features)} features: {', '.join(selected_features[:5])}...")
    return selected_features


def evaluate_model(predictor, test_data, target_column):
    """
    Evaluate a trained model on test data.
    
    Args:
        predictor: The predictor instance with a trained model
        test_data: DataFrame with test data
        target_column: Name of the target column
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating model...")
    
    # Drop rows with NaN in the target column
    valid_test_data = test_data.dropna(subset=[target_column]).copy()
    print(f"Dropped {len(test_data) - len(valid_test_data)} rows with NaN values in target column")
    
    if len(valid_test_data) == 0:
        print("No valid test data after dropping NaN values. Cannot evaluate model.")
        return {
            'error': 'No valid test data after dropping NaN values'
        }
    
    # Extract features and target from valid test data
    X_test = valid_test_data.drop(columns=[target_column])
    y_test = valid_test_data[target_column].astype(int)  # Ensure integer targets
    
    # Generate predictions
    predictions = predictor.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions.tolist(),
        'ground_truth': y_test.tolist()
    }


def run_example(
    dataset_path=None, 
    test_size=0.2, 
    random_state=42, 
    explain=False, 
    explainer_type='shap'
):
    """
    Run the universal data example for migraine prediction.
    
    Args:
        dataset_path (str, optional): Path to the dataset. If None, synthetic data will be used.
        test_size (float, optional): Proportion of data to use for testing. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        explain (bool, optional): Whether to generate model explanations. Defaults to False.
        explainer_type (str, optional): Type of explainer to use ('shap', 'lime', or 'feature_importance').
            Defaults to 'shap'.
    
    Returns:
        dict: Dictionary containing results from the example.
    """
    # Create Universal Data Adapter
    adapter = UniversalDataAdapter()
    
    # Load dataset or create synthetic data
    if dataset_path and os.path.exists(dataset_path):
        print(f"\nLoading data from {dataset_path}...")
        data = adapter.load_data(dataset_path)
    else:
        data = create_example_dataset()
    
    print(f"\nLoaded dataset with shape: {data.shape}")
    print(f"Columns: {', '.join(data.columns)}")
    
    # Detect schema
    print("\nDetecting schema...")
    schema = adapter.detect_schema(data)
    print(f"Detected target column: {schema['target_column']}")
    print(f"Detected date column: {schema['date_column']}")
    
    # Select features
    print("\nSelecting features...")
    selected_features = select_features(data, schema)
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Prepare data for training
    print("\nPreparing data for training...")
    # Filter to only include selected features and target column
    train_features = train_data[selected_features].copy()
    test_features = test_data[selected_features].copy()
    
    # Add target column if not already in selected features
    target_column = schema['target_column']
    if target_column not in selected_features:
        train_features[target_column] = train_data[target_column]
        test_features[target_column] = test_data[target_column]
    
    # Drop rows with NaN values in the target column and any features
    train_features = train_features.dropna()
    test_features = test_features.dropna()
    
    # Convert target column to categorical (0 or 1)
    train_features[target_column] = train_features[target_column].astype(int)
    test_features[target_column] = test_features[target_column].astype(int)
    
    # Store target column separately to avoid scaling it
    y_train = train_features[target_column].copy()
    y_test = test_features[target_column].copy()
    
    # Remove target column from features for scaling
    X_train = train_features.drop(columns=[target_column])
    X_test = test_features.drop(columns=[target_column])
    
    # Preprocess the features only (without the target column)
    X_train_scaled, scaler = adapter.preprocess_data(X_train, fit_scaler=True)
    X_test_scaled, _ = adapter.preprocess_data(X_test, scaler=scaler)
    
    # Add target column back after scaling
    train_data_prepared = X_train_scaled.copy()
    train_data_prepared[target_column] = y_train
    
    test_data_prepared = X_test_scaled.copy()
    test_data_prepared[target_column] = y_test
    
    # Initialize and create data handler
    print("\nCreating data handler...")
    data_handler = DataHandler(data_dir="data")
    
    # Add flexible_features attribute to support missing features
    data_handler.flexible_features = True
        
    # Initialize predictor, set target column and replace its data handler
    predictor = MigrainePredictorV2(model_dir="models", data_dir="data")
    predictor.target_column = schema['target_column']
    predictor.data_handler = data_handler
    
    # Train model
    print("\nTraining model...")
    
    # Ensure no NaN values in the data before training
    if train_data_prepared.isna().any().any():
        print("Warning: NaN values found in training data. Filling with 0.")
        train_data_prepared = train_data_prepared.fillna(0)
    
    # Train using the full prepared DataFrame - the method will extract X and y
    model_id = predictor.train(
        train_data_prepared, 
        model_name=f"model_{int(time.time())}",
        description="Model trained with universal data adapter",
        make_default=True
    )
    print(f"Trained model ID: {model_id}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluation_report = evaluate_model(predictor, test_data_prepared, schema['target_column'])
    
    # Store the features used by the model for explainability
    model_features = predictor.model.feature_names_in_ if hasattr(predictor.model, 'feature_names_in_') else None
    
    # Generate explanations if requested and available
    if explain and EXPLAINABILITY_AVAILABLE:
        print("\nGenerating model explanations...")
        
        # Make sure we're using the same features the model was trained on
        if model_features is not None:
            print(f"Using model's expected features for explanation ({len(model_features)} features)")
            # Ensure test data has the same features as the model expects
            X_test_for_explanation = test_data_prepared[model_features].copy()
        else:
            # Drop the target column for explanation
            X_test_for_explanation = test_data_prepared.drop(columns=[schema['target_column']])
            
        explainer, explanations = explain_model(
            predictor, 
            X_test_for_explanation, 
            test_data_prepared[schema['target_column']], 
            explainer_type=explainer_type, 
            n_samples=5
        )
        evaluation_report['explanations'] = explanations
    
    print("\nExample complete!")
    return {
        'predictor': predictor,
        'data_handler': data_handler,
        'schema': schema,
        'model_id': model_id,
        'evaluation': evaluation_report,
        'selected_features': selected_features,
        'test_data': test_data_prepared,
        'model_features': model_features
    }


def explain_model(predictor, X_test, y_test=None, explainer_type='shap', n_samples=5):
    """Generate explanations for model predictions using the explainability framework."""
    print("Generating model explanations...")
    
    try:
        # Get feature names
        feature_names = list(X_test.columns)
        print(f"Features for explanation: {feature_names}")
        
        # Add explainability module to path
        explainability_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "explainability")
        if not os.path.exists(explainability_path):
            explainability_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "explainability")
        
        if not os.path.exists(explainability_path):
            print("Explainability module not found at expected paths")
            raise ImportError("Explainability module not found")
            
        print(f"Using explainability module at: {explainability_path}")
        sys.path.insert(0, os.path.dirname(explainability_path))
        
        # Sample data for explanation
        if len(X_test) > n_samples:
            X_sample = X_test.sample(n_samples, random_state=42)
        else:
            X_sample = X_test
            
        # Import explainability components
        try:
            from explainability.explainer_factory import ExplainerFactory
            print("Successfully imported ExplainerFactory")
            
            # Create explainer
            factory = ExplainerFactory()
            explainer = factory.create_explainer(explainer_type, predictor.model, feature_names=feature_names)
            print(f"Successfully created {explainer_type} explainer")
            
            # Generate explanation
            print(f"Generating explanations for {len(X_sample)} samples")
            explanation = explainer.explain(X_sample)
            print(f"Successfully generated explanation")
            
            # Print explanation type and structure
            print(f"Explanation type: {type(explanation)}")
            if isinstance(explanation, dict):
                print(f"Explanation keys: {list(explanation.keys())}")
                
                # If the explanation contains shap_values, extract those directly
                if 'shap_values' in explanation:
                    shap_values = explanation['shap_values']
                    print(f"SHAP values type: {type(shap_values)}")
                    print(f"SHAP values shape: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'N/A'}")
                    
                    # Create feature importance from SHAP values
                    try:
                        # Get absolute mean of SHAP values across samples
                        if isinstance(shap_values, list):
                            # For multi-class models
                            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                        else:
                            # For binary/regression models
                            mean_abs_shap = np.abs(shap_values).mean(axis=0)
                            
                        # Create dictionary of feature importances
                        feature_importance_dict = {feature: float(importance) 
                                                 for feature, importance in zip(feature_names, mean_abs_shap)}
                        
                        # Display top features
                        if feature_importance_dict:
                            print("Top 5 important features:")
                            top_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                            for feature, importance in top_features:
                                print(f"  {feature}: {importance:.4f}")
                            
                            # Generate a visualization of feature importance
                            try:
                                plt.figure(figsize=(10, 6))
                                features = []
                                importances = []
                                
                                # Get top 10 features for visualization
                                for feature, importance in sorted(feature_importance_dict.items(), 
                                                                 key=lambda x: abs(x[1]), reverse=True)[:10]:
                                    features.append(feature)
                                    importances.append(abs(importance))  # Use absolute values for better visualization
                                
                                # Create horizontal bar chart
                                plt.barh(range(len(features)), importances, align='center')
                                plt.yticks(range(len(features)), features)
                                plt.xlabel('Feature Importance')
                                plt.title('Top 10 Important Features')
                                
                                # Create output directory if it doesn't exist
                                output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                    
                                # Save the plot
                                plot_path = os.path.join(output_dir, f"{explainer_type}_feature_importance.png")
                                plt.savefig(plot_path)
                                plt.close()
                                print(f"Feature importance plot saved to {plot_path}")
                            except Exception as plot_err:
                                print(f"Error generating feature importance plot: {str(plot_err)}")
                        
                        # Return the explanation with feature importance
                        return explainer, {
                            'explanation': explanation,
                            'feature_importance': feature_importance_dict,
                            'explained_samples': len(X_sample)
                        }
                    except Exception as e:
                        print(f"Error creating feature importance from SHAP values: {str(e)}")
            
            # Extract feature importance using the explainer method
            try:
                print("Attempting to get feature importance from explainer...")
                feature_importance = explainer.get_feature_importance()
                print(f"Feature importance type: {type(feature_importance)}")
                if isinstance(feature_importance, np.ndarray):
                    print(f"Feature importance shape: {feature_importance.shape}")
                    print(f"Feature importance sample: {feature_importance[:3] if len(feature_importance) >= 3 else feature_importance}")
                
                # Create a dictionary of feature importances based on the type
                feature_importance_dict = {}
                
                if isinstance(feature_importance, dict):
                    # Already a dictionary - convert any array values to scalars
                    scalar_dict = {}
                    for feature, importance in feature_importance.items():
                        if isinstance(importance, np.ndarray):
                            # If it's an array, take the mean of absolute values
                            scalar_dict[feature] = float(np.abs(importance).mean())
                        else:
                            # Try to convert to float directly
                            try:
                                scalar_dict[feature] = float(importance)
                            except (TypeError, ValueError):
                                print(f"Could not convert importance value for {feature} to float: {importance}")
                                # Use a default value or skip
                                scalar_dict[feature] = 0.0
                    feature_importance_dict = scalar_dict
                elif isinstance(feature_importance, np.ndarray):
                    # Convert numpy array to dictionary
                    if len(feature_importance.shape) == 1:
                        # 1D array
                        min_len = min(len(feature_names), len(feature_importance))
                        feature_importance_dict = {feature_names[i]: float(feature_importance[i]) 
                                                 for i in range(min_len)}
                    else:
                        # Multi-dimensional array
                        mean_imp = np.abs(feature_importance).mean(axis=0)
                        min_len = min(len(feature_names), len(mean_imp))
                        feature_importance_dict = {feature_names[i]: float(mean_imp[i]) 
                                                 for i in range(min_len)}
                else:
                    print(f"Unsupported feature importance type: {type(feature_importance)}")
                    # Try direct conversion if possible
                    try:
                        if hasattr(feature_importance, "__iter__"):
                            min_len = min(len(feature_names), len(list(feature_importance)))
                            feature_importance_dict = {feature_names[i]: float(list(feature_importance)[i]) 
                                                    for i in range(min_len)}
                    except Exception as conv_err:
                        print(f"Failed to convert feature importance: {str(conv_err)}")
                
                # Display top features
                if feature_importance_dict:
                    print("Top 5 important features:")
                    top_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    for feature, importance in top_features:
                        print(f"  {feature}: {importance:.4f}")
                    
                    # Generate a visualization of feature importance
                    try:
                        plt.figure(figsize=(10, 6))
                        features = []
                        importances = []
                        
                        # Get top 10 features for visualization
                        for feature, importance in sorted(feature_importance_dict.items(), 
                                                         key=lambda x: abs(x[1]), reverse=True)[:10]:
                            features.append(feature)
                            importances.append(abs(importance))  # Use absolute values for better visualization
                        
                        # Create horizontal bar chart
                        plt.barh(range(len(features)), importances, align='center')
                        plt.yticks(range(len(features)), features)
                        plt.xlabel('Feature Importance')
                        plt.title('Top 10 Important Features')
                        
                        # Create output directory if it doesn't exist
                        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            
                        # Save the plot
                        plot_path = os.path.join(output_dir, f"{explainer_type}_feature_importance.png")
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"Feature importance plot saved to {plot_path}")
                    except Exception as plot_err:
                        print(f"Error generating feature importance plot: {str(plot_err)}")
                
                # Return the explanation with feature importance
                return explainer, {
                    'explanation': explanation,
                    'feature_importance': feature_importance_dict,
                    'explained_samples': len(X_sample)
                }
                
            except Exception as e:
                print(f"Error extracting feature importance: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Return with limited explanation
                return explainer, {
                    'explanation': explanation,
                    'error': str(e),
                    'explained_samples': len(X_sample)
                }
                
        except ImportError as e:
            print(f"Failed to import explainability components: {str(e)}")
            
            # Fallback to basic explainability if possible
            try:
                print("Falling back to basic feature importance...")
                if hasattr(predictor.model, 'feature_importances_'):
                    importances = predictor.model.feature_importances_
                    min_len = min(len(feature_names), len(importances))
                    feature_importance_dict = {feature_names[i]: float(importances[i]) 
                                            for i in range(min_len)}
                    
                    print("Top 5 important features (from model):")
                    top_features = sorted(feature_importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    for feature, importance in top_features:
                        print(f"  {feature}: {importance:.4f}")
                        
                    # Generate a visualization of feature importance
                    try:
                        plt.figure(figsize=(10, 6))
                        features = []
                        importances = []
                        
                        # Get top 10 features for visualization
                        for feature, importance in sorted(feature_importance_dict.items(), 
                                                         key=lambda x: abs(x[1]), reverse=True)[:10]:
                            features.append(feature)
                            importances.append(abs(importance))  # Use absolute values for better visualization
                        
                        # Create horizontal bar chart
                        plt.barh(range(len(features)), importances, align='center')
                        plt.yticks(range(len(features)), features)
                        plt.xlabel('Feature Importance')
                        plt.title('Top 10 Important Features')
                        
                        # Create output directory if it doesn't exist
                        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                            
                        # Save the plot
                        plot_path = os.path.join(output_dir, f"{explainer_type}_feature_importance.png")
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"Feature importance plot saved to {plot_path}")
                    except Exception as plot_err:
                        print(f"Error generating feature importance plot: {str(plot_err)}")
                    
                    return None, {
                        'feature_importance': feature_importance_dict,
                        'fallback': 'Basic model feature importance'
                    }
            except Exception as fallback_err:
                print(f"Fallback explainability failed: {str(fallback_err)}")
                
            return None, {'error': f"Failed to import explainability components: {str(e)}"}
            
    except Exception as e:
        print(f"Error generating explanations: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, {'error': str(e)}


if __name__ == "__main__":
    # Run the example
    results = run_example(
        explain=True,
        explainer_type='shap'
    )
    
    # Access results
    try:
        predictor = results.get('predictor')
        schema = results.get('schema', {})
        model_id = results.get('model_id')
        
        print(f"\nModel trained and evaluated successfully. Model ID: {model_id}")
        
        # Check if explanations were generated
        if 'explanations' in results.get('evaluation', {}):
            print("Explanations generated successfully.")
        else:
            print("No explanations were generated.")
            
    except Exception as e:
        print(f"Error in generating explanations: {str(e)}")
        traceback.print_exc()

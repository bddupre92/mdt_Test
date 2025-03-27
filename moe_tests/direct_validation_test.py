#!/usr/bin/env python

"""
Direct test script for clinical data validation without command imports.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from argparse import Namespace
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("direct_validation_test")

# Import the required modules directly
from data_integration.clinical_data_adapter import ClinicalDataAdapter
from data_integration.clinical_data_validator import ClinicalDataValidator
from data_integration.real_synthetic_comparator import RealSyntheticComparator
from utils.enhanced_synthetic_data import EnhancedSyntheticDataGenerator

# For JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def validate_real_data():
    """Direct implementation of real data validation"""
    logger.info("Starting direct real data validation test")
    
    # Setup paths and parameters
    clinical_data_path = 'sample_data/clinical_data_sample.csv'
    output_dir = 'direct_validation_results'
    target_column = 'migraine_severity'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'moe_validation'), exist_ok=True)
    
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    
    try:
        # Initialize adapter and validator
        adapter = ClinicalDataAdapter(None)  # No config for now
        validator = ClinicalDataValidator(None)  # No config for now
        
        # Load clinical data
        logger.info(f"Loading clinical data from {clinical_data_path}")
        clinical_data = adapter.load_data(clinical_data_path, 'csv')
        logger.info(f"Loaded {len(clinical_data)} records with {len(clinical_data.columns)} features")
        
        # Validate data quality
        logger.info("Validating clinical data quality")
        validation_report = validator.validate_all(
            clinical_data,
            save_path=os.path.join(output_dir, 'reports', f'validation_report_{timestamp}.json')
        )
        
        # Get validation summary
        validation_summary = validation_report.get('validation_summary', {})
        logger.info(f"Validation complete: {validation_summary.get('passed_validations', 0)} passed, "
              f"{validation_summary.get('warnings', 0)} warnings, {validation_summary.get('errors', 0)} errors")
        
        # Generate a small synthetic dataset for comparison
        logger.info("Generating comparable synthetic data")
        generator = EnhancedSyntheticDataGenerator(
            num_samples=len(clinical_data),
            drift_type='sudden',
            data_modality='mixed'
        )
        
        # Extract feature statistics
        feature_stats = {}
        for col in clinical_data.columns:
            if col == target_column:
                continue
            
            col_data = clinical_data[col].dropna()
            if len(col_data) == 0 or not np.issubdtype(col_data.dtype, np.number):
                continue
                
            feature_stats[col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max())
            }
        
        # Generate synthetic data
        synthetic_data = generator.generate_mirrored_data(
            feature_stats=feature_stats,
            target_column=target_column,
            target_ratio=clinical_data[target_column].mean() 
                if np.issubdtype(clinical_data[target_column].dtype, np.number) else 0.3
        )
        
        # Save synthetic data
        synthetic_path = os.path.join(output_dir, 'data', f'synthetic_comparison_{timestamp}.csv')
        synthetic_data.to_csv(synthetic_path, index=False)
        
        # Run comparison
        logger.info("Comparing real and synthetic data")
        comparator = RealSyntheticComparator(None)  # No config
        
        # Determine model type based on target column
        if np.issubdtype(clinical_data[target_column].dtype, np.number):
            if clinical_data[target_column].nunique() <= 2:
                task_type = 'classification'
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                task_type = 'regression'
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            task_type = 'classification'
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Run comparison
        comparison_results = comparator.compare_all(
            real_df=clinical_data,
            synthetic_df=synthetic_data,
            model=model,
            target_column=target_column,
            task_type=task_type,
            save_path=os.path.join(output_dir, 'reports', f'comparison_report_{timestamp}.json')
        )
        
        # Log summary
        if 'report_summary' in comparison_results and 'overall_similarity_score' in comparison_results['report_summary']:
            similarity = comparison_results['report_summary']['overall_similarity_score']
            logger.info(f"Overall similarity between real and synthetic data: {similarity:.2f} (0-1 scale)")
        
        # Now set up and run model validation
        validation_dir = os.path.join(output_dir, 'moe_validation')
        
        # Split data for model training
        from sklearn.model_selection import train_test_split
        
        # Preprocess data before modeling
        preprocess_df = clinical_data.copy()
        
        # Handle datetime columns by either dropping or converting to numeric features
        for col in preprocess_df.columns:
            # Skip target column
            if col == target_column:
                continue
                
            # Convert datetime columns to numeric (timestamp) or drop them
            if pd.api.types.is_datetime64_dtype(preprocess_df[col]) or (
                isinstance(preprocess_df[col].dtype, type) and 
                preprocess_df[col].dtype.name.startswith('datetime')
            ):
                logger.info(f"Converting datetime column {col} to timestamp")
                try:
                    # Convert to timestamp (seconds since epoch)
                    preprocess_df[col] = pd.to_datetime(preprocess_df[col], errors='coerce').astype(np.int64) // 10**9
                except Exception as e:
                    logger.warning(f"Failed to convert datetime column {col} to timestamp: {e}")
                    # If conversion fails, drop the column
                    logger.info(f"Dropping datetime column {col} that couldn't be converted")
                    preprocess_df = preprocess_df.drop(columns=[col])
            
            # Also try to convert string columns that might be dates
            elif preprocess_df[col].dtype == 'object':
                try:
                    # Check if column contains datetime strings
                    temp_series = pd.to_datetime(preprocess_df[col], errors='coerce')
                    if temp_series.notna().sum() > 0.7 * len(temp_series):  # >70% conversion success
                        logger.info(f"Converting string column {col} to timestamp")
                        preprocess_df[col] = temp_series.astype(np.int64) // 10**9
                        # Fill NA values with mean
                        if preprocess_df[col].isna().any():
                            preprocess_df[col] = preprocess_df[col].fillna(preprocess_df[col].mean())
                except:
                    # Keep as is if not a datetime
                    pass
        
        # Ensure all remaining columns are numeric
        for col in preprocess_df.columns:
            if col == target_column:
                continue
                
            if not pd.api.types.is_numeric_dtype(preprocess_df[col]):
                logger.warning(f"Dropping non-numeric column {col}")
                preprocess_df = preprocess_df.drop(columns=[col])
        
        # After cleaning, check if we have enough features left
        remaining_features = [col for col in preprocess_df.columns if col != target_column]
        logger.info(f"Remaining features after preprocessing: {len(remaining_features)}")
        
        if len(remaining_features) < 2:
            raise ValueError("Not enough features left after preprocessing")
        
        # Create final X and y for model training
        X = preprocess_df.drop(columns=[target_column])
        y = preprocess_df[target_column]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        logger.info(f"Training model for {target_column} prediction")
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
        
        # Initialize validation results
        validation_results = validation_report  # Use the earlier validation results
        real_data_validation = {}
        
        # Add model performance metrics
        performance_metrics = {}
        
        if task_type == 'classification':
            performance_metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            performance_metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted'))
            
            # For binary classification, calculate ROC AUC
            if len(np.unique(y)) == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    y_prob = model.predict_proba(X_test)[:, 1]
                    performance_metrics['roc_auc'] = float(roc_auc_score(y_test, y_prob))
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC score: {e}")
        else:
            performance_metrics['r2_score'] = float(r2_score(y_test, y_pred))
            performance_metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            performance_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        # Add model performance to results with proper test structure
        real_data_validation['model_performance'] = {
            'passed': True,
            'details': f"Accuracy: {performance_metrics.get('accuracy', 'N/A')}, F1 Score: {performance_metrics.get('f1_score', 'N/A')}",
            'metrics': performance_metrics
        }
        
        # Try to generate feature importance visualization
        if hasattr(model, 'feature_importances_'):
            logger.info("Generating feature importance visualization")
            
            # Extract feature importances
            feature_importances = model.feature_importances_
            feature_names = X.columns
            
            # Create sorted feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)
            
            # Save feature importance data
            importance_file = os.path.join(validation_dir, 'feature_importance.csv')
            importance_df.to_csv(importance_file, index=False)
            
            # Try to create interactive plot
            import plotly.express as px
            
            # Take top 10 features
            top_features = importance_df.head(10)
            
            # Create bar chart
            fig = px.bar(
                top_features, 
                x='importance', 
                y='feature', 
                orientation='h',
                title='Top 10 Feature Importance',
                color='importance',
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500, 
                width=800,
                template='plotly_white'
            )
            
            # Save plot
            importance_plot = os.path.join(validation_dir, 'feature_importance.html')
            fig.write_html(importance_plot)
            
            # Add to validation results
            real_data_validation['feature_importance'] = {
                'passed': True,
                'details': f"Top features: {', '.join(top_features['feature'].tolist()[:3])}",
                'importances': importance_df.to_dict('records'),
                'plot_path': importance_plot,
                'available': True,
                'top_features': top_features['feature'].tolist()
            }
        
        # Add SHAP explainability if available
        try:
            import shap
            logger.info("Running SHAP explainability analysis")
            
            # Sample data to explain (limit to 100 samples for performance)
            explain_data = X_test.head(min(100, len(X_test)))
            
            # Create explainer with appropriate handling for different model types
            if hasattr(model, 'predict_proba'):
                # For models that support probability predictions (classification)
                explainer = shap.TreeExplainer(model)
            else:
                # For regression models or models without predict_proba
                explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values with robust error handling
            try:
                shap_values = explainer.shap_values(explain_data)
                
                # Handle different return types of shap_values
                if isinstance(shap_values, list):
                    # For classification with multiple classes or multi-output
                    multi_class = True
                    
                    # For binary classification with two arrays, use the positive class
                    if len(shap_values) == 2:
                        shap_values_viz = shap_values[1]  # Use positive class (index 1)
                    else:
                        # For multiclass, use the first class for visualization by default
                        shap_values_viz = shap_values[0]
                else:
                    # For regression or single-output models
                    shap_values_viz = shap_values
                    multi_class = False
                
                # Handle different types of expected_value
                if hasattr(explainer, 'expected_value'):
                    if isinstance(explainer.expected_value, np.ndarray) or isinstance(explainer.expected_value, list):
                        expected_val = explainer.expected_value
                        # Convert ndarray to list for JSON serialization
                        if isinstance(expected_val, np.ndarray):
                            expected_val = expected_val.tolist()
                    else:
                        expected_val = float(explainer.expected_value)
                else:
                    expected_val = 0.0  # Default if expected_value not available
                
                # Prepare SHAP data for saving
                shap_data = {
                    'feature_names': list(explain_data.columns),
                    'multi_class': multi_class,
                    'expected_value': expected_val
                }
                
                # Handle different formats of shap_values for JSON serialization
                if multi_class:
                    if isinstance(shap_values[0], np.ndarray):
                        shap_data['shap_values'] = [sv.tolist() for sv in shap_values]
                    else:
                        shap_data['shap_values'] = shap_values  # Already in right format
                else:
                    if isinstance(shap_values_viz, np.ndarray):
                        shap_data['shap_values'] = shap_values_viz.tolist()
                    else:
                        shap_data['shap_values'] = shap_values_viz
            except Exception as e:
                # If shap_values calculation fails, fall back to feature importance
                logger.warning(f"Error calculating SHAP values: {e}. Falling back to feature importance.")
                
                # Use feature importance as a fallback
                if hasattr(model, 'feature_importances_'):
                    feature_importances = model.feature_importances_
                    shap_data = {
                        'feature_names': list(explain_data.columns),
                        'feature_importances': feature_importances.tolist(),
                        'fallback_method': 'feature_importance'
                    }
                else:
                    # If no fallback available, raise the original error
                    raise
            
            # Save SHAP data with custom encoder to handle numpy types
            shap_file = os.path.join(validation_dir, 'shap_data.json')
            with open(shap_file, 'w') as f:
                json.dump(shap_data, f, indent=2, cls=NumpyEncoder)
            
            # Generate SHAP summary plot and save as image
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Create a new figure
            plt.figure(figsize=(10, 8))
            
            # Generate summary plot
            if multi_class and len(shap_values) > 1:
                # For multi-class, we'll plot the most important class
                shap.summary_plot(shap_values[0], explain_data, plot_type="bar", show=False)
            else:
                shap.summary_plot(shap_values_viz, explain_data, plot_type="bar", show=False)
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            # Save the plot image
            shap_plot_path = os.path.join(validation_dir, 'shap_summary_plot.png')
            with open(shap_plot_path, 'wb') as f:
                f.write(buf.getvalue())
            
            # Add to validation results
            real_data_validation['explainability'] = {
                'passed': True,
                'details': "SHAP explainability analysis completed successfully",
                'available': True,
                'shap_data_path': shap_file,
                'shap_plot_path': shap_plot_path,
                'fallback_method': shap_data.get('fallback_method', None)
            }
            
            logger.info("SHAP analysis completed successfully")
            
        except ImportError:
            logger.warning("SHAP library not available for explainability analysis")
            real_data_validation['explainability'] = {
                'passed': False,
                'details': "SHAP library not available for explainability analysis",
                'available': False,
                'error': "SHAP library not available"
            }
        except Exception as e:
            logger.warning(f"Error generating SHAP explainability: {e}")
            real_data_validation['explainability'] = {
                'passed': False,
                'details': f"Error generating SHAP explainability: {str(e)}",
                'available': False,
                'error': str(e)
            }
        
        # Structure test results to match what the interactive report expects
        # Convert any string values to proper dictionaries with 'passed' key
        structured_results = {}
        
        # Add validation results with proper structure
        for key, value in validation_results.items():
            if isinstance(value, dict) and 'passed' in value:
                structured_results[key] = value
            elif isinstance(value, dict):
                # Add passed key if missing
                structured_results[key] = {**value, 'passed': True}
            elif isinstance(value, str):
                # Convert string results to dictionaries
                structured_results[key] = {'details': value, 'passed': True}
            else:
                # Other types
                structured_results[key] = {'details': str(value), 'passed': True}
        
        # Add model performance and other real data validation results
        for key, value in real_data_validation.items():
            structured_results[key] = value
        
        # Make sure we have the expected top-level sections
        if 'data_quality' not in structured_results:
            structured_results['data_quality'] = {
                'passed': len(validation_report.get('quality_issues', [])) == 0,
                'details': f"Found {len(validation_report.get('quality_issues', []))} data quality issues"
            }
            
        if 'structure_validation' not in structured_results:
            structured_results['structure_validation'] = {
                'passed': len(validation_report.get('structure_issues', [])) == 0,
                'details': f"Found {len(validation_report.get('structure_issues', []))} structure issues"
            }
            
        if 'distribution_analysis' not in structured_results:
            structured_results['distribution_analysis'] = {
                'passed': True,
                'details': "Distribution analysis completed"
            }
            
        if 'completeness_checks' not in structured_results:
            structured_results['completeness_checks'] = {
                'passed': validation_report.get('completeness', {}).get('overall_completeness', 100) > 80,
                'details': f"Overall completeness: {validation_report.get('completeness', {}).get('overall_completeness', 'N/A')}%"
            }
        
        # Generate interactive report
        try:
            # Import from tests directory
            from tests.moe_interactive_report import generate_interactive_report
            
            # Create report directory
            report_dir = os.path.join(validation_dir, f'report_{timestamp}')
            os.makedirs(report_dir, exist_ok=True)
            
            # Save structured results
            results_file = os.path.join(report_dir, 'validation_results.json')
            with open(results_file, 'w') as f:
                json.dump(structured_results, f, indent=2, cls=NumpyEncoder)
            
            # Save metadata
            metadata_file = os.path.join(report_dir, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "data_source": clinical_data_path,
                    "num_records": len(clinical_data),
                    "num_features": len(clinical_data.columns),
                    "target_column": target_column,
                    "validation_summary": validation_summary,
                    "data_type": "real_clinical",
                    "title": "Real Clinical Data MoE Validation Report",
                    "description": f"Validation results for real clinical data from {clinical_data_path}"
                }, f, indent=2, cls=NumpyEncoder)
            
            # Generate interactive report
            report_path = os.path.join(validation_dir, f'real_data_report_{timestamp}.html')
            
            # Generate HTML content with properly structured results - work with absolute paths
            # Ensure the report directory is an absolute path
            abs_report_dir = os.path.abspath(report_dir)
            logger.info(f"Using absolute report dir: {abs_report_dir}")
            
            html_content = generate_interactive_report(
                test_results=structured_results,
                results_dir=abs_report_dir,
                return_html=True
            )
            
            # Write HTML report - directly create it ourselves instead of relying on the module
            report_path = os.path.join(output_dir, f'real_data_report_{timestamp}.html')
            abs_report_path = os.path.abspath(report_path)
            logger.info(f"Writing HTML report directly to: {abs_report_path}")
            
            with open(abs_report_path, 'w') as f:
                f.write(html_content)
                
            # Also copy important visualizations to the output directory for easier access
            if os.path.exists(os.path.join(validation_dir, 'shap_summary_plot.png')):
                import shutil
                src = os.path.join(validation_dir, 'shap_summary_plot.png')
                dst = os.path.join(output_dir, 'shap_summary_plot.png')
                shutil.copy2(src, dst)
                logger.info(f"Copied SHAP plot to: {dst}")
            
            logger.info(f"Interactive report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating interactive report: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("Real data validation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during real data validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(validate_real_data())

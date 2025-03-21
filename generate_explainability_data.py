"""
Generate sample explainability data to test the report visualization
"""
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Create the continuous explainability directory
results_dir = Path('results/moe_validation')
explainability_dir = results_dir / 'continuous_explainability'
explainability_dir.mkdir(parents=True, exist_ok=True)

print("Generating explainability data...")

# Generate sample feature importance data
features = ['heart_rate', 'stress_level', 'sleep_quality', 'hydration', 
           'barometric_pressure', 'screen_time', 'exercise_duration', 
           'light_exposure', 'medication_dosage', 'caffeine_intake']

# Generate sample explainer visualizations
for explainer_type in ['shap', 'feature_importance']:
    # Create feature importance
    feature_importance = {}
    for feature in features:
        importance = random.uniform(-1, 1)
        feature_importance[feature] = importance
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Sort features by absolute importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Plot top 8 features
    top_features = sorted_features[:8]
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    # Create horizontal bar chart
    plt.barh(names, values)
    plt.xlabel('Feature Importance')
    plt.title(f'Top Features - {explainer_type.replace("_", " ").title()} ({datetime.now().strftime("%Y-%m-%d %H:%M:%S")})')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = explainability_dir / f"explanation_{explainer_type}_{timestamp}.png"
    plt.savefig(viz_path)
    plt.close()
    
    print(f"Saved {explainer_type} visualization to {viz_path}")

# Generate sample continuous explanations log
log_entries = []
feature_trends = {}

# Initialize feature trends
for feature in features[:5]:  # Use top 5 features for trends
    feature_trends[feature] = []

# Generate logs for past 10 time points
for i in range(10):
    explanations = {}
    
    # Add SHAP explainer results
    shap_importance = {}
    for feature in features:
        # Generate feature importance with some consistency over time
        base_value = random.uniform(-0.8, 0.8)
        importance = base_value + random.uniform(-0.2, 0.2)
        shap_importance[feature] = importance
        
        # Add to trends for top features
        if feature in feature_trends:
            feature_trends[feature].append(importance)
    
    explanations['shap'] = {
        'feature_importance': shap_importance
    }
    
    # Add feature importance explainer results
    fi_importance = {}
    for feature in features:
        # Generate feature importance with some consistency over time
        base_value = abs(random.uniform(0, 1.0))
        importance = base_value + random.uniform(-0.1, 0.1)
        fi_importance[feature] = importance
    
    explanations['feature_importance'] = {
        'feature_importance': fi_importance
    }
    
    # Create log entry
    timestamp = (datetime.now().timestamp() - (10 - i) * 3600)  # One hour apart
    entry_time = datetime.fromtimestamp(timestamp).isoformat()
    
    log_entries.append({
        'timestamp': entry_time,
        'explanations': explanations
    })

# Save continuous explanations log
log_file = explainability_dir / 'continuous_explanations.json'
with open(log_file, 'w') as f:
    json.dump(log_entries, f, indent=2)

print(f"Saved continuous explanations log to {log_file}")

# Create a report-ready JSON file with explainability data
report_data = {
    'explanation_results': {
        'explainability': {
            'shap': {
                'feature_importance': shap_importance,
                'timestamp': datetime.now().isoformat()
            },
            'feature_importance': {
                'feature_importance': fi_importance,
                'timestamp': datetime.now().isoformat()
            }
        },
        'importance_trends': feature_trends
    }
}

# Save report data
report_data_file = results_dir / 'explainability_report_data.json'
with open(report_data_file, 'w') as f:
    json.dump(report_data, f, indent=2)

print(f"Saved report data to {report_data_file}")
print("Done generating explainability data.")

#!/usr/bin/env python
"""
Real Data Validation Runner for MoE Framework

This script integrates real clinical data with the MoE validation framework
and evaluates model performance on both real and synthetic data.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from cli.real_data_commands to bypass the CLI infrastructure
from cli.real_data_commands import RealDataValidationCommand
from data_integration.clinical_data_adapter import ClinicalDataAdapter
from data_integration.clinical_data_validator import ClinicalDataValidator
from data_integration.real_synthetic_comparator import RealSyntheticComparator
from utils.enhanced_synthetic_data import EnhancedSyntheticDataGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run MoE validation with real clinical data')
    
    parser.add_argument('--clinical-data', type=str, required=True,
                        help='Path to clinical data file (CSV, JSON, or Excel)')
    
    parser.add_argument('--data-format', type=str, default='csv',
                        choices=['csv', 'json', 'excel', 'parquet'],
                        help='Format of clinical data file')
    
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file for data integration')
    
    parser.add_argument('--target-column', type=str, default='migraine',
                        help='Target column name in clinical data')
    
    parser.add_argument('--output-dir', type=str, default='results/real_data_validation',
                        help='Directory to store validation results')
    
    parser.add_argument('--synthetic-compare', action='store_true',
                        help='Generate and compare with synthetic data')
    
    parser.add_argument('--drift-type', type=str, default='sudden',
                        choices=['sudden', 'gradual', 'recurring', 'none'],
                        help='Type of drift to analyze')
    
    parser.add_argument('--run-mode', type=str, default='full',
                        choices=['full', 'validation', 'comparison', 'report'],
                        help='Which parts of the process to run')
    
    return parser.parse_args()


def prepare_output_directory(output_dir):
    """Create output directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return timestamp


def load_and_validate_clinical_data(args, timestamp):
    """Load and validate clinical data."""
    print(f"Loading clinical data from {args.clinical_data}...")
    
    # Initialize adapter and validator
    adapter = ClinicalDataAdapter(args.config)
    validator = ClinicalDataValidator(args.config)
    
    # Load data
    try:
        clinical_data = adapter.load_data(args.clinical_data, args.data_format)
    except Exception as e:
        print(f"Error loading clinical data: {str(e)}")
        sys.exit(1)
    
    print(f"Loaded {len(clinical_data)} records with {len(clinical_data.columns)} features.")
    
    # Validate data
    print("Validating clinical data quality and compatibility...")
    validation_report = validator.validate_all(
        clinical_data, 
        save_path=os.path.join(args.output_dir, 'reports', f'validation_report_{timestamp}.json')
    )
    
    validation_summary = validation_report['validation_summary']
    print(f"Validation complete: {validation_summary['passed_validations']} checks passed, "
          f"{validation_summary['warnings']} warnings, {validation_summary['errors']} errors.")
    
    if validation_summary['errors'] > 0:
        print("WARNING: Data validation found errors. Review the validation report before proceeding.")
        print(f"Report saved to: {os.path.join(args.output_dir, 'reports', f'validation_report_{timestamp}.json')}")
    
    # Preprocess data
    print("Preprocessing clinical data...")
    processed_data = adapter.preprocess(clinical_data)
    
    # Save processed data
    processed_path = os.path.join(args.output_dir, 'data', f'processed_clinical_{timestamp}.csv')
    processed_data.to_csv(processed_path, index=False)
    print(f"Processed data saved to: {processed_path}")
    
    # Partition data by profile (if possible)
    try:
        print("Partitioning data by patient profiles...")
        profile_data = adapter.partition_by_profile(processed_data)
        
        for profile, data in profile_data.items():
            profile_path = os.path.join(args.output_dir, 'data', f'profile_{profile}_{timestamp}.csv')
            data.to_csv(profile_path, index=False)
            print(f"- {profile}: {len(data)} records saved to {profile_path}")
    except Exception as e:
        print(f"Could not partition data by profile: {str(e)}")
        profile_data = {'all_patients': processed_data}
    
    return processed_data, profile_data, validation_report


def generate_comparison_synthetic_data(clinical_data, args, timestamp):
    """Generate synthetic data mirroring clinical data structure for comparison."""
    print("Generating comparable synthetic data...")
    
    target_col = args.target_column
    
    # Check if target column exists
    if target_col not in clinical_data.columns:
        print(f"Target column '{target_col}' not found in clinical data.")
        print(f"Available columns: {', '.join(clinical_data.columns)}")
        sys.exit(1)
    
    # Get feature statistics from clinical data
    feature_stats = {}
    for col in clinical_data.columns:
        if col == target_col:
            continue
        
        # Get basic statistics
        col_data = clinical_data[col].dropna()
        if len(col_data) == 0:
            continue
            
        # Skip non-numeric columns for now
        if not np.issubdtype(col_data.dtype, np.number):
            continue
        
        feature_stats[col] = {
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max())
        }
    
    # Configure the synthetic data generator
    generator = EnhancedSyntheticDataGenerator(
        num_samples=len(clinical_data),
        drift_type=args.drift_type,
        data_modality='mixed'  # Generate mixed data type
    )
    
    # Generate data with similar feature distributions
    synthetic_data = generator.generate_mirrored_data(
        feature_stats=feature_stats,
        target_column=target_col,
        target_ratio=clinical_data[target_col].mean() if np.issubdtype(clinical_data[target_col].dtype, np.number) else 0.3
    )
    
    # Add any non-numeric columns as categorical
    for col in clinical_data.columns:
        if col in synthetic_data.columns:
            continue
            
        if not np.issubdtype(clinical_data[col].dtype, np.number):
            if clinical_data[col].nunique() < 10:  # Only mirror if it's a reasonable categorical
                # Sample from the same distribution
                synthetic_data[col] = np.random.choice(
                    clinical_data[col].dropna().values,
                    size=len(synthetic_data),
                    replace=True
                )
    
    # Save synthetic data
    synthetic_path = os.path.join(args.output_dir, 'data', f'synthetic_comparison_{timestamp}.csv')
    synthetic_data.to_csv(synthetic_path, index=False)
    print(f"Generated synthetic data saved to: {synthetic_path}")
    
    return synthetic_data


def compare_real_and_synthetic(clinical_data, synthetic_data, args, timestamp):
    """Compare real clinical data with synthetic data."""
    print("Comparing real and synthetic data...")
    
    # Initialize comparator
    comparator = RealSyntheticComparator(args.config)
    
    # Prepare basic model for performance comparison
    target_col = args.target_column
    
    # Determine task type based on target variable
    if np.issubdtype(clinical_data[target_col].dtype, np.number):
        if clinical_data[target_col].nunique() <= 2:
            task_type = 'classification'
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            task_type = 'regression'
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        task_type = 'classification'
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print(f"Using {task_type} model for performance comparison.")
    
    # Run all comparisons
    comparison_results = comparator.compare_all(
        real_df=clinical_data,
        synthetic_df=synthetic_data,
        model=model,
        target_column=target_col,
        task_type=task_type,
        save_path=os.path.join(args.output_dir, 'reports', f'comparison_report_{timestamp}.json')
    )
    
    # Print summary
    if 'report_summary' in comparison_results and 'overall_similarity_score' in comparison_results['report_summary']:
        similarity = comparison_results['report_summary']['overall_similarity_score']
        print(f"Overall similarity between real and synthetic data: {similarity:.2f} (0-1 scale)")
    
    return comparison_results


def run_moe_validation_with_real_data(clinical_data, synthetic_data, args, timestamp):
    """Run MoE validation with real clinical data."""
    print("Running MoE validation with real clinical data...")
    
    # Prepare output directory for validation
    validation_dir = os.path.join(args.output_dir, 'moe_validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    # Configure validation - need to update this based on real data structure
    validation_config = {
        'drift_type': args.drift_type,
        'output_dir': validation_dir,
        'data_source': 'real',
        'target_column': args.target_column,
        'timestamp': timestamp
    }
    
    # Save config for reference
    with open(os.path.join(validation_dir, f'validation_config_{timestamp}.json'), 'w') as f:
        json.dump(validation_config, f, indent=2)
    
    # Create expert types mapping based on column names
    expert_mapping = {}
    for col in clinical_data.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['heart', 'rate', 'temp', 'blood']):
            expert_mapping[col] = 'physiological'
        elif any(term in col_lower for term in ['weather', 'humid', 'pressure']):
            expert_mapping[col] = 'environmental'
        elif any(term in col_lower for term in ['stress', 'sleep', 'activity']):
            expert_mapping[col] = 'behavioral'
        elif any(term in col_lower for term in ['medication', 'drug', 'treatment']):
            expert_mapping[col] = 'medication'
        else:
            expert_mapping[col] = 'general'
    
    # Save expert mapping for reference
    with open(os.path.join(validation_dir, f'expert_mapping_{timestamp}.json'), 'w') as f:
        json.dump(expert_mapping, f, indent=2)
    
    # Run enhanced validation
    try:
        validation_results = run_enhanced_validation(
            real_data=clinical_data,
            synthetic_data=synthetic_data,
            drift_type=args.drift_type,
            output_dir=validation_dir,
            expert_mapping=expert_mapping,
            target_column=args.target_column
        )
        
        print(f"MoE validation complete. Results saved to: {validation_dir}")
        return validation_results
    except Exception as e:
        print(f"Error running MoE validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run real data validation using RealDataValidationCommand class."""
    args = parse_arguments()
    
    # Create a dictionary of args for RealDataValidationCommand
    command_args = {
        "clinical_data": args.clinical_data,
        "output_dir": args.output_dir,
        "target_column": args.target_column,
        "synthetic_compare": args.synthetic_compare,
        "run_mode": args.run_mode,
        "data_format": args.data_format,
        "drift_type": args.drift_type,
        "interactive_report": True  # Always generate an interactive report
    }
    
    print(f"Starting real data validation using RealDataValidationCommand...")
    
    # Create and execute the RealDataValidationCommand directly
    command = RealDataValidationCommand(command_args)
    
    print("Executing real data validation...")
    result = command.execute()
    
    print(f"Real data validation completed with exit code: {result}")
    print(f"Results saved to: {args.output_dir}")
    
    return result


if __name__ == "__main__":
    main()

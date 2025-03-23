"""
Interactive testing environment for migraine prediction system.
"""
import os
import json
import click
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

from app.core.data.generators.test import TestDataGenerator
from app.core.data.generators.synthetic import SyntheticDataGenerator, SyntheticConfig
from app.core.data.drift import DriftDetector
from app.core.models.database import DiaryEntry, Prediction, User

class TestEnvironment:
    def generate_patient_data(self, n_days: int, include_drift: bool = False) -> pd.DataFrame:
        """Generate patient data."""
        return self.test_generator.generate_time_series(
            n_days=n_days,
            drift_start=n_days // 2 if include_drift else None
        )
    
    def generate_production_data(self, n_patients: int, n_days: int) -> Dict[int, pd.DataFrame]:
        """Generate production-like data."""
        config = SyntheticConfig(
            n_patients=n_patients,
            time_range_days=n_days,
            missing_rate=0.05,
            drift_points=[n_days // 2]
        )
        generator = SyntheticDataGenerator(config)
        return generator.generate_dataset()
    
    def simulate_predictions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Simulate predictions and track performance."""
        predictions = []
        actual = []
        drift_detected = []
        
        # Initialize reference data for drift detection
        reference_data = data.iloc[:30]  # Use first 30 days as reference
        self.drift_detector.initialize_reference(reference_data)
        
        for idx, row in data.iterrows():
            # Make prediction
            features = {
                col: row[col] for col in self.test_generator.feature_configs.keys()
            }
            prob = self.test_generator.calculate_migraine_probability(features)
            predictions.append(prob)
            actual.append(row['migraine_occurred'])
            
            # Check for drift
            current_window = data.loc[:idx]
            drift_results = self.drift_detector.detect_drift(current_window)
            drift_detected.append(any(r.detected for r in drift_results))
        
        # Calculate metrics
        true_pos = sum(1 for p, a in zip(predictions, actual) if p > 0.5 and a)
        false_pos = sum(1 for p, a in zip(predictions, actual) if p > 0.5 and not a)
        true_neg = sum(1 for p, a in zip(predictions, actual) if p <= 0.5 and not a)
        false_neg = sum(1 for p, a in zip(predictions, actual) if p <= 0.5 and a)
        
        accuracy = (true_pos + true_neg) / len(predictions)
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        return {
            'predictions': predictions,
            'actual': actual,
            'drift_detected': drift_detected,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        }
    
    def save_results(self, results: Dict[str, Any], scenario_name: str):
        """Save simulation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.data_dir / 'results' / f"{scenario_name}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")

@click.group()
def cli():
    """Migraine Prediction Testing Environment"""
    pass

@cli.command()
@click.option('--days', default=90, help='Number of days to simulate')
@click.option('--drift/--no-drift', default=False, help='Include concept drift')
@click.option('--scenario', default='test', help='Scenario name for results')
def simulate(days: int, drift: bool, scenario: str):
    """Run a simulation with specified parameters."""
    env = TestEnvironment(Path('test_data'))
    
    print(f"\nGenerating {days} days of patient data {'with' if drift else 'without'} drift...")
    data = env.generate_patient_data(days, drift)
    
    print("\nSimulating predictions and monitoring drift...")
    results = env.simulate_predictions(data)
    
    print("\nSimulation Results:")
    print(f"Accuracy:  {results['metrics']['accuracy']:.2f}")
    print(f"Precision: {results['metrics']['precision']:.2f}")
    print(f"Recall:    {results['metrics']['recall']:.2f}")
    
    drift_days = sum(results['drift_detected'])
    print(f"\nDrift detected on {drift_days} days")
    
    env.save_results(results, scenario)

@cli.command()
@click.option('--patients', default=5, help='Number of patients')
@click.option('--days', default=90, help='Number of days per patient')
def generate_production(patients: int, days: int):
    """Generate production-like dataset."""
    env = TestEnvironment(Path('test_data'))
    
    print(f"\nGenerating production data for {patients} patients over {days} days...")
    datasets = env.generate_production_data(patients, days)
    
    # Save each patient's data
    for patient_id, data in datasets.items():
        output_file = env.data_dir / f"patient_{patient_id}_prod.csv"
        data.to_csv(output_file, index=False)
        print(f"Saved data for patient {patient_id} to: {output_file}")

@cli.command()
@click.option('--days', default=30, help='Number of days of data')
def generate_test(days: int):
    """Generate test dataset."""
    env = TestEnvironment(Path('test_data'))
    
    print(f"\nGenerating {days} days of test data...")
    data = env.generate_patient_data(days)
    
    # Save to CSV
    output_file = env.data_dir / f"test_data_{days}days.csv"
    data.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")

@cli.command()
def analyze_drift():
    """Analyze drift detection performance."""
    env = TestEnvironment(Path('test_data'))
    
    # Generate validation data
    print("\nGenerating validation data...")
    stable_data, drift_data = env.test_generator.generate_validation_set()
    
    # Analyze stable period
    print("\nAnalyzing stable period...")
    stable_results = env.simulate_predictions(stable_data)
    
    # Analyze drift period
    print("\nAnalyzing drift period...")
    drift_results = env.simulate_predictions(drift_data)
    
    print("\nResults:")
    print("Stable Period:")
    print(f"- Accuracy:  {stable_results['metrics']['accuracy']:.2f}")
    print(f"- Drift detected: {sum(stable_results['drift_detected'])} days")
    
    print("\nDrift Period:")
    print(f"- Accuracy:  {drift_results['metrics']['accuracy']:.2f}")
    print(f"- Drift detected: {sum(drift_results['drift_detected'])} days")

if __name__ == '__main__':
    cli()

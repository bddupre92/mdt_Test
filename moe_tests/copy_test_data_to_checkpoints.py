#!/usr/bin/env python
"""
Utility script to copy test performance data into the checkpoints directory
for testing the Performance Analysis Dashboard.
"""
import json
import os
from pathlib import Path
import datetime

# Define directories
test_dir = Path('test_data/performance_formats')
checkpoint_dir = Path('checkpoints/dev')

# Ensure the checkpoint directory exists
checkpoint_dir.mkdir(exist_ok=True, parents=True)

# Current timestamp for checkpoint naming
timestamp = datetime.datetime.now().strftime('%Y_%m_%d')

# Process each test data file
for data_file in test_dir.glob('*.json'):
    # Read the original data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Create checkpoint name
    checkpoint_name = f'checkpoint_{data_file.stem}_{timestamp}.json'
    
    # Create checkpoint data structure
    checkpoint_data = {
        'version': '1.0',
        'timestamp': datetime.datetime.now().isoformat(),
        'metadata': {
            'model_version': 'MoE-v2.0',
            'dataset': 'migraine-clinical-dataset-v2',
            'experiment_id': f'Test-{data_file.stem}'
        }
    }
    
    # Add performance metrics
    if 'performance_metrics' not in data:
        checkpoint_data['performance_metrics'] = data
    else:
        checkpoint_data.update(data)
    
    # Write the checkpoint file
    with open(checkpoint_dir / checkpoint_name, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f'Created {checkpoint_name}')

print("All test data copied to checkpoints directory.")

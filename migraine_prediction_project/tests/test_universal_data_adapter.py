#!/usr/bin/env python
"""
Integration tests for the Universal Data Adapter.
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules to test
from migraine_prediction_project.src.migraine_model.universal_data_adapter import UniversalDataAdapter
from migraine_prediction_project.src.migraine_model.feature_meta_optimizer import MetaFeatureSelector
from migraine_prediction_project.examples.clinical_sythetic_data import generate_synthetic_data


class TestUniversalDataAdapter(unittest.TestCase):
    """Test cases for the Universal Data Adapter."""

    @classmethod
    def setUpClass(cls):
        """Set up test data and adapter."""
        # Create a temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate synthetic data
        cls.synthetic_data = generate_synthetic_data(
            num_patients=20,
            num_days=30,
            female_percentage=0.5,
            missing_data_rate=0.1,
            anomaly_rate=0.02
        )
        
        # Save test data to temp directory
        cls.test_data_path = os.path.join(cls.temp_dir.name, 'test_data.csv')
        cls.synthetic_data.to_csv(cls.test_data_path, index=False)
        
        # Create an adapter
        cls.adapter = UniversalDataAdapter(
            data_dir=cls.temp_dir.name,
            auto_feature_selection=True,
            verbose=False
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        cls.temp_dir.cleanup()
    
    def test_load_data(self):
        """Test loading data from file."""
        data = self.adapter.load_data(self.test_data_path)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, self.synthetic_data.shape)
    
    def test_detect_schema(self):
        """Test automatic schema detection."""
        schema = self.adapter.detect_schema(self.synthetic_data)
        
        # Check that critical columns were detected
        self.assertIsNotNone(self.adapter.target_column)
        self.assertIsNotNone(self.adapter.date_column)
        
        # Check that schema contains expected data
        self.assertIn('feature_map', schema)
        self.assertIn('target_column', schema)
        self.assertIn('date_column', schema)
    
    def test_auto_select_features(self):
        """Test automatic feature selection."""
        selected_features = self.adapter.auto_select_features(
            self.synthetic_data,
            max_features=10
        )
        
        # Check that features were selected
        self.assertIsInstance(selected_features, list)
        self.assertLessEqual(len(selected_features), 10)
        
        # Check that selected_features is stored in the adapter
        self.assertEqual(selected_features, self.adapter.selected_features)
    
    def test_add_derived_features(self):
        """Test adding derived features."""
        # Add some derived features
        enriched_data = self.adapter.add_derived_features(self.synthetic_data)
        
        # Check that new features were added
        self.assertGreaterEqual(enriched_data.shape[1], self.synthetic_data.shape[1])
    
    def test_prepare_training_data(self):
        """Test preparing data for training."""
        # First detect schema and select features
        self.adapter.detect_schema(self.synthetic_data)
        self.adapter.auto_select_features(self.synthetic_data, max_features=10)
        
        # Split data into training and test sets
        training_data = self.adapter.prepare_training_data(
            self.synthetic_data,
            test_size=0.3,
            random_state=42
        )
        
        # Check that the expected components are in the result
        self.assertIn('X_train', training_data)
        self.assertIn('X_test', training_data)
        self.assertIn('y_train', training_data)
        self.assertIn('y_test', training_data)
        self.assertIn('feature_names', training_data)
        
        # Check splitting ratios
        x_train_ratio = len(training_data['X_train']) / len(self.synthetic_data)
        self.assertAlmostEqual(x_train_ratio, 0.7, delta=0.05)
    
    def test_schema_with_different_column_names(self):
        """Test schema detection with non-standard column names."""
        # Create synthetic data with different column names
        data = self.synthetic_data.copy()
        rename_map = {
            'heart_rate': 'pulse_bpm',
            'temperature': 'body_temp_celsius',
            'migraine': 'headache_occurred'
        }
        renamed_data = data.rename(columns=rename_map)
        
        # Detect schema with renamed columns
        schema = self.adapter.detect_schema(renamed_data)
        
        # Verify that the target column was detected correctly
        self.assertEqual(self.adapter.target_column, 'headache_occurred')
        
        # Verify that feature mapping worked
        feature_map = schema['feature_map']
        self.assertIn('heart_rate', feature_map)
        self.assertEqual(feature_map['heart_rate'], 'pulse_bpm')
    
    def test_meta_feature_selection(self):
        """Test meta-optimization for feature selection if available."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from meta.meta_learner import MetaLearner
            
            # Prepare data
            X = self.synthetic_data.drop(columns=['migraine'])
            y = self.synthetic_data['migraine']
            
            # Create meta-feature selector
            base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            meta_selector = MetaFeatureSelector(
                base_model=base_model,
                n_features=5,
                scoring='roc_auc',
                meta_method='de',
                surrogate='rf',
                verbose=False
            )
            
            # Run meta-feature selection
            meta_selector.fit(X, y, feature_names=X.columns.tolist())
            
            # Verify results
            self.assertIsNotNone(meta_selector.selected_features_)
            self.assertLessEqual(len(meta_selector.selected_features_), 5)
            
            # Get and verify feature importance
            importance = meta_selector.get_feature_importance()
            self.assertIsInstance(importance, dict)
            self.assertGreaterEqual(len(importance), 1)
            
        except ImportError:
            # Skip test if meta modules not available
            self.skipTest("Meta-optimization modules not available")


if __name__ == '__main__':
    unittest.main()

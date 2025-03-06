#!/usr/bin/env python3
"""
Unit tests for the migraine data handling functionality.
"""

import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import migraine data handlers
from src.migraine_model.data_handler import DataHandler
from src.migraine_model.new_data_migraine_predictor import MigrainePredictorV2


class TestDataHandler(unittest.TestCase):
    """Test the DataHandler class for import and derived features."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create test data with core columns
        self.core_data = pd.DataFrame({
            'sleep_hours': [7.5, 6.2, 8.1, 5.5],
            'stress_level': [5, 8, 3, 9],
            'weather_pressure': [1015, 1020, 1005, 1010],
            'heart_rate': [72, 80, 68, 85],
            'hormonal_level': [4.5, 6.2, 3.8, 7.1],
            'migraine_occurred': [0, 1, 0, 1]
        })
        
        # Save test data
        self.core_data_path = os.path.join(self.data_dir, "core_data.csv")
        self.core_data.to_csv(self.core_data_path, index=False)
        
        # Create test data with additional columns
        self.expanded_data = self.core_data.copy()
        self.expanded_data['screen_time_hours'] = [3.5, 7.5, 2.0, 6.0]
        self.expanded_data['hydration_ml'] = [2000, 1200, 2500, 800]
        
        # Save expanded test data
        self.expanded_data_path = os.path.join(self.data_dir, "expanded_data.csv")
        self.expanded_data.to_csv(self.expanded_data_path, index=False)
        
        # Initialize data handler
        self.data_handler = DataHandler(data_dir=self.data_dir)

    def tearDown(self):
        """Clean up test environment."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error removing temporary directory: {e}")

    def test_import_csv(self):
        """Test importing data from CSV."""
        imported_data = self.data_handler.import_data(self.core_data_path)
        self.assertEqual(len(imported_data), len(self.core_data))
        self.assertTrue(all(col in imported_data.columns for col in self.core_data.columns))

    def test_add_new_columns(self):
        """Test adding new columns to schema."""
        # First initialize with core data
        self.data_handler.import_data(self.core_data_path)
        original_columns = self.data_handler.get_feature_list()
        
        # Then import with new columns
        imported_data = self.data_handler.import_data(
            self.expanded_data_path, 
            add_new_columns=True
        )
        
        updated_columns = self.data_handler.get_feature_list()
        
        # Check new columns were added
        self.assertTrue('screen_time_hours' in updated_columns)
        self.assertTrue('hydration_ml' in updated_columns)
        self.assertTrue(len(updated_columns) > len(original_columns))
        self.assertTrue(all(col in imported_data.columns for col in self.expanded_data.columns))

    def test_derived_features(self):
        """Test adding and calculating derived features."""
        # Import data first
        imported_data = self.data_handler.import_data(self.core_data_path)
        
        # Add a derived feature
        self.data_handler.add_derived_feature(
            name="stress_sleep_ratio",
            formula="df['stress_level'] / df['sleep_hours']"
        )
        
        # Process data to calculate derived features
        processed_data = self.data_handler.process_data(imported_data)
        
        # Check derived feature was calculated
        self.assertTrue('stress_sleep_ratio' in processed_data.columns)
        
        # Check formula was applied correctly
        expected_values = self.core_data['stress_level'] / self.core_data['sleep_hours']
        pd.testing.assert_series_equal(
            processed_data['stress_sleep_ratio'],
            expected_values,
            check_names=False
        )

    def test_validate_missing_columns(self):
        """Test validating missing columns."""
        # Import data first to set up the schema
        self.data_handler.import_data(self.core_data_path)
        
        # Create data with missing required columns
        test_data = self.core_data.drop(columns=['heart_rate'])
        
        # Try to validate through process_data which checks columns
        with self.assertRaises(Exception):
            result = self.data_handler.process_data(test_data)


class TestMigrainePredictorV2(unittest.TestCase):
    """Test the MigrainePredictorV2 class for training and prediction."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.model_dir = os.path.join(self.temp_dir, "models")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create test data
        np.random.seed(42)
        n_samples = 100
        
        self.test_data = pd.DataFrame({
            'sleep_hours': np.random.normal(7, 1, n_samples),
            'stress_level': np.random.randint(1, 11, n_samples),
            'weather_pressure': np.random.normal(1013, 10, n_samples),
            'heart_rate': np.random.normal(75, 8, n_samples),
            'hormonal_level': np.random.normal(5, 2, n_samples),
            'migraine_occurred': np.random.binomial(1, 0.3, n_samples),
        })
        
        # Add some correlation between features and target
        for i in range(len(self.test_data)):
            if self.test_data.loc[i, 'stress_level'] > 7 and self.test_data.loc[i, 'sleep_hours'] < 6:
                self.test_data.loc[i, 'migraine_occurred'] = 1
                
        # Save test data
        self.test_data_path = os.path.join(self.data_dir, "test_data.csv")
        self.test_data.to_csv(self.test_data_path, index=False)
        
        # Create test data with new columns
        self.new_data = self.test_data.copy()
        self.new_data['screen_time_hours'] = np.random.normal(4.5, 2, n_samples)
        self.new_data['hydration_ml'] = np.random.normal(1500, 500, n_samples)
        
        # Save new test data
        self.new_data_path = os.path.join(self.data_dir, "new_test_data.csv")
        self.new_data.to_csv(self.new_data_path, index=False)
        
        # Initialize predictor
        self.predictor = MigrainePredictorV2(
            model_dir=self.model_dir,
            data_dir=self.data_dir
        )

    def tearDown(self):
        """Clean up test environment."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error removing temporary directory: {e}")

    def test_train_base_model(self):
        """Test training a base model with core features."""
        model_id = self.predictor.train_with_new_data(
            data_path=self.test_data_path,
            model_name="test_model",
            description="Test model with core features"
        )
        
        # Check model was created
        self.assertIsNotNone(model_id)
        
        # Check default model information exists in predictor
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.model_id)

    def test_add_new_columns_during_training(self):
        """Test adding new columns during training."""
        # First train baseline model
        self.predictor.train_with_new_data(
            data_path=self.test_data_path,
            model_name="base_model"
        )
        
        # Get initial feature columns
        initial_features = self.predictor.feature_columns.copy()
        
        # Train with new data and add new columns
        self.predictor.train_with_new_data(
            data_path=self.new_data_path,
            model_name="new_model",
            add_new_columns=True
        )
        
        # Check that new features were added
        updated_features = self.predictor.feature_columns
        self.assertTrue(len(updated_features) > len(initial_features))
        self.assertTrue('screen_time_hours' in updated_features)
        self.assertTrue('hydration_ml' in updated_features)

    def test_derived_features_in_training(self):
        """Test adding derived features and using them in training."""
        # Add derived feature
        self.predictor.add_derived_feature(
            name="stress_sleep_ratio",
            formula="df['stress_level'] / df['sleep_hours']"
        )
        
        # Train model with derived feature
        model_id = self.predictor.train_with_new_data(
            data_path=self.test_data_path,
            model_name="derived_model"
        )
        
        # Check model was created
        self.assertIsNotNone(model_id)
        
        # Check derived feature is in feature columns
        self.assertTrue('stress_sleep_ratio' in self.predictor.feature_columns)

    def test_predict_with_missing_features(self):
        """Test making predictions with missing features."""
        # First train a model
        self.predictor.train_with_new_data(
            data_path=self.new_data_path,
            model_name="complete_model",
            add_new_columns=True
        )
        
        # Create test data with missing features
        test_data = pd.DataFrame({
            'sleep_hours': [6.2, 8.1],
            'stress_level': [9, 3],
            'weather_pressure': [1020, 1005],
            # Missing 'heart_rate'
            'hormonal_level': [6.2, 4.8],
            'screen_time_hours': [7.5, 2.0],
            # Missing 'hydration_ml'
        })
        
        # Make predictions
        predictions = self.predictor.predict_with_missing_features(test_data)
        
        # Check predictions format
        self.assertEqual(len(predictions), 2)
        self.assertTrue('prediction' in predictions[0])
        self.assertTrue('probability' in predictions[0])
        self.assertTrue('feature_importances' in predictions[0])

    def test_schema_info(self):
        """Test getting schema information."""
        # First train with original data
        self.predictor.train_with_new_data(
            data_path=self.test_data_path,
            model_name="schema_test_model"
        )
        
        # Add a derived feature
        self.predictor.add_derived_feature(
            name="stress_sleep_ratio",
            formula="df['stress_level'] / df['sleep_hours']"
        )
        
        # Then train with new data and add new columns
        self.predictor.train_with_new_data(
            data_path=self.new_data_path,
            model_name="schema_test_model_2",
            add_new_columns=True
        )
        
        # Get schema info
        schema_info = self.predictor.get_schema_info()
        
        # Check schema structure
        self.assertTrue('version' in schema_info)
        self.assertTrue('core_features' in schema_info)
        self.assertTrue('optional_features' in schema_info)
        self.assertTrue('derived_features' in schema_info)
        
        # Check content
        self.assertTrue('sleep_hours' in schema_info['core_features'])
        self.assertTrue('screen_time_hours' in schema_info['optional_features'])
        self.assertTrue('stress_sleep_ratio' in schema_info['derived_features'])


if __name__ == '__main__':
    unittest.main()

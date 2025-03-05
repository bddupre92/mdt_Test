"""
Unit tests for the MigrainePredictor class.
"""

import os
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from migraine_model.migraine_predictor import MigrainePredictor


class TestMigrainePredictor(unittest.TestCase):
    """
    Test cases for the MigrainePredictor class.
    """
    
    def setUp(self):
        """
        Set up the test environment.
        """
        # Create temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        os.environ["MIGRAINE_MODELS_DIR"] = self.temp_dir
        
        # Create test data
        self.test_data = pd.DataFrame({
            "patient_id": range(1, 101),
            "date": pd.date_range(start="2024-01-01", periods=100),
            "sleep_hours": np.random.uniform(4, 10, 100),
            "stress_level": np.random.uniform(1, 10, 100),
            "weather_pressure": np.random.uniform(980, 1030, 100),
            "heart_rate": np.random.uniform(60, 100, 100),
            "hormonal_level": np.random.uniform(1, 10, 100),
            "migraine_occurred": np.random.randint(0, 2, 100),
        })
        
        # Create predictor
        self.predictor = MigrainePredictor()
    
    def tearDown(self):
        """
        Clean up the test environment.
        """
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_train_and_predict(self):
        """
        Test training a model and making predictions.
        """
        # Train model
        model_id = self.predictor.train(self.test_data)
        
        # Check if model was trained and saved
        self.assertIsNotNone(model_id)
        
        # Create test data for prediction (a smaller subset)
        test_subset = self.test_data.iloc[:3].copy()
        
        # Make predictions
        predictions = self.predictor.predict(test_subset)
        
        # Check predictions
        self.assertEqual(len(predictions), len(test_subset))
        for pred in predictions:
            self.assertIn(pred, [0, 1])
    
    def test_evaluate(self):
        """
        Test evaluating a model.
        """
        # Train model
        self.predictor.train(self.test_data)
        
        # Evaluate model
        metrics = self.predictor.evaluate(self.test_data)
        
        # Check metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("feature_importance", metrics)
    
    def test_save_and_load_model(self):
        """
        Test saving and loading a model.
        """
        # Train and save model
        model_id = self.predictor.train(self.test_data)
        
        # Create a new predictor instance
        new_predictor = MigrainePredictor()
        
        # Load the model
        new_predictor.load_model(model_id)
        
        # Create test data for prediction (a smaller subset)
        test_subset = self.test_data.iloc[:3].copy()
        
        # Make predictions with the loaded model
        predictions = new_predictor.predict(test_subset)
        
        # Check predictions
        self.assertEqual(len(predictions), len(test_subset))
        for pred in predictions:
            self.assertIn(pred, [0, 1])


if __name__ == "__main__":
    unittest.main()

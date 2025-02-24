"""
test_data_generation.py
-----------------------
Unit tests for data generation and preprocessing.
"""

import unittest
from data.generate_synthetic import generate_synthetic_data
from data.preprocessing import preprocess_data

class TestDataGeneration(unittest.TestCase):
    def test_generate_synthetic(self):
        df = generate_synthetic_data(num_days=30, random_seed=0)
        self.assertFalse(df.empty)
        self.assertIn('migraine_occurred', df.columns)
    
    def test_preprocessing(self):
        df = generate_synthetic_data(num_days=30)
        df_clean = preprocess_data(df, strategy_numeric='mean', scale_method='minmax')
        self.assertFalse(df_clean.isna().any().any(), "Should have no missing after imputation")
        # Check scaling range
        for col in df_clean.select_dtypes(include=['float','int']):
            min_val = df_clean[col].min()
            max_val = df_clean[col].max()
            self.assertTrue(min_val >= 0 and max_val <= 1 or col in ['migraine_occurred','severity'],
                            f"{col} not scaled or is a label")

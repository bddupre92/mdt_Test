"""
test_domain_integration.py
--------------------------
Ensures domain_knowledge features are added as expected.
"""

import unittest
import pandas as pd
from data.domain_knowledge import add_migraine_features

class TestDomainIntegration(unittest.TestCase):
    def test_add_migraine_features(self):
        df = pd.DataFrame({
            'sleep_hours': [4, 6, 3],
            'weather_pressure': [1010, 1006, 1002],
            'stress_level': [8, 4, 9]
        })
        out = add_migraine_features(df)
        self.assertIn('poor_sleep', out.columns)
        self.assertIn('big_pressure_drop', out.columns)
        self.assertIn('trigger_count', out.columns)
        # Check correctness
        self.assertEqual(out.loc[0,'poor_sleep'], 1)  # sleep<5
        self.assertEqual(out.loc[1,'poor_sleep'], 0)
        # big_pressure_drop from row0->row1 = 1006-1010= -4 => yes
        # etc.

"""
test_full_pipeline.py
---------------------
End-to-end integration test: generate data, preprocess, train models, evaluate,
and test drift detection in a complete pipeline.
"""

import unittest
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil
import torch
import torch.nn as nn

from data.generate_synthetic import generate_synthetic_data
from data.preprocessing import preprocess_data
from data.domain_knowledge import add_migraine_features
from models.sklearn_model import SklearnModel
from models.torch_model import TorchModel
from drift_detection.statistical import ks_drift_test
from drift_detection.performance_monitor import DDM
from optimizers.aco import AntColonyOptimizer
from meta.meta_learner import MetaLearner

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        # Generate larger dataset
        self.df = generate_synthetic_data(num_days=1000)
        self.df = self.df.dropna(subset=['migraine_occurred'])
        
        # Preprocess
        self.df_clean = preprocess_data(
            self.df, 
            strategy_numeric='mean',
            scale_method='minmax',
            exclude_cols=['migraine_occurred','severity']
        )
        self.df_feat = add_migraine_features(self.df_clean)
        
        # Prepare features
        self.features = [c for c in self.df_feat.columns 
                        if c not in ['migraine_occurred','severity']]
        self.X = self.df_feat[self.features].values
        self.y = self.df_feat['migraine_occurred'].values.astype(int)
        
        # Split indices
        self.split_idx = int(0.7 * len(self.X))
        self.X_train = self.X[:self.split_idx]
        self.y_train = self.y[:self.split_idx]
        self.X_test = self.X[self.split_idx:]
        self.y_test = self.y[self.split_idx:]
    
    def test_pipeline_performance(self):
        """Test complete pipeline with performance metrics"""
        # Initialize models
        rf_model = SklearnModel(RandomForestClassifier(n_estimators=100))
        
        # Train and evaluate
        start_time = time.time()
        rf_model.train(self.X_train, self.y_train)
        train_time = time.time() - start_time
        
        # Get predictions
        y_pred = rf_model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred)
        }
        
        # Performance assertions
        self.assertLess(train_time, 5.0, "Training too slow")
        self.assertGreater(metrics['accuracy'], 0.7, "Poor accuracy")
        self.assertGreater(metrics['f1'], 0.6, "Poor F1 score")
    
    def test_memory_efficiency(self):
        """Test memory usage during pipeline execution"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run complete pipeline
        rf_model = SklearnModel(RandomForestClassifier(n_estimators=100))
        rf_model.train(self.X_train, self.y_train)
        _ = rf_model.predict(self.X_test)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        self.assertLess(memory_increase, 500, 
                       "Pipeline memory usage too high")
    
    def test_torch_integration(self):
        """Test PyTorch model integration"""
        input_dim = self.X_train.shape[1]
        
        # Initialize PyTorch model
        net = SimpleNN(input_dim)
        torch_model = TorchModel(
            net,
            criterion=nn.BCELoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={'lr': 0.001},
            batch_size=32,
            num_epochs=5
        )
        
        # Train and evaluate
        torch_model.train(self.X_train, self.y_train)
        accuracy = accuracy_score(self.y_test, torch_model.predict(self.X_test))
        
        self.assertGreater(accuracy, 0.6, 
                          "PyTorch model performing poorly")
    
    def test_optimization_integration(self):
        """Test optimization algorithm integration"""
        def objective_func(params):
            n_estimators = int(params[0] * 190 + 10)  # 10-200 trees
            rf = RandomForestClassifier(n_estimators=n_estimators)
            rf.fit(self.X_train, self.y_train)
            return -accuracy_score(self.y_test, rf.predict(self.X_test))
        
        optimizer = AntColonyOptimizer(dim=1, bounds=[(0,1)])
        best_solution, best_score = optimizer.optimize(objective_func)
        
        self.assertLess(-best_score, 0.3, 
                       "Optimization failed to find good parameters")
    
    def test_drift_detection_integration(self):
        """Test drift detection in complete pipeline"""
        # Train initial model
        rf_model = SklearnModel(RandomForestClassifier(n_estimators=100))
        rf_model.train(self.X_train, self.y_train)
        
        # Initialize drift detectors
        stat_detector = lambda x, y: ks_drift_test(
            pd.DataFrame(x), pd.DataFrame(y), '0'
        )
        perf_detector = DDM()
        
        # Simulate concept drift
        drift_X = self.X_test.copy()
        drift_X += np.random.normal(0, 2, size=drift_X.shape)  # Add noise
        
        # Check statistical drift
        p_value = stat_detector(self.X_train, drift_X)
        self.assertLess(p_value, 0.05, "Failed to detect statistical drift")
        
        # Check performance drift
        drift_detected = False
        y_pred = rf_model.predict(drift_X)
        for i, (pred, true) in enumerate(zip(y_pred, self.y_test)):
            if perf_detector.update(pred == true):
                drift_detected = True
                break
        
        self.assertTrue(drift_detected, 
                       "Failed to detect performance drift")
    
    def test_meta_learner_integration(self):
        """Test meta-learner in complete pipeline"""
        # Create multiple optimizers
        optimizers = [
            AntColonyOptimizer(dim=1, bounds=[(0,1)]),
            AntColonyOptimizer(dim=1, bounds=[(0,1)])  # Different config
        ]
        
        ml = MetaLearner(method='bayesian')
        ml.set_algorithms(optimizers)
        
        def objective_func(params):
            n_estimators = int(params[0] * 190 + 10)
            rf = RandomForestClassifier(n_estimators=n_estimators)
            rf.fit(self.X_train, self.y_train)
            return -accuracy_score(self.y_test, rf.predict(self.X_test))
        
        # Run optimization with meta-learner
        context = {'data_size': len(self.X_train)}
        optimizer = ml.select_algorithm(context)
        solution, score = optimizer.optimize(objective_func)
        
        self.assertIsNotNone(solution, "Meta-learner failed to select optimizer")
        self.assertLess(-score, 0.3, "Meta-learner solution performing poorly")

"""
Model manager for migraine prediction system.
"""
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class ModelManager:
    """Manage model lifecycle including training, prediction, and adaptation."""
    
    def __init__(self, config):
        """Initialize model manager.
        
        Args:
            config: Configuration object containing model parameters and thresholds
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = {}
        self.last_training_time = None
        
        # Drift detection state
        self.drift_buffer = []
        self.reference_stats = {}
        self.drift_scores = []  # Track drift scores over time
        self.drift_trends = {}  # Track drift trends per feature
        self.last_drift_time = None
        self.ema_alpha = 0.3  # EMA smoothing factor
        
        logger.info(f"Initialized ModelManager with config: {config}")
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train model on provided data.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Dict containing training results
        """
        logger.info(f"Starting model training with data shape: X={X.shape}, y={y.shape}")
        logger.debug(f"Feature statistics:\n{X.describe().to_dict()}")
        
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        logger.debug(f"Scaled feature statistics:\n{pd.DataFrame(X_scaled, columns=X.columns).describe().to_dict()}")
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=self.config.model_params.n_estimators,
            max_depth=self.config.model_params.max_depth,
            min_samples_split=self.config.model_params.min_samples_split,
            random_state=self.config.model_params.random_state
        )
        
        logger.info(f"Training RandomForestClassifier with params: {self.model.get_params()}")
        
        # Train model
        try:
            self.model.fit(X_scaled, y)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
        
        # Calculate cross-validation score
        try:
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=self.config.training_params.cross_validation_folds,
                scoring='roc_auc'
            )
            logger.info(f"Cross-validation ROC-AUC scores: mean={np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            raise
        
        # Get feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        logger.debug(f"Feature importance: {json.dumps(self.feature_importance, indent=2)}")
        
        # Record training time
        self.last_training_time = datetime.now()
        
        # Initialize reference statistics
        self._update_reference_stats(X)
        logger.debug(f"Updated reference statistics: {json.dumps(self.reference_stats, indent=2)}")
        
        # Reset drift detection state
        self.drift_buffer = []
        self.drift_scores = []
        self.drift_trends = {}
        self.last_drift_time = None
        
        results = {
            'status': 'success',
            'cv_score_mean': float(np.mean(cv_scores)),
            'cv_score_std': float(np.std(cv_scores)),
            'selected_features': self.feature_names,
            'feature_importance': self.feature_importance,
            'best_params': self.model.get_params()
        }
        
        logger.info(f"Training completed with results: {json.dumps(results, indent=2)}")
        return results
    
    def predict(self, features: Union[pd.DataFrame, Dict[str, float]]) -> Dict[str, Any]:
        """Make prediction.
        
        Args:
            features: Feature data as DataFrame or dict
            
        Returns:
            Dict containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        logger.debug(f"Input features shape: {features_df.shape}")
        logger.debug(f"Feature statistics: {features_df.describe().to_dict()}")
            
        # Ensure all required features are present
        missing_features = [f for f in self.feature_names if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Scale features
        try:
            X_scaled = self.scaler.transform(features_df[self.feature_names])
            logger.debug(f"Scaled features shape: {X_scaled.shape}")
        except Exception as e:
            logger.error(f"Feature scaling failed: {str(e)}")
            raise ValueError(f"Feature scaling failed: {str(e)}")
            
        # Make prediction
        try:
            proba = self.model.predict_proba(X_scaled)[:, 1]
            prediction = (proba > 0.5).astype(int)
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Check for drift
            drift_result = self.check_drift(features_df)
            
            result = {
                'probability': float(proba[0]),
                'prediction': bool(prediction[0]),
                'drift_detected': drift_result['drift_detected'],
                'drift_severity': drift_result['severity'],
                'drift_features': drift_result['drift_features'],
                'feature_importance': feature_importance
            }
            
            logger.info(f"Prediction: {result['prediction']}, Probability: {result['probability']:.3f}")
            if result['drift_detected']:
                logger.warning(f"Drift detected with severity {result['drift_severity']:.3f} in features: {result['drift_features']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def _update_reference_stats(self, data: pd.DataFrame):
        """Update reference statistics for drift detection."""
        for feature in self.feature_names:
            self.reference_stats[feature] = {
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std()),
                'q25': float(data[feature].quantile(0.25)),
                'q75': float(data[feature].quantile(0.75))
            }
    
    def check_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for concept drift in new data.
        
        Args:
            new_data: New feature data to check for drift
            
        Returns:
            Dict containing drift detection results
        """
        if len(self.drift_buffer) < self.config.drift_params.window_size:
            self.drift_buffer.append(new_data)
            logger.debug(f"Building drift buffer: {len(self.drift_buffer)}/{self.config.drift_params.window_size}")
            return {'drift_detected': False, 'severity': 0.0, 'drift_features': []}
            
        # Compare distributions
        reference = pd.concat(self.drift_buffer)
        drift_features = []
        feature_severities = {}
        
        for feature in self.feature_names:
            # Calculate drift indicators
            ref_data = reference[feature].values
            new_data_values = new_data[feature].values
            
            # 1. KS test
            ks_stat, p_value = stats.ks_2samp(ref_data, new_data_values)
            
            # 2. Mean shift
            mean_shift = abs(np.mean(new_data_values) - np.mean(ref_data)) / (np.std(ref_data) + 1e-6)
            mean_shift_score = np.tanh(mean_shift / 2)  # Squash large values
            
            # Calculate combined severity
            severity = 0.6 * mean_shift_score + 0.4 * ks_stat
            
            # Update feature trend
            if feature not in self.drift_trends:
                self.drift_trends[feature] = []
            self.drift_trends[feature].append(severity)
            
            # Calculate trend if enough points
            trend = 0
            if len(self.drift_trends[feature]) >= 5:
                recent_scores = self.drift_trends[feature][-5:]
                try:
                    trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0] * 1000
                except:
                    trend = 0
            
            logger.debug(f"Feature {feature} drift metrics: "
                        f"KS stat={ks_stat:.3f}, p-value={p_value:.3e}, "
                        f"mean_shift={mean_shift:.3f}, severity={severity:.3f}, "
                        f"trend={trend:.3f}")
            
            # Check drift conditions
            drift_detected = False
            if mean_shift > 1.5 and p_value < 0.05:  # Primary condition - relaxed thresholds
                drift_detected = True
            elif p_value < 1e-8 and ks_stat > 0.4:  # Secondary condition - relaxed thresholds
                drift_detected = True
            elif mean_shift > 2.0:  # Additional condition for strong mean shifts
                drift_detected = True
            
            if drift_detected:
                drift_features.append(feature)
                feature_severities[feature] = severity
        
        # Overall drift detection
        drift_detected = len(drift_features) > 0
        overall_severity = max(feature_severities.values()) if feature_severities else 0
        
        # Update drift buffer
        if drift_detected:
            if self.last_drift_time is None or len(self.drift_buffer) >= 40:  # Minimum interval
                self.drift_buffer = []  # Reset buffer
                self._update_reference_stats(new_data)
                self.last_drift_time = len(self.drift_buffer)
                logger.warning(f"Drift detected with severity {overall_severity:.3f} in features: {drift_features}")
            else:
                drift_detected = False  # Suppress drift if too soon
        else:
            self.drift_buffer.append(new_data)
            if len(self.drift_buffer) > self.config.drift_params.window_size:
                self.drift_buffer.pop(0)
                self._update_reference_stats(pd.concat(self.drift_buffer))
        
        return {
            'drift_detected': drift_detected,
            'severity': overall_severity,
            'drift_features': drift_features,
            'feature_severities': feature_severities
        }
    
    def save_model(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'last_training_time': self.last_training_time,
            'config': self.config,
            'reference_stats': self.reference_stats
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.last_training_time = model_data['last_training_time']
        self.config = model_data['config']
        self.reference_stats = model_data.get('reference_stats', {})
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.feature_importance:
            raise ValueError("Model not trained yet")
        return self.feature_importance.copy()

"""
Personalization Layer Module

This module implements the personalization layer for the MoE framework, enabling
adaptation to individual patient profiles. It builds on the existing explainability
framework and integrates with the gating network to adjust expert weights based on
patient-specific characteristics.
"""
import os
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
from utils.json_utils import NumpyEncoder
from explainability import adapt_explainer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonalizationLayer:
    """
    Implements the personalization layer for patient adaptation in the MoE framework.
    
    This layer sits between the gating network and the expert models, adjusting the
    weights produced by the gating network based on patient-specific profiles and
    historical response patterns. It also provides adaptive thresholds and contextual
    adjustments based on patient-specific patterns.
    """
    
    def __init__(self, 
                 results_dir: str = 'results/personalization',
                 adaptation_rate: float = 0.2,
                 profile_update_threshold: float = 0.1,
                 base_prediction_threshold: float = 0.5,
                 threshold_adaptation_rate: float = 0.1):
        """
        Initialize the personalization layer.
        
        Parameters:
        -----------
        results_dir : str
            Directory to store personalization results and patient profiles
        adaptation_rate : float
            Rate at which to adapt the expert weights (0.0-1.0)
        profile_update_threshold : float
            Threshold for updating patient profiles based on new observations
        base_prediction_threshold : float
            Base threshold for binary classification predictions
        threshold_adaptation_rate : float
            Rate at which to adapt prediction thresholds (0.0-1.0)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir = self.results_dir / "patient_profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.adaptation_rate = adaptation_rate
        self.profile_update_threshold = profile_update_threshold
        self.base_prediction_threshold = base_prediction_threshold
        self.threshold_adaptation_rate = threshold_adaptation_rate
        
        self.patient_profiles = {}
        self.feature_importance_cache = {}
        self.response_history = {}
        self.adaptive_thresholds = {}
        self.threshold_history = {}
        self.contextual_factors = {}
        
        # Load existing profiles if available
        self._load_existing_profiles()
    
    def _load_existing_profiles(self):
        """Load existing patient profiles from disk."""
        profile_files = list(self.profiles_dir.glob("*.json"))
        for profile_file in profile_files:
            try:
                patient_id = profile_file.stem
                with open(profile_file, 'r') as f:
                    self.patient_profiles[patient_id] = json.load(f)
                logger.info(f"Loaded profile for patient {patient_id}")
            except Exception as e:
                logger.warning(f"Error loading profile {profile_file}: {str(e)}")
    
    def _save_patient_profile(self, patient_id: str):
        """Save patient profile to disk."""
        if patient_id not in self.patient_profiles:
            logger.warning(f"No profile exists for patient {patient_id}")
            return
        
        profile_path = self.profiles_dir / f"{patient_id}.json"
        with open(profile_path, 'w') as f:
            json.dump(self.patient_profiles[patient_id], f, cls=NumpyEncoder, indent=2)
        logger.info(f"Saved profile for patient {patient_id}")
    
    def has_patient_profile(self, patient_id: str) -> bool:
        """Check if a patient profile exists.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
            
        Returns:
        --------
        bool
            True if the patient profile exists, False otherwise
        """
        return patient_id in self.patient_profiles
    
    def register_patient_profile(self, patient_id: str, profile_data: Dict[str, Any]) -> None:
        """Register a patient profile for personalized adaptations.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        profile_data : Dict[str, Any]
            Patient profile data including demographics and preferences
        """
        if patient_id in self.patient_profiles:
            # Update existing profile
            self.patient_profiles[patient_id].update(profile_data)
            logger.info(f"Updated existing profile for patient {patient_id}")
        else:
            # Create new profile
            self.patient_profiles[patient_id] = {
                "patient_id": patient_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                **profile_data
            }
            logger.info(f"Created new profile for patient {patient_id}")
        
        # Save the profile
        self._save_patient_profile(patient_id)
    
    def adapt_quality_scores(self, quality_scores: Dict[str, float], patient_id: str, data: pd.DataFrame) -> Dict[str, float]:
        """Adapt quality scores based on patient-specific characteristics.
        
        Parameters:
        -----------
        quality_scores : Dict[str, float]
            Original quality scores
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Input data for the current prediction
            
        Returns:
        --------
        Dict[str, float]
            Adapted quality scores
        """
        if not self.has_patient_profile(patient_id):
            logger.warning(f"No profile exists for patient {patient_id}, returning original scores")
            return quality_scores
        
        profile = self.patient_profiles[patient_id]
        adapted_scores = quality_scores.copy()
        
        # Apply adaptations based on patient profile
        if 'response_patterns' in profile:
            patterns = profile['response_patterns']
            
            # Apply adaptations based on known response patterns
            if 'completeness_sensitivity' in patterns:
                if 'completeness' in adapted_scores:
                    factor = patterns['completeness_sensitivity']
                    adapted_scores['completeness'] = min(1.0, adapted_scores['completeness'] * factor)
            
            if 'consistency_sensitivity' in patterns:
                if 'consistency' in adapted_scores:
                    factor = patterns['consistency_sensitivity']
                    adapted_scores['consistency'] = min(1.0, adapted_scores['consistency'] * factor)
            
            if 'timeliness_sensitivity' in patterns:
                if 'timeliness' in adapted_scores:
                    factor = patterns['timeliness_sensitivity']
                    adapted_scores['timeliness'] = min(1.0, adapted_scores['timeliness'] * factor)
        
        # Record adaptation in history
        if 'adaptation_history' not in profile:
            profile['adaptation_history'] = []
        
        profile['adaptation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'original_scores': quality_scores,
            'adapted_scores': adapted_scores
        })
        
        # Trim history if it gets too long
        if len(profile['adaptation_history']) > 100:
            profile['adaptation_history'] = profile['adaptation_history'][-100:]
        
        # Update profile's updated_at timestamp
        profile['updated_at'] = datetime.now().isoformat()
        
        # Save the updated profile
        self._save_patient_profile(patient_id)
        
        return adapted_scores
    
    def get_quality_adaptation_factors(self, patient_id: str, data: pd.DataFrame) -> Dict[str, float]:
        """Get quality adaptation factors for a specific patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Input data for the current prediction
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of adaptation factors for each quality metric
        """
        if not self.has_patient_profile(patient_id):
            # Return default factors (no adaptation)
            return {'completeness': 1.0, 'consistency': 1.0, 'timeliness': 1.0}
        
        profile = self.patient_profiles[patient_id]
        adaptation_factors = {'completeness': 1.0, 'consistency': 1.0, 'timeliness': 1.0}
        
        # Apply adaptations based on patient profile
        if 'response_patterns' in profile:
            patterns = profile['response_patterns']
            
            # Get adaptation factors from response patterns
            if 'completeness_sensitivity' in patterns:
                adaptation_factors['completeness'] = patterns['completeness_sensitivity']
            
            if 'consistency_sensitivity' in patterns:
                adaptation_factors['consistency'] = patterns['consistency_sensitivity']
            
            if 'timeliness_sensitivity' in patterns:
                adaptation_factors['timeliness'] = patterns['timeliness_sensitivity']
        
        # Apply additional context-specific adaptations based on current data
        # This could analyze the current data and adjust factors accordingly
        # For example, if we detect that the patient has more missing values than usual,
        # we might increase the completeness factor to make the system more sensitive to completeness issues
        
        # For demonstration, we'll just use a simple heuristic based on data characteristics
        if data is not None and not data.empty:
            # Check for missing values
            missing_rate = data.isnull().mean().mean()
            if missing_rate > 0.2:  # If more than 20% missing values
                adaptation_factors['completeness'] *= 1.2  # Increase completeness sensitivity
            
            # Check for data consistency (using standard deviation as a proxy)
            try:
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    avg_std = data[numeric_cols].std().mean()
                    if avg_std > 1.5:  # If high standard deviation
                        adaptation_factors['consistency'] *= 1.2  # Increase consistency sensitivity
            except Exception:
                pass  # Ignore errors in consistency calculation
        
        # Record this adaptation in the profile
        if 'adaptation_factors_history' not in profile:
            profile['adaptation_factors_history'] = []
        
        profile['adaptation_factors_history'].append({
            'timestamp': datetime.now().isoformat(),
            'factors': adaptation_factors
        })
        
        # Trim history if it gets too long
        if len(profile['adaptation_factors_history']) > 100:
            profile['adaptation_factors_history'] = profile['adaptation_factors_history'][-100:]
        
        # Update profile's updated_at timestamp
        profile['updated_at'] = datetime.now().isoformat()
        
        # Save the updated profile
        self._save_patient_profile(patient_id)
        
        return adaptation_factors
    
    def create_patient_profile(self, 
                              patient_id: str, 
                              demographic_data: Dict[str, Any] = None,
                              initial_data: pd.DataFrame = None,
                              explainer = None) -> Dict[str, Any]:
        """
        Create a new patient profile or update an existing one with initial data.
        
        Parameters:
        -----------
        patient_id : str
            Unique identifier for the patient
        demographic_data : Dict[str, Any], optional
            Demographic information about the patient
        initial_data : pd.DataFrame, optional
            Initial data for feature analysis
        explainer : object, optional
            Explainability engine to analyze feature importance
            
        Returns:
        --------
        Dict[str, Any]
            The created or updated patient profile
        """
        # Initialize profile if it doesn't exist
        if patient_id not in self.patient_profiles:
            logger.info(f"Creating new profile for patient {patient_id}")
            self.patient_profiles[patient_id] = {
                "patient_id": patient_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "demographics": demographic_data or {},
                "feature_importance": {},
                "expert_preferences": {},
                "response_patterns": {},
                "adaptation_history": []
            }
        
        # Update with demographic data if provided
        if demographic_data:
            self.patient_profiles[patient_id]["demographics"] = demographic_data
            self.patient_profiles[patient_id]["updated_at"] = datetime.now().isoformat()
        
        # Analyze initial data if provided
        if initial_data is not None and explainer is not None:
            self._analyze_patient_data(patient_id, initial_data, explainer)
        
        # Save the profile
        self._save_patient_profile(patient_id)
        
        return self.patient_profiles[patient_id]
    
    def _analyze_patient_data(self, 
                             patient_id: str, 
                             data: pd.DataFrame,
                             explainer) -> Dict[str, Any]:
        """
        Analyze patient data to extract feature importance and response patterns.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        data : pd.DataFrame
            Patient data for analysis
        explainer : object
            Explainability engine to analyze feature importance
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        if patient_id not in self.patient_profiles:
            logger.warning(f"No profile exists for patient {patient_id}, creating one")
            self.create_patient_profile(patient_id)
        
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "feature_importance": {},
            "detected_patterns": {}
        }
        
        # Extract feature importance using the explainer
        try:
            # Create a callable mock model for explainability analysis
            class MockModel:
                def __init__(self):
                    # Add feature_importances_ attribute for explainers that look for it
                    self.feature_importances_ = {feature: 0.1 for feature in data.columns}
                    self.feature_names_in_ = list(data.columns)
                
                def __call__(self, X):
                    """Make model callable for SHAP compatibility."""
                    # Return random probabilities between 0.0 and 1.0
                    if isinstance(X, pd.DataFrame):
                        return np.random.random(len(X))
                    else:
                        return np.random.random(len(X))
                        
                def predict_proba(self, X):
                    """Predict probabilities for both classes."""
                    probs = []
                    for _ in range(len(X)):
                        p = np.random.random() * 0.5  # Keep probabilities reasonable
                        probs.append([1-p, p])  # [no_migraine, migraine]
                    return np.array(probs)
            
            mock_model = MockModel()
            
            # Adapt the explainer if it doesn't have explain_model method
            if hasattr(explainer, 'explain_model'):
                adapted_explainer = explainer
            else:
                adapted_explainer = adapt_explainer(explainer)
                
            importance = adapted_explainer.explain_model(mock_model, data)
            
            if isinstance(importance, dict):
                # Extract the actual feature importance values if nested
                feature_imp = importance.get('feature_importance', importance)
                if not isinstance(feature_imp, dict):
                    feature_imp = importance  # Fallback if structure is unexpected
                
                # Store feature importance in profile
                self.patient_profiles[patient_id]["feature_importance"] = feature_imp
                analysis_results["feature_importance"] = feature_imp
                
                # Cache feature importance for quick access
                self.feature_importance_cache[patient_id] = feature_imp
                
                # Detect feature categories (physiological, environmental, behavioral)
                feature_categories = {
                    "physiological": sum(importance.get(f, 0) for f in importance if "physio" in f),
                    "environmental": sum(importance.get(f, 0) for f in importance 
                                     if f in ["temperature", "humidity", "barometric_pressure"]),
                    "behavioral": sum(importance.get(f, 0) for f in importance if "behavior" in f)
                }
                
                # Normalize category importance
                category_sum = sum(feature_categories.values())
                if category_sum > 0:
                    feature_categories = {k: v/category_sum for k, v in feature_categories.items()}
                
                # Store category importance in profile
                self.patient_profiles[patient_id]["feature_categories"] = feature_categories
                analysis_results["feature_categories"] = feature_categories
                
                # Update expert preferences based on feature categories
                # Higher weight for experts that specialize in important feature categories
                expert_preferences = {}
                for category, importance in feature_categories.items():
                    if importance > 0.2:  # Only consider significant categories
                        expert_preferences[category] = importance
                
                self.patient_profiles[patient_id]["expert_preferences"] = expert_preferences
                analysis_results["expert_preferences"] = expert_preferences
        except Exception as e:
            logger.warning(f"Error analyzing feature importance: {str(e)}")
        
        # Detect response patterns (this would be more sophisticated in a real implementation)
        # For now, we'll implement a basic version that looks for temporal patterns
        try:
            if 'timestamp' in data.columns and 'target' in data.columns:
                # Store historical responses for pattern detection
                self.response_history.setdefault(patient_id, [])
                
                for _, row in data.iterrows():
                    self.response_history[patient_id].append({
                        'timestamp': row['timestamp'],
                        'target': row['target']
                    })
                
                # Limit history size
                self.response_history[patient_id] = self.response_history[patient_id][-100:]
                
                # Simple pattern detection (e.g., increasing trend, decreasing trend)
                if len(self.response_history[patient_id]) >= 3:
                    recent = self.response_history[patient_id][-3:]
                    targets = [r['target'] for r in recent]
                    
                    if all(targets[i] < targets[i+1] for i in range(len(targets)-1)):
                        pattern = "increasing"
                    elif all(targets[i] > targets[i+1] for i in range(len(targets)-1)):
                        pattern = "decreasing"
                    else:
                        pattern = "fluctuating"
                    
                    self.patient_profiles[patient_id]["response_patterns"]["trend"] = pattern
                    analysis_results["detected_patterns"]["trend"] = pattern
        except Exception as e:
            logger.warning(f"Error detecting response patterns: {str(e)}")
        
        # Update profile
        self.patient_profiles[patient_id]["updated_at"] = datetime.now().isoformat()
        self._save_patient_profile(patient_id)
        
        return analysis_results
    
    def personalize_expert_weights(self, 
                                 patient_id: str, 
                                 base_weights: Dict[int, float],
                                 features: pd.DataFrame = None) -> Dict[int, float]:
        """
        Adjust expert weights based on patient profile.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        base_weights : Dict[int, float]
            Base weights from the gating network
        features : pd.DataFrame, optional
            Current features for contextual adaptation
            
        Returns:
        --------
        Dict[int, float]
            Personalized expert weights
        """
        if patient_id not in self.patient_profiles:
            logger.warning(f"No profile for patient {patient_id}, using base weights")
            return base_weights
        
        # Get patient profile
        profile = self.patient_profiles[patient_id]
        
        # Initialize personalized weights with base weights
        personalized_weights = dict(base_weights)
        
        # Apply expert preferences from patient profile
        expert_preferences = profile.get("expert_preferences", {})
        if expert_preferences:
            # Map expert IDs to their specialties
            expert_specialties = {
                0: "physiological",  # Assuming expert 0 specializes in physiological features
                1: "environmental",   # Assuming expert 1 specializes in environmental features
                2: "behavioral"       # Assuming expert 2 specializes in behavioral features
            }
            
            # Adjust weights based on preferences
            for expert_id, specialty in expert_specialties.items():
                if expert_id in personalized_weights and specialty in expert_preferences:
                    # Increase weight for experts that match patient's important feature categories
                    preference_factor = 1.0 + (expert_preferences[specialty] * self.adaptation_rate)
                    personalized_weights[expert_id] *= preference_factor
            
            # Normalize weights to sum to 1
            weight_sum = sum(personalized_weights.values())
            if weight_sum > 0:
                personalized_weights = {k: v/weight_sum for k, v in personalized_weights.items()}
        
        # Log the personalization
        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "base_weights": base_weights,
            "personalized_weights": personalized_weights
        }
        
        # Record the adaptation
        profile.setdefault("adaptation_history", []).append(adaptation_record)
        
        # Limit history size
        if len(profile["adaptation_history"]) > 20:
            profile["adaptation_history"] = profile["adaptation_history"][-20:]
        
        # Update profile
        profile["updated_at"] = datetime.now().isoformat()
        self._save_patient_profile(patient_id)
        
        return personalized_weights
    
    def update_profile_from_feedback(self, 
                                   patient_id: str, 
                                   feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update patient profile based on explicit feedback.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        feedback : Dict[str, Any]
            Feedback data to incorporate into profile
            
        Returns:
        --------
        Dict[str, Any]
            Updated patient profile
        """
        if patient_id not in self.patient_profiles:
            logger.warning(f"No profile for patient {patient_id}, creating one")
            self.create_patient_profile(patient_id)
        
        profile = self.patient_profiles[patient_id]
        
        # Update profile with feedback
        if "trigger_sensitivity" in feedback:
            profile.setdefault("trigger_sensitivity", {}).update(feedback["trigger_sensitivity"])
        
        if "symptom_severity" in feedback:
            profile.setdefault("symptom_severity", {}).update(feedback["symptom_severity"])
        
        if "treatment_effectiveness" in feedback:
            profile.setdefault("treatment_effectiveness", {}).update(feedback["treatment_effectiveness"])
        
        # Record the feedback event
        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "feedback_data": feedback
        }
        
        profile.setdefault("feedback_history", []).append(feedback_record)
        
        # Limit history size
        if len(profile.get("feedback_history", [])) > 20:
            profile["feedback_history"] = profile["feedback_history"][-20:]
        
        # Update profile
        profile["updated_at"] = datetime.now().isoformat()
        self._save_patient_profile(patient_id)
        
        return profile
    
    def get_personalization_metrics(self, patient_id: str) -> Dict[str, Any]:
        """
        Get metrics about the personalization effectiveness for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
            
        Returns:
        --------
        Dict[str, Any]
            Personalization metrics
        """
        if patient_id not in self.patient_profiles:
            logger.warning(f"No profile for patient {patient_id}")
            return {}
        
        profile = self.patient_profiles[patient_id]
        
        # Extract adaptation history
        adaptation_history = profile.get("adaptation_history", [])
        threshold_history = self.threshold_history.get(patient_id, [])
        
        # Calculate metrics
        metrics = {
            "profile_age_days": 0,
            "adaptation_count": len(adaptation_history),
            "avg_weight_adjustment": 0.0,
            "personalization_impact": 0.0,
            "threshold_adaptations": len(threshold_history),
            "avg_threshold_adjustment": 0.0,
            "current_threshold": self.get_adaptive_threshold(patient_id)
        }
        
        # Calculate profile age
        if "created_at" in profile:
            created_at = datetime.fromisoformat(profile["created_at"])
            metrics["profile_age_days"] = (datetime.now() - created_at).days
        
        # Calculate average weight adjustment and personalization impact
        if adaptation_history:
            weight_diffs = []
            for record in adaptation_history:
                base = record.get("base_weights", {})
                pers = record.get("personalized_weights", {})
                
                # Calculate weight differences for each expert
                for expert_id in base:
                    if expert_id in pers:
                        weight_diffs.append(abs(pers[expert_id] - base[expert_id]))
            
            if weight_diffs:
                metrics["avg_weight_adjustment"] = sum(weight_diffs) / len(weight_diffs)
                
                # Simple metric to quantify personalization impact
                metrics["personalization_impact"] = metrics["avg_weight_adjustment"] * 100
        
        # Calculate threshold adaptation metrics
        if threshold_history:
            threshold_adjustments = [abs(record["new_threshold"] - record["base_threshold"]) 
                                    for record in threshold_history]
            
            if threshold_adjustments:
                metrics["avg_threshold_adjustment"] = sum(threshold_adjustments) / len(threshold_adjustments)
        
        return metrics
    
    def get_adaptive_threshold(self, patient_id: str, features: pd.DataFrame = None) -> float:
        """
        Get an adaptive threshold for the patient based on their profile.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        features : pd.DataFrame, optional
            Current features for contextual adaptation
            
        Returns:
        --------
        float
            Personalized prediction threshold
        """
        # Start with the base threshold
        if patient_id in self.adaptive_thresholds:
            threshold = self.adaptive_thresholds[patient_id]
        else:
            threshold = self.base_prediction_threshold
            
        # Apply contextual adjustments if features are provided
        if features is not None and not features.empty:
            contextual_adjustment = self._calculate_contextual_adjustment(patient_id, features)
            threshold += contextual_adjustment
            
        # Ensure threshold is between 0 and 1
        threshold = max(0.1, min(0.9, threshold))
        
        return threshold
    
    def update_adaptive_threshold(self, patient_id: str, prediction_result: Dict[str, Any]) -> float:
        """
        Update the adaptive threshold based on prediction results.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        prediction_result : Dict[str, Any]
            Results from prediction including true/false positives/negatives
            
        Returns:
        --------
        float
            Updated threshold value
        """
        if patient_id not in self.patient_profiles:
            logger.warning(f"No profile for patient {patient_id}, using base threshold")
            return self.base_prediction_threshold
            
        # Get current threshold
        current_threshold = self.get_adaptive_threshold(patient_id)
        
        # Extract prediction performance
        true_positives = prediction_result.get('true_positives', 0)
        false_positives = prediction_result.get('false_positives', 0)
        true_negatives = prediction_result.get('true_negatives', 0)
        false_negatives = prediction_result.get('false_negatives', 0)
        
        # Calculate metrics
        total_predictions = true_positives + false_positives + true_negatives + false_negatives
        if total_predictions == 0:
            return current_threshold
            
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Determine adjustment direction and magnitude
        # If high false positives, increase threshold to improve precision
        # If high false negatives, decrease threshold to improve recall
        adjustment = 0.0
        
        if false_positives > false_negatives and precision < 0.7:
            # Too many false alarms, increase threshold
            adjustment = self.threshold_adaptation_rate * (false_positives / total_predictions)
        elif false_negatives > false_positives and recall < 0.7:
            # Too many missed events, decrease threshold
            adjustment = -self.threshold_adaptation_rate * (false_negatives / total_predictions)
        
        # Apply adjustment
        new_threshold = current_threshold + adjustment
        
        # Ensure threshold is between 0.1 and 0.9
        new_threshold = max(0.1, min(0.9, new_threshold))
        
        # Record the threshold update
        threshold_record = {
            "timestamp": datetime.now().isoformat(),
            "base_threshold": current_threshold,
            "adjustment": adjustment,
            "new_threshold": new_threshold,
            "precision": precision,
            "recall": recall,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
        
        # Store the update in history
        if patient_id not in self.threshold_history:
            self.threshold_history[patient_id] = []
        self.threshold_history[patient_id].append(threshold_record)
        
        # Limit history size
        if len(self.threshold_history[patient_id]) > 20:
            self.threshold_history[patient_id] = self.threshold_history[patient_id][-20:]
        
        # Update the stored threshold
        self.adaptive_thresholds[patient_id] = new_threshold
        
        # Add to patient profile
        profile = self.patient_profiles[patient_id]
        profile["current_threshold"] = new_threshold
        profile["updated_at"] = datetime.now().isoformat()
        self._save_patient_profile(patient_id)
        
        return new_threshold
    
    def _calculate_contextual_adjustment(self, patient_id: str, features: pd.DataFrame) -> float:
        """
        Calculate contextual adjustment for threshold based on current features.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        features : pd.DataFrame
            Current feature values
            
        Returns:
        --------
        float
            Contextual adjustment value for threshold
        """
        if features.empty:
            return 0.0
            
        # Initialize adjustment
        adjustment = 0.0
        
        # Get patient profile
        if patient_id not in self.patient_profiles:
            return adjustment
            
        profile = self.patient_profiles[patient_id]
        
        # Get feature importance from profile
        feature_importance = profile.get("feature_importance", {})
        if not feature_importance:
            return adjustment
            
        # Check for important features in extreme values
        for feature, importance in feature_importance.items():
            if feature in features.columns and importance > 0.1:
                # Get the current value
                current_value = features[feature].iloc[0] if not features[feature].empty else None
                
                if current_value is not None:
                    # Check if we have historical values for this feature
                    history_key = f"{feature}_history"
                    if history_key in profile and profile[history_key]:
                        history = profile[history_key]
                        
                        # Calculate percentile of current value in history
                        if len(history) >= 5:
                            sorted_history = sorted(history)
                            min_val = sorted_history[0]
                            max_val = sorted_history[-1]
                            
                            if max_val > min_val:
                                # Normalize the value to 0-1 range
                                normalized = (current_value - min_val) / (max_val - min_val)
                                
                                # If value is in extreme ranges (top or bottom 10%), adjust threshold
                                if normalized > 0.9:  # High extreme
                                    # For high risk features, decrease threshold when values are high
                                    adjustment -= importance * 0.05
                                elif normalized < 0.1:  # Low extreme
                                    # For high risk features, increase threshold when values are low
                                    adjustment += importance * 0.05
        
        # Store the contextual factors for this patient
        if patient_id not in self.contextual_factors:
            self.contextual_factors[patient_id] = []
            
        # Record the contextual adjustment
        contextual_record = {
            "timestamp": datetime.now().isoformat(),
            "features": {f: features[f].iloc[0] for f in features.columns 
                       if f in feature_importance and not pd.isna(features[f].iloc[0])},
            "adjustment": adjustment
        }
        
        self.contextual_factors[patient_id].append(contextual_record)
        
        # Limit history size
        if len(self.contextual_factors[patient_id]) > 20:
            self.contextual_factors[patient_id] = self.contextual_factors[patient_id][-20:]
            
        return adjustment
    
    def apply_contextual_adjustments(self, patient_id: str, 
                                    prediction_proba: float, 
                                    features: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply contextual adjustments to prediction probability based on current features.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        prediction_proba : float
            Original prediction probability
        features : pd.DataFrame
            Current feature values
            
        Returns:
        --------
        Dict[str, Any]
            Adjusted prediction information
        """
        if patient_id not in self.patient_profiles or features.empty:
            return {"adjusted_proba": prediction_proba, "adjustment": 0.0, "reasons": []}
            
        profile = self.patient_profiles[patient_id]
        feature_importance = profile.get("feature_importance", {})
        
        # Initialize adjustment and reasons
        proba_adjustment = 0.0
        adjustment_reasons = []
        
        # Get personalized threshold
        threshold = self.get_adaptive_threshold(patient_id)
        
        # Analyze features for contextual adjustments
        if feature_importance and not features.empty:
            # Ensure feature_importance is a flat dictionary of feature->importance values
            if isinstance(feature_importance, dict) and 'feature_importance' in feature_importance:
                feature_importance = feature_importance.get('feature_importance', {})
                
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Consider top 5 important features for adjustments
            for feature, importance in sorted_features[:5]:
                if feature in features.columns and importance > 0.05:
                    # Get the current value
                    if not features[feature].empty:
                        current_value = features[feature].iloc[0]
                        
                        # Check if we have history for this feature in profile
                        history_key = f"{feature}_history"
                        if history_key in profile and len(profile[history_key]) >= 5:
                            history = profile[history_key]
                            
                            # Check if current value is extreme compared to history
                            sorted_history = sorted(history)
                            percentile_25 = sorted_history[int(len(sorted_history) * 0.25)]
                            percentile_75 = sorted_history[int(len(sorted_history) * 0.75)]
                            feature_range = percentile_75 - percentile_25
                            
                            if feature_range > 0:
                                # Detect if value is significantly outside the interquartile range
                                if current_value > percentile_75 + feature_range:
                                    # Value is unusually high - adjust probability
                                    feature_adjustment = importance * 0.05
                                    proba_adjustment += feature_adjustment
                                    adjustment_reasons.append(f"{feature} is unusually high (+{feature_adjustment:.3f})")
                                    
                                elif current_value < percentile_25 - feature_range:
                                    # Value is unusually low - adjust probability
                                    feature_adjustment = -importance * 0.05
                                    proba_adjustment += feature_adjustment
                                    adjustment_reasons.append(f"{feature} is unusually low ({feature_adjustment:.3f})")
        
        # Apply combined adjustment
        adjusted_proba = max(0.0, min(1.0, prediction_proba + proba_adjustment))
        
        # Record the contextual adjustment
        adjustment_record = {
            "timestamp": datetime.now().isoformat(),
            "original_proba": prediction_proba,
            "adjustment": proba_adjustment,
            "adjusted_proba": adjusted_proba,
            "threshold": threshold,
            "reasons": adjustment_reasons
        }
        
        if "contextual_adjustments" not in profile:
            profile["contextual_adjustments"] = []
            
        profile["contextual_adjustments"].append(adjustment_record)
        
        # Limit history size
        if len(profile["contextual_adjustments"]) > 20:
            profile["contextual_adjustments"] = profile["contextual_adjustments"][-20:]
            
        # Update profile
        profile["updated_at"] = datetime.now().isoformat()
        self._save_patient_profile(patient_id)
        
        return {
            "original_proba": prediction_proba,
            "adjusted_proba": adjusted_proba,
            "adjustment": proba_adjustment,
            "threshold": threshold,
            "reasons": adjustment_reasons
        }

"""
Selective Expert Retraining Module

This module provides functionality to selectively retrain only the experts most affected
by concept drift, rather than retraining the entire mixture-of-experts model.
"""
import os
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelectiveExpertRetrainer:
    """
    Provides functionality to selectively retrain experts based on drift impact.
    """
    
    def __init__(self, 
                impact_threshold: float = 0.3, 
                results_dir: str = 'results/moe_validation'):
        """
        Initialize the selective expert retrainer
        
        Parameters:
        -----------
        impact_threshold : float
            Threshold to determine which experts need retraining (0-1)
        results_dir : str
            Directory to save retraining results
        """
        self.impact_threshold = impact_threshold
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.retraining_log = self.results_dir / "retraining_log.json"
        
        # Initialize retraining log if it doesn't exist
        if not self.retraining_log.exists():
            with open(self.retraining_log, 'w') as f:
                json.dump([], f)
    
    def identify_experts_for_retraining(self, 
                                       expert_impacts: Dict[str, float], 
                                       drift_score: float) -> List[str]:
        """
        Identify which experts need retraining based on drift impact
        
        Parameters:
        -----------
        expert_impacts : Dict[str, float]
            Dictionary mapping expert names to drift impact scores
        drift_score : float
            Overall drift score
            
        Returns:
        --------
        List[str]
            List of expert names that need retraining
        """
        if drift_score < self.impact_threshold:
            logger.info(f"Overall drift score {drift_score} below threshold {self.impact_threshold}, no retraining needed")
            return []
            
        experts_to_retrain = []
        
        for expert_name, impact in expert_impacts.items():
            if impact >= self.impact_threshold:
                experts_to_retrain.append(expert_name)
                logger.info(f"Expert '{expert_name}' needs retraining (impact: {impact:.4f})")
                
        return experts_to_retrain
    
    def retrain_selected_experts(self, 
                               moe_model: Any, 
                               experts_to_retrain: List[str],
                               new_data: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Selectively retrain only the experts that need retraining
        
        Parameters:
        -----------
        moe_model : Any
            The mixture-of-experts model
        experts_to_retrain : List[str]
            List of expert names that need retraining
        new_data : Any
            New data to retrain the experts on
            
        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            Updated model and retraining metrics
        """
        if not experts_to_retrain:
            return moe_model, {"retraining_performed": False}
            
        try:
            # Start retraining metrics
            retraining_metrics = {
                "retraining_performed": True,
                "start_time": datetime.now().isoformat(),
                "experts_retrained": experts_to_retrain,
                "per_expert_metrics": {}
            }
            
            # Ensure the model has a get_expert method
            if not hasattr(moe_model, 'get_expert') or not hasattr(moe_model, 'replace_expert'):
                logger.error("Model does not support expert access/replacement")
                return moe_model, {"retraining_performed": False, "error": "Model does not support expert access/replacement"}
            
            # Get train/test split from new data
            X_train, X_test, y_train, y_test = self._split_data(new_data)
            
            # For each expert to retrain
            for expert_name in experts_to_retrain:
                logger.info(f"Retraining expert: {expert_name}")
                
                try:
                    # Get the expert
                    expert = moe_model.get_expert(expert_name)
                    
                    # Store original performance
                    original_performance = self._evaluate_expert(expert, X_test, y_test)
                    
                    # Retrain the expert
                    expert.fit(X_train, y_train)
                    
                    # Evaluate new performance
                    new_performance = self._evaluate_expert(expert, X_test, y_test)
                    
                    # Replace the expert in the model
                    moe_model.replace_expert(expert_name, expert)
                    
                    # Store metrics
                    retraining_metrics["per_expert_metrics"][expert_name] = {
                        "original_performance": original_performance,
                        "new_performance": new_performance,
                        "improvement": new_performance - original_performance
                    }
                    
                    logger.info(f"Expert '{expert_name}' retrained. Performance improvement: {new_performance - original_performance:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error retraining expert '{expert_name}': {str(e)}")
                    retraining_metrics["per_expert_metrics"][expert_name] = {
                        "error": str(e)
                    }
            
            # Finalize metrics
            retraining_metrics["end_time"] = datetime.now().isoformat()
            
            # Log retraining
            self._log_retraining(retraining_metrics)
            
            return moe_model, retraining_metrics
            
        except Exception as e:
            logger.error(f"Error in selective retraining: {str(e)}")
            return moe_model, {"retraining_performed": False, "error": str(e)}
    
    def _split_data(self, data: Any) -> Tuple:
        """
        Split data into train/test sets
        
        Parameters:
        -----------
        data : Any
            Data to split
            
        Returns:
        --------
        Tuple
            X_train, X_test, y_train, y_test
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Check if data is a tuple of (X, y)
            if isinstance(data, tuple) and len(data) == 2:
                X, y = data
                return train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Check if data is a dictionary with 'X' and 'y' keys
            elif isinstance(data, dict) and 'X' in data and 'y' in data:
                return train_test_split(data['X'], data['y'], test_size=0.2, random_state=42)
            
            # Otherwise, assume structured array-like with last column as target
            else:
                if hasattr(data, 'iloc'):  # pandas DataFrame
                    X = data.iloc[:, :-1]
                    y = data.iloc[:, -1]
                else:  # numpy array
                    X = data[:, :-1]
                    y = data[:, -1]
                return train_test_split(X, y, test_size=0.2, random_state=42)
                
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            # If splitting fails, return data as is for both train and test
            return data, data, data, data
    
    def _evaluate_expert(self, expert: Any, X: Any, y: Any) -> float:
        """
        Evaluate an expert's performance
        
        Parameters:
        -----------
        expert : Any
            Expert model to evaluate
        X : Any
            Features for evaluation
        y : Any
            Target for evaluation
            
        Returns:
        --------
        float
            Performance metric (higher is better)
        """
        try:
            from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
            
            # Make predictions
            if hasattr(expert, 'predict_proba'):
                y_pred = expert.predict_proba(X)
                if y_pred.shape[1] == 2:  # Binary classification
                    y_pred = y_pred[:, 1]
                    return roc_auc_score(y, y_pred)
            
            # Regular prediction for classification or regression
            y_pred = expert.predict(X)
            
            # Choose appropriate metric based on unique values in y
            if len(np.unique(y)) <= 5:  # Classification (assuming <= 5 classes)
                return accuracy_score(y, y_pred)
            else:  # Regression
                return r2_score(y, y_pred)
                
        except Exception as e:
            logger.warning(f"Error evaluating expert: {str(e)}")
            return 0.0
    
    def _log_retraining(self, metrics: Dict[str, Any]) -> None:
        """
        Log retraining metrics
        
        Parameters:
        -----------
        metrics : Dict[str, Any]
            Retraining metrics to log
        """
        try:
            # Read existing logs
            with open(self.retraining_log, 'r') as f:
                logs = json.load(f)
                
            # Add new log
            logs.append(metrics)
            
            # Write updated logs
            with open(self.retraining_log, 'w') as f:
                json.dump(logs, f, indent=2)
                
            logger.info(f"Retraining log saved to {self.retraining_log}")
        except Exception as e:
            logger.error(f"Could not log retraining: {str(e)}")
            
    def get_retraining_history(self) -> List[Dict[str, Any]]:
        """
        Get retraining history
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of retraining events with metrics
        """
        try:
            with open(self.retraining_log, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not read retraining history: {str(e)}")
            return []

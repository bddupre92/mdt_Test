"""
MLflow experiment tracking for migraine prediction system.
"""
import mlflow
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ExperimentTracker:
    """Track experiments and model performance."""
    
    def __init__(self, experiment_name: str = "migraine_prediction"):
        self.experiment_name = experiment_name
        self._setup_tracking()
    
    def _setup_tracking(self):
        """Setup MLflow tracking."""
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new tracking run."""
        mlflow.start_run(run_name=run_name)
    
    def end_run(self) -> None:
        """End current tracking run."""
        mlflow.end_run()
    
    def log_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                            stage: str = "validation") -> Dict[str, float]:
        """Log model performance metrics."""
        metrics = {
            f"{stage}_accuracy": accuracy_score(y_true, y_pred),
            f"{stage}_precision": precision_score(y_true, y_pred),
            f"{stage}_recall": recall_score(y_true, y_pred),
            f"{stage}_f1": f1_score(y_true, y_pred),
            f"{stage}_auc": roc_auc_score(y_true, y_pred)
        }
        
        mlflow.log_metrics(metrics)
        return metrics
    
    def log_feature_importance(self, feature_importance: Dict[str, float]) -> None:
        """Log feature importance scores."""
        mlflow.log_dict(feature_importance, "feature_importance.json")
        
        # Create and log feature importance plot
        plt.figure(figsize=(10, 6))
        features = list(feature_importance.keys())
        scores = list(feature_importance.values())
        
        plt.barh(features, scores)
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()
    
    def log_drift_detection(self, drift_results: Dict[str, Any]) -> None:
        """Log drift detection results."""
        mlflow.log_dict(drift_results, "drift_detection.json")
        
        # Log drift metrics
        metrics = {
            "drift_detection_time": drift_results.get("detection_time", 0),
            "drift_magnitude": drift_results.get("magnitude", 0),
            "affected_features_count": len(drift_results.get("affected_features", []))
        }
        mlflow.log_metrics(metrics)
    
    def log_meta_optimizer_run(self, optimizer_results: Dict[str, Any]) -> None:
        """Log meta-optimizer results."""
        mlflow.log_dict(optimizer_results, "meta_optimizer_results.json")
        
        # Log optimization metrics
        metrics = {
            "optimization_time": optimizer_results.get("time_taken", 0),
            "best_fitness": optimizer_results.get("best_fitness", 0),
            "generations": optimizer_results.get("generations", 0),
            "algorithm_switches": optimizer_results.get("algorithm_switches", 0)
        }
        mlflow.log_metrics(metrics)
        
        # Log parameter selection history
        param_history = optimizer_results.get("parameter_history", [])
        if param_history:
            param_df = pd.DataFrame(param_history)
            mlflow.log_table("parameter_history.json", param_df)
    
    def log_synthetic_data_stats(self, data_stats: Dict[str, Any]) -> None:
        """Log statistics about generated synthetic data."""
        mlflow.log_dict(data_stats, "synthetic_data_stats.json")
        
        # Log key metrics about the synthetic data
        metrics = {
            "total_records": data_stats.get("total_records", 0),
            "missing_rate": data_stats.get("missing_rate", 0),
            "drift_points": len(data_stats.get("drift_points", [])),
            "feature_count": data_stats.get("feature_count", 0)
        }
        mlflow.log_metrics(metrics)
    
    def log_training_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log training metadata."""
        mlflow.log_dict(metadata, "training_metadata.json")
        
        # Log relevant parameters
        params = {
            "model_type": metadata.get("model_type", "unknown"),
            "optimizer": metadata.get("optimizer", "unknown"),
            "batch_size": metadata.get("batch_size", 0),
            "epochs": metadata.get("epochs", 0)
        }
        mlflow.log_params(params)
    
    def create_summary(self) -> Dict[str, Any]:
        """Create summary of current run."""
        run = mlflow.active_run()
        if not run:
            return {}
        
        metrics = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.metrics
        params = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.params
        
        return {
            "run_id": run.info.run_id,
            "start_time": run.info.start_time,
            "metrics": metrics,
            "parameters": params
        }

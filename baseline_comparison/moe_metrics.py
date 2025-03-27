"""
MoE Evaluation Metrics

This module provides specialized evaluation metrics for Mixture of Experts (MoE)
models, enabling detailed analysis of expert contributions, confidence levels,
gating network quality, and temporal prediction performance.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import baseline comparison components
from baseline_comparison.comparison_metrics import compute_baseline_metrics

logger = logging.getLogger(__name__)

class MoEMetricsCalculator:
    """
    Calculates and analyzes MoE-specific metrics to evaluate model performance
    and provide insights into the mixture of experts behavior.
    """
    
    def __init__(self, output_dir: str = "results/moe_metrics"):
        """
        Initialize the MoE metrics calculator.
        
        Args:
            output_dir: Directory to save metric results and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = {}
        
    def compute_expert_contribution_metrics(
        self, 
        expert_contributions: Dict[str, List[float]],
        prediction_errors: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics related to expert contributions.
        
        Args:
            expert_contributions: Dictionary mapping expert names to their contribution 
                                 weights for each prediction
            prediction_errors: Optional array of prediction errors to correlate with
                              expert contributions
                              
        Returns:
            Dictionary of expert contribution metrics
        """
        metrics = {}
        
        # Calculate basic contribution statistics
        contribution_matrix = np.array(list(expert_contributions.values())).T
        metrics["total_predictions"] = contribution_matrix.shape[0]
        
        # Calculate mean and std dev of contributions per expert
        metrics["expert_mean_contributions"] = {
            expert: np.mean(contributions) 
            for expert, contributions in expert_contributions.items()
        }
        
        metrics["expert_std_contributions"] = {
            expert: np.std(contributions) 
            for expert, contributions in expert_contributions.items()
        }
        
        # Calculate frequency of expert dominance (when an expert has highest weight)
        dominant_experts = np.argmax(contribution_matrix, axis=1)
        expert_names = list(expert_contributions.keys())
        metrics["expert_dominance_counts"] = {
            expert_names[i]: np.sum(dominant_experts == i) 
            for i in range(len(expert_names))
        }
        
        metrics["expert_dominance_percentage"] = {
            expert: count / metrics["total_predictions"] * 100
            for expert, count in metrics["expert_dominance_counts"].items()
        }
        
        # Calculate contribution diversity (entropy)
        # Higher values indicate more balanced expert utilization
        contribution_entropy = -np.sum(
            contribution_matrix * np.log2(np.clip(contribution_matrix, 1e-10, 1.0)), 
            axis=1
        )
        metrics["mean_contribution_entropy"] = np.mean(contribution_entropy)
        metrics["max_possible_entropy"] = np.log2(len(expert_contributions))
        metrics["normalized_entropy"] = metrics["mean_contribution_entropy"] / metrics["max_possible_entropy"]
        
        # If prediction errors are provided, correlate with expert contributions
        if prediction_errors is not None:
            metrics["error_contribution_correlation"] = {}
            for expert, contributions in expert_contributions.items():
                corr = np.corrcoef(contributions, np.abs(prediction_errors))[0, 1]
                metrics["error_contribution_correlation"][expert] = corr
        
        return metrics
    
    def compute_confidence_metrics(
        self,
        confidence_scores: np.ndarray,
        actual_errors: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate metrics related to prediction confidence.
        
        Args:
            confidence_scores: Array of confidence scores for each prediction
            actual_errors: Array of actual prediction errors
            
        Returns:
            Dictionary of confidence-related metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics["mean_confidence"] = np.mean(confidence_scores)
        metrics["min_confidence"] = np.min(confidence_scores)
        metrics["max_confidence"] = np.max(confidence_scores)
        
        # Calculate calibration metrics
        # Correlation between confidence and accuracy
        # Negative correlation is good (higher confidence -> lower error)
        metrics["confidence_error_correlation"] = np.corrcoef(confidence_scores, np.abs(actual_errors))[0, 1]
        
        # Divide into confidence bins and calculate mean error in each bin
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidence_scores, bin_edges) - 1
        
        bin_mean_errors = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) > 0:
                bin_mean_errors.append(np.mean(np.abs(actual_errors[bin_mask])))
                bin_counts.append(np.sum(bin_mask))
            else:
                bin_mean_errors.append(np.nan)
                bin_counts.append(0)
                
        metrics["bin_edges"] = bin_edges.tolist()
        metrics["bin_mean_errors"] = bin_mean_errors
        metrics["bin_counts"] = bin_counts
        
        # Calculate expected calibration error (ECE)
        # Lower is better, indicates how well calibrated the confidence is
        bin_weights = np.array(bin_counts) / len(confidence_scores)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate expected confidence in each bin
        bin_mean_confidences = []
        for i in range(num_bins):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) > 0:
                bin_mean_confidences.append(np.mean(confidence_scores[bin_mask]))
            else:
                bin_mean_confidences.append(np.nan)
        
        # Expected calibration error
        valid_bins = ~np.isnan(bin_mean_errors) & ~np.isnan(bin_mean_confidences)
        if np.sum(valid_bins) > 0:
            # Scale errors to 0-1 range for proper comparison with confidence
            max_error = np.max(np.abs(actual_errors))
            normalized_bin_errors = np.array(bin_mean_errors) / max_error
            
            # For well-calibrated models: confidence should be 1 - normalized_error
            expected_confidences = 1 - normalized_bin_errors
            
            # ECE is the weighted average absolute difference
            ece = np.sum(
                bin_weights[valid_bins] * 
                np.abs(np.array(bin_mean_confidences)[valid_bins] - expected_confidences[valid_bins])
            )
            metrics["expected_calibration_error"] = ece
        else:
            metrics["expected_calibration_error"] = np.nan
        
        return metrics
    
    def compute_gating_network_metrics(
        self,
        expert_weights: Dict[str, List[float]],
        expert_errors: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Calculate metrics to evaluate gating network quality.
        
        Args:
            expert_weights: Dictionary mapping expert names to weights assigned by
                           the gating network for each prediction
            expert_errors: Dictionary mapping expert names to individual prediction
                          errors for each expert
                          
        Returns:
            Dictionary of gating network quality metrics
        """
        metrics = {}
        
        # Convert to numpy arrays for easier processing
        experts = list(expert_weights.keys())
        weights_array = np.array([expert_weights[expert] for expert in experts])
        errors_array = np.array([expert_errors[expert] for expert in experts])
        
        # Transpose to have samples as the first dimension
        weights_array = weights_array.T
        errors_array = errors_array.T
        
        # Calculate how often the gating network selects the best expert
        # (Expert with lowest error)
        best_expert_indices = np.argmin(errors_array, axis=1)
        selected_expert_indices = np.argmax(weights_array, axis=1)
        
        optimal_selection_rate = np.mean(selected_expert_indices == best_expert_indices)
        metrics["optimal_expert_selection_rate"] = optimal_selection_rate
        
        # Calculate regret: difference between selected expert error and best expert error
        selected_expert_errors = np.array([
            errors_array[i, selected_expert_indices[i]] 
            for i in range(len(selected_expert_indices))
        ])
        best_expert_errors = np.array([
            errors_array[i, best_expert_indices[i]] 
            for i in range(len(best_expert_indices))
        ])
        
        regret = selected_expert_errors - best_expert_errors
        metrics["mean_regret"] = np.mean(regret)
        metrics["max_regret"] = np.max(regret)
        
        # Calculate weight concentration (how much weight on top expert)
        top_weight_ratio = np.max(weights_array, axis=1) / np.sum(weights_array, axis=1)
        metrics["mean_top_weight_ratio"] = np.mean(top_weight_ratio)
        
        # Calculate correlation between weights and inverse errors
        # Higher values are better (more weight should go to experts with lower error)
        weight_error_correlations = []
        for i in range(weights_array.shape[0]):
            # Calculate inverse errors (higher is better)
            inverse_errors = 1.0 / (np.abs(errors_array[i]) + 1e-10)
            
            # Normalize to sum to 1 like weights
            normalized_inverse_errors = inverse_errors / np.sum(inverse_errors)
            
            # Calculate correlation
            corr = np.corrcoef(weights_array[i], normalized_inverse_errors)[0, 1]
            weight_error_correlations.append(corr)
        
        metrics["mean_weight_error_correlation"] = np.mean(weight_error_correlations)
        metrics["weight_error_correlation_std"] = np.std(weight_error_correlations)
        
        return metrics
    
    def compute_temporal_metrics(
        self,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        timestamps: np.ndarray,
        expert_contributions: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate specialized metrics for time series predictions.
        
        Args:
            predictions: Array of model predictions
            actual_values: Array of actual values
            timestamps: Array of timestamps (sorted chronologically)
            expert_contributions: Optional dictionary of expert contributions over time
            
        Returns:
            Dictionary of temporal performance metrics
        """
        metrics = {}
        
        # Ensure inputs are numpy arrays
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        timestamps = np.array(timestamps)
        
        # Calculate standard time series metrics
        errors = actual_values - predictions
        metrics["rmse"] = np.sqrt(mean_squared_error(actual_values, predictions))
        metrics["mae"] = mean_absolute_error(actual_values, predictions)
        metrics["r2"] = r2_score(actual_values, predictions)
        
        # Calculate error statistics over time
        n_segments = min(10, len(predictions) // 10)  # Split into segments
        if n_segments > 1:
            segment_indices = np.array_split(np.arange(len(predictions)), n_segments)
            
            metrics["temporal_segment_rmse"] = []
            metrics["temporal_segment_mae"] = []
            
            for segment in segment_indices:
                seg_predictions = predictions[segment]
                seg_actual = actual_values[segment]
                seg_error = np.abs(seg_actual - seg_predictions)
                
                metrics["temporal_segment_rmse"].append(
                    np.sqrt(mean_squared_error(seg_actual, seg_predictions))
                )
                metrics["temporal_segment_mae"].append(
                    mean_absolute_error(seg_actual, seg_predictions)
                )
        
        # Calculate error autocorrelation (consecutive errors tend to be similar?)
        if len(errors) > 2:
            error_autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1]
            metrics["error_autocorrelation"] = error_autocorr
        
        # Analyze expert contribution trends if provided
        if expert_contributions:
            metrics["expert_temporal_trends"] = {}
            
            for expert, contributions in expert_contributions.items():
                # Calculate moving average of contributions (window size = 10% of data)
                window_size = max(1, len(contributions) // 10)
                padding = np.ones(window_size - 1) * contributions[0]
                padded_contributions = np.concatenate([padding, contributions])
                
                moving_avg = np.convolve(
                    padded_contributions, 
                    np.ones(window_size) / window_size, 
                    mode='valid'
                )
                
                # Calculate whether contribution is increasing/decreasing
                if len(moving_avg) > 2:
                    # Linear regression slope
                    x = np.arange(len(moving_avg))
                    A = np.vstack([x, np.ones(len(x))]).T
                    slope, _ = np.linalg.lstsq(A, moving_avg, rcond=None)[0]
                    
                    metrics["expert_temporal_trends"][expert] = {
                        "slope": slope,
                        "start_contribution": moving_avg[0],
                        "end_contribution": moving_avg[-1],
                        "change_percentage": (moving_avg[-1] - moving_avg[0]) / (moving_avg[0] + 1e-10) * 100
                    }
        
        return metrics
    
    def compute_personalization_metrics(
        self,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        patient_ids: np.ndarray,
        expert_contributions: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics to evaluate personalization effectiveness.
        
        Args:
            predictions: Array of model predictions
            actual_values: Array of actual values
            patient_ids: Array of patient identifiers
            expert_contributions: Optional dictionary of expert contributions
            
        Returns:
            Dictionary of personalization performance metrics
        """
        metrics = {}
        
        # Get unique patients
        unique_patients = np.unique(patient_ids)
        metrics["num_patients"] = len(unique_patients)
        
        # Calculate per-patient metrics
        patient_metrics = {}
        for patient in unique_patients:
            patient_mask = (patient_ids == patient)
            
            if np.sum(patient_mask) > 0:
                patient_preds = predictions[patient_mask]
                patient_actual = actual_values[patient_mask]
                
                # Calculate error metrics for this patient
                patient_metrics[patient] = {
                    "rmse": np.sqrt(mean_squared_error(patient_actual, patient_preds)),
                    "mae": mean_absolute_error(patient_actual, patient_preds),
                    "sample_count": np.sum(patient_mask)
                }
                
                # Add expert contributions if available
                if expert_contributions:
                    patient_metrics[patient]["expert_contributions"] = {
                        expert: np.mean(np.array(contributions)[patient_mask])
                        for expert, contributions in expert_contributions.items()
                    }
                    
                    # Get dominant expert for this patient
                    patient_metrics[patient]["dominant_expert"] = max(
                        patient_metrics[patient]["expert_contributions"].items(),
                        key=lambda x: x[1]
                    )[0]
        
        metrics["per_patient"] = patient_metrics
        
        # Calculate overall personalization metrics
        per_patient_rmse = [m["rmse"] for m in patient_metrics.values()]
        metrics["mean_patient_rmse"] = np.mean(per_patient_rmse)
        metrics["patient_rmse_std"] = np.std(per_patient_rmse)
        metrics["patient_rmse_range"] = np.max(per_patient_rmse) - np.min(per_patient_rmse)
        
        # Analyze expert specialization by patient
        if expert_contributions:
            metrics["expert_patient_specialization"] = {}
            
            # For each expert, calculate the variance in contribution across patients
            # Higher variance = more specialization (expert specializes in certain patients)
            for expert in expert_contributions.keys():
                mean_contribs_by_patient = [
                    m["expert_contributions"][expert] 
                    for m in patient_metrics.values() 
                    if "expert_contributions" in m
                ]
                
                if mean_contribs_by_patient:
                    metrics["expert_patient_specialization"][expert] = {
                        "mean": np.mean(mean_contribs_by_patient),
                        "std": np.std(mean_contribs_by_patient),
                        "max_contribution": np.max(mean_contribs_by_patient),
                        "min_contribution": np.min(mean_contribs_by_patient),
                        "range": np.max(mean_contribs_by_patient) - np.min(mean_contribs_by_patient)
                    }
            
            # Calculate dominant expert distribution
            dominant_experts = [m.get("dominant_expert") for m in patient_metrics.values() 
                              if "dominant_expert" in m]
            
            if dominant_experts:
                # Count occurrences of each expert as dominant
                from collections import Counter
                dominant_counts = Counter(dominant_experts)
                
                metrics["dominant_expert_distribution"] = {
                    expert: count / len(dominant_experts) * 100
                    for expert, count in dominant_counts.items()
                }
        
        return metrics
    
    def compute_all_metrics(
        self,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        expert_contributions: Dict[str, List[float]],
        confidence_scores: np.ndarray,
        expert_errors: Dict[str, List[float]],
        timestamps: Optional[np.ndarray] = None,
        patient_ids: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute all available MoE metrics.
        
        Args:
            predictions: Array of final model predictions
            actual_values: Array of actual values
            expert_contributions: Dictionary mapping expert names to contribution weights
            confidence_scores: Array of confidence scores for each prediction
            expert_errors: Dictionary mapping expert names to individual prediction errors
            timestamps: Optional array of timestamps for temporal analysis
            patient_ids: Optional array of patient IDs for personalization analysis
            
        Returns:
            Dictionary containing all computed metrics
        """
        all_metrics = {}
        
        # Calculate prediction errors
        errors = actual_values - predictions
        
        # Compute standard baseline metrics
        standard_metrics = compute_baseline_metrics(predictions, actual_values)
        all_metrics["standard"] = standard_metrics
        
        # Compute expert contribution metrics
        all_metrics["expert_contribution"] = self.compute_expert_contribution_metrics(
            expert_contributions, errors
        )
        
        # Compute confidence metrics
        all_metrics["confidence"] = self.compute_confidence_metrics(
            confidence_scores, errors
        )
        
        # Compute gating network metrics
        all_metrics["gating_network"] = self.compute_gating_network_metrics(
            expert_contributions, expert_errors
        )
        
        # Compute temporal metrics if timestamps provided
        if timestamps is not None:
            all_metrics["temporal"] = self.compute_temporal_metrics(
                predictions, actual_values, timestamps, expert_contributions
            )
        
        # Compute personalization metrics if patient IDs provided
        if patient_ids is not None:
            all_metrics["personalization"] = self.compute_personalization_metrics(
                predictions, actual_values, patient_ids, expert_contributions
            )
        
        return all_metrics
    
    def save_metrics(self, metrics: Dict[str, Any], name: str = "moe_metrics") -> str:
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            name: Name for the metrics file
            
        Returns:
            Path to the saved metrics file
        """
        metrics_file = self.output_dir / f"{name}.json"
        
        # Convert numpy values to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        serializable_metrics = convert_numpy(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
        logger.info(f"Saved MoE metrics to {metrics_file}")
        return str(metrics_file)
    
    def visualize_metrics(self, metrics: Dict[str, Any], name: str = "moe_metrics") -> List[str]:
        """
        Create visualizations of key MoE metrics.
        
        Args:
            metrics: Dictionary of metrics to visualize
            name: Base name for visualization files
            
        Returns:
            List of paths to generated visualization files
        """
        visualization_paths = []
        
        # Set up plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'figure.figsize': (12, 8)})
        
        # 1. Expert contribution visualization
        if "expert_contribution" in metrics:
            contrib_metrics = metrics["expert_contribution"]
            
            if "expert_dominance_percentage" in contrib_metrics:
                fig, ax = plt.subplots()
                experts = list(contrib_metrics["expert_dominance_percentage"].keys())
                dominance = list(contrib_metrics["expert_dominance_percentage"].values())
                
                # Sort by dominance percentage
                sorted_indices = np.argsort(dominance)[::-1]
                sorted_experts = [experts[i] for i in sorted_indices]
                sorted_dominance = [dominance[i] for i in sorted_indices]
                
                ax.bar(sorted_experts, sorted_dominance)
                ax.set_xlabel('Expert')
                ax.set_ylabel('Dominance Percentage (%)')
                ax.set_title('Expert Dominance in Predictions')
                ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                path = self.output_dir / f"{name}_expert_dominance.png"
                plt.savefig(path)
                plt.close()
                visualization_paths.append(str(path))
        
        # 2. Confidence calibration and metrics visualization
        if "confidence" in metrics:
            conf_metrics = metrics["confidence"]
            
            if "bin_edges" in conf_metrics and "bin_mean_errors" in conf_metrics:
                # 2a. Confidence calibration plot
                fig, ax = plt.subplots()
                
                bin_centers = [(conf_metrics["bin_edges"][i] + conf_metrics["bin_edges"][i+1])/2 
                              for i in range(len(conf_metrics["bin_edges"])-1)]
                
                ax.plot(bin_centers, conf_metrics["bin_mean_errors"], 'o-', label='Mean Error')
                
                # Add ideal calibration line (diagonal)
                max_error = max(conf_metrics["bin_mean_errors"])
                ideal_x = np.linspace(0, 1, 100)
                ideal_y = max_error * (1 - ideal_x)
                ax.plot(ideal_x, ideal_y, '--', label='Ideal Calibration')
                
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Mean Absolute Error')
                ax.set_title('Confidence Calibration Plot')
                ax.legend()
                
                plt.tight_layout()
                path = self.output_dir / f"{name}_confidence_calibration.png"
                plt.savefig(path)
                plt.close()
                visualization_paths.append(str(path))
                
                # 2b. Confidence distribution histogram
                if "mean_confidence" in conf_metrics:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create histogram with KDE
                    if hasattr(metrics, "_raw_confidence_scores"):
                        sns.histplot(metrics["_raw_confidence_scores"], kde=True, ax=ax)
                    elif "bin_edges" in conf_metrics and "bin_counts" in conf_metrics:
                        # Use bin data to approximate histogram
                        ax.bar(bin_centers, conf_metrics["bin_counts"], 
                        width=(conf_metrics["bin_edges"][1]-conf_metrics["bin_edges"][0]),
                            alpha=0.7, label='Frequency')
                        # Add mean confidence line
                        ax.axvline(conf_metrics["mean_confidence"], color='red', linestyle='--', 
                            label=f'Mean: {conf_metrics["mean_confidence"]:.3f}')
                    
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Confidence Scores')
                    ax.legend()
                    
                    plt.tight_layout()
                    path = self.output_dir / f"{name}_confidence_distribution.png"
                    plt.savefig(path)
                    plt.close()
                    visualization_paths.append(str(path))
                
                # 2c. Confidence vs Error scatter plot
                if "confidence_error_correlation" in conf_metrics:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if hasattr(metrics, "_raw_confidence_scores") and hasattr(metrics, "_raw_errors"):
                        # Create scatter plot with density coloring
                        scatter = ax.scatter(
                            metrics["_raw_confidence_scores"], 
                            np.abs(metrics["_raw_errors"]),
                            alpha=0.5, c=metrics["_raw_confidence_scores"], cmap='viridis'
                        )
                        plt.colorbar(scatter, label='Confidence')
                    else:
                        # Create simulated scatter using bin data
                        for i, count in enumerate(conf_metrics["bin_counts"]):
                            if count > 0:
                                # Simulate points in this bin
                                conf_val = bin_centers[i]
                                error_val = conf_metrics["bin_mean_errors"][i]
                                ax.scatter([conf_val] * count, [error_val] * count, 
                                          alpha=0.5, c='blue')
                    
                    # Add correlation value to plot
                    corr_val = conf_metrics["confidence_error_correlation"]
                    ax.text(0.05, 0.95, f'Correlation: {corr_val:.3f}', 
                           transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
                    
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Absolute Error')
                    ax.set_title('Confidence vs Error Relationship')
                    
                    plt.tight_layout()
                    path = self.output_dir / f"{name}_confidence_vs_error.png"
                    plt.savefig(path)
                    plt.close()
                    visualization_paths.append(str(path))
        
        # 3. Temporal metrics visualization if available
        if "temporal" in metrics:
            temp_metrics = metrics["temporal"]
            
            if "temporal_segment_rmse" in temp_metrics:
                fig, ax = plt.subplots()
                
                segments = range(len(temp_metrics["temporal_segment_rmse"]))
                ax.plot(segments, temp_metrics["temporal_segment_rmse"], 'o-', label='RMSE')
                ax.plot(segments, temp_metrics["temporal_segment_mae"], 's-', label='MAE')
                
                ax.set_xlabel('Time Segment')
                ax.set_ylabel('Error')
                ax.set_title('Prediction Error Over Time')
                ax.legend()
                
                plt.tight_layout()
                path = self.output_dir / f"{name}_temporal_error.png"
                plt.savefig(path)
                plt.close()
                visualization_paths.append(str(path))
        
        # 4. Patient personalization visualization if available
        if "personalization" in metrics:
            pers_metrics = metrics["personalization"]
            
            if "per_patient" in pers_metrics:
                # Extract per-patient RMSE values
                patients = list(pers_metrics["per_patient"].keys())
                rmse_values = [pers_metrics["per_patient"][p]["rmse"] for p in patients]
                
                # Sort by RMSE
                sorted_indices = np.argsort(rmse_values)
                sorted_patients = [patients[i] for i in sorted_indices]
                sorted_rmse = [rmse_values[i] for i in sorted_indices]
                
                fig, ax = plt.subplots()
                ax.bar(sorted_patients, sorted_rmse)
                ax.set_xlabel('Patient ID')
                ax.set_ylabel('RMSE')
                ax.set_title('Prediction Error by Patient')
                ax.tick_params(axis='x', rotation=90)
                
                plt.tight_layout()
                path = self.output_dir / f"{name}_patient_rmse.png"
                plt.savefig(path)
                plt.close()
                visualization_paths.append(str(path))
                
                # If we have expert contributions per patient
                if "expert_patient_specialization" in pers_metrics:
                    # Create heatmap of expert specialization
                    if "dominant_expert_distribution" in pers_metrics:
                        fig, ax = plt.subplots()
                        
                        experts = list(pers_metrics["dominant_expert_distribution"].keys())
                        percentages = list(pers_metrics["dominant_expert_distribution"].values())
                        
                        # Sort by percentage
                        sorted_indices = np.argsort(percentages)[::-1]
                        sorted_experts = [experts[i] for i in sorted_indices]
                        sorted_percentages = [percentages[i] for i in sorted_indices]
                        
                        ax.bar(sorted_experts, sorted_percentages)
                        ax.set_xlabel('Expert')
                        ax.set_ylabel('% of Patients')
                        ax.set_title('Dominant Expert Distribution Across Patients')
                        ax.tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        path = self.output_dir / f"{name}_expert_patient_distribution.png"
                        plt.savefig(path)
                        plt.close()
                        visualization_paths.append(str(path))
        
        # Add gating network visualization if available
        if "gating_network" in metrics:
            gating_metrics = metrics["gating_network"]
            
            if "optimal_expert_selection_rate" in gating_metrics:
                # Create a gating effectiveness plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                metrics_to_plot = [
                    ("optimal_expert_selection_rate", "Optimal Selection Rate"),
                    ("mean_top_weight_ratio", "Mean Top Weight Ratio"),
                    ("mean_weight_error_correlation", "Weight-Error Correlation")
                ]
                
                values = []
                labels = []
                colors = ['#2C7BB6', '#D7191C', '#FDAE61']
                
                for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
                    if metric_key in gating_metrics:
                        values.append(gating_metrics[metric_key])
                        labels.append(metric_label)
                
                # Create bar chart
                bars = ax.bar(labels, values, color=colors[:len(values)])
                
                # Add text labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
                
                ax.set_ylim(0, 1.1)
                ax.set_ylabel('Score')
                ax.set_title('Gating Network Effectiveness Metrics')
                
                plt.tight_layout()
                path = self.output_dir / f"{name}_gating_effectiveness.png"
                plt.savefig(path)
                plt.close()
                visualization_paths.append(str(path))
                
                # Add regret visualization if available
                if "mean_regret" in gating_metrics and "max_regret" in gating_metrics:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    regret_vals = [gating_metrics["mean_regret"], gating_metrics["max_regret"]]
                    regret_labels = ["Mean Regret", "Max Regret"]
                    
                    ax.bar(regret_labels, regret_vals, color=['#66C2A5', '#FC8D62'])
                    ax.set_ylabel('Regret (Error Difference)')
                    ax.set_title('Gating Network Regret Analysis')
                    
                    plt.tight_layout()
                    path = self.output_dir / f"{name}_gating_regret.png"
                    plt.savefig(path)
                    plt.close()
                    visualization_paths.append(str(path))
                    
        logger.info(f"Created {len(visualization_paths)} visualizations in {self.output_dir}")
        return visualization_paths
        
    def visualize_comparison(self, moe_metrics: Dict[str, Any], baseline_metrics: Dict[str, Dict[str, Any]], 
                            name: str = "moe_comparison") -> List[str]:
        """
        Create visualizations comparing MoE with baseline approaches.
        
        Args:
            moe_metrics: Dictionary of MoE metrics
            baseline_metrics: Dictionary mapping baseline names to their metrics
            name: Base name for visualization files
            
        Returns:
            List of paths to generated visualization files
        """
        visualization_paths = []
        
        # Set up plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'figure.figsize': (12, 8)})
        
        # 1. Performance comparison across key metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metric_keys = ['rmse', 'mae', 'r2', 'training_time']
        metric_names = ['RMSE', 'MAE', 'R²', 'Training Time (s)']
        metric_lower_better = [True, True, False, True]  # For RMSE, MAE and time, lower is better
        
        for i, (metric_key, metric_name, lower_better) in enumerate(zip(metric_keys, metric_names, metric_lower_better)):
            baseline_values = []
            baseline_labels = []
            
            # Get baseline values
            for baseline_name, metrics in baseline_metrics.items():
                # Get the metric value, handling different potential locations
                metric_value = None
                if metric_key in metrics:
                    metric_value = metrics[metric_key]
                elif 'performance' in metrics and metric_key in metrics['performance']:
                    metric_value = metrics['performance'][metric_key]
                
                if metric_value is not None:
                    baseline_values.append(metric_value)
                    baseline_labels.append(baseline_name)
            
            # Get MoE value
            moe_value = None
            if metric_key in moe_metrics:
                moe_value = moe_metrics[metric_key]
            elif 'performance' in moe_metrics and metric_key in moe_metrics['performance']:
                moe_value = moe_metrics['performance'][metric_key]
            
            if moe_value is not None:
                baseline_values.append(moe_value)
                baseline_labels.append('MoE')
            
            # Create plot if we have data
            if baseline_values:
                # Plot in the current subplot
                ax = axes[i]
                
                # Create bar colors (highlight MoE)
                colors = ['#1f77b4'] * len(baseline_values)
                if 'MoE' in baseline_labels:
                    moe_idx = baseline_labels.index('MoE')
                    colors[moe_idx] = '#d62728'  # Highlight MoE in red
                
                # Sort by value if needed
                if lower_better:
                    # Sort ascending for metrics where lower is better
                    sorted_idx = np.argsort(baseline_values)
                else:
                    # Sort descending for metrics where higher is better
                    sorted_idx = np.argsort(baseline_values)[::-1]
                
                sorted_values = [baseline_values[i] for i in sorted_idx]
                sorted_labels = [baseline_labels[i] for i in sorted_idx]
                sorted_colors = [colors[i] for i in sorted_idx]
                
                # Create bar chart
                bars = ax.bar(sorted_labels, sorted_values, color=sorted_colors)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(sorted_values),
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
                
                ax.set_title(metric_name)
                ax.set_ylabel(metric_name)
                ax.tick_params(axis='x', rotation=45)
                
                # Add a note about which direction is better
                direction = "Lower is better" if lower_better else "Higher is better"
                ax.text(0.5, 0.01, direction, transform=ax.transAxes, ha='center', 
                        fontsize=10, style='italic')
        
        plt.tight_layout()
        path = self.output_dir / f"{name}_performance_comparison.png"
        plt.savefig(path)
        plt.close()
        visualization_paths.append(str(path))
        
        # 2. Expert contribution vs baseline selection frequency comparison
        if 'expert_contribution' in moe_metrics:
            # Get MoE expert contributions
            moe_experts = []
            moe_values = []
            
            if 'expert_dominance_percentage' in moe_metrics['expert_contribution']:
                contrib_data = moe_metrics['expert_contribution']['expert_dominance_percentage']
                for expert, value in contrib_data.items():
                    moe_experts.append(expert)
                    moe_values.append(value)
            
            # Sort by contribution
            sorted_idx = np.argsort(moe_values)[::-1]  # Descending
            moe_experts = [moe_experts[i] for i in sorted_idx]
            moe_values = [moe_values[i] for i in sorted_idx]
            
            # Find any baseline selection frequencies
            baseline_has_selections = False
            for baseline_name, metrics in baseline_metrics.items():
                if 'selections' in metrics or 'algorithm_selections' in metrics:
                    baseline_has_selections = True
                    break
            
            if baseline_has_selections:
                # Create comparative visualization
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Plot MoE expert contributions
                ax.bar(moe_experts, moe_values, alpha=0.7, label='MoE Expert Contribution %')
                
                # Add baseline selection frequencies on the same axes
                for baseline_name, metrics in baseline_metrics.items():
                    selections = None
                    if 'selections' in metrics:
                        selections = metrics['selections']
                    elif 'algorithm_selections' in metrics:
                        selections = metrics['algorithm_selections']
                    
                    if selections:
                        # Count frequencies
                        freq_dict = {}
                        for alg in selections:
                            freq_dict[alg] = freq_dict.get(alg, 0) + 1
                        
                        # Convert to percentages
                        total = sum(freq_dict.values())
                        for alg in freq_dict:
                            freq_dict[alg] = (freq_dict[alg] / total) * 100
                        
                        # Get values for moe_experts to align bars
                        values = [freq_dict.get(expert, 0) for expert in moe_experts]
                        
                        # Plot as line to differentiate from bars
                        ax.plot(moe_experts, values, 'o-', label=f'{baseline_name} Selection %')
                
                ax.set_xlabel('Experts/Algorithms')
                ax.set_ylabel('Percentage (%)')
                ax.set_title('MoE Expert Contributions vs Baseline Selection Frequencies')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                path = self.output_dir / f"{name}_expert_vs_baseline.png"
                plt.savefig(path)
                plt.close()
                visualization_paths.append(str(path))
        
        # 3. Create radar chart comparing MoE with baselines
        # Get metrics that are common to MoE and at least one baseline
        common_metrics = set()
        for metric in ['rmse', 'mae', 'r2', 'training_time', 'inference_time']:
            if (metric in moe_metrics or 
                ('performance' in moe_metrics and metric in moe_metrics['performance'])):
                # Check if at least one baseline has this metric
                for baseline_metrics_dict in baseline_metrics.values():
                    if ((metric in baseline_metrics_dict) or 
                        ('performance' in baseline_metrics_dict and 
                         metric in baseline_metrics_dict['performance'])):
                        common_metrics.add(metric)
                        break
        
        if len(common_metrics) >= 3:  # Need at least 3 metrics for meaningful radar
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Convert metrics to list and add readable labels
            metrics_list = sorted(list(common_metrics))
            metrics_labels = [m.replace('_', ' ').upper() for m in metrics_list]
            
            # Get normalized values for each approach
            approaches = list(baseline_metrics.keys()) + ['MoE']
            values_by_approach = {}
            
            # For each metric, find min and max across all approaches for normalization
            metric_min = {m: float('inf') for m in metrics_list}
            metric_max = {m: float('-inf') for m in metrics_list}
            
            for approach in approaches:
                if approach == 'MoE':
                    metrics_dict = moe_metrics
                else:
                    metrics_dict = baseline_metrics[approach]
                
                approach_values = []
                for metric in metrics_list:
                    value = None
                    if metric in metrics_dict:
                        value = metrics_dict[metric]
                    elif 'performance' in metrics_dict and metric in metrics_dict['performance']:
                        value = metrics_dict['performance'][metric]
                    
                    if value is not None:
                        metric_min[metric] = min(metric_min[metric], value)
                        metric_max[metric] = max(metric_max[metric], value)
            
            # Now normalize values and create the radar chart
            for approach in approaches:
                if approach == 'MoE':
                    metrics_dict = moe_metrics
                else:
                    metrics_dict = baseline_metrics[approach]
                
                approach_values = []
                for metric in metrics_list:
                    value = None
                    if metric in metrics_dict:
                        value = metrics_dict[metric]
                    elif 'performance' in metrics_dict and metric in metrics_dict['performance']:
                        value = metrics_dict['performance'][metric]
                    
                    if value is not None:
                        # Normalize based on min-max, but invert for metrics where lower is better
                        if metric in ['rmse', 'mae', 'training_time', 'inference_time']:
                            # Lower is better, so invert
                            if metric_max[metric] > metric_min[metric]:  # Avoid division by zero
                                normalized = 1 - ((value - metric_min[metric]) / 
                                                 (metric_max[metric] - metric_min[metric]))
                            else:
                                normalized = 1.0  # All values are the same
                        else:
                            # Higher is better (r²)
                            if metric_max[metric] > metric_min[metric]:  # Avoid division by zero
                                normalized = (value - metric_min[metric]) / (metric_max[metric] - metric_min[metric])
                            else:
                                normalized = 1.0  # All values are the same
                        approach_values.append(normalized)
                    else:
                        approach_values.append(0)  # Missing values get zero
                
                # Close the radar loop by repeating first value
                values_closed = approach_values + [approach_values[0]]
                labels_closed = metrics_labels + [metrics_labels[0]]
                
                # Plot this approach
                angle = np.linspace(0, 2*np.pi, len(labels_closed))
                ax.plot(angle, values_closed, 'o-', linewidth=2, 
                        label=approach, alpha=0.8)
                ax.fill(angle, values_closed, alpha=0.1)
            
            # Set radar chart properties
            ax.set_thetagrids(np.degrees(angle[:-1]), labels_closed[:-1])
            ax.set_ylim(0, 1)
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=45)
            ax.set_title('Performance Comparison (normalized)', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            
            plt.tight_layout()
            path = self.output_dir / f"{name}_radar_comparison.png"
            plt.savefig(path)
            plt.close()
            visualization_paths.append(str(path))
        
        logger.info(f"Created {len(visualization_paths)} comparison visualizations in {self.output_dir}")
        return visualization_paths

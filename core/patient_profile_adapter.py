"""
Patient Profile Adapter Module

This module implements the patient profile adaptation mechanisms for the MoE framework,
focusing on online adaptation, reinforcement learning for adaptation, and profile-specific
evaluation metrics.
"""
import os
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatientProfileAdapter:
    """
    Implements online adaptation and evaluation for patient profiles in the MoE framework.
    
    This module works with the PersonalizationLayer to provide real-time profile
    refinement, reinforcement learning components, and evaluation metrics for
    adaptation effectiveness.
    """
    
    def __init__(self, 
                 personalization_layer,
                 results_dir: str = 'results/patient_adaptation',
                 learning_rate: float = 0.05,
                 exploration_rate: float = 0.1,
                 reward_decay: float = 0.9):
        """
        Initialize the patient profile adapter.
        
        Parameters:
        -----------
        personalization_layer : PersonalizationLayer
            Reference to the personalization layer
        results_dir : str
            Directory to store adaptation results and evaluation metrics
        learning_rate : float
            Learning rate for reinforcement learning components
        exploration_rate : float
            Exploration rate for trying new adaptations
        reward_decay : float
            Discount factor for future rewards
        """
        self.personalization_layer = personalization_layer
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.reward_decay = reward_decay
        
        # State tracking for reinforcement learning
        self.state_history = {}
        self.action_history = {}
        self.reward_history = {}
        self.q_values = {}
        
        # Profile-specific metrics
        self.metrics = {}
        
    def update_profile_online(self, 
                              patient_id: str, 
                              new_data: pd.DataFrame,
                              prediction_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update a patient profile in real-time with new observation data.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        new_data : pd.DataFrame
            New observation data
        prediction_results : Dict[str, Any], optional
            Results from recent predictions for this patient
            
        Returns:
        --------
        Dict[str, Any]
            Updated profile information
        """
        if patient_id not in self.personalization_layer.patient_profiles:
            logger.warning(f"No existing profile for patient {patient_id}, creating one")
            return self.personalization_layer.create_patient_profile(patient_id, initial_data=new_data)
        
        profile = self.personalization_layer.patient_profiles[patient_id]
        
        # Initialize tracking dictionaries if needed
        self.state_history.setdefault(patient_id, [])
        self.action_history.setdefault(patient_id, [])
        self.reward_history.setdefault(patient_id, [])
        
        # Extract current state features
        current_state = self._extract_state_features(new_data, profile)
        
        # Record current state
        self.state_history[patient_id].append(current_state)
        
        # Determine adaptation action using reinforcement learning
        action = self._select_adaptation_action(patient_id, current_state)
        
        # Record action
        self.action_history[patient_id].append(action)
        
        # Apply the adaptation action
        update_results = self._apply_adaptation_action(patient_id, action, new_data)
        
        # Calculate reward based on prediction_results if available
        if prediction_results:
            reward = self._calculate_reward(patient_id, action, prediction_results)
            self.reward_history[patient_id].append(reward)
            
            # Update Q-values based on reward
            self._update_q_values(patient_id, current_state, action, reward)
        
        # Limit history size
        max_history = 100
        if len(self.state_history[patient_id]) > max_history:
            self.state_history[patient_id] = self.state_history[patient_id][-max_history:]
        if len(self.action_history[patient_id]) > max_history:
            self.action_history[patient_id] = self.action_history[patient_id][-max_history:]
        if len(self.reward_history[patient_id]) > max_history:
            self.reward_history[patient_id] = self.reward_history[patient_id][-max_history:]
            
        # Update profile-specific metrics
        self._update_profile_metrics(patient_id)
        
        return update_results
    
    def _extract_state_features(self, data: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract state features from data and profile for RL state representation.
        
        Parameters:
        -----------
        data : pd.DataFrame
            New observation data
        profile : Dict[str, Any]
            Patient profile
            
        Returns:
        --------
        Dict[str, float]
            State features
        """
        state = {}
        
        # Extract feature category importance from profile
        if "feature_categories" in profile:
            for category, importance in profile["feature_categories"].items():
                state[f"category_{category}"] = importance
                
        # Calculate recent trend from data if temporal data is available
        if "timestamp" in data.columns and "target" in data.columns:
            # Sort by timestamp
            sorted_data = data.sort_values("timestamp")
            
            # Calculate trend over the last few observations
            if len(sorted_data) >= 3:
                recent_values = sorted_data["target"].values[-3:]
                if all(recent_values[i] < recent_values[i+1] for i in range(len(recent_values)-1)):
                    state["trend"] = 1.0  # Increasing trend
                elif all(recent_values[i] > recent_values[i+1] for i in range(len(recent_values)-1)):
                    state["trend"] = -1.0  # Decreasing trend
                else:
                    state["trend"] = 0.0  # No clear trend
            
        # Add other relevant features from current data sample
        if len(data) > 0:
            latest_row = data.iloc[-1]
            for col in data.columns:
                if col not in ["patient_id", "timestamp", "target"]:
                    # Normalize the value (simple min-max scaling using profile history)
                    value = latest_row[col]
                    history_key = f"{col}_history"
                    
                    if history_key not in profile:
                        profile[history_key] = [value]
                    else:
                        profile[history_key].append(value)
                        # Limit history size
                        profile[history_key] = profile[history_key][-100:]
                    
                    # Calculate normalized value
                    history = profile[history_key]
                    min_val = min(history)
                    max_val = max(history)
                    
                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)
                    else:
                        normalized = 0.5  # Default if no variation
                    
                    state[f"feature_{col}"] = normalized
        
        return state
    
    def _select_adaptation_action(self, patient_id: str, state: Dict[str, float]) -> Dict[str, Any]:
        """
        Select an adaptation action using reinforcement learning.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        state : Dict[str, float]
            Current state features
            
        Returns:
        --------
        Dict[str, Any]
            Selected adaptation action
        """
        # Define possible adaptation actions
        actions = [
            {"type": "increase_adaptation_rate", "value": 0.05},
            {"type": "decrease_adaptation_rate", "value": -0.05},
            {"type": "boost_physiological", "expert_id": 0, "value": 0.1},
            {"type": "boost_environmental", "expert_id": 1, "value": 0.1},
            {"type": "boost_behavioral", "expert_id": 2, "value": 0.1},
            {"type": "no_action", "value": 0}
        ]
        
        # Convert state to a string key for q-values dictionary
        state_key = self._state_to_key(state)
        
        # Initialize Q-values for this state if they don't exist
        if state_key not in self.q_values:
            self.q_values[state_key] = {i: 0.0 for i in range(len(actions))}
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            # Random exploration
            action_idx = np.random.choice(range(len(actions)))
        else:
            # Greedy exploitation
            q_vals = self.q_values[state_key]
            action_idx = max(q_vals, key=q_vals.get)
        
        # Get the selected action
        selected_action = actions[action_idx]
        selected_action["idx"] = action_idx  # Add the index for reference
        
        return selected_action
    
    def _apply_adaptation_action(self, patient_id: str, action: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply the selected adaptation action to the patient profile.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        action : Dict[str, Any]
            Adaptation action to apply
        data : pd.DataFrame
            New observation data
            
        Returns:
        --------
        Dict[str, Any]
            Results of the adaptation
        """
        profile = self.personalization_layer.patient_profiles[patient_id]
        action_type = action["type"]
        action_value = action.get("value", 0)
        
        results = {
            "action_applied": action_type,
            "value_change": action_value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Apply the action based on its type
        if action_type == "no_action":
            results["details"] = "No adaptation applied"
            
        elif action_type == "increase_adaptation_rate":
            # Increase the adaptation rate
            old_rate = self.personalization_layer.adaptation_rate
            new_rate = min(0.5, old_rate + action_value)  # Cap at 0.5
            self.personalization_layer.adaptation_rate = new_rate
            results["details"] = f"Adaptation rate increased from {old_rate:.3f} to {new_rate:.3f}"
            
        elif action_type == "decrease_adaptation_rate":
            # Decrease the adaptation rate
            old_rate = self.personalization_layer.adaptation_rate
            new_rate = max(0.05, old_rate + action_value)  # Minimum of 0.05
            self.personalization_layer.adaptation_rate = new_rate
            results["details"] = f"Adaptation rate decreased from {old_rate:.3f} to {new_rate:.3f}"
            
        elif action_type.startswith("boost_"):
            # Boost a specific expert type
            expert_id = action.get("expert_id")
            if expert_id is not None:
                # Get the expert specialty name
                specialty = action_type.replace("boost_", "")
                
                # Update expert preferences in the profile
                if "expert_preferences" not in profile:
                    profile["expert_preferences"] = {}
                
                if specialty in profile["expert_preferences"]:
                    old_pref = profile["expert_preferences"][specialty]
                    new_pref = min(1.0, old_pref + action_value)  # Cap at 1.0
                    profile["expert_preferences"][specialty] = new_pref
                    results["details"] = f"{specialty} expert preference increased from {old_pref:.3f} to {new_pref:.3f}"
                else:
                    profile["expert_preferences"][specialty] = action_value
                    results["details"] = f"{specialty} expert preference set to {action_value:.3f}"
        
        # Record the adaptation in profile history
        if "adaptation_actions" not in profile:
            profile["adaptation_actions"] = []
            
        profile["adaptation_actions"].append({
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "value": action_value,
            "details": results.get("details", "")
        })
        
        # Limit history size
        if len(profile["adaptation_actions"]) > 50:
            profile["adaptation_actions"] = profile["adaptation_actions"][-50:]
        
        # Update profile
        profile["updated_at"] = datetime.now().isoformat()
        self.personalization_layer._save_patient_profile(patient_id)
        
        return results
    
    def _calculate_reward(self, patient_id: str, action: Dict[str, Any], prediction_results: Dict[str, Any]) -> float:
        """
        Calculate reward for the reinforcement learning based on prediction results.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        action : Dict[str, Any]
            Action that was taken
        prediction_results : Dict[str, Any]
            Results from prediction
            
        Returns:
        --------
        float
            Calculated reward
        """
        # Extract prediction performance metrics
        true_positives = prediction_results.get("true_positives", 0)
        true_negatives = prediction_results.get("true_negatives", 0)
        false_positives = prediction_results.get("false_positives", 0)
        false_negatives = prediction_results.get("false_negatives", 0)
        
        # Calculate f1 score as the primary reward signal
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Base reward on f1 score
        reward = f1
        
        # Penalty for no-action when performance is poor
        if action["type"] == "no_action" and f1 < 0.5:
            reward -= 0.2
            
        # Additional reward for actions that improve over previous results
        if len(self.reward_history.get(patient_id, [])) > 0:
            prev_rewards = self.reward_history[patient_id]
            if len(prev_rewards) > 0:
                prev_reward = prev_rewards[-1]
                if reward > prev_reward:
                    reward += 0.1  # Bonus for improvement
        
        return reward
    
    def _update_q_values(self, patient_id: str, state: Dict[str, float], action: Dict[str, Any], reward: float):
        """
        Update Q-values based on reward using Q-learning algorithm.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        state : Dict[str, float]
            Current state
        action : Dict[str, Any]
            Action taken
        reward : float
            Reward received
        """
        state_key = self._state_to_key(state)
        action_idx = action.get("idx", 0)
        
        # Get current Q-value
        if state_key not in self.q_values:
            self.q_values[state_key] = {i: 0.0 for i in range(6)}  # 6 possible actions
            
        current_q = self.q_values[state_key][action_idx]
        
        # Simple Q-learning update
        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        # where s' is the next state, a' is the best action in s'
        
        # Since we don't have the next state yet, use simpler update
        # Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
        self.q_values[state_key][action_idx] = current_q + self.learning_rate * (reward - current_q)
    
    def _state_to_key(self, state: Dict[str, float]) -> str:
        """
        Convert state dictionary to a string key for the Q-values table.
        
        Parameters:
        -----------
        state : Dict[str, float]
            State dictionary
            
        Returns:
        --------
        str
            String key representing the state
        """
        # Quantize continuous values to reduce state space
        quantized_state = {}
        for k, v in state.items():
            # Quantize to 0.2 increments
            quantized_state[k] = round(v * 5) / 5
            
        # Convert to sorted string representation
        return str(sorted(quantized_state.items()))
    
    def _update_profile_metrics(self, patient_id: str):
        """
        Update metrics tracking for a specific patient profile.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        """
        if patient_id not in self.personalization_layer.patient_profiles:
            return
            
        profile = self.personalization_layer.patient_profiles[patient_id]
        
        # Initialize metrics dict for this patient if needed
        if patient_id not in self.metrics:
            self.metrics[patient_id] = {
                "adaptation_count": 0,
                "adaptation_effectiveness": 0.0,
                "reward_history": [],
                "expert_weight_history": [],
                "last_updated": datetime.now().isoformat()
            }
            
        # Update adaptation count
        adaptation_actions = profile.get("adaptation_actions", [])
        self.metrics[patient_id]["adaptation_count"] = len(adaptation_actions)
        
        # Update reward history
        rewards = self.reward_history.get(patient_id, [])
        self.metrics[patient_id]["reward_history"] = rewards
        
        # Calculate overall adaptation effectiveness
        if rewards:
            recent_rewards = rewards[-10:] if len(rewards) > 10 else rewards
            self.metrics[patient_id]["adaptation_effectiveness"] = sum(recent_rewards) / len(recent_rewards)
            
        # Update timestamp
        self.metrics[patient_id]["last_updated"] = datetime.now().isoformat()
    
    def get_adaptation_effectiveness(self, patient_id: str) -> Dict[str, Any]:
        """
        Get adaptation effectiveness metrics for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
            
        Returns:
        --------
        Dict[str, Any]
            Adaptation metrics
        """
        if patient_id not in self.metrics:
            logger.warning(f"No metrics available for patient {patient_id}")
            return {}
            
        return self.metrics[patient_id]
    
    def visualize_adaptation_impact(self, patient_id: str, output_path: str = None):
        """
        Create visualization of adaptation impact for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier
        output_path : str, optional
            Path to save the visualization
            
        Returns:
        --------
        str
            Path to the saved visualization
        """
        if patient_id not in self.metrics or patient_id not in self.reward_history:
            logger.warning(f"Not enough data to visualize adaptation for patient {patient_id}")
            return None
            
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot reward history
        rewards = self.reward_history[patient_id]
        axs[0].plot(rewards, marker='o')
        axs[0].set_title(f"Reward History for Patient {patient_id}")
        axs[0].set_xlabel("Adaptation Step")
        axs[0].set_ylabel("Reward")
        axs[0].grid(True)
        
        # Plot adaptation actions
        profile = self.personalization_layer.patient_profiles[patient_id]
        actions = profile.get("adaptation_actions", [])
        
        if actions:
            action_types = [a["action"] for a in actions]
            action_values = [a["value"] for a in actions]
            
            # Count action types
            action_counts = {}
            for action in action_types:
                action_counts[action] = action_counts.get(action, 0) + 1
                
            # Plot action distribution
            labels = list(action_counts.keys())
            sizes = list(action_counts.values())
            axs[1].bar(labels, sizes)
            axs[1].set_title(f"Adaptation Actions for Patient {patient_id}")
            axs[1].set_xlabel("Action Type")
            axs[1].set_ylabel("Count")
            axs[1].grid(True)
            
        plt.tight_layout()
        
        # Save figure if output path is provided
        if output_path:
            save_path = output_path
        else:
            save_path = self.results_dir / f"adaptation_impact_{patient_id}.png"
            
        plt.savefig(save_path)
        logger.info(f"Saved adaptation visualization to {save_path}")
        
        plt.close(fig)
        return str(save_path)
    
    def compare_profiles(self, patient_ids: List[str], metric: str = "adaptation_effectiveness") -> Dict[str, Any]:
        """
        Compare adaptation effectiveness across multiple patient profiles.
        
        Parameters:
        -----------
        patient_ids : List[str]
            List of patient identifiers to compare
        metric : str
            Metric to use for comparison
            
        Returns:
        --------
        Dict[str, Any]
            Comparison results
        """
        results = {
            "metric": metric,
            "patient_values": {},
            "best_performing": None,
            "worst_performing": None,
            "average": 0.0
        }
        
        valid_metrics = []
        
        # Collect metric values for each patient
        for patient_id in patient_ids:
            if patient_id in self.metrics:
                metric_value = self.metrics[patient_id].get(metric, 0.0)
                results["patient_values"][patient_id] = metric_value
                valid_metrics.append(metric_value)
        
        # Calculate statistics
        if valid_metrics:
            results["average"] = sum(valid_metrics) / len(valid_metrics)
            
            # Find best and worst performing
            if results["patient_values"]:
                results["best_performing"] = max(results["patient_values"].items(), key=lambda x: x[1])[0]
                results["worst_performing"] = min(results["patient_values"].items(), key=lambda x: x[1])[0]
                
        return results

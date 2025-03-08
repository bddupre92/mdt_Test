"""
Track and analyze optimizer selection patterns.
"""
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict


class SelectionTracker:
    """Track and analyze optimizer selection patterns."""
    
    def __init__(self, tracker_file: Optional[str] = None):
        """
        Initialize selection tracker.
        
        Args:
            tracker_file: Optional path to save/load selection data
        """
        self.tracker_file = tracker_file
        self.selections = defaultdict(list)  # {problem_type: [(optimizer, success, score)]}
        
        if tracker_file and Path(tracker_file).exists():
            self.load_data()
            
    def record_selection(self, 
                        problem_type: str,
                        optimizer: str,
                        features: Dict[str, float],
                        success: bool,
                        score: float) -> None:
        """
        Record an optimizer selection.
        
        Args:
            problem_type: Type of problem (e.g., 'sphere', 'rastrigin')
            optimizer: Name of selected optimizer
            features: Problem features when selection was made
            success: Whether optimization was successful
            score: Final score achieved
        """
        self.selections[problem_type].append({
            'optimizer': optimizer,
            'features': features,
            'success': success,
            'score': score,
        })
        
        if self.tracker_file:
            self.save_data()
            
    def get_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the complete selection history.
        
        Returns:
            Dictionary mapping problem types to lists of selection records
        """
        # Convert defaultdict to regular dict for serialization
        history = {}
        for problem_type, selections in self.selections.items():
            # Convert any numpy values to standard Python types for serialization
            processed_selections = []
            for selection in selections:
                processed_selection = {}
                for key, value in selection.items():
                    if key == 'features' and isinstance(value, dict):
                        processed_selection[key] = {k: float(v) if isinstance(v, np.generic) else v 
                                                  for k, v in value.items()}
                    elif isinstance(value, np.generic):
                        processed_selection[key] = value.item()
                    else:
                        processed_selection[key] = value
                processed_selections.append(processed_selection)
            history[problem_type] = processed_selections
        
        return history
            
    def get_selection_stats(self, problem_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get statistics about optimizer selections.
        
        Args:
            problem_type: Optional problem type to filter by
            
        Returns:
            DataFrame with selection statistics
        """
        data = []
        problems = [problem_type] if problem_type else self.selections.keys()
        
        for prob in problems:
            if prob not in self.selections:
                continue
                
            # Count selections and successes per optimizer
            optimizer_stats = defaultdict(lambda: {'selections': 0, 'successes': 0, 'total_score': 0.0})
            
            for record in self.selections[prob]:
                opt = record['optimizer']
                optimizer_stats[opt]['selections'] += 1
                optimizer_stats[opt]['successes'] += int(record['success'])
                optimizer_stats[opt]['total_score'] += record['score']
                
            # Calculate statistics
            for opt, stats in optimizer_stats.items():
                data.append({
                    'problem_type': prob,
                    'optimizer': opt,
                    'selections': stats['selections'],
                    'success_rate': stats['successes'] / stats['selections'],
                    'avg_score': stats['total_score'] / stats['selections']
                })
                
        return pd.DataFrame(data)
        
    def get_feature_correlations(self, problem_type: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze correlations between features and optimizer success.
        
        Args:
            problem_type: Optional problem type to filter by
            
        Returns:
            Dictionary mapping features to their correlation with success for each optimizer
        """
        all_records = []
        problems = [problem_type] if problem_type else self.selections.keys()
        
        for prob in problems:
            all_records.extend(self.selections[prob])
            
        if not all_records:
            return {}
            
        # Convert to DataFrame for analysis
        records_df = pd.DataFrame(all_records)
        feature_names = list(all_records[0]['features'].keys())
        
        # Extract features into separate columns
        for feat in feature_names:
            records_df[f'feature_{feat}'] = records_df['features'].apply(lambda x: x[feat])
            
        correlations = {}
        for opt in records_df['optimizer'].unique():
            opt_data = records_df[records_df['optimizer'] == opt]
            if len(opt_data) < 2:  # Need at least 2 samples for correlation
                continue
                
            opt_corr = {}
            for feat in feature_names:
                feat_col = f'feature_{feat}'
                corr = opt_data[feat_col].corr(opt_data['success'])
                if not np.isnan(corr):
                    opt_corr[feat] = corr
                    
            correlations[opt] = opt_corr
            
        return correlations
        
    def update_correlations(self, problem_type: str, optimizer_states: Dict[str, Any]) -> None:
        """
        Update correlations between optimizer states and performance.
        
        Args:
            problem_type: Type of problem
            optimizer_states: Dictionary of optimizer states
        """
        # Extract state metrics for correlation analysis
        for optimizer, state in optimizer_states.items():
            state_dict = state.to_dict() if hasattr(state, 'to_dict') else state
            
            # Record the state metrics for this optimizer
            if problem_type not in self.selections:
                self.selections[problem_type] = []
                
            # Find the last entry for this optimizer if it exists
            for entry in reversed(self.selections[problem_type]):
                if entry['optimizer'] == optimizer:
                    # Update with state metrics
                    entry['state_metrics'] = state_dict
                    break
        
        # Save updated data
        if self.tracker_file:
            self.save_data()
            
    def save_data(self) -> None:
        """Save selection data to file."""
        if self.tracker_file:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.selections, f, indent=2)
                
    def load_data(self) -> None:
        """Load selection data from file."""
        if self.tracker_file and Path(self.tracker_file).exists():
            with open(self.tracker_file, 'r') as f:
                self.selections = defaultdict(list, json.load(f))

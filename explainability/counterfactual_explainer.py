"""
counterfactual_explainer.py
-------------------------
Counterfactual explanation implementation for MoE framework
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import json
from .base_explainer import BaseExplainer

# Import counterfactual libraries
try:
    import alibi
    from alibi.explainers import CounterfactualProto
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False
    
try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False

# TensorFlow compatibility layer for Keras backend
try:
    import tensorflow as tf
    # Add a compatibility layer for the get_session function
    if not hasattr(tf.keras.backend, 'get_session'):
        def get_session():
            # Return the current TF session or create a new one
            return tf.compat.v1.keras.backend.get_session()
        # Add the function to the keras backend
        tf.keras.backend.get_session = get_session
        
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class CounterfactualExplainer(BaseExplainer):
    """
    Explainer using counterfactual methods
    
    This explainer generates counterfactual explanations that show how input features
    would need to change to achieve a different prediction outcome.
    """
    
    def __init__(self, model=None, feature_names: Optional[List[str]] = None, 
                 method: str = 'alibi', categorical_features: Optional[List[int]] = None,
                 feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 **kwargs):
        """
        Initialize Counterfactual explainer
        
        Args:
            model: Pre-trained model to explain
            feature_names: List of feature names
            method: Counterfactual method to use ('alibi' or 'dice')
            categorical_features: Indices of categorical features
            feature_ranges: Dictionary mapping feature names to (min, max) ranges
            **kwargs: Additional parameters for counterfactual explainer
        """
        super().__init__('Counterfactual', model, feature_names)
        self.method = method.lower()
        self.categorical_features = categorical_features or []
        self.feature_ranges = feature_ranges or {}
        self.explainer_kwargs = kwargs
        self.explainer = None
        self.counterfactuals = None
        
        # Check if required libraries are available
        if self.method == 'alibi' and not ALIBI_AVAILABLE:
            raise ImportError("Alibi library is required for 'alibi' method. Install with 'pip install alibi'")
        if self.method == 'dice' and not DICE_AVAILABLE:
            raise ImportError("DiCE library is required for 'dice' method. Install with 'pip install dice-ml'")
        
        # Define supported plot types
        self.supported_plot_types = [
            'feature_change', 'comparison', 'radar', 'parallel'
        ]
    
    def _create_explainer(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Create the appropriate counterfactual explainer
        
        Args:
            X: Training data for the explainer
        """
        if isinstance(X, pd.DataFrame) and self.feature_names is None:
            self.feature_names = list(X.columns)
        
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        if self.method == 'alibi':
            # Create Alibi CounterfactualProto explainer
            if not hasattr(self.model, 'predict') and hasattr(self.model, 'predict_proba'):
                # If model doesn't have predict but has predict_proba, create a wrapper
                class ModelWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def predict(self, X):
                        probs = self.model.predict_proba(X)
                        return np.argmax(probs, axis=1)
                
                self.model_wrapper = ModelWrapper(self.model)
                predict_fn = self.model_wrapper.predict
            else:
                predict_fn = self.model.predict
            
            # Create the explainer with appropriate parameters for Alibi version
            try:
                # Try the newer Alibi API first
                self.explainer = CounterfactualProto(
                    predictor=predict_fn,  # Newer versions use 'predictor' instead of 'predict_fn'
                    shape=X_np[0].shape,
                    kappa=self.explainer_kwargs.get('kappa', 0.0),
                    beta=self.explainer_kwargs.get('beta', 0.1),
                    cat_vars=self.categorical_features,
                    **{k: v for k, v in self.explainer_kwargs.items() 
                       if k not in ['kappa', 'beta', 'feature_names']}
                )
            except TypeError:
                # Fall back to older Alibi API
                self.explainer = CounterfactualProto(
                    predict=predict_fn,  # Older versions use 'predict'
                    shape=X_np[0].shape,
                    kappa=self.explainer_kwargs.get('kappa', 0.0),
                    beta=self.explainer_kwargs.get('beta', 0.1),
                    cat_vars=self.categorical_features,
                    **{k: v for k, v in self.explainer_kwargs.items() 
                       if k not in ['kappa', 'beta', 'feature_names']}
                )
            # Fit the explainer with training data
            self.explainer.fit(X_np)
            
        elif self.method == 'dice':
            # Create DiCE explainer
            # Convert data to DataFrame if it's not already
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = X
            
            # Identify continuous and categorical features
            if self.categorical_features:
                categorical_names = [self.feature_names[i] for i in self.categorical_features]
                continuous_names = [f for f in self.feature_names if f not in categorical_names]
            else:
                categorical_names = []
                continuous_names = self.feature_names
            
            # Create data and model objects for DiCE
            d = dice_ml.Data(
                dataframe=X_df,
                continuous_features=continuous_names,
                categorical_features=categorical_names
            )
            
            # Create appropriate model wrapper based on model type
            if hasattr(self.model, 'predict_proba'):
                m = dice_ml.Model(model=self.model, backend='sklearn')
            else:
                # For custom models, create a function wrapper
                def custom_predict(x):
                    return self.model.predict(x)
                
                m = dice_ml.Model(model=custom_predict, backend='sklearn')
            
            # Create the DiCE explainer
            self.explainer = dice_ml.Dice(d, m)
        else:
            raise ValueError(f"Unsupported counterfactual method: {self.method}")
    
    def explain(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None, 
               target_class: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate counterfactual explanations for the provided data
        
        Args:
            X: Input features to explain
            y: Optional target values
            target_class: Target class for counterfactual (for classification)
            **kwargs: Additional parameters for specific counterfactual methods
            
        Returns:
            Dictionary containing explanation data
        """
        # Convert to numpy if DataFrame
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        
        # If X is a batch, we'll explain the first instance by default
        instance_idx = kwargs.get('instance_idx', 0)
        instance = X_np[instance_idx].reshape(1, -1)
        
        # Create explainer if not already created
        if self.explainer is None:
            self._create_explainer(X)
        
        # Generate counterfactuals
        if self.method == 'alibi':
            # For regression tasks, we need to specify a desired outcome
            # instead of a target class
            desired_outcome = kwargs.get('desired_outcome', None)
            
            try:
                # Try the newer Alibi API first
                explanation = self.explainer.explain(instance, desired_outcome=desired_outcome)
            except TypeError as type_error:
                # Fall back to older Alibi API or different parameter names
                try:
                    explanation = self.explainer.explain(instance, target_class=target_class)
                except Exception as e:
                    # Check for Keras backend error
                    if "get_session" in str(e):
                        logger.warning("Encountered Keras backend get_session error. Attempting workaround...")
                        try:
                            # Try to use TensorFlow compatibility layer
                            if TF_AVAILABLE:
                                explanation = self.explainer.explain(instance, desired_outcome=desired_outcome)
                            else:
                                raise ImportError("TensorFlow not available for compatibility layer")
                        except Exception as tf_error:
                            logger.error(f"TensorFlow compatibility layer failed: {tf_error}")
                            # If all else fails, create a minimal explanation structure
                            return {
                                'original_instance': instance[0],
                                'counterfactuals': [],
                                'feature_names': self.feature_names,
                                'original_prediction': self.model.predict(instance)[0],
                                'method': 'alibi',
                                'success': False,
                                'error_message': f"Original error: {e}. TF error: {tf_error}"
                            }
                    else:
                        # If all else fails, create a minimal explanation structure
                        logger.warning(f"Error generating counterfactual with Alibi: {e}")
                        return {
                            'original_instance': instance[0],
                            'counterfactuals': [],
                            'feature_names': self.feature_names,
                            'original_prediction': self.model.predict(instance)[0],
                            'method': 'alibi',
                            'success': False,
                            'error_message': str(e)
                        }
            
            self.counterfactuals = explanation
            
            # Try to extract counterfactual instances based on different Alibi versions
            try:
                # Newer Alibi versions
                if hasattr(explanation, 'cf') and 'X' in explanation.cf:
                    cf_instances = explanation.cf['X']
                    success = explanation.cf.get('success', False)
                # Older Alibi versions or different return structure
                elif hasattr(explanation, 'data') and hasattr(explanation.data, 'get'):
                    cf_instances = explanation.data.get('counterfactual_instances', [])
                    success = explanation.success if hasattr(explanation, 'success') else False
                else:
                    # Fallback for unknown structure
                    cf_instances = []
                    success = False
            except Exception as e:
                logger.warning(f"Error extracting counterfactual instances: {e}")
                cf_instances = []
                success = False
            
            # Create explanation data
            explanation_data = {
                'original_instance': instance[0],
                'counterfactuals': cf_instances if len(cf_instances) > 0 else [],
                'feature_names': self.feature_names,
                'original_prediction': self.model.predict(instance)[0],
                'method': 'alibi',
                'success': success,
            }
            
            # Add counterfactual prediction if available
            if len(cf_instances) > 0:
                cf_instance = cf_instances[0].reshape(1, -1) if cf_instances[0].ndim == 1 else cf_instances[0]
                explanation_data['counterfactual_prediction'] = self.model.predict(cf_instance)[0]
                explanation_data['feature_changes'] = cf_instance.flatten() - instance[0]
            
        elif self.method == 'dice':
            # For DiCE, convert to DataFrame
            if not isinstance(X, pd.DataFrame):
                X_df = pd.DataFrame(X, columns=self.feature_names)
                instance_df = pd.DataFrame(instance, columns=self.feature_names)
            else:
                X_df = X
                instance_df = X.iloc[[instance_idx]]
            
            # Generate counterfactuals
            dice_exp = self.explainer.generate_counterfactuals(
                instance_df, 
                total_CFs=kwargs.get('num_counterfactuals', 3),
                desired_class=target_class
            )
            self.counterfactuals = dice_exp
            
            # Extract counterfactual instances
            cf_instances = dice_exp.cf_examples_list[0].final_cfs_df
            
            # Calculate feature changes for the first counterfactual
            if len(cf_instances) > 0:
                cf_instance = cf_instances.iloc[0].values
                feature_changes = cf_instance - instance[0]
                
                # Get predictions
                orig_pred = self.model.predict(instance)[0]
                cf_pred = self.model.predict(cf_instance.reshape(1, -1))[0]
                
                explanation_data = {
                    'original_instance': instance[0],
                    'counterfactual_instance': cf_instance,
                    'feature_changes': feature_changes,
                    'feature_names': self.feature_names,
                    'original_prediction': orig_pred,
                    'counterfactual_prediction': cf_pred,
                    'method': 'dice',
                    'success': True,
                    'all_counterfactuals': cf_instances.values
                }
            else:
                explanation_data = {
                    'original_instance': instance[0],
                    'counterfactual_instance': None,
                    'feature_changes': None,
                    'feature_names': self.feature_names,
                    'original_prediction': self.model.predict(instance)[0],
                    'counterfactual_prediction': None,
                    'method': 'dice',
                    'success': False
                }
        
        self.explanation_data = explanation_data
        self.last_explanation = explanation_data
        return explanation_data
    
    def plot(self, plot_type: str = 'feature_change', **kwargs) -> Any:
        """
        Generate plots for counterfactual explanations
        
        Args:
            plot_type: Type of plot to generate
            **kwargs: Additional parameters for specific plot types
            
        Returns:
            Matplotlib figure, Plotly figure, or HTML string depending on plot type
        """
        if self.last_explanation is None:
            raise ValueError("No explanation available. Run explain() first.")
        
        if not self.last_explanation.get('success', False):
            logger.warning("Counterfactual generation was not successful. Plot may be empty.")
        
        if plot_type not in self.supported_plot_types:
            raise ValueError(f"Unsupported plot type: {plot_type}. Choose from {self.supported_plot_types}")
        
        # Extract data from explanation
        original = self.last_explanation['original_instance']
        feature_names = self.last_explanation['feature_names']
        
        # Check for counterfactuals in both formats (for compatibility)
        counterfactual = self.last_explanation.get('counterfactual_instance')
        counterfactuals = self.last_explanation.get('counterfactuals', [])
        
        # If we have counterfactuals in the new format but not the old format, use the first one
        if counterfactual is None and counterfactuals and len(counterfactuals) > 0:
            counterfactual = counterfactuals[0]
            # Also update feature_changes if needed
            if 'feature_changes' not in self.last_explanation and counterfactual is not None:
                self.last_explanation['feature_changes'] = counterfactual - original
                
        # Ensure feature_names are available
        if not feature_names and self.feature_names is not None:
            feature_names = self.feature_names
            self.last_explanation['feature_names'] = feature_names
        elif not feature_names:
            # Create default feature names if none are available
            num_features = len(original)
            feature_names = [f'Feature {i+1}' for i in range(num_features)]
            self.last_explanation['feature_names'] = feature_names
        
        if counterfactual is None:
            logger.warning("No counterfactual instance available. Creating a simple plot of original instance.")
            # Create a simple bar chart of the original instance
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(feature_names))
            ax.bar(x, original, color='blue', alpha=0.7, label='Original Instance')
            ax.set_xlabel('Features')
            ax.set_ylabel('Value')
            ax.set_title('Original Instance (No Counterfactual Found)')
            ax.set_xticks(x)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            return fig
        
        if plot_type == 'feature_change':
            # Create a bar plot showing feature changes
            feature_changes = self.last_explanation['feature_changes']
            
            # Use Plotly for interactive visualization
            fig = go.Figure()
            
            # Add bar for each feature change
            fig.add_trace(go.Bar(
                x=feature_names,
                y=feature_changes,
                marker_color=['red' if x < 0 else 'green' for x in feature_changes],
                name='Feature Changes'
            ))
            
            # Add horizontal line at y=0
            fig.add_shape(
                type='line',
                x0=-0.5,
                y0=0,
                x1=len(feature_names)-0.5,
                y1=0,
                line=dict(color='black', width=1, dash='dash')
            )
            
            # Update layout
            fig.update_layout(
                title='Feature Changes for Counterfactual Explanation',
                xaxis_title='Features',
                yaxis_title='Change in Value',
                template='plotly_white',
                height=500,
                width=800,
                xaxis=dict(tickangle=-45)
            )
            
            return fig
            
        elif plot_type == 'comparison':
            # Create a comparison plot between original and counterfactual
            
            # Create a DataFrame for plotting
            df = pd.DataFrame({
                'Feature': feature_names * 2,
                'Value': np.concatenate([original, counterfactual]),
                'Type': ['Original'] * len(feature_names) + ['Counterfactual'] * len(feature_names)
            })
            
            # Use Plotly for interactive visualization
            fig = px.bar(
                df, 
                x='Feature', 
                y='Value', 
                color='Type',
                barmode='group',
                title='Comparison: Original vs Counterfactual',
                color_discrete_map={'Original': 'blue', 'Counterfactual': 'orange'}
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Features',
                yaxis_title='Value',
                template='plotly_white',
                height=500,
                width=800,
                xaxis=dict(tickangle=-45)
            )
            
            return fig
            
        elif plot_type == 'radar':
            # Create a radar plot comparing original and counterfactual
            
            # Normalize values for better visualization
            min_vals = np.min([original, counterfactual], axis=0)
            max_vals = np.max([original, counterfactual], axis=0)
            
            # Avoid division by zero
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1
            
            # Normalize to 0-1 range
            orig_norm = (original - min_vals) / ranges
            cf_norm = (counterfactual - min_vals) / ranges
            
            # Create radar plot
            fig = go.Figure()
            
            # Add original instance
            fig.add_trace(go.Scatterpolar(
                r=orig_norm,
                theta=feature_names,
                fill='toself',
                name='Original'
            ))
            
            # Add counterfactual instance
            fig.add_trace(go.Scatterpolar(
                r=cf_norm,
                theta=feature_names,
                fill='toself',
                name='Counterfactual'
            ))
            
            # Update layout
            fig.update_layout(
                title='Radar Plot: Original vs Counterfactual',
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                height=600,
                width=800
            )
            
            return fig
            
        elif plot_type == 'parallel':
            # Create parallel coordinates plot
            
            # Combine original and counterfactual into a DataFrame
            df = pd.DataFrame([original, counterfactual], columns=feature_names)
            df['type'] = ['Original', 'Counterfactual']
            
            # Create parallel coordinates plot
            fig = px.parallel_coordinates(
                df,
                dimensions=feature_names,
                color='type',
                color_continuous_scale=px.colors.sequential.Blues,
                title='Parallel Coordinates: Original vs Counterfactual'
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                width=900
            )
            
            return fig
    
    def visualize(self, explanation: Dict[str, Any], plot_type: str = 'comparison', **kwargs) -> Any:
        """
        Visualize the counterfactual explanation
        
        Args:
            explanation: Explanation data from explain() method
            plot_type: Type of plot to generate ('comparison', 'feature_change', 'radar', 'parallel')
            **kwargs: Additional parameters for visualization
            
        Returns:
            Matplotlib figure or Plotly figure depending on plot type
        """
        # Ensure feature_names are available
        if 'feature_names' not in explanation and self.feature_names is not None:
            explanation['feature_names'] = self.feature_names
        elif 'feature_names' not in explanation and self.feature_names is None:
            # Create default feature names if none are available
            if 'original_instance' in explanation:
                num_features = len(explanation['original_instance'])
                explanation['feature_names'] = [f'Feature {i+1}' for i in range(num_features)]
            else:
                logger.error("Cannot visualize: No feature names available and cannot determine number of features")
                return None
        # Store the explanation data for plotting
        self.last_explanation = explanation
        
        # Call the plot method to generate the visualization
        return self.plot(plot_type=plot_type, **kwargs)
    
    def save_explanation(self, filepath: Union[str, Path], format: str = 'json') -> str:
        """
        Save explanation data to file
        
        Args:
            filepath: Path to save the explanation
            format: Format to save ('json' or 'html')
            
        Returns:
            Path to the saved file
        """
        if self.last_explanation is None:
            raise ValueError("No explanation available. Run explain() first.")
        
        filepath = Path(filepath)
        
        if format.lower() == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = {
                'original_instance': self.last_explanation['original_instance'].tolist(),
                'feature_names': self.last_explanation['feature_names'],
                'original_prediction': float(self.last_explanation['original_prediction']),
                'method': self.last_explanation['method'],
                'success': self.last_explanation['success']
            }
            
            # Add counterfactual data if available
            if self.last_explanation.get('counterfactual_instance') is not None:
                json_data.update({
                    'counterfactual_instance': self.last_explanation['counterfactual_instance'].tolist(),
                    'feature_changes': self.last_explanation['feature_changes'].tolist(),
                    'counterfactual_prediction': float(self.last_explanation['counterfactual_prediction'])
                })
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            return str(filepath)
        
        elif format.lower() == 'html':
            # Generate HTML report with interactive visualizations
            html_content = self.to_html(**kwargs)
            
            # Save to HTML file
            with open(filepath, 'w') as f:
                f.write(html_content)
            
            return str(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Choose from ['json', 'html']")
    
    def to_html(self, **kwargs) -> str:
        """
        Convert explanation to HTML format with interactive visualizations
        
        Args:
            **kwargs: Additional parameters for HTML generation
            
        Returns:
            HTML string
        """
        if self.last_explanation is None:
            raise ValueError("No explanation available. Run explain() first.")
        
        # Create HTML content
        html_parts = []
        
        # Add header
        html_parts.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Counterfactual Explanation</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 40px; }
                .plot { width: 100%; height: 500px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Counterfactual Explanation</h1>
                </div>
        """)
        
        # Add summary section
        html_parts.append("""
                <div class="section">
                    <h2>Summary</h2>
                    <p>This report shows counterfactual explanations that indicate how features would need to change to achieve a different prediction.</p>
        """)
        
        # Add prediction information
        orig_pred = self.last_explanation['original_prediction']
        cf_pred = self.last_explanation.get('counterfactual_prediction')
        
        html_parts.append(f"""
                    <div>
                        <p><strong>Original Prediction:</strong> {orig_pred:.4f}</p>
        """)
        
        if cf_pred is not None:
            html_parts.append(f"""
                        <p><strong>Counterfactual Prediction:</strong> {cf_pred:.4f}</p>
                        <p><strong>Prediction Difference:</strong> {cf_pred - orig_pred:.4f}</p>
            """)
        
        html_parts.append("""
                    </div>
                </div>
        """)
        
        # Add feature comparison table
        html_parts.append("""
                <div class="section">
                    <h2>Feature Comparison</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Original Value</th>
                            <th>Counterfactual Value</th>
                            <th>Change</th>
                            <th>Percent Change</th>
                        </tr>
        """)
        
        # Add rows for each feature
        for i, feature in enumerate(self.last_explanation['feature_names']):
            orig_val = self.last_explanation['original_instance'][i]
            
            if self.last_explanation.get('counterfactual_instance') is not None:
                cf_val = self.last_explanation['counterfactual_instance'][i]
                change = cf_val - orig_val
                
                # Calculate percent change, handling division by zero
                if orig_val != 0:
                    pct_change = (change / abs(orig_val)) * 100
                    pct_change_str = f"{pct_change:.2f}%"
                else:
                    pct_change_str = "N/A"
                
                # Color code based on direction of change
                if change > 0:
                    change_color = "green"
                elif change < 0:
                    change_color = "red"
                else:
                    change_color = "black"
                
                html_parts.append(f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{orig_val:.4f}</td>
                            <td>{cf_val:.4f}</td>
                            <td style="color: {change_color}">{change:.4f}</td>
                            <td style="color: {change_color}">{pct_change_str}</td>
                        </tr>
                """)
            else:
                html_parts.append(f"""
                        <tr>
                            <td>{feature}</td>
                            <td>{orig_val:.4f}</td>
                            <td>N/A</td>
                            <td>N/A</td>
                            <td>N/A</td>
                        </tr>
                """)
        
        html_parts.append("""
                    </table>
                </div>
        """)
        
        # Add visualizations
        if self.last_explanation.get('counterfactual_instance') is not None:
            html_parts.append("""
                <div class="section">
                    <h2>Visualizations</h2>
            """)
            
            # Add feature change plot
            fig_change = self.plot('feature_change')
            html_parts.append("""
                    <div class="plot-container">
                        <h3>Feature Changes</h3>
                        <div class="plot" id="feature-change-plot"></div>
                    </div>
                    <script>
                        var featureChangePlotData = 
            """)
            html_parts.append(fig_change.to_json())
            html_parts.append("""
                        ;
                        Plotly.newPlot('feature-change-plot', featureChangePlotData.data, featureChangePlotData.layout);
                    </script>
            """)
            
            # Add comparison plot
            fig_comparison = self.plot('comparison')
            html_parts.append("""
                    <div class="plot-container">
                        <h3>Original vs Counterfactual Comparison</h3>
                        <div class="plot" id="comparison-plot"></div>
                    </div>
                    <script>
                        var comparisonPlotData = 
            """)
            html_parts.append(fig_comparison.to_json())
            html_parts.append("""
                        ;
                        Plotly.newPlot('comparison-plot', comparisonPlotData.data, comparisonPlotData.layout);
                    </script>
            """)
            
            # Add radar plot
            fig_radar = self.plot('radar')
            html_parts.append("""
                    <div class="plot-container">
                        <h3>Radar Plot</h3>
                        <div class="plot" id="radar-plot"></div>
                    </div>
                    <script>
                        var radarPlotData = 
            """)
            html_parts.append(fig_radar.to_json())
            html_parts.append("""
                        ;
                        Plotly.newPlot('radar-plot', radarPlotData.data, radarPlotData.layout);
                    </script>
            """)
            
            html_parts.append("""
                </div>
            """)
        
        # Close HTML
        html_parts.append("""
            </div>
        </body>
        </html>
        """)
        
        return "".join(html_parts)

"""
Expert visualization module for MoE framework.

This module provides specialized visualization functions for the MoE framework,
focusing on expert-specific analysis such as expert weights, agreement metrics,
confidence analysis, and contribution patterns.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.cm as cm
import glob
from datetime import datetime


def extract_expert_data_from_workflow(workflow_data, expert_names=None):
    """
    Extract expert-related data from a workflow JSON file.
    
    Args:
        workflow_data: A dictionary containing workflow data
        expert_names: A list of expert names to extract (if None, extract all)
        
    Returns:
        A dictionary with expert data, including:
        - weights: DataFrame of expert weights over time
        - contributions: DataFrame of expert contributions per sample
        - agreement: DataFrame of expert agreement metrics
        - confidence: DataFrame of expert confidence metrics
    """
    result = {}
    
    try:
        # Extract expert weights over time
        if 'gating_weights' in workflow_data:
            weights_data = workflow_data['gating_weights']
            if isinstance(weights_data, dict) and weights_data:
                # Convert to DataFrame
                weights_df = pd.DataFrame(weights_data)
                
                # Filter expert names if provided
                if expert_names:
                    weights_df = weights_df[[col for col in weights_df.columns if col in expert_names]]
                
                # Add time/iteration column if not present
                if 'time' not in weights_df.columns and 'iteration' not in weights_df.columns:
                    weights_df['iteration'] = list(range(len(weights_df)))
                
                result['weights'] = weights_df
                
        # Extract expert contributions per sample
        if 'expert_contributions' in workflow_data:
            contrib_data = workflow_data['expert_contributions']
            if isinstance(contrib_data, list) and contrib_data:
                # Convert to DataFrame
                contrib_rows = []
                
                for sample_idx, sample_contrib in enumerate(contrib_data):
                    for expert_name, value in sample_contrib.items():
                        if expert_names is None or expert_name in expert_names:
                            contrib_rows.append({
                                'sample_id': sample_idx,
                                'expert_name': expert_name,
                                'contribution': value
                            })
                
                if contrib_rows:
                    result['contributions'] = pd.DataFrame(contrib_rows)
                    
        # Extract expert agreement metrics
        if 'expert_agreement' in workflow_data:
            agreement_data = workflow_data['expert_agreement']
            if isinstance(agreement_data, dict) and agreement_data:
                # Convert to DataFrame
                agreement_rows = []
                
                for metric_name, values in agreement_data.items():
                    if isinstance(values, dict):
                        for expert1, expert2_values in values.items():
                            if expert_names is None or expert1 in expert_names:
                                for expert2, value in expert2_values.items():
                                    if expert_names is None or expert2 in expert_names:
                                        agreement_rows.append({
                                            'metric': metric_name,
                                            'expert1': expert1,
                                            'expert2': expert2,
                                            'value': value
                                        })
                
                if agreement_rows:
                    result['agreement'] = pd.DataFrame(agreement_rows)
                    
        # Extract expert confidence metrics
        if 'expert_confidence' in workflow_data:
            confidence_data = workflow_data['expert_confidence']
            if isinstance(confidence_data, dict) and confidence_data:
                # Convert to DataFrame
                confidence_rows = []
                
                for expert_name, values in confidence_data.items():
                    if expert_names is None or expert_name in expert_names:
                        if isinstance(values, dict):
                            for sample_id, value in values.items():
                                confidence_rows.append({
                                    'expert_name': expert_name,
                                    'sample_id': sample_id,
                                    'confidence': value
                                })
                        elif isinstance(values, list):
                            for sample_id, value in enumerate(values):
                                confidence_rows.append({
                                    'expert_name': expert_name,
                                    'sample_id': sample_id,
                                    'confidence': value
                                })
                
                if confidence_rows:
                    result['confidence'] = pd.DataFrame(confidence_rows)
    
    except Exception as e:
        st.error(f"Error extracting expert data from workflow: {str(e)}")
        
    return result


def load_workflow_expert_data(workflow_dir=".workflow_tracking", max_workflows=5):
    """
    Load expert data from workflow JSON files.
    
    Args:
        workflow_dir: The directory containing workflow JSON files
        max_workflows: The maximum number of workflows to load
        
    Returns:
        A list of dictionaries with expert data from each workflow
    """
    workflow_data_list = []
    
    try:
        # Check if workflow directory exists
        if not os.path.exists(workflow_dir):
            st.warning(f"Workflow directory not found: {workflow_dir}")
            return workflow_data_list
        
        # Find workflow JSON files
        workflow_files = []
        for root, _, files in os.walk(workflow_dir):
            for file in files:
                if file.endswith('.json') and not file.startswith('.'):
                    workflow_files.append(os.path.join(root, file))
        
        # Sort by modification time (newest first)
        workflow_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Load expert data from each workflow
        for i, workflow_file in enumerate(workflow_files[:max_workflows]):
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)
                
                expert_data = extract_expert_data_from_workflow(workflow_data)
                
                if expert_data:
                    expert_data['workflow_file'] = workflow_file
                    expert_data['workflow_name'] = os.path.basename(workflow_file).replace('.json', '')
                    workflow_data_list.append(expert_data)
                    
            except Exception as e:
                st.error(f"Error loading workflow file {workflow_file}: {str(e)}")
                continue
                
            # Stop if we've loaded the maximum number of workflows
            if i >= max_workflows - 1:
                break
    
    except Exception as e:
        st.error(f"Error loading workflow expert data: {str(e)}")
        
    return workflow_data_list


def create_expert_agreement_matrix(agreement_df, metric='correlation', 
                                 title="Expert Agreement Matrix", 
                                 colorscale="YlGnBu"):
    """
    Create a heatmap visualization of expert agreement.
    
    Args:
        agreement_df: A pandas DataFrame with expert agreement data
        metric: The agreement metric to visualize
        title: The title of the plot
        colorscale: The colorscale to use
        
    Returns:
        A plotly figure object
    """
    if agreement_df is None or agreement_df.empty:
        return None
    
    # Filter for the specified metric
    if 'metric' in agreement_df.columns:
        df = agreement_df[agreement_df['metric'] == metric].copy()
    else:
        df = agreement_df.copy()
    
    # Get unique experts
    experts = sorted(list(set(df['expert1'].unique()) | set(df['expert2'].unique())))
    
    # Create empty matrix
    matrix = pd.DataFrame(index=experts, columns=experts)
    
    # Fill matrix with agreement values
    for _, row in df.iterrows():
        matrix.loc[row['expert1'], row['expert2']] = row['value']
        
    # Fill diagonal with 1.0 (perfect agreement with self)
    for expert in experts:
        matrix.loc[expert, expert] = 1.0
        
    # Fill in symmetric values (if not already present)
    for i in range(len(experts)):
        for j in range(i+1, len(experts)):
            expert1, expert2 = experts[i], experts[j]
            
            # If one direction is missing, copy from the other
            if pd.isna(matrix.loc[expert1, expert2]) and not pd.isna(matrix.loc[expert2, expert1]):
                matrix.loc[expert1, expert2] = matrix.loc[expert2, expert1]
            elif pd.isna(matrix.loc[expert2, expert1]) and not pd.isna(matrix.loc[expert1, expert2]):
                matrix.loc[expert2, expert1] = matrix.loc[expert1, expert2]
    
    # Create heatmap
    fig = px.imshow(
        matrix,
        labels=dict(x="Expert", y="Expert", color=metric.capitalize()),
        x=experts,
        y=experts,
        color_continuous_scale=colorscale,
        title=f"{title} ({metric.capitalize()})"
    )
    
    # Set layout
    fig.update_layout(
        width=700,
        height=600,
        xaxis_title="Expert",
        yaxis_title="Expert",
        coloraxis_colorbar=dict(
            title=metric.capitalize(),
            lenmode="fraction", 
            len=0.75
        )
    )
    
    # Add text annotations
    for i, expert1 in enumerate(experts):
        for j, expert2 in enumerate(experts):
            value = matrix.loc[expert1, expert2]
            if not pd.isna(value):
                fig.add_annotation(
                    x=expert2,
                    y=expert1,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color="black" if value < 0.7 else "white")
                )
    
    return fig


def create_expert_confidence_chart(confidence_df, 
                                 title="Expert Confidence Distribution",
                                 color_discrete_sequence=px.colors.qualitative.Plotly):
    """
    Create a box plot or violin plot of expert confidence distributions.
    
    Args:
        confidence_df: A pandas DataFrame with expert confidence data
        title: The title of the plot
        color_discrete_sequence: The color sequence to use
        
    Returns:
        A plotly figure object
    """
    if confidence_df is None or confidence_df.empty:
        return None
    
    # Create violin plot
    fig = px.violin(
        confidence_df,
        x="expert_name",
        y="confidence",
        color="expert_name",
        box=True,  # include box plot inside the violin
        points="all",  # show all points
        title=title,
        color_discrete_sequence=color_discrete_sequence,
        labels={
            "expert_name": "Expert Model",
            "confidence": "Confidence Score"
        }
    )
    
    # Set layout
    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Expert Model",
        yaxis_title="Confidence Score",
        showlegend=False
    )
    
    # Add mean confidence as annotations
    for expert in confidence_df['expert_name'].unique():
        mean_conf = confidence_df[confidence_df['expert_name'] == expert]['confidence'].mean()
        
        fig.add_annotation(
            x=expert,
            y=mean_conf,
            text=f"Mean: {mean_conf:.3f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    return fig


def create_expert_contribution_chart(contribution_df, 
                                   title="Expert Contribution Analysis",
                                   color_discrete_sequence=px.colors.qualitative.Plotly,
                                   chart_type="bar"):
    """
    Create a chart showing expert contributions.
    
    Args:
        contribution_df: A pandas DataFrame with expert contribution data
        title: The title of the plot
        color_discrete_sequence: The color sequence to use
        chart_type: The type of chart to create ("bar", "pie", or "treemap")
        
    Returns:
        A plotly figure object
    """
    if contribution_df is None or contribution_df.empty:
        return None
    
    # Check if the contribution column is numeric
    try:
        # Try to convert contribution to numeric if it's not already
        if contribution_df['contribution'].dtype == 'object':
            contribution_df['contribution'] = pd.to_numeric(contribution_df['contribution'], errors='coerce')
            # Drop any NaN values that resulted from conversion failures
            contribution_df = contribution_df.dropna(subset=['contribution'])
            
            if contribution_df.empty:
                print("Warning: No numeric contribution values found after conversion")
                # Create a dummy chart with placeholder data
                fig = px.bar(
                    x=["No numeric data available"],
                    y=[0],
                    title=f"{title} (No valid data)"
                )
                fig.update_layout(width=800, height=500)
                return fig
    except Exception as e:
        print(f"Error converting contributions to numeric: {str(e)}")
        # Create a dummy chart with placeholder data
        fig = px.bar(
            x=["Error in data processing"],
            y=[0],
            title=f"{title} (Data error)"
        )
        fig.update_layout(width=800, height=500)
        return fig
    
    # Calculate mean contribution per expert
    try:
        mean_contrib = contribution_df.groupby('expert_name')['contribution'].mean().reset_index()
        mean_contrib = mean_contrib.sort_values('contribution', ascending=False)
    except Exception as e:
        print(f"Error calculating mean contributions: {str(e)}")
        # Create a dummy chart with placeholder data
        fig = px.bar(
            x=["Error calculating means"],
            y=[0],
            title=f"{title} (Calculation error)"
        )
        fig.update_layout(width=800, height=500)
        return fig
    
    if chart_type == "pie":
        # Create pie chart
        fig = px.pie(
            mean_contrib,
            values='contribution',
            names='expert_name',
            title=f"{title} (Average)",
            color_discrete_sequence=color_discrete_sequence
        )
        
        # Set layout
        fig.update_layout(
            width=700,
            height=500
        )
        
        # Update traces
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            hole=0.3
        )
    
    elif chart_type == "treemap":
        # Create treemap
        fig = px.treemap(
            mean_contrib,
            path=['expert_name'],
            values='contribution',
            title=f"{title} (Average)",
            color='contribution',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=mean_contrib['contribution'].mean()
        )
        
        # Set layout
        fig.update_layout(
            width=800,
            height=600
        )
        
    else:  # Default to bar chart
        # Create bar chart
        fig = px.bar(
            mean_contrib,
            x='expert_name',
            y='contribution',
            color='expert_name',
            title=f"{title} (Average)",
            color_discrete_sequence=color_discrete_sequence,
            labels={
                "expert_name": "Expert Model",
                "contribution": "Average Contribution"
            }
        )
        
        # Set layout
        fig.update_layout(
            width=800,
            height=500,
            xaxis_title="Expert Model",
            yaxis_title="Average Contribution",
            showlegend=False
        )
        
        # Add value annotations
        for i, row in mean_contrib.iterrows():
            fig.add_annotation(
                x=row['expert_name'],
                y=row['contribution'],
                text=f"{row['contribution']:.3f}",
                showarrow=False,
                yshift=10
            )
    
    return fig


def create_expert_weight_evolution(weights_df, 
                                 title="Expert Weight Evolution",
                                 color_discrete_sequence=px.colors.qualitative.Plotly):
    """
    Create a line chart visualization of expert weight evolution over time.
    
    Args:
        weights_df: A pandas DataFrame with expert weight data
        title: The title of the plot
        color_discrete_sequence: The color sequence to use
        
    Returns:
        A plotly figure object
    """
    if weights_df is None or weights_df.empty:
        return None
    
    # Determine time column
    time_cols = [col for col in weights_df.columns 
                if any(t in col.lower() for t in ['time', 'timestamp', 'iteration'])]
    
    if time_cols:
        time_col = time_cols[0]
    else:
        # Create an iteration column if no time column exists
        weights_df = weights_df.reset_index()
        time_col = 'index'
    
    # Get expert columns (non-time columns)
    expert_cols = [col for col in weights_df.columns if col != time_col]
    
    if not expert_cols:
        return None
    
    # Create line chart
    fig = go.Figure()
    
    for i, expert in enumerate(expert_cols):
        color_idx = i % len(color_discrete_sequence)
        
        fig.add_trace(go.Scatter(
            x=weights_df[time_col],
            y=weights_df[expert],
            mode='lines+markers',
            name=expert,
            line=dict(
                color=color_discrete_sequence[color_idx],
                width=2
            ),
            marker=dict(
                size=6,
                symbol='circle'
            )
        ))
    
    # Set layout
    fig.update_layout(
        title=title,
        xaxis_title=time_col.capitalize(),
        yaxis_title="Expert Weight",
        width=800,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add a horizontal line at 1/n_experts
    n_experts = len(expert_cols)
    equal_weight = 1.0 / n_experts
    
    fig.add_shape(
        type="line",
        x0=weights_df[time_col].min(),
        y0=equal_weight,
        x1=weights_df[time_col].max(),
        y1=equal_weight,
        line=dict(
            color="rgba(0,0,0,0.5)",
            width=1,
            dash="dash"
        )
    )
    
    fig.add_annotation(
        x=weights_df[time_col].max(),
        y=equal_weight,
        text=f"Equal weight (1/{n_experts})",
        showarrow=False,
        xshift=10,
        textangle=0,
        xanchor="left"
    )
    
    return fig


def plot_expert_dominance_regions(contribution_df, title="Expert Dominance Regions", with_colorbar=True):
    """
    Create a scatter plot showing regions where different experts dominate.
    
    Args:
        contribution_df: DataFrame with expert_name, contribution, and sample_id columns
        title: Title of the plot
        with_colorbar: Whether to include a colorbar showing contribution strength
    
    Returns:
        fig: Plotly figure
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    
    # Check if contribution_df is None or empty
    if contribution_df is None or contribution_df.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Expert Contribution Data Available",
            annotations=[
                dict(
                    text="No expert contribution data found in the selected workflow.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=16)
                )
            ]
        )
        return fig
    
    # Try to import sklearn for dimensionality reduction
    has_sklearn = False
    try:
        from sklearn.decomposition import PCA
        has_sklearn = True
    except ImportError:
        # If sklearn is not available, we'll use a simpler projection method
        pass
    
    try:
        # Create a pivot table to get expert contributions by sample
        pivot_df = contribution_df.pivot_table(
            index="sample_id", 
            columns="expert_name", 
            values="contribution",
            fill_value=0
        )
        
        # Get unique expert names
        expert_names = contribution_df['expert_name'].unique()
        num_experts = len(expert_names)
        
        # We need at least 2 dimensions for a meaningful visualization
        if num_experts < 2:
            # Create dummy dimensions if we have fewer than 2 experts
            if num_experts == 1:
                expert_name = expert_names[0]
                # Create a 1D plot spreading points along a line
                contributions = contribution_df[contribution_df['expert_name'] == expert_name]['contribution']
                x = contributions.values
                y = np.zeros_like(x)  # Just zeros for the second dimension
                
                # Create the figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=contributions,
                        colorscale='Viridis',
                        showscale=with_colorbar,
                        colorbar=dict(title="Contribution")
                    ),
                    text=[f"{expert_name}: {c:.2f}" for c in contributions],
                    hoverinfo="text"
                ))
                
                fig.update_layout(
                    title=f"{title} (Single Expert: {expert_name})",
                    xaxis=dict(title=f"{expert_name} Contribution"),
                    yaxis=dict(
                        title="",
                        showticklabels=False,
                        zeroline=True
                    )
                )
                return fig
            else:
                # No experts case - return empty plot with message
                fig = go.Figure()
                fig.update_layout(
                    title="No Expert Data Available",
                    annotations=[dict(
                        text="No expert data found in the selected workflow.",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        font=dict(size=16)
                    )]
                )
                return fig
        
        # Prepare data for dimensionality reduction
        X = pivot_df.values
        
        # Determine the dominant expert for each sample
        dominant_experts = pivot_df.idxmax(axis=1)
        
        # Project the data to 2D space
        if has_sklearn and X.shape[1] > 2:
            # Use PCA for dimensionality reduction if available
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            explained_var = pca.explained_variance_ratio_
            axis_labels = [f"PC1 ({explained_var[0]:.2%})", f"PC2 ({explained_var[1]:.2%})"]
        else:
            # Fallback: use the first two experts or a simple projection
            if X.shape[1] >= 2:
                # Just use the first two experts
                X_2d = X[:, :2]
                axis_labels = [f"{expert_names[0]} Contribution", f"{expert_names[1]} Contribution"]
            else:
                # Create a synthetic second dimension
                X_2d = np.column_stack((X, np.random.random(X.shape[0]) * 0.1))
                axis_labels = [f"{expert_names[0]} Contribution", "Random Dimension"]
        
        # Create a DataFrame with the 2D coordinates and dominant expert
        plot_df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'dominant_expert': dominant_experts,
            'max_contribution': pivot_df.max(axis=1)
        })
        
        # Create the figure
        fig = go.Figure()
        
        # Add traces for each expert
        for expert in expert_names:
            subset = plot_df[plot_df['dominant_expert'] == expert]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset['x'],
                    y=subset['y'],
                    mode='markers',
                    name=expert,
                    marker=dict(
                        size=10,
                        color=subset['max_contribution'] if with_colorbar else None,
                        colorscale='Viridis' if with_colorbar else None,
                        showscale=with_colorbar,
                        colorbar=dict(title="Contribution Strength") if with_colorbar else None
                    ),
                    text=[f"{expert}: {c:.2f}" for c in subset['max_contribution']],
                    hoverinfo="text"
                ))
        
        # Update layout with better styling
        fig.update_layout(
            title=title,
            xaxis=dict(title=axis_labels[0]),
            yaxis=dict(title=axis_labels[1]),
            legend_title="Dominant Expert",
            template="plotly_white"
        )
        
        return fig
    
    except Exception as e:
        # Fallback to basic visualization if any error occurs
        fig = go.Figure()
        fig.update_layout(
            title="Error Creating Expert Visualization",
            annotations=[dict(
                text=f"An error occurred: {str(e)}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=14)
            )]
        )
        return fig


def create_expert_ensemble_dashboard(workflow_data_list):
    """
    Create a dashboard of expert ensemble visualizations.
    
    Args:
        workflow_data_list: A list of dictionaries with expert data from each workflow
        
    Returns:
        A list of plotly figures
    """
    figures = []
    
    if not workflow_data_list:
        st.warning("No workflow data available for expert ensemble visualization.")
        return figures
    
    # Use the most recent workflow by default
    workflow_data = workflow_data_list[0]
    
    # Create expert weight evolution visualization
    if 'weights' in workflow_data:
        weights_fig = create_expert_weight_evolution(
            workflow_data['weights'],
            title=f"Expert Weight Evolution ({workflow_data['workflow_name']})"
        )
        if weights_fig:
            figures.append(("Expert Weight Evolution", weights_fig))
    
    # Create expert contribution chart
    if 'contributions' in workflow_data:
        contrib_fig = create_expert_contribution_chart(
            workflow_data['contributions'],
            title=f"Expert Contribution Analysis ({workflow_data['workflow_name']})"
        )
        if contrib_fig:
            figures.append(("Expert Contribution Analysis", contrib_fig))
    
    # Create expert agreement matrix
    if 'agreement' in workflow_data:
        agreement_fig = create_expert_agreement_matrix(
            workflow_data['agreement'],
            title=f"Expert Agreement Matrix ({workflow_data['workflow_name']})"
        )
        if agreement_fig:
            figures.append(("Expert Agreement Matrix", agreement_fig))
    
    # Create expert confidence chart
    if 'confidence' in workflow_data:
        confidence_fig = create_expert_confidence_chart(
            workflow_data['confidence'],
            title=f"Expert Confidence Distribution ({workflow_data['workflow_name']})"
        )
        if confidence_fig:
            figures.append(("Expert Confidence Distribution", confidence_fig))
    
    # Create expert dominance regions visualization
    if 'contributions' in workflow_data:
        dominance_fig = plot_expert_dominance_regions(
            workflow_data['contributions'],
            title=f"Expert Dominance Regions ({workflow_data['workflow_name']})"
        )
        if dominance_fig:
            figures.append(("Expert Dominance Regions", dominance_fig))
    
    return figures 
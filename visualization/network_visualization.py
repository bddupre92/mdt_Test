"""
Network Visualization for Feature Interactions

This module provides visualization tools for feature interactions, 
cross-modal correlations, and causal relationships in the MoE framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

from core.theoretical_metrics import measure_mutual_information, extract_causal_relationships

logger = logging.getLogger(__name__)

def create_feature_correlation_network(data: pd.DataFrame, 
                                      threshold: float = 0.3,
                                      include_negative: bool = True) -> Dict[str, Any]:
    """
    Create network visualization of feature correlations.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with feature data
    threshold : float
        Minimum absolute correlation to include in network
    include_negative : bool
        Whether to include negative correlations
        
    Returns:
    --------
    Dict[str, Any]
        Network visualization data for plotly
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (features)
    for feature in data.columns:
        G.add_node(feature)
    
    # Add edges (correlations above threshold)
    for i, feat1 in enumerate(data.columns):
        for j, feat2 in enumerate(data.columns):
            if i < j:  # Only upper triangle to avoid duplicates
                correlation = corr_matrix.loc[feat1, feat2]
                if include_negative:
                    if abs(correlation) >= threshold:
                        G.add_edge(feat1, feat2, weight=correlation)
                else:
                    if correlation >= threshold:
                        G.add_edge(feat1, feat2, weight=correlation)
    
    # Generate layout
    pos = nx.spring_layout(G)
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
    
    node_trace.marker.color = node_adjacencies
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        # Determine color based on correlation
        if weight >= 0:
            color = f'rgba(0, 100, 0, {min(1, abs(weight) * 1.5)})'  # Green for positive
        else:
            color = f'rgba(200, 0, 0, {min(1, abs(weight) * 1.5)})'  # Red for negative
        
        # Edge width based on correlation strength
        width = abs(weight) * 3
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"{edge[0]} -- {edge[1]}: {weight:.3f}",
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig_data = edge_traces + [node_trace]
    
    return {
        'data': fig_data,
        'layout': go.Layout(
            title='Feature Correlation Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ),
        'network': G,
        'positions': pos
    }

def create_mutual_information_network(data: pd.DataFrame, 
                                    threshold: float = 0.2) -> Dict[str, Any]:
    """
    Create network visualization of mutual information between features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with feature data
    threshold : float
        Minimum mutual information to include in network
        
    Returns:
    --------
    Dict[str, Any]
        Network visualization data for plotly
    """
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (features)
    for feature in data.columns:
        G.add_node(feature)
    
    # Calculate mutual information between all pairs of features
    for i, feat1 in enumerate(data.columns):
        for j, feat2 in enumerate(data.columns):
            if i < j:  # Only upper triangle to avoid duplicates
                try:
                    mi = measure_mutual_information(data[feat1].values, data[feat2].values)
                    if mi >= threshold:
                        G.add_edge(feat1, feat2, weight=mi)
                except Exception as e:
                    logger.warning(f"Error calculating MI between {feat1} and {feat2}: {e}")
    
    # Generate layout
    pos = nx.spring_layout(G)
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=15,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
    
    node_trace.marker.color = node_adjacencies
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        
        # Edge width based on mutual information
        width = weight * 5
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=f'rgba(100, 100, 255, {min(1, weight * 2)})'),
            hoverinfo='text',
            text=f"{edge[0]} -- {edge[1]}: MI={weight:.3f}",
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig_data = edge_traces + [node_trace]
    
    return {
        'data': fig_data,
        'layout': go.Layout(
            title='Feature Mutual Information Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        ),
        'network': G,
        'positions': pos
    }

def create_causal_network(data: pd.DataFrame, 
                         target_col: str,
                         threshold: float = 0.05) -> Dict[str, Any]:
    """
    Create network visualization of causal relationships.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with time series data
    target_col : str
        Target column for causality analysis
    threshold : float
        Significance threshold for causality
        
    Returns:
    --------
    Dict[str, Any]
        Network visualization data for plotly
    """
    # Extract causal relationships
    causal_results = extract_causal_relationships(data, target_col, method='granger', significance=threshold)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for col in data.columns:
        G.add_node(col)
    
    # Add directed edges
    for source, result in causal_results.items():
        if result['causal']:
            G.add_edge(source, target_col, 
                     weight=result['strength'],
                     lag=result.get('significant_lags', [0])[0] if result.get('significant_lags') else 0)
    
    # Generate layout
    pos = nx.spring_layout(G)
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        
        # Target node has special color
        if node == target_col:
            node_colors.append('rgba(255, 0, 0, 0.8)')
        else:
            node_colors.append('rgba(0, 116, 217, 0.8)')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=15,
            line_width=2
        )
    )
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        lag = edge[2].get('lag', 0)
        
        # Edge width based on causal strength
        width = weight * 5
        
        # Calculate arrow position (80% along the edge)
        xa = x0 * 0.2 + x1 * 0.8
        ya = y0 * 0.2 + y1 * 0.8
        
        # Line trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=f'rgba(50, 50, 50, {min(1, weight + 0.3)})'),
            hoverinfo='text',
            text=f"{edge[0]} causes {edge[1]} (p={1-weight:.3f}, lag={lag})",
            mode='lines'
        )
        edge_traces.append(edge_trace)
        
        # Arrow trace
        arrow = go.Scatter(
            x=[xa],
            y=[ya],
            mode='markers',
            marker=dict(
                symbol='triangle-right',
                size=10,
                color='rgba(50, 50, 50, 0.8)',
                angle=np.degrees(np.arctan2(y1-y0, x1-x0))
            ),
            hoverinfo='none'
        )
        edge_traces.append(arrow)
    
    # Create figure
    fig_data = edge_traces + [node_trace]
    
    return {
        'data': fig_data,
        'layout': go.Layout(
            title=f'Causal Network (Target: {target_col})',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[
                dict(
                    text="Direction of causality →",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.01, y=0.01
                )
            ]
        ),
        'network': G,
        'positions': pos,
        'causal_results': causal_results
    }

def create_multimodal_correlation_network(data: pd.DataFrame, 
                                         modality_groups: Dict[str, List[str]],
                                         threshold: float = 0.3) -> Dict[str, Any]:
    """
    Create network visualization highlighting correlations between different modalities.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with feature data
    modality_groups : Dict[str, List[str]]
        Dictionary mapping modality names to lists of features
    threshold : float
        Minimum absolute correlation to include
        
    Returns:
    --------
    Dict[str, Any]
        Network visualization data for plotly
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (features) with modality attribute
    for modality, features in modality_groups.items():
        for feature in features:
            if feature in data.columns:
                G.add_node(feature, modality=modality)
    
    # Add edges (correlations above threshold)
    for i, feat1 in enumerate(data.columns):
        if not any(feat1 in features for features in modality_groups.values()):
            continue
            
        for j, feat2 in enumerate(data.columns):
            if i < j and any(feat2 in features for features in modality_groups.values()):
                correlation = corr_matrix.loc[feat1, feat2]
                
                if abs(correlation) >= threshold:
                    # Get modalities
                    mod1 = next((m for m, feats in modality_groups.items() if feat1 in feats), "unknown")
                    mod2 = next((m for m, feats in modality_groups.items() if feat2 in feats), "unknown")
                    
                    # Add edge with modality info
                    G.add_edge(feat1, feat2, weight=correlation, cross_modal=(mod1 != mod2))
    
    # Generate layout - group by modality
    pos = nx.spring_layout(G)
    
    # Adjust positions to group by modality
    modality_centers = {}
    spacing = 2.0
    
    for i, modality in enumerate(modality_groups.keys()):
        angle = 2 * np.pi * i / len(modality_groups)
        modality_centers[modality] = np.array([spacing * np.cos(angle), spacing * np.sin(angle)])
    
    # Adjust node positions based on modality
    for node in G.nodes:
        modality = G.nodes[node]['modality']
        if modality in modality_centers:
            # Move node towards its modality center
            center = modality_centers[modality]
            pos[node] = pos[node] * 0.3 + center * 0.7
    
    # Create node traces by modality
    node_traces = []
    modality_colors = {
        'physiological': 'rgba(255, 0, 0, 0.8)',
        'environmental': 'rgba(0, 128, 0, 0.8)',
        'behavioral': 'rgba(0, 0, 255, 0.8)',
        'demographic': 'rgba(128, 0, 128, 0.8)',
        'medical': 'rgba(255, 165, 0, 0.8)'
    }
    
    for modality in modality_groups.keys():
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            if G.nodes[node].get('modality') == modality:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))
        
        if node_x:  # Only create trace if we have nodes
            color = modality_colors.get(modality.lower(), 'rgba(100, 100, 100, 0.8)')
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                hoverinfo='text',
                name=modality,
                marker=dict(
                    color=color,
                    size=15,
                    line_width=2
                )
            )
            node_traces.append(node_trace)
    
    # Create edge traces
    edge_traces = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        cross_modal = edge[2].get('cross_modal', False)
        
        # Determine color based on correlation and cross-modality
        if cross_modal:
            if weight >= 0:
                color = f'rgba(255, 100, 0, {min(1, abs(weight) * 1.5)})'  # Orange for positive cross-modal
            else:
                color = f'rgba(128, 0, 128, {min(1, abs(weight) * 1.5)})'  # Purple for negative cross-modal
        else:
            if weight >= 0:
                color = f'rgba(0, 100, 0, {min(1, abs(weight) * 1.5)})'  # Green for positive
            else:
                color = f'rgba(200, 0, 0, {min(1, abs(weight) * 1.5)})'  # Red for negative
        
        # Edge width based on correlation strength
        width = abs(weight) * 3
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"{edge[0]} -- {edge[1]}: {weight:.3f}" + (" (cross-modal)" if cross_modal else ""),
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig_data = edge_traces + node_traces
    
    return {
        'data': fig_data,
        'layout': go.Layout(
            title='Multi-modal Feature Correlation Network',
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                title='Modalities',
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        ),
        'network': G,
        'positions': pos
    }

def generate_network_html(network_data: Dict[str, Any], filename: str) -> str:
    """
    Generate standalone HTML file with network visualization.
    
    Parameters:
    -----------
    network_data : Dict[str, Any]
        Network visualization data from one of the create_*_network functions
    filename : str
        Output HTML filename
        
    Returns:
    --------
    str:
        Path to the generated HTML file
    """
    fig = go.Figure(
        data=network_data['data'],
        layout=network_data['layout']
    )
    
    # Export to HTML
    html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
    
    with open(filename, 'w') as f:
        f.write(html_content)
    
    return filename

def network_to_json(network_data: Dict[str, Any]) -> str:
    """
    Convert network data to JSON for embedding in reports.
    
    Parameters:
    -----------
    network_data : Dict[str, Any]
        Network visualization data
        
    Returns:
    --------
    str:
        JSON string of network data for Plotly
    """
    # Convert networkx objects to serializable format
    serializable_data = {
        'data': network_data['data'],
        'layout': network_data['layout']
    }
    
    return json.dumps(serializable_data, cls=PlotlyJSONEncoder)

class PlotlyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Plotly objects"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, go.Figure):
            return obj.to_dict()
        return super().default(obj)

"""
Algorithm Visualization Module

This module provides visualizations for algorithm selection, performance prediction,
and convergence properties of different optimization algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

from core.theoretical_metrics import calculate_convergence_rate, analyze_complexity_scaling

logger = logging.getLogger(__name__)

def create_algorithm_selection_visualization(algorithm_data: Dict[str, Dict[str, Any]],
                                            problem_characteristics: List[str]) -> Dict[str, Any]:
    """
    Create visualization for algorithm selection based on problem characteristics.
    
    Parameters:
    -----------
    algorithm_data : Dict[str, Dict[str, Any]]
        Dictionary mapping algorithm names to their performance metrics across different problems
    problem_characteristics : List[str]
        List of problem characteristic dimensions to visualize
        
    Returns:
    --------
    Dict[str, Any]
        Visualization data for plotly
    """
    if len(problem_characteristics) < 2:
        raise ValueError("Need at least two problem characteristics for 2D visualization")
    
    # Extract the first two characteristics for 2D visualization
    char1, char2 = problem_characteristics[:2]
    
    # Create data for heatmap
    algorithms = list(algorithm_data.keys())
    
    # Get unique values for each characteristic
    char1_values = sorted(set(data[char1] for data in algorithm_data.values() if char1 in data))
    char2_values = sorted(set(data[char2] for data in algorithm_data.values() if char2 in data))
    
    # Create empty performance matrix
    performance_matrix = np.zeros((len(char1_values), len(char2_values), len(algorithms)))
    performance_matrix[:] = np.nan
    
    # Fill performance matrix
    for i, algo in enumerate(algorithms):
        if algo in algorithm_data:
            if char1 in algorithm_data[algo] and char2 in algorithm_data[algo]:
                char1_idx = char1_values.index(algorithm_data[algo][char1])
                char2_idx = char2_values.index(algorithm_data[algo][char2])
                performance_matrix[char1_idx, char2_idx, i] = algorithm_data[algo].get('performance', 0)
    
    # Find best algorithm for each problem characteristic combination
    best_algorithm_idx = np.nanargmax(performance_matrix, axis=2)
    
    # Create heatmap data
    heatmap_data = []
    for i, c1 in enumerate(char1_values):
        for j, c2 in enumerate(char2_values):
            if not np.isnan(performance_matrix[i, j, :]).all():
                best_algo_idx = best_algorithm_idx[i, j]
                best_algo = algorithms[best_algo_idx]
                performance = performance_matrix[i, j, best_algo_idx]
                
                heatmap_data.append({
                    char1: c1,
                    char2: c2,
                    'best_algorithm': best_algo,
                    'performance': performance
                })
    
    # Create DataFrame for visualization
    df = pd.DataFrame(heatmap_data)
    
    # Create 2D heatmap for algorithm selection
    fig = px.density_heatmap(
        df,
        x=char1,
        y=char2,
        z='performance',
        color_continuous_scale='Viridis',
        title=f'Algorithm Selection Heatmap ({char1} vs {char2})'
    )
    
    # Add text annotations for best algorithm
    annotations = []
    for _, row in df.iterrows():
        annotations.append(
            dict(
                x=row[char1],
                y=row[char2],
                text=row['best_algorithm'],
                showarrow=False,
                font=dict(
                    color='white',
                    size=10
                )
            )
        )
    
    fig.update_layout(annotations=annotations)
    
    # If we have a third characteristic, create multiple heatmaps
    if len(problem_characteristics) > 2:
        char3 = problem_characteristics[2]
        char3_values = sorted(set(data[char3] for data in algorithm_data.values() if char3 in data))
        
        # Create subplot with one heatmap per value of char3
        subplots = make_subplots(
            rows=1, cols=len(char3_values),
            subplot_titles=[f"{char3}={val}" for val in char3_values]
        )
        
        for k, val3 in enumerate(char3_values):
            # Filter data for this value of char3
            filtered_data = [d for d in heatmap_data if algorithm_data[d['best_algorithm']].get(char3) == val3]
            
            if filtered_data:
                df_filtered = pd.DataFrame(filtered_data)
                
                heatmap = px.density_heatmap(
                    df_filtered,
                    x=char1,
                    y=char2,
                    z='performance',
                    color_continuous_scale='Viridis'
                )
                
                for trace in heatmap.data:
                    subplots.add_trace(trace, row=1, col=k+1)
        
        fig = subplots
    
    return {
        'figure': fig,
        'data': heatmap_data,
        'characteristics': problem_characteristics[:3],
        'algorithms': algorithms
    }

def create_performance_prediction_visualization(actual_performance: List[float],
                                              predicted_performance: List[float],
                                              algorithm_names: List[str],
                                              problem_ids: List[str]) -> Dict[str, Any]:
    """
    Visualize performance prediction accuracy.
    
    Parameters:
    -----------
    actual_performance : List[float]
        Actual performance values
    predicted_performance : List[float]
        Predicted performance values
    algorithm_names : List[str]
        Names of algorithms corresponding to each performance value
    problem_ids : List[str]
        Problem identifiers
        
    Returns:
    --------
    Dict[str, Any]
        Visualization data for plotly
    """
    if len(actual_performance) != len(predicted_performance) or len(actual_performance) != len(algorithm_names):
        raise ValueError("Input arrays must have the same length")
    
    # Create a dataframe for visualization
    df = pd.DataFrame({
        'actual': actual_performance,
        'predicted': predicted_performance,
        'algorithm': algorithm_names,
        'problem': problem_ids
    })
    
    # Calculate prediction error
    df['error'] = np.abs(df['predicted'] - df['actual'])
    df['relative_error'] = df['error'] / np.abs(df['actual'])
    
    # Create scatter plot for actual vs. predicted
    scatter_fig = px.scatter(
        df, x='actual', y='predicted', 
        color='algorithm', symbol='algorithm',
        hover_data=['problem', 'error', 'relative_error'],
        title='Performance Prediction Accuracy',
        labels={'actual': 'Actual Performance', 'predicted': 'Predicted Performance'}
    )
    
    # Add 45-degree line (perfect prediction)
    min_val = min(df['actual'].min(), df['predicted'].min())
    max_val = max(df['actual'].max(), df['predicted'].max())
    
    scatter_fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    # Create error violin plot by algorithm
    violin_fig = px.violin(
        df, x='algorithm', y='relative_error',
        box=True, points="all",
        title='Prediction Error by Algorithm',
        labels={'relative_error': 'Relative Error', 'algorithm': 'Algorithm'}
    )
    
    # Calculate error statistics
    error_stats = df.groupby('algorithm')['relative_error'].agg(['mean', 'median', 'std']).reset_index()
    
    # Create bar chart for average error
    bar_fig = px.bar(
        error_stats, x='algorithm', y='mean',
        error_y='std',
        title='Average Prediction Error by Algorithm',
        labels={'mean': 'Mean Relative Error', 'algorithm': 'Algorithm'}
    )
    
    return {
        'scatter_plot': scatter_fig,
        'violin_plot': violin_fig,
        'bar_chart': bar_fig,
        'error_stats': error_stats.to_dict('records'),
        'data': df.to_dict('records')
    }

def create_convergence_visualization(optimization_trajectories: Dict[str, np.ndarray],
                                    theoretical_curves: Optional[Dict[str, callable]] = None) -> Dict[str, Any]:
    """
    Create visualization for algorithm convergence properties.
    
    Parameters:
    -----------
    optimization_trajectories : Dict[str, np.ndarray]
        Dictionary mapping algorithm names to their optimization trajectories
    theoretical_curves : Dict[str, callable], optional
        Dictionary mapping curve names to functions that generate theoretical convergence curves
        
    Returns:
    --------
    Dict[str, Any]
        Visualization data for plotly
    """
    # Create figure with log-scaled y-axis
    fig = go.Figure()
    
    # Plot empirical trajectories
    for algo_name, trajectory in optimization_trajectories.items():
        # Calculate error as distance from final value
        final_value = trajectory[-1]
        errors = np.abs(trajectory - final_value)
        errors = np.maximum(errors, 1e-10)  # Avoid log(0)
        
        iterations = np.arange(len(trajectory))
        
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=errors,
                mode='lines+markers',
                name=f'{algo_name} (empirical)',
                line=dict(width=2)
            )
        )
        
        # Calculate convergence rate
        convergence_results = calculate_convergence_rate(trajectory)
        
        # Add theoretical curve based on estimated rate
        rate = convergence_results['asymptotic_rate']
        
        if np.isfinite(rate) and 0 < rate < 1:
            # Create theoretical curve: error[0] * rate^t
            theo_errors = errors[0] * np.power(rate, iterations)
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=theo_errors,
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    name=f'{algo_name} (theoretical rate={rate:.3f})'
                )
            )
    
    # Add any additional theoretical curves
    if theoretical_curves:
        max_iter = max(len(traj) for traj in optimization_trajectories.values())
        x = np.arange(max_iter)
        
        for curve_name, curve_func in theoretical_curves.items():
            try:
                y = curve_func(x)
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        line=dict(dash='dot', width=1.5),
                        name=curve_name
                    )
                )
            except Exception as e:
                logger.warning(f"Error generating theoretical curve {curve_name}: {e}")
    
    # Add LaTeX annotations for convergence rates
    annotations = [
        dict(
            x=0.02,
            y=0.95,
            xref="paper",
            yref="paper",
            text="\\(O(c^t)\\): Linear convergence rate",
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        ),
        dict(
            x=0.02,
            y=0.87,
            xref="paper",
            yref="paper",
            text="\\(O(t^{-1})\\): Sub-linear convergence",
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        ),
        dict(
            x=0.02,
            y=0.79,
            xref="paper",
            yref="paper",
            text="\\(O(t^{-2})\\): Quadratic convergence",
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
    ]
    
    # Update layout
    fig.update_layout(
        title='Algorithm Convergence Properties',
        xaxis_title='Iteration',
        yaxis_title='Error (log scale)',
        yaxis_type='log',
        annotations=annotations,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Calculate convergence summary
    convergence_summary = {}
    for algo_name, trajectory in optimization_trajectories.items():
        convergence_summary[algo_name] = calculate_convergence_rate(trajectory)
    
    return {
        'figure': fig,
        'convergence_summary': convergence_summary
    }

def create_complexity_visualization(algorithm_runtimes: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Create visualization for algorithm complexity scaling.
    
    Parameters:
    -----------
    algorithm_runtimes : Dict[str, Dict[str, List[float]]]
        Dictionary mapping algorithm names to dictionaries containing:
        - 'dimensions': list of problem dimensions
        - 'runtimes': list of corresponding runtimes
        
    Returns:
    --------
    Dict[str, Any]
        Visualization data for plotly
    """
    fig = go.Figure()
    
    # Create scatter plot for each algorithm
    for algo_name, data in algorithm_runtimes.items():
        dimensions = data['dimensions']
        runtimes = data['runtimes']
        
        fig.add_trace(
            go.Scatter(
                x=dimensions,
                y=runtimes,
                mode='markers+lines',
                name=f'{algo_name} (empirical)',
                line=dict(width=2)
            )
        )
        
        # Analyze complexity scaling
        complexity_results = analyze_complexity_scaling(dimensions, runtimes)
        
        # Create theoretical curve based on estimated complexity
        if complexity_results['r_squared'] > 0.8:  # Good fit
            x_theo = np.linspace(min(dimensions), max(dimensions), 100)
            y_theo = [complexity_results['fit_curve'](x) for x in x_theo]
            
            fig.add_trace(
                go.Scatter(
                    x=x_theo,
                    y=y_theo,
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    name=f'{algo_name} ({complexity_results["complexity_class"]})'
                )
            )
    
    # Add LaTeX annotations for complexity classes
    annotations = [
        dict(
            x=0.02,
            y=0.95,
            xref="paper",
            yref="paper",
            text="Common complexity classes:",
            showarrow=False,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        ),
        dict(
            x=0.02,
            y=0.90,
            xref="paper",
            yref="paper",
            text="\\(O(1)\\): Constant time",
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.85,
            xref="paper",
            yref="paper",
            text="\\(O(\\log n)\\): Logarithmic",
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.80,
            xref="paper",
            yref="paper",
            text="\\(O(n)\\): Linear",
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.75,
            xref="paper",
            yref="paper",
            text="\\(O(n \\log n)\\): Linearithmic",
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.70,
            xref="paper",
            yref="paper",
            text="\\(O(n^2)\\): Quadratic",
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.65,
            xref="paper",
            yref="paper",
            text="\\(O(n^3)\\): Cubic",
            showarrow=False
        ),
        dict(
            x=0.02,
            y=0.60,
            xref="paper",
            yref="paper",
            text="\\(O(2^n)\\): Exponential",
            showarrow=False
        )
    ]
    
    # Update layout
    fig.update_layout(
        title='Algorithm Complexity Scaling',
        xaxis_title='Problem Dimension (n)',
        yaxis_title='Runtime (seconds)',
        annotations=annotations,
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Generate text summary
    complexity_summary = {}
    for algo_name, data in algorithm_runtimes.items():
        complexity_results = analyze_complexity_scaling(data['dimensions'], data['runtimes'])
        complexity_summary[algo_name] = {
            'complexity_class': complexity_results['complexity_class'],
            'r_squared': complexity_results['r_squared']
        }
    
    return {
        'figure': fig,
        'complexity_summary': complexity_summary
    }

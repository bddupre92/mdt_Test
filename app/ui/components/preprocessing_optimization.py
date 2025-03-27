"""
Preprocessing Optimization Component

This module provides Streamlit UI components for configuring and visualizing the optimization
of preprocessing pipelines using evolutionary algorithms and other optimization techniques.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go


def render_optimization_section():
    """Render UI for configuring preprocessing pipeline optimization."""
    st.subheader("Pipeline Optimization")
    
    # Get current configuration
    config = st.session_state.preprocessing_config
    optimization = config.get('optimization', {})
    
    # Enable optimization
    enable_optimization = st.checkbox(
        "Enable Pipeline Optimization",
        value=optimization.get('enable', False),
        key="enable_optimization_checkbox"
    )
    
    if enable_optimization:
        optimization_params = optimization.get('params', {})
        
        # Optimization objective
        st.write("**Optimization Objective**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.text_input(
                "Target Column",
                value=optimization_params.get('target_col', ''),
                key="target_col_input"
            )
            optimization_params['target_col'] = target_col
            
            metric = st.selectbox(
                "Optimization Metric",
                options=['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'r2', 'mse', 'mae'],
                index=['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'r2', 'mse', 'mae'].index(
                    optimization_params.get('metric', 'accuracy')
                ),
                key="metric_select"
            )
            optimization_params['metric'] = metric
            
        with col2:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=int(optimization_params.get('cv_folds', 5)),
                step=1,
                key="cv_folds_slider"
            )
            optimization_params['cv_folds'] = cv_folds
            
            task_type = st.selectbox(
                "Task Type",
                options=['classification', 'regression'],
                index=['classification', 'regression'].index(
                    optimization_params.get('task_type', 'classification')
                ),
                key="task_type_select"
            )
            optimization_params['task_type'] = task_type
        
        # Evolutionary algorithm settings
        st.write("---")
        st.write("**Evolutionary Algorithm Settings**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            algorithm = st.selectbox(
                "Algorithm",
                options=['genetic', 'particle_swarm', 'ant_colony'],
                index=['genetic', 'particle_swarm', 'ant_colony'].index(
                    optimization_params.get('algorithm', 'genetic')
                ),
                key="algorithm_select"
            )
            optimization_params['algorithm'] = algorithm
            
        with col2:
            population_size = st.slider(
                "Population Size",
                min_value=10,
                max_value=100,
                value=int(optimization_params.get('population_size', 30)),
                step=5,
                key="population_size_slider"
            )
            optimization_params['population_size'] = population_size
            
        with col3:
            generations = st.slider(
                "Generations/Iterations",
                min_value=5,
                max_value=50,
                value=int(optimization_params.get('generations', 10)),
                step=5,
                key="generations_slider"
            )
            optimization_params['generations'] = generations
            
        # Algorithm-specific parameters
        if algorithm == 'genetic':
            col1, col2 = st.columns(2)
            
            with col1:
                crossover_prob = st.slider(
                    "Crossover Probability",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(optimization_params.get('crossover_prob', 0.8)),
                    step=0.05,
                    key="crossover_prob_slider"
                )
                optimization_params['crossover_prob'] = crossover_prob
                
            with col2:
                mutation_prob = st.slider(
                    "Mutation Probability",
                    min_value=0.01,
                    max_value=0.5,
                    value=float(optimization_params.get('mutation_prob', 0.2)),
                    step=0.01,
                    key="mutation_prob_slider"
                )
                optimization_params['mutation_prob'] = mutation_prob
                
        elif algorithm == 'particle_swarm':
            col1, col2 = st.columns(2)
            
            with col1:
                inertia = st.slider(
                    "Inertia Weight",
                    min_value=0.1,
                    max_value=1.0,
                    value=float(optimization_params.get('inertia', 0.7)),
                    step=0.05,
                    key="inertia_slider"
                )
                optimization_params['inertia'] = inertia
                
            with col2:
                cognitive = st.slider(
                    "Cognitive Weight",
                    min_value=0.1,
                    max_value=2.0,
                    value=float(optimization_params.get('cognitive', 1.5)),
                    step=0.1,
                    key="cognitive_slider"
                )
                optimization_params['cognitive'] = cognitive
                
                social = st.slider(
                    "Social Weight",
                    min_value=0.1,
                    max_value=2.0,
                    value=float(optimization_params.get('social', 1.5)),
                    step=0.1,
                    key="social_slider"
                )
                optimization_params['social'] = social
                
        elif algorithm == 'ant_colony':
            col1, col2 = st.columns(2)
            
            with col1:
                pheromone_factor = st.slider(
                    "Pheromone Factor",
                    min_value=0.1,
                    max_value=2.0,
                    value=float(optimization_params.get('pheromone_factor', 1.0)),
                    step=0.1,
                    key="pheromone_factor_slider"
                )
                optimization_params['pheromone_factor'] = pheromone_factor
                
            with col2:
                evaporation_rate = st.slider(
                    "Evaporation Rate",
                    min_value=0.01,
                    max_value=0.5,
                    value=float(optimization_params.get('evaporation_rate', 0.1)),
                    step=0.01,
                    key="evaporation_rate_slider"
                )
                optimization_params['evaporation_rate'] = evaporation_rate
                
        # Advanced optimization settings
        with st.expander("Advanced Optimization Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                early_stopping = st.checkbox(
                    "Enable Early Stopping",
                    value=optimization_params.get('early_stopping', True),
                    key="early_stopping_checkbox"
                )
                optimization_params['early_stopping'] = early_stopping
                
                if early_stopping:
                    patience = st.slider(
                        "Patience",
                        min_value=1,
                        max_value=10,
                        value=int(optimization_params.get('patience', 3)),
                        step=1,
                        key="patience_slider"
                    )
                    optimization_params['patience'] = patience
                    
            with col2:
                n_jobs = st.slider(
                    "Parallel Jobs",
                    min_value=1,
                    max_value=8,
                    value=int(optimization_params.get('n_jobs', 1)),
                    step=1,
                    key="n_jobs_slider"
                )
                optimization_params['n_jobs'] = n_jobs
                
                timeout = st.number_input(
                    "Optimization Timeout (seconds)",
                    min_value=60,
                    max_value=3600,
                    value=int(optimization_params.get('timeout', 600)),
                    step=60,
                    key="timeout_input"
                )
                optimization_params['timeout'] = timeout
                
        # Operations to optimize
        st.write("---")
        st.write("**Operations to Optimize**")
        
        # Get all available operations from the configuration
        all_operations = []
        
        # Basic operations
        for op_name, op_config in config.get('operations', {}).items():
            if op_config.get('include', False):
                all_operations.append(op_name)
                
        # Advanced operations
        for op_name, op_config in config.get('advanced_operations', {}).items():
            if op_config.get('include', False):
                all_operations.append(op_name)
                
        # Domain operations
        for op_name, op_config in config.get('domain_operations', {}).items():
            if op_config.get('include', False):
                all_operations.append(op_name)
                
        # If no operations are included, show a message
        if not all_operations:
            st.warning("No operations are currently included in the pipeline. Enable operations in the other tabs to optimize them.")
        else:
            # Default to all operations if none are specified
            default_ops_to_optimize = optimization_params.get('operations_to_optimize', all_operations)
            
            # Ensure all default operations exist in the current pipeline
            default_ops_to_optimize = [op for op in default_ops_to_optimize if op in all_operations]
            
            operations_to_optimize = st.multiselect(
                "Select Operations to Optimize",
                options=all_operations,
                default=default_ops_to_optimize,
                key="operations_to_optimize_multiselect"
            )
            optimization_params['operations_to_optimize'] = operations_to_optimize
            
        # Add help text explaining optimization
        with st.expander("About Pipeline Optimization"):
            st.markdown("""
            **Pipeline Optimization** uses evolutionary algorithms to find the best combination of preprocessing operations and parameters.
            
            How it works:
            1. **Population Initialization**: Creates a set of random pipeline configurations
            2. **Evaluation**: Tests each configuration using cross-validation
            3. **Selection**: Keeps the best-performing configurations
            4. **Evolution**: Creates new configurations through crossover and mutation
            5. **Iteration**: Repeats the process for multiple generations
            
            Benefits:
            - Discovers optimal preprocessing strategies automatically
            - Reduces manual trial-and-error
            - Can find non-obvious parameter combinations
            - Optimizes for specific metrics relevant to your task
            
            The optimization process may take several minutes depending on the dataset size and number of generations.
            """)
            
        # Visualization of optimization results
        st.write("---")
        st.write("**Optimization Results Visualization**")
        
        # Check if optimization results exist in session state
        if 'optimization_results' in st.session_state and st.session_state.optimization_results:
            results = st.session_state.optimization_results
            
            # Show best pipeline configuration
            st.write("**Best Pipeline Configuration**")
            st.json(results['best_pipeline'])
            
            # Show performance metrics
            st.write("**Performance Metrics**")
            st.write(f"Best Score: {results['best_score']:.4f}")
            
            # Plot optimization progress
            fig = px.line(
                x=list(range(1, len(results['history']) + 1)),
                y=results['history'],
                labels={'x': 'Generation', 'y': f'{metric.upper()} Score'},
                title='Optimization Progress'
            )
            st.plotly_chart(fig)
            
            # Plot parameter importance
            if 'parameter_importance' in results:
                param_imp = results['parameter_importance']
                fig = px.bar(
                    x=list(param_imp.keys()),
                    y=list(param_imp.values()),
                    labels={'x': 'Parameter', 'y': 'Importance'},
                    title='Parameter Importance'
                )
                st.plotly_chart(fig)
        else:
            st.info("Run the optimization to see results here.")
            
        # Update optimization parameters
        optimization['params'] = optimization_params
        
    optimization['enable'] = enable_optimization
    config['optimization'] = optimization
    st.session_state.preprocessing_config = config

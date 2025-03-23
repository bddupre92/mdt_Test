"""
Optimizer analyzer module for visualization and analysis of optimization results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import time
import itertools
from typing import Dict, List, Tuple, Any, Optional

class OptimizerAnalyzer:
    def __init__(self, optimizers, results=None):
        self.optimizers = optimizers
        self.results = results if results is not None else {}
        self.performance_data = []
        self.convergence_data = []
        self.algorithm_selection_data = []
        self.parameter_adaptation_data = []
        self.scalability_data = []
        self.robustness_data = []
        
    def add_result(self, func_name, optimizer_name, run, result):
        if func_name not in self.results:
            self.results[func_name] = {}
        if optimizer_name not in self.results[func_name]:
            self.results[func_name][optimizer_name] = []
        self.results[func_name][optimizer_name].append(result)
        
        # Add to performance data for visualizations
        self.performance_data.append({
            'function': func_name,
            'optimizer': optimizer_name,
            'run': run,
            'best_score': result.best_score,
            'execution_time': result.execution_time,
            'evaluations': result.evaluations
        })
        
        # Add convergence data
        for i, score in enumerate(result.convergence_curve):
            self.convergence_data.append({
                'function': func_name,
                'optimizer': optimizer_name,
                'run': run,
                'iteration': i,
                'score': score
            })
            
        # Handle algorithm selections if present
        if hasattr(result, 'algorithm_selections') and result.algorithm_selections is not None:
            self.add_algorithm_selection(func_name, optimizer_name, result.algorithm_selections)
            
        # Handle parameter adaptation if present
        if hasattr(result, 'parameter_history') and result.parameter_history is not None:
            self.add_parameter_adaptation(func_name, optimizer_name, result.parameter_history)
    
    def add_scalability_result(self, dimension, optimizer_name, execution_time, best_score, evaluations):
        """Add data for scalability analysis."""
        self.scalability_data.append({
            'dimension': dimension,
            'optimizer': optimizer_name,
            'execution_time': execution_time,
            'best_score': best_score,
            'evaluations': evaluations
        })
        
    def add_robustness_result(self, noise_level, optimizer_name, func_name, best_score, execution_time):
        """Add data for robustness to noise analysis."""
        self.robustness_data.append({
            'noise_level': noise_level,
            'optimizer': optimizer_name,
            'function': func_name,
            'best_score': best_score,
            'execution_time': execution_time
        })
        
    def add_algorithm_selection(self, func_name, optimizer_name, algorithm_selections):
        """Add algorithm selection data from meta-optimizers.
        
        Args:
            func_name: Name of the function being optimized
            optimizer_name: Name of the optimizer (typically Meta-Optimizer)
            algorithm_selections: List of selected algorithms over iterations
        """
        try:
            # Handle numpy array types
            selections = []
            # Convert all elements to native Python types
            for alg in algorithm_selections:
                if hasattr(alg, 'decode'):
                    # Handle numpy string types (np.str_)
                    selections.append(alg.decode('utf-8'))
                elif hasattr(alg, 'item'):
                    # Handle numpy scalar types
                    selections.append(alg.item())
                elif hasattr(alg, '__str__'):
                    # Handle numpy string representation
                    selections.append(str(alg))
                else:
                    selections.append(alg)
                
            # Add to algorithm selection data
            for i, algo in enumerate(selections):
                self.algorithm_selection_data.append({
                    'function': func_name,
                    'optimizer': optimizer_name,
                    'iteration': i,
                    'selected_algorithm': algo
                })
        except Exception as e:
            logging.error(f"Error processing algorithm selections: {e}")
            
    def add_parameter_adaptation(self, func_name, optimizer_name, parameter_history):
        """Add parameter adaptation data from optimizers.
        
        Args:
            func_name: Name of the function being optimized
            optimizer_name: Name of the optimizer
            parameter_history: List of parameter dictionaries over iterations
        """
        try:
            for i, params in enumerate(parameter_history):
                # Convert parameter dictionary to a format that can be stored
                for param_name, param_value in params.items():
                    # Convert numpy types to Python primitives
                    if hasattr(param_value, 'tolist'):
                        value = param_value.tolist()
                    elif hasattr(param_value, 'item'):
                        value = param_value.item()
                    else:
                        value = param_value
                        
                    self.parameter_adaptation_data.append({
                        'function': func_name,
                        'optimizer': optimizer_name,
                        'iteration': i,
                        'parameter': param_name,
                        'value': value
                    })
        except Exception as e:
            logging.error(f"Error processing parameter history: {e}")
        
    def plot_performance_heatmap(self):
        """Plot performance heatmap showing average scores for each optimizer on each function."""
        # Create a summary DataFrame for the heatmap
        data = []
        for func_name, func_results in self.results.items():
            for opt_name, opt_results in func_results.items():
                avg_score = np.mean([r.best_score for r in opt_results])
                data.append({
                    'function': func_name,
                    'optimizer': opt_name,
                    'score': avg_score
                })
        
        df = pd.DataFrame(data)
        if df.empty:
            logging.warning("No data available for performance heatmap")
            return None
        
        # Create pivot table for the heatmap
        pivot_df = df.pivot(index='function', columns='optimizer', values='score')
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Performance Heatmap: Average Score by Function and Optimizer')
        plt.tight_layout()
        return plt.gcf()
    
    def get_performance_data(self):
        """Return performance data as a DataFrame."""
        return pd.DataFrame(self.performance_data)
    
    def plot_convergence_curves(self, func_name=None):
        """
        Plot convergence curves showing how the best fitness value changes over iterations
        for each algorithm on the specified benchmark function.
        """
        if not self.convergence_data:
            logging.warning("No convergence data available")
            return None
        
        convergence_df = pd.DataFrame(self.convergence_data)
        if convergence_df.empty:
            logging.warning("Convergence DataFrame is empty")
            return None
        
        # Filter by function if specified
        if func_name:
            convergence_df = convergence_df[convergence_df['function'] == func_name]
            if convergence_df.empty:
                logging.warning(f"No convergence data for function {func_name}")
                return None
        
        # Group by function, optimizer, and iteration, then calculate mean and std of scores
        grouped = convergence_df.groupby(['function', 'optimizer', 'iteration'])['score'].agg(['mean', 'std']).reset_index()
        
        # Create a new figure for each function
        functions = grouped['function'].unique()
        figs = []
        
        for func in functions:
            func_data = grouped[grouped['function'] == func]
            plt.figure(figsize=(12, 8))
            
            for optimizer in func_data['optimizer'].unique():
                opt_data = func_data[func_data['optimizer'] == optimizer]
                plt.plot(opt_data['iteration'], opt_data['mean'], label=optimizer)
                
                # Add shaded area for standard deviation
                plt.fill_between(
                    opt_data['iteration'],
                    opt_data['mean'] - opt_data['std'],
                    opt_data['mean'] + opt_data['std'],
                    alpha=0.2
                )
            
            plt.xlabel('Iterations')
            plt.ylabel('Fitness Value')
            plt.title(f'Convergence Curve for {func} Function')
            plt.yscale('log')  # Log scale often works better for optimization plots
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            figs.append((func, plt.gcf()))
            
        return figs
    
    def plot_performance_profiles(self):
        """
        Plot performance profiles showing the fraction of problems that each
        algorithm solves within a factor τ of the best result.
        """
        if not self.performance_data:
            logging.warning("No performance data available")
            return None
            
        performance_df = pd.DataFrame(self.performance_data)
        if performance_df.empty:
            logging.warning("Performance DataFrame is empty")
            return None
            
        # Get best score for each function across all optimizers and runs
        best_scores = performance_df.groupby('function')['best_score'].min().reset_index()
        best_scores.rename(columns={'best_score': 'best_overall'}, inplace=True)
        
        # Merge with main dataframe
        merged_df = pd.merge(performance_df, best_scores, on='function')
        
        # Calculate ratio to best score
        merged_df['ratio'] = merged_df['best_score'] / merged_df['best_overall']
        
        # Create performance profile
        plt.figure(figsize=(12, 8))
        
        # Create tau values (x-axis)
        tau_values = np.logspace(0, 2, 100)  # From 1 to 100
        
        for optimizer in merged_df['optimizer'].unique():
            opt_data = merged_df[merged_df['optimizer'] == optimizer]
            
            # Calculate probability for each tau
            probabilities = []
            for tau in tau_values:
                prob = (opt_data['ratio'] <= tau).mean()
                probabilities.append(prob)
            
            plt.plot(tau_values, probabilities, label=optimizer)
        
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Performance Ratio τ (log scale)')
        plt.ylabel('Probability P(ratio ≤ τ)')
        plt.title('Performance Profiles')
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_algorithm_selection_frequency(self):
        """
        Plot stacked area chart showing which base algorithms the
        meta-optimizer selects over time for different problems.
        """
        if not self.algorithm_selection_data:
            logging.warning("No algorithm selection data available")
            return None
            
        selection_df = pd.DataFrame(self.algorithm_selection_data)
        if selection_df.empty:
            logging.warning("Algorithm selection DataFrame is empty")
            return None
            
        # Group by function and iteration
        functions = selection_df['function'].unique()
        figs = []
        
        for func in functions:
            func_data = selection_df[selection_df['function'] == func]
            
            # Count occurrences of each algorithm at each iteration
            pivot_data = pd.crosstab(
                index=func_data['iteration'], 
                columns=func_data['selected_algorithm']
            )
            
            # Convert to proportions
            proportions = pivot_data.div(pivot_data.sum(axis=1), axis=0)
            
            plt.figure(figsize=(12, 8))
            proportions.plot.area(figsize=(12, 8), alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('Selection Proportion')
            plt.title(f'Algorithm Selection Frequency for {func} Function')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Selected Algorithm')
            plt.tight_layout()
            
            figs.append((func, plt.gcf()))
            
        return figs
    
    def plot_parameter_adaptation(self, optimizer_name=None, parameter=None):
        """
        Plot how parameters change over iterations for different optimizers and problems.
        """
        if not self.parameter_adaptation_data:
            logging.warning("No parameter adaptation data available")
            return None
            
        param_df = pd.DataFrame(self.parameter_adaptation_data)
        if param_df.empty:
            logging.warning("Parameter adaptation DataFrame is empty")
            return None
            
        # Filter by optimizer if specified
        if optimizer_name:
            param_df = param_df[param_df['optimizer'] == optimizer_name]
            
        # Filter by parameter if specified
        if parameter:
            param_df = param_df[param_df['parameter'] == parameter]
            
        # Group by function, optimizer, parameter, and iteration
        functions = param_df['function'].unique()
        optimizers = param_df['optimizer'].unique()
        parameters = param_df['parameter'].unique()
        
        figs = []
        
        # Create a new figure for each function-optimizer-parameter combination
        for func in functions:
            for opt in optimizers:
                for param in parameters:
                    filtered_df = param_df[
                        (param_df['function'] == func) & 
                        (param_df['optimizer'] == opt) & 
                        (param_df['parameter'] == param)
                    ]
                    
                    if not filtered_df.empty:
                        plt.figure(figsize=(12, 8))
                        
                        # Group by iteration and calculate mean and std
                        grouped = filtered_df.groupby('iteration')['value'].agg(['mean', 'std']).reset_index()
                        
                        plt.plot(grouped['iteration'], grouped['mean'], label=f'{param} (mean)')
                        
                        # Add shaded area for standard deviation
                        plt.fill_between(
                            grouped['iteration'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.2
                        )
                        
                        plt.xlabel('Iteration')
                        plt.ylabel('Parameter Value')
                        plt.title(f'Parameter Adaptation for {param} ({opt}, {func})')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.tight_layout()
                        
                        figs.append((f"{func}_{opt}_{param}", plt.gcf()))
        
        return figs
    
    def plot_scalability_analysis(self):
        """
        Plot performance metrics (solution quality, execution time) as a 
        function of problem dimensionality.
        """
        if not self.scalability_data:
            logging.warning("No scalability data available")
            return None
            
        scalability_df = pd.DataFrame(self.scalability_data)
        if scalability_df.empty:
            logging.warning("Scalability DataFrame is empty")
            return None
            
        # Create separate plots for execution time and best score
        figs = []
        
        # Execution time vs dimension
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=scalability_df,
            x='dimension',
            y='execution_time',
            hue='optimizer',
            marker='o'
        )
        plt.xlabel('Problem Dimension')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Scalability: Execution Time vs Problem Dimension')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Optimizer')
        plt.tight_layout()
        figs.append(('execution_time', plt.gcf()))
        
        # Best score vs dimension
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=scalability_df,
            x='dimension',
            y='best_score',
            hue='optimizer',
            marker='o'
        )
        plt.xlabel('Problem Dimension')
        plt.ylabel('Best Score')
        plt.title('Scalability: Solution Quality vs Problem Dimension')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Optimizer')
        plt.tight_layout()
        figs.append(('best_score', plt.gcf()))
        
        return figs
    
    def plot_statistical_significance(self, alpha=0.05):
        """
        Create boxplots with statistical significance indicators.
        """
        if not self.performance_data:
            logging.warning("No performance data available")
            return None
            
        performance_df = pd.DataFrame(self.performance_data)
        if performance_df.empty:
            logging.warning("Performance DataFrame is empty")
            return None
            
        # Create figure for boxplot
        plt.figure(figsize=(14, 10))
        
        # Create boxplot
        ax = sns.boxplot(data=performance_df, x='optimizer', y='best_score', hue='function')
        plt.yscale('log')
        plt.title('Performance Comparison with Statistical Significance')
        plt.xticks(rotation=45)
        
        # Perform statistical tests
        from scipy import stats
        import itertools
        
        # Collect p-values
        p_values = {}
        
        # Get unique optimizers
        optimizers = performance_df['optimizer'].unique()
        
        # Perform t-test for each pair of optimizers
        for opt1, opt2 in itertools.combinations(optimizers, 2):
            scores1 = performance_df[performance_df['optimizer'] == opt1]['best_score']
            scores2 = performance_df[performance_df['optimizer'] == opt2]['best_score']
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(scores1, scores2)
            p_values[(opt1, opt2)] = p_val
            
            # Add significance annotation if significant
            if p_val < alpha:
                # Get x-coordinates of optimizers
                x1 = list(optimizers).index(opt1)
                x2 = list(optimizers).index(opt2)
                
                # Get y-coordinates (maximum score + some offset)
                y = max(scores1.max(), scores2.max()) * 1.1
                
                # Add annotation
                plt.plot([x1, x2], [y, y], 'k-')
                plt.plot([x1, x1], [y, y*0.95], 'k-')
                plt.plot([x2, x2], [y, y*0.95], 'k-')
                plt.text((x1+x2)/2, y*1.05, f'p={p_val:.3f}', ha='center')
        
        plt.tight_layout()
        return plt.gcf(), p_values
    
    def plot_computational_efficiency(self):
        """
        Create bar chart comparing computational resources required by each algorithm.
        """
        if not self.performance_data:
            logging.warning("No performance data available")
            return None
            
        performance_df = pd.DataFrame(self.performance_data)
        if performance_df.empty:
            logging.warning("Performance DataFrame is empty")
            return None
            
        # Group by optimizer and calculate means
        grouped = performance_df.groupby('optimizer').agg({
            'best_score': 'mean',
            'execution_time': 'mean',
            'evaluations': 'mean'
        }).reset_index()
        
        # Create bar charts for execution time and function evaluations
        figs = []
        
        # Execution time comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(data=grouped, x='optimizer', y='execution_time')
        plt.xlabel('Optimizer')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title('Computational Efficiency: Execution Time Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        figs.append(('execution_time', plt.gcf()))
        
        # Function evaluations comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(data=grouped, x='optimizer', y='evaluations')
        plt.xlabel('Optimizer')
        plt.ylabel('Average Function Evaluations')
        plt.title('Computational Efficiency: Function Evaluations Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        figs.append(('evaluations', plt.gcf()))
        
        return figs
    
    def plot_robustness_to_noise(self):
        """
        Plot performance degradation as noise is introduced to the objective function.
        """
        if not self.robustness_data:
            logging.warning("No robustness data available")
            return None
            
        robustness_df = pd.DataFrame(self.robustness_data)
        if robustness_df.empty:
            logging.warning("Robustness DataFrame is empty")
            return None
            
        # Group by noise level and optimizer
        figs = []
        
        # Plot for each function
        for func in robustness_df['function'].unique():
            func_data = robustness_df[robustness_df['function'] == func]
            
            plt.figure(figsize=(12, 8))
            sns.lineplot(
                data=func_data,
                x='noise_level',
                y='best_score',
                hue='optimizer',
                marker='o'
            )
            plt.xlabel('Noise Level')
            plt.ylabel('Best Score')
            plt.title(f'Robustness to Noise: {func} Function')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Optimizer')
            plt.tight_layout()
            
            figs.append((func, plt.gcf()))
            
        return figs
    
    def plot_decision_boundary(self, meta_optimizer, problem_function, bounds=None):
        """
        For 2D problems, create heatmap showing which algorithm the meta-optimizer 
        chooses for different regions of the search space.
        """
        if bounds is None:
            bounds = [(-5, 5), (-5, 5)]  # Default bounds
            
        # Create a grid of points
        x = np.linspace(bounds[0][0], bounds[0][1], 50)
        y = np.linspace(bounds[1][0], bounds[1][1], 50)
        X, Y = np.meshgrid(x, y)
        
        # Initialize empty array for algorithm choices
        Z = np.zeros_like(X, dtype=int)
        
        # Use meta-optimizer to select algorithm for each point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                
                # Get algorithm choice (assuming meta_optimizer has a method to select algorithm)
                if hasattr(meta_optimizer, 'select_algorithm_for_point'):
                    algorithm_idx = meta_optimizer.select_algorithm_for_point(point, problem_function)
                    Z[i, j] = algorithm_idx
                else:
                    logging.warning("Meta-optimizer doesn't have select_algorithm_for_point method")
                    return None
        
        # Get available algorithms
        if hasattr(meta_optimizer, 'available_optimizers'):
            algorithms = meta_optimizer.available_optimizers
        else:
            algorithms = [f"Algorithm {i}" for i in range(len(np.unique(Z)))]
            
        # Plot decision boundary
        plt.figure(figsize=(12, 10))
        plt.contourf(X, Y, Z, levels=len(algorithms), cmap='viridis')
        plt.colorbar(ticks=range(len(algorithms)), label='Selected Algorithm')
        plt.contour(X, Y, problem_function(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape), 
                   levels=10, colors='k', alpha=0.3)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Decision Boundary for Algorithm Selection')
        plt.tight_layout()
        
        return plt.gcf()
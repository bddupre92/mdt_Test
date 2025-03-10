#!/usr/bin/env python3
"""
Analyze benchmark results from the comprehensive benchmark suite.

This script processes results from multiple benchmark runs and generates
comparative visualizations and statistical analyses.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any, Tuple

# Configure matplotlib
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save analysis results (defaults to results-dir/analysis)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()

def collect_results(results_dir: str) -> Dict[str, Any]:
    """
    Collect results from all benchmark runs in the specified directory
    
    Args:
        results_dir: Directory containing benchmark results
        
    Returns:
        Dictionary of collected results
    """
    results_dir = Path(results_dir)
    
    # Find all benchmark result directories
    benchmark_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not benchmark_dirs:
        print(f"No benchmark result directories found in {results_dir}")
        sys.exit(1)
        
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    
    # Collect results from each directory
    collected_results = {}
    
    for benchmark_dir in benchmark_dirs:
        # Look for JSON results files
        json_files = list(benchmark_dir.glob("**/benchmark_results.json"))
        
        if not json_files:
            print(f"No benchmark_results.json found in {benchmark_dir}")
            continue
            
        # Process each JSON file
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract configuration from directory name
                config = benchmark_dir.name
                collected_results[config] = data
                print(f"Loaded results from {json_file}")
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return collected_results

def extract_performance_metrics(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract performance metrics from collected results
    
    Args:
        results: Dictionary of collected results
        
    Returns:
        DataFrame with performance metrics
    """
    # Prepare data for DataFrame
    data = []
    
    for config, config_results in results.items():
        # Extract dimensions from config name
        dimensions = int(config.split('D_')[0]) if 'D_' in config else None
        
        for func_name, func_results in config_results.items():
            # Extract baseline and meta optimizer metrics
            baseline_fitness = func_results.get('baseline_best_fitness_avg', None)
            meta_fitness = func_results.get('meta_best_fitness_avg', None)
            
            if baseline_fitness is not None and meta_fitness is not None:
                # Calculate improvement percentage
                improvement = (baseline_fitness - meta_fitness) / abs(baseline_fitness) * 100
                
                # Add to data
                data.append({
                    'Configuration': config,
                    'Function': func_name,
                    'Dimensions': dimensions,
                    'Baseline Fitness': baseline_fitness,
                    'Meta Optimizer Fitness': meta_fitness,
                    'Improvement (%)': improvement,
                    'Baseline Std': func_results.get('baseline_best_fitness_std', None),
                    'Meta Optimizer Std': func_results.get('meta_best_fitness_std', None),
                    'Baseline Evaluations': func_results.get('baseline_evaluations_avg', None),
                    'Meta Optimizer Evaluations': func_results.get('meta_evaluations_avg', None),
                    'Baseline Time': func_results.get('baseline_time_avg', None),
                    'Meta Optimizer Time': func_results.get('meta_time_avg', None),
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def analyze_performance_by_dimension(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze performance differences by dimension
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save analysis results
    """
    # Filter out rows with missing dimensions
    df_with_dims = df.dropna(subset=['Dimensions'])
    
    if df_with_dims.empty:
        print("No dimension data available for analysis")
        return
    
    # Group by dimensions
    dimension_groups = df_with_dims.groupby('Dimensions')
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot average improvement by dimension
    dimension_improvement = dimension_groups['Improvement (%)'].mean()
    dimension_improvement.plot(kind='bar', yerr=dimension_groups['Improvement (%)'].std())
    
    plt.title('Average Improvement by Dimension')
    plt.xlabel('Dimensions')
    plt.ylabel('Improvement (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'improvement_by_dimension.png')
    plt.close()
    
    # Create detailed table
    dimension_stats = dimension_groups.agg({
        'Improvement (%)': ['mean', 'std', 'min', 'max'],
        'Baseline Fitness': 'mean',
        'Meta Optimizer Fitness': 'mean',
        'Baseline Time': 'mean',
        'Meta Optimizer Time': 'mean'
    })
    
    # Save table as CSV
    dimension_stats.to_csv(output_dir / 'dimension_analysis.csv')
    
    # Also create a markdown table for the report
    with open(output_dir / 'dimension_analysis.md', 'w') as f:
        f.write("# Performance Analysis by Dimension\n\n")
        f.write("| Dimensions | Avg. Improvement (%) | Min Improvement (%) | Max Improvement (%) | Baseline Fitness | Meta Optimizer Fitness |\n")
        f.write("|------------|----------------------|---------------------|---------------------|-----------------|------------------------|\n")
        
        for dim, stats in dimension_stats.iterrows():
            f.write(f"| {dim:.0f} | {stats[('Improvement (%)', 'mean')]:.2f} ± {stats[('Improvement (%)', 'std')]:.2f} | ")
            f.write(f"{stats[('Improvement (%)', 'min')]:.2f} | {stats[('Improvement (%)', 'max')]:.2f} | ")
            f.write(f"{stats[('Baseline Fitness', 'mean')]:.6f} | {stats[('Meta Optimizer Fitness', 'mean')]:.6f} |\n")

def analyze_performance_by_function(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze performance differences by function type
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save analysis results
    """
    # Group by function
    function_groups = df.groupby('Function')
    
    # Create plot
    plt.figure(figsize=(16, 8))
    
    # Plot average improvement by function
    function_improvement = function_groups['Improvement (%)'].mean().sort_values(ascending=False)
    
    # Calculate error bars
    yerr = function_groups['Improvement (%)'].std().reindex(function_improvement.index)
    
    # Plot
    bars = function_improvement.plot(kind='bar', yerr=yerr)
    
    # Add value labels on top of bars
    for i, v in enumerate(function_improvement):
        bars.text(i, v + (5 if v > 0 else -10), f"{v:.1f}%", ha='center')
    
    plt.title('Average Improvement by Function Type')
    plt.xlabel('Function')
    plt.ylabel('Improvement (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'improvement_by_function.png')
    plt.close()
    
    # Create heatmap for function performance across different dimensions
    # First, create a pivot table
    if 'Dimensions' in df.columns and not df['Dimensions'].isna().all():
        pivot_data = df.pivot_table(
            values='Improvement (%)', 
            index='Function',
            columns='Dimensions',
            aggfunc='mean'
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, fmt=".1f")
        plt.title('Improvement (%) by Function and Dimension')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / 'function_dimension_heatmap.png')
        plt.close()
    
    # Create detailed table
    function_stats = function_groups.agg({
        'Improvement (%)': ['mean', 'std', 'min', 'max', 'count'],
        'Baseline Fitness': 'mean',
        'Meta Optimizer Fitness': 'mean',
    }).sort_values(('Improvement (%)', 'mean'), ascending=False)
    
    # Save table as CSV
    function_stats.to_csv(output_dir / 'function_analysis.csv')
    
    # Also create a markdown table for the report
    with open(output_dir / 'function_analysis.md', 'w') as f:
        f.write("# Performance Analysis by Function Type\n\n")
        f.write("| Function | Avg. Improvement (%) | Min Improvement (%) | Max Improvement (%) | Samples | Baseline Fitness | Meta Optimizer Fitness |\n")
        f.write("|----------|----------------------|---------------------|---------------------|---------|-----------------|------------------------|\n")
        
        for func, stats in function_stats.iterrows():
            f.write(f"| {func} | {stats[('Improvement (%)', 'mean')]:.2f} ± {stats[('Improvement (%)', 'std')]:.2f} | ")
            f.write(f"{stats[('Improvement (%)', 'min')]:.2f} | {stats[('Improvement (%)', 'max')]:.2f} | ")
            f.write(f"{stats[('Improvement (%)', 'count')]:.0f} | ")
            f.write(f"{stats[('Baseline Fitness', 'mean')]:.6f} | {stats[('Meta Optimizer Fitness', 'mean')]:.6f} |\n")

def analyze_problem_type_characteristics(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Analyze performance differences by problem type characteristics
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save analysis results
    """
    # Classify functions by characteristics
    problem_types = {
        'unimodal': ['sphere', 'rosenbrock'],
        'multimodal': ['ackley', 'rastrigin', 'griewank', 'levy'],
        'noisy': ['noisy_sphere', 'noisy_rosenbrock', 'noisy_ackley'],
        'dynamic_linear': ['dynamic_sphere_linear', 'dynamic_rosenbrock_linear', 'dynamic_ackley_linear'],
        'dynamic_oscillatory': ['dynamic_sphere_oscillatory', 'dynamic_rosenbrock_oscillatory'],
        'dynamic_random': ['dynamic_sphere_random', 'dynamic_rosenbrock_random']
    }
    
    # Create a new column for problem type
    def determine_problem_type(func_name):
        for p_type, functions in problem_types.items():
            if any(f in func_name.lower() for f in functions):
                return p_type
        return 'other'
    
    df['Problem Type'] = df['Function'].apply(determine_problem_type)
    
    # Group by problem type
    problem_type_groups = df.groupby('Problem Type')
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot average improvement by problem type
    problem_type_improvement = problem_type_groups['Improvement (%)'].mean().sort_values(ascending=False)
    
    # Calculate error bars
    yerr = problem_type_groups['Improvement (%)'].std().reindex(problem_type_improvement.index)
    
    # Plot
    bars = problem_type_improvement.plot(kind='bar', yerr=yerr)
    
    # Add value labels on top of bars
    for i, v in enumerate(problem_type_improvement):
        bars.text(i, v + (5 if v > 0 else -10), f"{v:.1f}%", ha='center')
    
    plt.title('Average Improvement by Problem Type')
    plt.xlabel('Problem Type')
    plt.ylabel('Improvement (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_dir / 'improvement_by_problem_type.png')
    plt.close()
    
    # Create heatmap for problem type performance across different dimensions
    if 'Dimensions' in df.columns and not df['Dimensions'].isna().all():
        pivot_data = df.pivot_table(
            values='Improvement (%)', 
            index='Problem Type',
            columns='Dimensions',
            aggfunc='mean'
        )
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, fmt=".1f")
        plt.title('Improvement (%) by Problem Type and Dimension')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / 'heatmap_problem_type_dimension.png')
        plt.close()
    
    # Create detailed table
    problem_type_stats = problem_type_groups.agg({
        'Improvement (%)': ['mean', 'std', 'min', 'max', 'count'],
        'Baseline Fitness': 'mean',
        'Meta Optimizer Fitness': 'mean'
    })
    
    # Save table as CSV
    problem_type_stats.to_csv(output_dir / 'problem_type_analysis.csv')
    
    # Also create a markdown table for the report
    with open(output_dir / 'problem_type_analysis.md', 'w') as f:
        f.write("# Performance Analysis by Problem Type\n\n")
        f.write("| Problem Type | Avg. Improvement (%) | Min Improvement (%) | Max Improvement (%) | Sample Size | Baseline Fitness | Meta Optimizer Fitness |\n")
        f.write("|--------------|----------------------|---------------------|---------------------|-------------|-----------------|------------------------|\n")
        
        for p_type, stats in problem_type_stats.iterrows():
            f.write(f"| {p_type} | {stats[('Improvement (%)', 'mean')]:.2f} ± {stats[('Improvement (%)', 'std')]:.2f} | ")
            f.write(f"{stats[('Improvement (%)', 'min')]:.2f} | {stats[('Improvement (%)', 'max')]:.2f} | ")
            f.write(f"{stats[('Improvement (%)', 'count')]:.0f} | ")
            f.write(f"{stats[('Baseline Fitness', 'mean')]:.6f} | {stats[('Meta Optimizer Fitness', 'mean')]:.6f} |\n")

def perform_statistical_tests(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Perform enhanced statistical tests for performance comparison
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save analysis results
    """
    # Create a directory for statistical test results
    stat_dir = output_dir / 'statistical_tests'
    stat_dir.mkdir(exist_ok=True)
    
    # Prepare results
    stat_results = []
    
    # Perform paired tests across all functions
    baseline_fitness = df['Baseline Fitness'].values
    meta_fitness = df['Meta Optimizer Fitness'].values
    
    # T-test
    t_stat, t_p = stats.ttest_rel(baseline_fitness, meta_fitness)
    stat_results.append({
        'Test': 'Paired t-test',
        'Metric': 'Fitness',
        'Statistic': t_stat,
        'p-value': t_p,
        'Significant': t_p < 0.05
    })
    
    # Wilcoxon signed-rank test
    try:
        w_stat, w_p = stats.wilcoxon(baseline_fitness, meta_fitness)
        stat_results.append({
            'Test': 'Wilcoxon signed-rank test',
            'Metric': 'Fitness',
            'Statistic': w_stat,
            'p-value': w_p,
            'Significant': w_p < 0.05
        })
    except Exception as e:
        print(f"Warning: Could not perform Wilcoxon test: {e}")
    
    # For each problem type, perform separate tests
    for problem_type, group in df.groupby('Problem Type'):
        baseline = group['Baseline Fitness'].values
        meta = group['Meta Optimizer Fitness'].values
        
        # Skip if sample is too small
        if len(baseline) < 5:
            continue
        
        # T-test
        t_stat, t_p = stats.ttest_rel(baseline, meta)
        stat_results.append({
            'Test': 'Paired t-test',
            'Group': problem_type,
            'Metric': 'Fitness',
            'Statistic': t_stat,
            'p-value': t_p,
            'Significant': t_p < 0.05
        })
        
        # Wilcoxon signed-rank test
        try:
            w_stat, w_p = stats.wilcoxon(baseline, meta)
            stat_results.append({
                'Test': 'Wilcoxon signed-rank test',
                'Group': problem_type,
                'Metric': 'Fitness',
                'Statistic': w_stat,
                'p-value': w_p,
                'Significant': w_p < 0.05
            })
        except Exception as e:
            print(f"Warning: Could not perform Wilcoxon test for {problem_type}: {e}")
    
    # For each dimension, perform separate tests
    if 'Dimensions' in df.columns and not df['Dimensions'].isna().all():
        for dim, group in df.groupby('Dimensions'):
            baseline = group['Baseline Fitness'].values
            meta = group['Meta Optimizer Fitness'].values
            
            # Skip if sample is too small
            if len(baseline) < 5:
                continue
            
            # T-test
            t_stat, t_p = stats.ttest_rel(baseline, meta)
            stat_results.append({
                'Test': 'Paired t-test',
                'Group': f"{dim}D",
                'Metric': 'Fitness',
                'Statistic': t_stat,
                'p-value': t_p,
                'Significant': t_p < 0.05
            })
            
            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = stats.wilcoxon(baseline, meta)
                stat_results.append({
                    'Test': 'Wilcoxon signed-rank test',
                    'Group': f"{dim}D",
                    'Metric': 'Fitness',
                    'Statistic': w_stat,
                    'p-value': w_p,
                    'Significant': w_p < 0.05
                })
            except Exception as e:
                print(f"Warning: Could not perform Wilcoxon test for {dim}D: {e}")
    
    # Create DataFrame
    stat_df = pd.DataFrame(stat_results)
    
    # Save results
    stat_df.to_csv(stat_dir / 'statistical_tests.csv', index=False)
    
    # Create markdown report
    with open(stat_dir / 'statistical_analysis.md', 'w') as f:
        f.write("# Statistical Significance Analysis\n\n")
        
        # Overall tests
        f.write("## Overall Performance Comparison\n\n")
        overall_tests = stat_df[~stat_df['Test'].str.contains('Group')]
        
        if not overall_tests.empty:
            f.write("| Test | Statistic | p-value | Significant |\n")
            f.write("|------|-----------|---------|-------------|\n")
            
            for _, row in overall_tests.iterrows():
                f.write(f"| {row['Test']} | {row['Statistic']:.4f} | {row['p-value']:.4e} | {'Yes' if row['Significant'] else 'No'} |\n")
        
        # By problem type
        f.write("\n## Performance by Problem Type\n\n")
        type_tests = stat_df[stat_df['Group'].notna() & ~stat_df['Group'].astype(str).str.contains('D')]
        
        if not type_tests.empty:
            f.write("| Problem Type | Test | Statistic | p-value | Significant |\n")
            f.write("|--------------|------|-----------|---------|-------------|\n")
            
            for _, row in type_tests.iterrows():
                f.write(f"| {row['Group']} | {row['Test']} | {row['Statistic']:.4f} | {row['p-value']:.4e} | {'Yes' if row['Significant'] else 'No'} |\n")
        
        # By dimension
        f.write("\n## Performance by Dimension\n\n")
        dim_tests = stat_df[stat_df['Group'].notna() & stat_df['Group'].astype(str).str.contains('D')]
        
        if not dim_tests.empty:
            f.write("| Dimension | Test | Statistic | p-value | Significant |\n")
            f.write("|-----------|------|-----------|---------|-------------|\n")
            
            for _, row in dim_tests.iterrows():
                f.write(f"| {row['Group']} | {row['Test']} | {row['Statistic']:.4f} | {row['p-value']:.4e} | {'Yes' if row['Significant'] else 'No'} |\n")
        
        # Summary
        f.write("\n## Summary\n\n")
        significant_count = stat_df['Significant'].sum()
        total_count = len(stat_df)
        
        f.write(f"Out of {total_count} statistical tests performed, {significant_count} ({significant_count/total_count*100:.1f}%) ")
        f.write("showed statistically significant differences between the baseline selector and Meta Optimizer.\n\n")
        
        if significant_count / total_count > 0.5:
            f.write("**Conclusion**: There is strong statistical evidence that the Meta Optimizer outperforms the baseline selector.\n")
        elif significant_count / total_count > 0.25:
            f.write("**Conclusion**: There is moderate statistical evidence that the Meta Optimizer outperforms the baseline selector.\n")
        else:
            f.write("**Conclusion**: There is limited statistical evidence that the Meta Optimizer outperforms the baseline selector.\n")

def analyze_algorithm_selection_patterns(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Analyze patterns in algorithm selection
    
    Args:
        results: Dictionary of collected results
        output_dir: Directory to save analysis results
    """
    # Create a directory for algorithm selection analysis
    alg_dir = output_dir / 'algorithm_selection'
    alg_dir.mkdir(exist_ok=True)
    
    # Extract algorithm selections for each function and configuration
    selections = []
    
    for config, config_results in results.items():
        # Extract dimensions from config name
        dimensions = int(config.split('D_')[0]) if 'D_' in config else None
        
        # Determine if the function is dynamic
        is_dynamic = 'dynamic' in config.lower()
        drift_type = None
        if is_dynamic:
            if 'linear' in config.lower():
                drift_type = 'linear'
            elif 'oscillatory' in config.lower():
                drift_type = 'oscillatory'
            elif 'random' in config.lower():
                drift_type = 'random'
        
        for func_name, func_results in config_results.items():
            # Skip if no algorithm selection data
            if 'baseline_selected_algorithms' not in func_results or 'meta_selected_algorithms' not in func_results:
                continue
            
            # Determine problem type
            problem_type = 'other'
            if 'sphere' in func_name.lower():
                problem_type = 'sphere'
            elif 'rosenbrock' in func_name.lower():
                problem_type = 'rosenbrock'
            elif 'ackley' in func_name.lower():
                problem_type = 'ackley'
            elif 'rastrigin' in func_name.lower():
                problem_type = 'rastrigin'
            
            # Count algorithm selections
            baseline_algs = func_results['baseline_selected_algorithms']
            meta_algs = func_results['meta_selected_algorithms']
            
            baseline_counts = {}
            for alg in baseline_algs:
                baseline_counts[alg] = baseline_counts.get(alg, 0) + 1
                
            meta_counts = {}
            for alg in meta_algs:
                meta_counts[alg] = meta_counts.get(alg, 0) + 1
            
            # Add to selections
            selections.append({
                'Configuration': config,
                'Function': func_name,
                'Problem Type': problem_type,
                'Dimensions': dimensions,
                'Is Dynamic': is_dynamic,
                'Drift Type': drift_type,
                'Baseline Algorithm Counts': baseline_counts,
                'Meta Algorithm Counts': meta_counts,
                'Total Trials': len(baseline_algs)
            })
    
    # Convert to DataFrame
    selection_df = pd.DataFrame(selections)
    
    # Only proceed if we have data
    if selection_df.empty:
        print("No algorithm selection data available")
        return
    
    # Save the raw data
    selection_df.to_csv(alg_dir / 'algorithm_selection_data.csv')
    
    # Analyze algorithm selection by problem type
    problem_types = selection_df['Problem Type'].unique()
    
    # Create plots for algorithm selection by problem type
    plt.figure(figsize=(15, 8))
    
    # For each problem type, create a subplot showing algorithm distribution
    for i, p_type in enumerate(problem_types):
        type_data = selection_df[selection_df['Problem Type'] == p_type]
        
        # Skip if we don't have enough data
        if len(type_data) < 2:
            continue
        
        # Aggregate algorithm counts across all configurations for this problem type
        baseline_algs = {}
        meta_algs = {}
        
        for _, row in type_data.iterrows():
            baseline_counts = row['Baseline Algorithm Counts']
            meta_counts = row['Meta Algorithm Counts']
            
            for alg, count in baseline_counts.items():
                baseline_algs[alg] = baseline_algs.get(alg, 0) + count
                
            for alg, count in meta_counts.items():
                meta_algs[alg] = meta_algs.get(alg, 0) + count
        
        # Convert to percentage
        baseline_total = sum(baseline_algs.values())
        meta_total = sum(meta_algs.values())
        
        baseline_pct = {k: v/baseline_total*100 for k, v in baseline_algs.items()}
        meta_pct = {k: v/meta_total*100 for k, v in meta_algs.items()}
        
        # Create side-by-side bar chart
        plt.subplot(len(problem_types), 2, i*2+1)
        plt.bar(baseline_pct.keys(), baseline_pct.values())
        plt.title(f'{p_type} - Baseline')
        plt.ylabel('Selection %')
        plt.xticks(rotation=45)
        
        plt.subplot(len(problem_types), 2, i*2+2)
        plt.bar(meta_pct.keys(), meta_pct.values())
        plt.title(f'{p_type} - Meta Optimizer')
        plt.ylabel('Selection %')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(alg_dir / 'algorithm_selection_by_problem_type.png')
    plt.close()
    
    # Analyze by dimension if available
    if 'Dimensions' in selection_df.columns and not selection_df['Dimensions'].isna().all():
        dimensions = selection_df['Dimensions'].unique()
        
        plt.figure(figsize=(15, 8))
        
        for i, dim in enumerate(dimensions):
            dim_data = selection_df[selection_df['Dimensions'] == dim]
            
            # Skip if we don't have enough data
            if len(dim_data) < 2:
                continue
            
            # Aggregate algorithm counts
            baseline_algs = {}
            meta_algs = {}
            
            for _, row in dim_data.iterrows():
                baseline_counts = row['Baseline Algorithm Counts']
                meta_counts = row['Meta Algorithm Counts']
                
                for alg, count in baseline_counts.items():
                    baseline_algs[alg] = baseline_algs.get(alg, 0) + count
                    
                for alg, count in meta_counts.items():
                    meta_algs[alg] = meta_algs.get(alg, 0) + count
            
            # Convert to percentage
            baseline_total = sum(baseline_algs.values())
            meta_total = sum(meta_algs.values())
            
            baseline_pct = {k: v/baseline_total*100 for k, v in baseline_algs.items()}
            meta_pct = {k: v/meta_total*100 for k, v in meta_algs.items()}
            
            # Create side-by-side bar chart
            plt.subplot(len(dimensions), 2, i*2+1)
            plt.bar(baseline_pct.keys(), baseline_pct.values())
            plt.title(f'{dim}D - Baseline')
            plt.ylabel('Selection %')
            plt.xticks(rotation=45)
            
            plt.subplot(len(dimensions), 2, i*2+2)
            plt.bar(meta_pct.keys(), meta_pct.values())
            plt.title(f'{dim}D - Meta Optimizer')
            plt.ylabel('Selection %')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(alg_dir / 'algorithm_selection_by_dimension.png')
        plt.close()
    
    # Create summary markdown
    with open(alg_dir / 'algorithm_selection_analysis.md', 'w') as f:
        f.write("# Algorithm Selection Pattern Analysis\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This analysis examines algorithm selection patterns across {len(selection_df)} benchmark configurations.\n\n")
        
        # Summary by problem type
        f.write("## Selection Patterns by Problem Type\n\n")
        
        for p_type in problem_types:
            type_data = selection_df[selection_df['Problem Type'] == p_type]
            
            if len(type_data) < 2:
                continue
                
            f.write(f"### {p_type}\n\n")
            
            # Aggregate algorithm counts
            baseline_algs = {}
            meta_algs = {}
            
            for _, row in type_data.iterrows():
                baseline_counts = row['Baseline Algorithm Counts']
                meta_counts = row['Meta Algorithm Counts']
                
                for alg, count in baseline_counts.items():
                    baseline_algs[alg] = baseline_algs.get(alg, 0) + count
                    
                for alg, count in meta_counts.items():
                    meta_algs[alg] = meta_algs.get(alg, 0) + count
            
            # Convert to percentage
            baseline_total = sum(baseline_algs.values())
            meta_total = sum(meta_algs.values())
            
            baseline_pct = {k: v/baseline_total*100 for k, v in baseline_algs.items()}
            meta_pct = {k: v/meta_total*100 for k, v in meta_algs.items()}
            
            # Most frequently selected algorithms
            baseline_most = max(baseline_pct.items(), key=lambda x: x[1])
            meta_most = max(meta_pct.items(), key=lambda x: x[1])
            
            f.write(f"For {p_type} problems:\n\n")
            f.write(f"- Baseline most frequently selects **{baseline_most[0]}** ({baseline_most[1]:.1f}% of trials)\n")
            f.write(f"- Meta Optimizer most frequently selects **{meta_most[0]}** ({meta_most[1]:.1f}% of trials)\n\n")
            
            # Create table
            f.write("**Baseline Selection Distribution:**\n\n")
            f.write("| Algorithm | Selection % |\n")
            f.write("|-----------|------------|\n")
            
            for alg, pct in sorted(baseline_pct.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {alg} | {pct:.1f}% |\n")
            
            f.write("\n**Meta Optimizer Selection Distribution:**\n\n")
            f.write("| Algorithm | Selection % |\n")
            f.write("|-----------|------------|\n")
            
            for alg, pct in sorted(meta_pct.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {alg} | {pct:.1f}% |\n")
            
            f.write("\n")
        
        # Conclusions
        f.write("## Key Observations\n\n")
        
        # Find cases where baseline and meta optimizer select different algorithms
        different_selections = []
        
        for p_type in problem_types:
            type_data = selection_df[selection_df['Problem Type'] == p_type]
            
            if len(type_data) < 2:
                continue
                
            # Aggregate algorithm counts
            baseline_algs = {}
            meta_algs = {}
            
            for _, row in type_data.iterrows():
                baseline_counts = row['Baseline Algorithm Counts']
                meta_counts = row['Meta Algorithm Counts']
                
                for alg, count in baseline_counts.items():
                    baseline_algs[alg] = baseline_algs.get(alg, 0) + count
                    
                for alg, count in meta_counts.items():
                    meta_algs[alg] = meta_algs.get(alg, 0) + count
            
            # Most frequently selected algorithms
            baseline_most = max(baseline_algs.items(), key=lambda x: x[1])[0]
            meta_most = max(meta_algs.items(), key=lambda x: x[1])[0]
            
            if baseline_most != meta_most:
                different_selections.append((p_type, baseline_most, meta_most))
        
        if different_selections:
            f.write("Problems where the baseline and Meta Optimizer select different algorithms:\n\n")
            f.write("| Problem Type | Baseline Preference | Meta Optimizer Preference |\n")
            f.write("|--------------|---------------------|---------------------------|\n")
            
            for p_type, baseline, meta in different_selections:
                f.write(f"| {p_type} | {baseline} | {meta} |\n")
        else:
            f.write("The baseline selector and Meta Optimizer generally select the same algorithms for similar problems.\n")

def generate_summary_report(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate a summary report of the benchmark results
    
    Args:
        df: DataFrame with performance metrics
        output_dir: Directory to save analysis results
    """
    with open(output_dir / 'summary_report.md', 'w') as f:
        f.write("# Benchmark Analysis Summary Report\n\n")
        
        f.write("## Overall Performance\n\n")
        
        # Overall statistics
        overall_improvement = df['Improvement (%)'].mean()
        improvement_std = df['Improvement (%)'].std()
        positive_improvements = (df['Improvement (%)'] > 0).sum()
        negative_improvements = (df['Improvement (%)'] < 0).sum()
        
        f.write(f"- **Average Improvement**: {overall_improvement:.2f}% ± {improvement_std:.2f}%\n")
        f.write(f"- **Positive Improvements**: {positive_improvements} ({positive_improvements/len(df)*100:.1f}% of cases)\n")
        f.write(f"- **Negative Improvements**: {negative_improvements} ({negative_improvements/len(df)*100:.1f}% of cases)\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Best performing functions
        best_functions = df.groupby('Function')['Improvement (%)'].mean().nlargest(3)
        f.write("### Best Performing Functions\n\n")
        for func, improvement in best_functions.items():
            f.write(f"- **{func}**: {improvement:.2f}%\n")
        f.write("\n")
        
        # Worst performing functions
        worst_functions = df.groupby('Function')['Improvement (%)'].mean().nsmallest(3)
        f.write("### Challenging Functions\n\n")
        for func, improvement in worst_functions.items():
            f.write(f"- **{func}**: {improvement:.2f}%\n")
        f.write("\n")
        
        # Performance by dimension (if available)
        if 'Dimensions' in df.columns and not df['Dimensions'].isna().all():
            dimension_improvement = df.groupby('Dimensions')['Improvement (%)'].mean()
            f.write("### Performance by Dimension\n\n")
            for dim, improvement in dimension_improvement.items():
                f.write(f"- **{dim:.0f}D**: {improvement:.2f}%\n")
            f.write("\n")
        
        f.write("## Conclusion\n\n")
        
        if overall_improvement > 5:
            f.write("The Meta Optimizer shows **significant improvements** over the baseline algorithm selector in most test cases. ")
        elif overall_improvement > 0:
            f.write("The Meta Optimizer shows **modest improvements** over the baseline algorithm selector. ")
        else:
            f.write("The Meta Optimizer does not consistently outperform the baseline algorithm selector. ")
            
        f.write("See the detailed analysis reports for more information on specific performance characteristics.\n\n")
        
        f.write("## Recommended Next Steps\n\n")
        
        f.write("1. **Train the SATzilla-inspired selector** to improve its algorithm selection accuracy\n")
        f.write("2. **Analyze feature importance** to understand which problem characteristics most influence performance\n")
        f.write("3. **Test on real-world problems** to validate the benchmark results in practical applications\n")

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir) / 'analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Collecting results from {args.results_dir}")
    results = collect_results(args.results_dir)
    
    if not results:
        print("No results found. Exiting.")
        return
    
    print(f"Found results for {len(results)} configurations")
    
    # Extract metrics
    print("Extracting performance metrics")
    df = extract_performance_metrics(results)
    
    if df.empty:
        print("No performance metrics found. Exiting.")
        return
    
    # Save DataFrame
    df.to_csv(output_dir / 'performance_metrics.csv', index=False)
    
    # Run analyses
    print("Analyzing performance by dimension")
    analyze_performance_by_dimension(df, output_dir)
    
    print("Analyzing performance by function")
    analyze_performance_by_function(df, output_dir)
    
    print("Analyzing problem type characteristics")
    analyze_problem_type_characteristics(df, output_dir)
    
    print("Performing statistical tests")
    perform_statistical_tests(df, output_dir)
    
    print("Analyzing algorithm selection patterns")
    analyze_algorithm_selection_patterns(results, output_dir)
    
    print("Generating summary report")
    generate_summary_report(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

    # Generate index file for easy navigation
    with open(output_dir / 'index.md', 'w') as f:
        f.write("# Comprehensive Benchmark Analysis Results\n\n")
        f.write("This directory contains the complete analysis of the benchmark comparison between the Meta Optimizer and SATzilla baseline.\n\n")
        
        f.write("## Main Reports\n\n")
        f.write("- [Summary Report](summary_report.md)\n")
        f.write("- [Problem Type Analysis](problem_type_analysis.md)\n")
        f.write("- [Dimension Analysis](dimension_analysis.md)\n")
        
        f.write("\n## Statistical Analysis\n\n")
        f.write("- [Statistical Tests](statistical_tests.md)\n")
        
        f.write("\n## Algorithm Selection Analysis\n\n")
        f.write("- [Algorithm Selection Patterns](algorithm_selection_patterns.md)\n")
        
        f.write("\n## Key Visualizations\n\n")
        f.write("- [Improvement by Problem Type](improvement_by_problem_type.png)\n")
        f.write("- [Improvement by Function](improvement_by_function.png)\n")
        f.write("- [Improvement by Dimension](improvement_by_dimension.png)\n")
        
        f.write("\n## Raw Data\n\n")
        f.write("- [Performance Metrics](performance_metrics.csv)\n")
        
        print(f"Generated index file at {output_dir}/index.md")

if __name__ == "__main__":
    main() 
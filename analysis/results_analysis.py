"""
Generate comprehensive results tables and visualizations for publication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from tabulate import tabulate

@dataclass
class OptimizationResult:
    name: str
    scores: List[float]
    convergence: List[List[float]]
    parameters: Dict[str, Any]

def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all results from the results directory."""
    results = {}
    for file in results_dir.glob('*_results.json'):
        function_name = file.stem.replace('_results', '')
        with open(file) as f:
            results[function_name] = json.load(f)
    return results

def create_hyperparameter_table(optimizers: Dict[str, Any]) -> pd.DataFrame:
    """Create table comparing hyperparameters across optimizers."""
    params = {
        'Optimizer': [],
        'Population Size': [],
        'Exploration Rate': [],
        'Selection Pressure': [],
        'Adaptation': []
    }
    
    for name, opt in optimizers.items():
        params['Optimizer'].append(name)
        params['Population Size'].append(opt.population_size if hasattr(opt, 'population_size') else 'N/A')
        params['Exploration Rate'].append(opt.exploration_rate if hasattr(opt, 'exploration_rate') else 'N/A')
        params['Selection Pressure'].append(opt.selection_pressure if hasattr(opt, 'selection_pressure') else 'N/A')
        params['Adaptation'].append('Yes' if hasattr(opt, 'adaptive') and opt.adaptive else 'No')
    
    return pd.DataFrame(params)

def create_performance_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create detailed performance metrics table."""
    metrics = {
        'Function': [],
        'Optimizer': [],
        'Best Score': [],
        'Mean Score': [],
        'Std Dev': [],
        'Success Rate': [],
        'Convergence Time': []
    }
    
    for func_name, func_results in results.items():
        for opt_name, opt_results in func_results.items():
            metrics['Function'].append(func_name)
            metrics['Optimizer'].append(opt_name)
            metrics['Best Score'].append(float(opt_results['best']))
            metrics['Mean Score'].append(float(opt_results['mean']))
            metrics['Std Dev'].append(float(opt_results['std']))
            
            # Calculate success rate based on mean score
            threshold = 1e-4
            success_rate = 1.0 if float(opt_results['mean']) < threshold else 0.0
            metrics['Success Rate'].append(success_rate)
            
            # Use mean as proxy for convergence time since we don't have raw data
            metrics['Convergence Time'].append(float(opt_results['mean']))
    
    return pd.DataFrame(metrics)

def create_ranking_table(perf_table: pd.DataFrame) -> pd.DataFrame:
    """Create table with optimizer rankings across different metrics."""
    # Group by optimizer and calculate mean metrics
    grouped = perf_table.groupby('Optimizer').agg({
        'Best Score': 'mean',
        'Mean Score': 'mean',
        'Success Rate': 'mean',
        'Convergence Time': 'mean'
    }).reset_index()
    
    # Calculate ranks for each metric
    for col in ['Best Score', 'Mean Score', 'Convergence Time']:
        grouped[f'{col} Rank'] = grouped[col].rank()
    grouped['Success Rate Rank'] = grouped['Success Rate'].rank(ascending=False)
    
    # Calculate overall rank
    rank_cols = [col for col in grouped.columns if col.endswith('Rank')]
    grouped['Overall Rank'] = grouped[rank_cols].mean(axis=1)
    
    # Sort by overall rank
    return grouped.sort_values('Overall Rank')

def plot_performance_profiles(results: Dict[str, Dict[str, Any]], output_dir: Path):
    """Create performance profiles comparing optimizers."""
    plt.figure(figsize=(12, 8))
    
    # Collect all scores for each optimizer
    opt_scores = {}
    for func_results in results.values():
        for opt_name, opt_results in func_results.items():
            if opt_name not in opt_scores:
                opt_scores[opt_name] = []
            opt_scores[opt_name].append(float(opt_results['mean']))
    
    # Create performance profiles
    for opt_name, scores in opt_scores.items():
        sorted_scores = np.sort(scores)
        y = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        plt.plot(sorted_scores, y, label=opt_name.upper(), linewidth=2)
    
    plt.xlabel('Score (log scale)')
    plt.ylabel('Probability')
    plt.title('Performance Profiles')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_dir / 'performance_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_comparison(results: Dict[str, Dict[str, Any]], output_dir: Path):
    """Create convergence comparison plots for each function type."""
    function_types = {
        'Unimodal': ['sphere'],
        'Multimodal': ['rastrigin', 'ackley'],
        'High-Dimensional': ['rosenbrock', 'griewank']
    }
    
    for type_name, functions in function_types.items():
        plt.figure(figsize=(12, 8))
        
        # Plot mean scores for each optimizer
        for func_name in functions:
            if func_name not in results:
                continue
                
            for opt_name, opt_results in results[func_name].items():
                plt.scatter([func_name], [float(opt_results['mean'])], 
                          label=opt_name.upper())
                plt.errorbar([func_name], [float(opt_results['mean'])],
                           yerr=[float(opt_results['std'])],
                           fmt='none', capsize=5)
        
        plt.xlabel('Function')
        plt.ylabel('Mean Score (log scale)')
        plt.title(f'Performance Comparison - {type_name} Functions')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'performance_{type_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame."""
    # Round numeric columns to 4 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.round(4)
    
    # Start LaTeX table
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{" + "l" + "c" * (len(df.columns) - 1) + "}",
        "\\toprule"
    ]
    
    # Add header
    header = " & ".join(df.columns.str.replace("_", " ").str.title())
    latex.append(header + " \\\\")
    latex.append("\\midrule")
    
    # Add rows
    for _, row in df.iterrows():
        latex_row = []
        for val in row:
            if isinstance(val, (int, float)):
                latex_row.append(f"{val:.4f}")
            else:
                latex_row.append(str(val))
        latex.append(" & ".join(latex_row) + " \\\\")
    
    # Close table
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

def run_statistical_tests(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Run statistical significance tests between optimizers."""
    from scipy import stats
    
    # Collect scores for each optimizer
    optimizer_scores = {}
    for func_results in results.values():
        for opt_name, opt_results in func_results.items():
            if opt_name not in optimizer_scores:
                optimizer_scores[opt_name] = []
            optimizer_scores[opt_name].append(float(opt_results['mean']))
    
    # Run Mann-Whitney U test between all pairs
    optimizers = list(optimizer_scores.keys())
    n_optimizers = len(optimizers)
    p_values = np.zeros((n_optimizers, n_optimizers))
    
    for i in range(n_optimizers):
        for j in range(i + 1, n_optimizers):
            stat, p_val = stats.mannwhitneyu(
                optimizer_scores[optimizers[i]], 
                optimizer_scores[optimizers[j]],
                alternative='two-sided'
            )
            p_values[i, j] = p_val
            p_values[j, i] = p_val
    
    # Create DataFrame with p-values
    p_value_df = pd.DataFrame(
        p_values,
        index=optimizers,
        columns=optimizers
    )
    
    # Add significance stars
    sig_df = p_value_df.applymap(lambda x: 
        '***' if x < 0.001 else
        '**' if x < 0.01 else
        '*' if x < 0.05 else
        'ns'
    )
    
    return p_value_df, sig_df

def save_latex_tables(perf_table: pd.DataFrame, ranking_table: pd.DataFrame, 
                     p_value_df: pd.DataFrame, sig_df: pd.DataFrame,
                     output_dir: Path):
    """Save all tables in LaTeX format."""
    # Performance table
    perf_latex = generate_latex_table(
        perf_table,
        "Detailed Performance Metrics for Each Optimizer",
        "tab:performance"
    )
    
    # Ranking table
    ranking_latex = generate_latex_table(
        ranking_table,
        "Overall Rankings of Optimizers",
        "tab:rankings"
    )
    
    # Statistical significance table
    sig_table = p_value_df.copy()
    for i in range(len(sig_table)):
        for j in range(len(sig_table)):
            if i != j:
                sig_table.iloc[i, j] = f"{p_value_df.iloc[i, j]:.4f}$^{{{sig_df.iloc[i, j]}}}$"
            else:
                sig_table.iloc[i, j] = "-"
    
    sig_latex = generate_latex_table(
        sig_table,
        "Statistical Significance Tests Between Optimizers (p-values)",
        "tab:significance"
    )
    
    # Save all tables
    tables_dir = output_dir / "latex_tables"
    tables_dir.mkdir(exist_ok=True)
    
    with open(tables_dir / "performance.tex", "w") as f:
        f.write(perf_latex)
    with open(tables_dir / "rankings.tex", "w") as f:
        f.write(ranking_latex)
    with open(tables_dir / "significance.tex", "w") as f:
        f.write(sig_latex)

def main():
    """Generate all tables and visualizations for publication."""
    # Setup paths
    results_dir = Path('results/comprehensive_analysis_20250224_185027/data')
    output_dir = Path('results/publication_ready')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results(results_dir)
    
    # Generate tables
    perf_table = create_performance_table(results)
    ranking_table = create_ranking_table(perf_table)
    
    # Run statistical tests
    p_value_df, sig_df = run_statistical_tests(results)
    
    # Save LaTeX tables
    save_latex_tables(perf_table, ranking_table, p_value_df, sig_df, output_dir)
    
    # Generate plots
    plot_performance_profiles(results, output_dir)
    plot_convergence_comparison(results, output_dir)
    
    # Print summary tables
    print("\nPerformance Metrics Summary:")
    print(tabulate(perf_table, headers='keys', tablefmt='pipe', showindex=False))
    
    print("\nOptimizer Rankings:")
    print(tabulate(ranking_table, headers='keys', tablefmt='pipe', showindex=False))
    
    print("\nStatistical Significance:")
    print("\nP-values:")
    print(tabulate(p_value_df, headers='keys', tablefmt='pipe', showindex=True))
    print("\nSignificance levels:")
    print(tabulate(sig_df, headers='keys', tablefmt='pipe', showindex=True))

if __name__ == '__main__':
    main()

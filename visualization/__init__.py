"""
Visualization package for MoE framework.

This package contains visualization modules for the MoE framework, 
including publication-ready visualizations, expert contribution visualizations,
validation visualizations, and more.
"""

# Publication visualizations
from .publication_viz import (
    export_publication_figure,
    create_expert_contribution_heatmap,
    create_expert_weights_timeline,
    create_ablation_study_chart,
    create_kfold_validation_chart,
    create_patient_subgroup_analysis,
    load_ablation_study_results,
    load_kfold_validation_results,
    generate_publication_figures
)

# Expert visualizations
from .expert_viz import (
    extract_expert_data_from_workflow,
    load_workflow_expert_data,
    create_expert_agreement_matrix,
    create_expert_confidence_chart,
    create_expert_contribution_chart,
    create_expert_weight_evolution,
    plot_expert_dominance_regions,
    create_expert_ensemble_dashboard
)

# Validation visualizations
from .validation_viz import (
    load_validation_reports,
    extract_kfold_results,
    extract_ablation_results,
    create_kfold_boxplot,
    create_ablation_barchart,
    create_model_comparison_radar,
    create_metric_correlation_heatmap,
    create_validation_summary_dashboard,
    generate_validation_report_pdf
)

__all__ = [
    # Publication
    'export_publication_figure',
    'create_expert_contribution_heatmap', 
    'create_expert_weights_timeline',
    'create_ablation_study_chart',
    'create_kfold_validation_chart',
    'create_patient_subgroup_analysis',
    'load_ablation_study_results',
    'load_kfold_validation_results',
    'generate_publication_figures',
    
    # Expert
    'extract_expert_data_from_workflow',
    'load_workflow_expert_data',
    'create_expert_agreement_matrix',
    'create_expert_confidence_chart',
    'create_expert_contribution_chart',
    'create_expert_weight_evolution',
    'plot_expert_dominance_regions',
    'create_expert_ensemble_dashboard',
    
    # Validation
    'load_validation_reports',
    'extract_kfold_results',
    'extract_ablation_results',
    'create_kfold_boxplot',
    'create_ablation_barchart',
    'create_model_comparison_radar',
    'create_metric_correlation_heatmap',
    'create_validation_summary_dashboard',
    'generate_validation_report_pdf'
]

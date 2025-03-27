# MoE Framework Architecture

This document provides a comprehensive overview of the MoE (Mixture of Experts) framework architecture, detailing each component, its purpose, and connections to other components. This documentation serves as a living reference that will be updated as new components are added or existing ones are modified.

> **Update (2025-03-25)**: Added MoE Framework Interfaces section to document the unified interfaces created as part of Critical Path 3, Step 1. Updated to reflect completion of MoE metrics implementation, baseline framework integration, and validation framework.

## Overview

The MoE framework is designed with evolutionary computation (EC) algorithms as its core differentiator. The architecture follows these key principles:

1. **Extend existing components** rather than creating parallel implementations
2. **Use inheritance and composition** to add new functionality
3. **Maintain backward compatibility** with existing interfaces
4. **Reuse existing configuration systems** and parameter structures

## MoE Framework Interfaces

The MoE Framework interfaces define the core contracts that all components must adhere to, ensuring consistent integration and interaction patterns throughout the system. These interfaces have been implemented as part of Critical Path 3, Step 1.

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `Configurable` | `moe_framework/interfaces/base.py` | Base interface for all configurable components | Used by all components that require configuration capabilities |
| `Persistable` | `moe_framework/interfaces/base.py` | Base interface for all components that can be persisted | Used by all components that need state saving/loading |
| `DataStructure` | `moe_framework/interfaces/base.py` | Base class for all data structures | Parent class for `PatientContext`, `PredictionResult`, and `SystemState` |
| `PatientContext` | `moe_framework/interfaces/base.py` | Data structure for patient context information | Used throughout the pipeline for personalization |
| `PredictionResult` | `moe_framework/interfaces/base.py` | Data structure for prediction results | Used to provide structured output from pipelines |
| `SystemState` | `moe_framework/interfaces/base.py` | Data structure for system state information | Used for checkpointing and state persistence |
| `ExpertModel` | `moe_framework/interfaces/expert.py` | Base interface for all expert models | Defines the core API that all expert models must implement, including methods for training, prediction, evaluation, and feature importance. Each expert specializes in a specific domain of data (physiological, behavioral, etc.) |
| `ExpertRegistry` | `moe_framework/interfaces/expert.py` | Registry for managing expert models | Singleton pattern implementation that maintains a catalog of available experts and their metadata. Allows for dynamic discovery and instantiation of expert models at runtime |
| `GatingNetwork` | `moe_framework/interfaces/gating.py` | Base interface for gating networks | Defines the contract for all gating networks, which are responsible for determining the weights assigned to each expert's predictions at inference time. The weighting can be static or dynamic based on input data characteristics |
| `QualityAwareGating` | `moe_framework/interfaces/gating.py` | Extension for quality-aware gating | Specialized gating network that adjusts expert weights based on data quality metrics. Implements mechanisms to reduce the influence of experts when their input data is of low quality or contains missing values |
| `DriftAwareGating` | `moe_framework/interfaces/gating.py` | Extension for drift-aware gating | Gating network that monitors for concept drift in the data stream and adjusts expert weights accordingly. Helps the system adapt to changing data distributions without requiring full retraining |
| `Pipeline` | `moe_framework/interfaces/pipeline.py` | Base interface for end-to-end pipelines | Parent class for all pipeline implementations |
| `TrainingPipeline` | `moe_framework/interfaces/pipeline.py` | Extension for training workflows | Used for implementing training pipelines |
| `CheckpointingPipeline` | `moe_framework/interfaces/pipeline.py` | Extension for checkpointing capabilities | Used for implementing resumable pipelines |
| `Optimizer` | `moe_framework/interfaces/optimizer.py` | Base interface for optimizers | Parent class for all optimizer implementations |
| `ExpertSpecificOptimizer` | `moe_framework/interfaces/optimizer.py` | Extension for expert-specific optimization | Used for domain-specific parameter tuning |
| `OptimizerFactory` | `moe_framework/interfaces/optimizer.py` | Factory for creating optimizer instances | Used to create appropriate optimizers for different contexts |

## App Directory Structure and UI Components

> **Update (2025-03-25)**: Added comprehensive documentation of the app directory structure and UI components, with focus on the recently completed Performance Analysis Dashboard.

The application layer is organized into a modular structure that promotes separation of concerns and maintainability. Below is the detailed organization of the `/app` directory.

### Main Directory Structure

| Directory | Purpose | Components |
|-----------|---------|------------|
| `/app/api` | API endpoints and services | REST API endpoints, service interfaces |
| `/app/core` | Core application logic | Models, services, security, monitoring |
| `/app/reporting` | Reporting capabilities | Report generation, data export |
| `/app/ui` | User interface components | Dashboards, UI components, utilities |
| `/app/visualization` | Visualization utilities | Plotting functions, interactive visualizations |

### UI Layer Organization

The UI layer consists of several main dashboard files that serve as entry points for different application functionalities:

| Dashboard | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `BenchmarkDashboard` | `app/ui/benchmark_dashboard.py` | Dashboard for benchmarking optimization algorithms | Provides tools for running, visualizing, and comparing optimization benchmarks |
| `PerformanceAnalysisDashboard` | `app/ui/performance_analysis_dashboard.py` | Dashboard for MoE performance analysis | Comprehensive UI for analyzing MoE performance metrics, integrating expert benchmarks, gating evaluation, and end-to-end metrics |
| `ResultsDashboard` | `app/ui/results_dashboard.py` | Dashboard for results management | Manages experiment results, allows export and visualization |

### Performance Analysis Components

The Performance Analysis Dashboard is built using a modular architecture with specialized components for different aspects of MoE performance evaluation:

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `PerformanceAnalysis` | `app/ui/components/performance_analysis.py` | Main performance analysis UI | Core component that integrates all performance metrics views and provides the main user interface |
| `ExpertBenchmarks` | `app/ui/components/performance_views/expert_benchmarks.py` | Expert model evaluation | Visualizes individual expert model performance metrics, feature importance, and error analysis |
| `GatingAnalysis` | `app/ui/components/performance_views/gating_analysis.py` | Gating network analysis | Analyzes expert selection quality, weight distributions, and decision boundaries of the gating network |
| `TemporalAnalysis` | `app/ui/components/performance_views/temporal_analysis.py` | Performance over time | Tracks performance metrics over time, with trend analysis and drift detection visualization |
| `BaselineComparisons` | `app/ui/components/performance_views/baseline_comparisons.py` | Comparison with baselines | Compares MoE performance against baseline models with statistical significance testing |
| `StatisticalTests` | `app/ui/components/performance_views/statistical_tests.py` | Statistical analysis | Provides parametric and non-parametric tests for model comparison with visualizations |
| `OverallMetrics` | `app/ui/components/performance_views/overall_metrics.py` | Overall performance metrics | Displays aggregated performance metrics (RMSE, MAE, R²) with context and interpretation |
| `Visualizations` | `app/ui/components/performance_views/visualizations.py` | Advanced visualizations | Provides specialized visualizations including performance profiles, reliability diagrams, and critical difference plots |

### Preprocessing Pipeline Components

The preprocessing system uses a modular approach with a drag-and-drop interface for pipeline construction:

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `DragDropPipeline` | `app/ui/components/drag_drop_pipeline.py` | Drag-and-drop pipeline builder | Interactive UI for constructing preprocessing pipelines with drag-and-drop functionality |
| `PreprocessingPipelineComponent` | `app/ui/components/preprocessing_pipeline_component.py` | Main preprocessing pipeline | Core component managing the execution and visualization of preprocessing pipelines |
| `PreprocessingBasicOps` | `app/ui/components/preprocessing_basic_ops.py` | Basic preprocessing operations | Implementation of fundamental preprocessing operations (normalization, imputation, etc.) |
| `PreprocessingAdvancedOps` | `app/ui/components/preprocessing_advanced_ops.py` | Advanced preprocessing | Complex preprocessing operations (dimensionality reduction, feature extraction) |
| `PreprocessingDomainOps` | `app/ui/components/preprocessing_domain_ops.py` | Domain-specific operations | Medical domain-specific preprocessing operations (physiological signal processing, etc.) |
| `PreprocessingOptimization` | `app/ui/components/preprocessing_optimization.py` | Pipeline optimization | Tools for optimizing preprocessing pipeline parameters and configuration |
| `PreprocessingResults` | `app/ui/components/preprocessing_results.py` | Results visualization | Visualization and analysis of preprocessing results and impact on model performance |

### Other Key Components

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `Comparison` | `app/ui/components/comparison.py` | Benchmark comparison | Component for comparing optimization benchmark results with interactive reports |
| `DataConfiguration` | `app/ui/components/data_configuration.py` | Data configuration UI | Interface for configuring data sources, formats, and parameters |
| `DataConfigConnector` | `app/ui/components/data_config_connector.py` | Data configuration connector | Connects UI data configuration to backend data sources |
| `Overview` | `app/ui/components/overview.py` | Overview dashboard | Provides summary views and entry points to different application functionalities |
| `ReportGenerator` | `app/ui/components/report_generator.py` | Report generation | Creates comprehensive reports for benchmarks and performance analysis |
| `ResultsManagement` | `app/ui/components/results_management.py` | Results management | Manages experiment results, artifacts, and metadata |

### Component Integration Pattern

Components in the application follow a consistent integration pattern:

1. **Modular Design**: Each component is self-contained with clear responsibilities
2. **Standardized Interfaces**: Components use consistent input/output patterns
3. **Event-Driven Communication**: Components communicate via events to maintain loose coupling
4. **State Management**: SystemState is used for persistent storage of component state
5. **Configurable Visualization**: Components provide multiple visualization options

This architecture ensures that new components can be easily added and existing ones modified without disrupting the overall system. The Performance Analysis Dashboard demonstrates this pattern by integrating multiple specialized components into a cohesive user experience.

## Component Map

### Recently Added Components

| Component | File Path | Description | Purpose |
|-----------|-----------|-------------|--------|
| `TimeSeriesValidator` | `moe_framework/validation/time_series_validation.py` | Specialized cross-validation for time-series data | Provides validation strategies that respect temporal dependencies and prevent data leakage in medical time-series data |
| `BenchmarkRunner` | `moe_framework/benchmark/performance_benchmarks.py` | Performance benchmarking framework | Measures and compares performance metrics (execution time, memory usage, accuracy) across different MoE configurations |
| `MoEMetricsCalculator` | `baseline_comparison/moe_metrics.py` | Comprehensive MoE metrics calculation | Computes and visualizes expert contribution metrics, confidence metrics, and gating network quality metrics |
| `MoEBaselineAdapter` | `baseline_comparison/moe_adapter.py` | Adapter for baseline framework integration | Integrates the MoE pipeline with the existing baseline comparison framework |
| `MoEBaselineComparison` | `baseline_comparison/moe_comparison.py` | Extends baseline comparison system | Provides MoE-specific comparison capabilities and validation summaries |
| `MoEComparisonCommand` | `cli/moe_commands.py` | CLI command for MoE comparisons | Adds command-line interface for running MoE comparisons |
| `AdaptiveIntegrationTests` | `moe_tests/test_adaptive_integration_strategies.py` | Tests for adaptive integration strategies | Validates the correctness of confidence-based and quality-aware integration strategies across various scenarios |
| `PerformanceBenchmarkTests` | `moe_tests/test_performance_benchmarks.py` | Tests for performance benchmarking | Validates the benchmarking framework functionality and result comparisons |

### Time Series Validation Strategies

| Strategy | Implementation | Description | Use Case |
|----------|----------------|-------------|----------|
| `RollingWindowSplit` | `TimeSeriesValidator.rolling_window_split()` | Creates temporal cross-validation folds using fixed-size windows that roll forward in time | Medical data with evolving patterns over time |
| `PatientAwareSplit` | `TimeSeriesValidator.patient_aware_split()` | Creates folds ensuring no patient data leaks between training and testing sets | Clinical studies where patient identity is crucial |
| `ExpandingWindowSplit` | `TimeSeriesValidator.expanding_window_split()` | Creates folds with progressively larger training sets, maintaining temporal order | Long-term studies with accumulating historical data |

### MoE Metrics System

| Metric Category | Implementation | Description | Purpose |
|----------------|----------------|-------------|--------|
| `Expert Contribution Metrics` | `MoEMetricsCalculator.compute_expert_contribution_metrics()` | Computes metrics related to expert contributions | Measures how much each expert contributes to predictions, expert dominance, and contribution diversity |
| `Confidence Metrics` | `MoEMetricsCalculator.compute_confidence_metrics()` | Computes metrics related to prediction confidence | Evaluates confidence calibration, confidence-error correlation, and expected calibration error (ECE) |
| `Gating Network Metrics` | `MoEMetricsCalculator.compute_gating_network_metrics()` | Evaluates gating network quality | Measures optimal expert selection rate, regret, and weight-error correlation |
| `Temporal Metrics` | `MoEMetricsCalculator.compute_temporal_metrics()` | Computes specialized metrics for time series predictions | Analyzes prediction performance over time and temporal expert contribution patterns |
| `Personalization Metrics` | `MoEMetricsCalculator.compute_personalization_metrics()` | Evaluates personalization effectiveness | Measures expert specialization by patient and personalized prediction accuracy |
| `Visualization` | `MoEMetricsCalculator.visualize_metrics()` | Generates visualizations of key metrics | Creates plots and charts to visually analyze MoE performance |

Below is a detailed map of the framework components, organized by their functional areas.

### Examples and Demonstrations

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `phase1_demo.py` | `examples/phase1_demo.py` | Demonstrates the use of all Phase 1 components | Uses `FileDataConnector`, `DataQualityAssessment`, `UploadManager`, and `ExecutionPipeline` |
| `preprocessing_pipeline_example.py` | `examples/preprocessing_pipeline_example.py` | Demonstrates the use of domain-specific preprocessing operations | Uses all preprocessing operations and pipeline components |
| `expert_models_example.py` | `examples/expert_models_example.py` | Demonstrates the use of expert models in a complete workflow | Uses all expert models with their respective preprocessing operations and optimizers |
| `expert_optimizer_integration_example.py` | `examples/expert_optimizer_integration_example.py` | Demonstrates the use of expert-optimizer integration for hyperparameter optimization | Uses all expert models with their specialized optimizers, showing domain-specific hyperparameter optimization and early stopping configuration |

### Data Connectors

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `BaseDataConnector` | `moe_framework/data_connectors/base_connector.py` | Abstract base class defining the standard interface for all data connectors | Used by all data connector implementations |
| `FileDataConnector` | `moe_framework/data_connectors/file_connector.py` | Connector for loading data from file-based sources (CSV, Excel, etc.) | Inherits from `BaseDataConnector`; Used by data processing components |
| `DataQualityAssessment` | `moe_framework/data_connectors/data_quality.py` | Assesses data quality and provides metrics for EC algorithm selection | Used by `FileDataConnector` and Meta-Optimizer for algorithm selection |

### Expert Models

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `BaseExpert` | `moe_framework/experts/base_expert.py` | Abstract base class for all expert models | Defines common interface for all experts; Provides quality metrics tracking, feature importance calculation, hyperparameter optimization, and early stopping capabilities |
| `PhysiologicalExpert` | `moe_framework/experts/physiological_expert.py` | Expert model for physiological data | Integrates with `PhysiologicalSignalProcessor` and Differential Evolution optimizer |
| `EnvironmentalExpert` | `moe_framework/experts/environmental_expert.py` | Expert model for environmental data | Integrates with `EnvironmentalTriggerAnalyzer` and Evolution Strategy optimizer |
| `BehavioralExpert` | `moe_framework/experts/behavioral_expert.py` | Expert model for behavioral data | Uses Ant Colony Optimization for feature selection |
| `MedicationHistoryExpert` | `moe_framework/experts/medication_history_expert.py` | Expert model for medication history data | Integrates with `MedicationNormalizer` and uses hybrid evolutionary approach |
| `ExpertOptimizerIntegration` | `moe_framework/experts/expert_optimizer_integration.py` | Connects expert models with their specialized optimizers | Manages optimizer creation, hyperparameter optimization, and early stopping for each expert type. Includes robust error handling, domain-specific evaluation functions, and fallback mechanisms for different optimizer interfaces. |

### Meta-Optimization Components

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `MetaOptimizer` | `meta_optimizer/meta/meta_optimizer.py` | Dynamically selects the best EC algorithm for each expert model | Consumes data quality metrics; Manages EC algorithms |
| `Meta_Learner` | `meta_optimizer/meta/meta_learner.py` | Forms the core of the gating network's adaptive weighting mechanism | Used by `GatingNetwork`; Consumes data quality metrics |
| `ProblemAnalysis` | `meta_optimizer/meta/problem_analysis.py` | Analyzes problem characteristics to guide optimizer selection | Used by `MetaOptimizer` to characterize problems |
| `OptimizationHistory` | `meta_optimizer/meta/optimization_history.py` | Tracks optimization history for performance analysis | Used by `MetaOptimizer` to improve selection over time |
| `SelectionTracker` | `meta_optimizer/meta/selection_tracker.py` | Tracks algorithm selection decisions | Used for analysis and improvement of selection logic |
| `ExpertEvaluationFunctions` | `meta_optimizer/evaluation/expert_evaluation_functions.py` | Specialized evaluation functions for different expert domains | Contains domain-specific metrics mapped to expert types: RMSE with smoothness penalty for physiological data, MAE with lag penalty for environmental data, weighted RMSE/MAE for behavioral data, and treatment response score for medication history |

### Evolutionary Computation Algorithms

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `BaseOptimizer` | `meta_optimizer/optimizers/base_optimizer.py` | Abstract base class for all optimization algorithms | Parent class for all EC algorithm implementations |
| `DifferentialEvolution` | `meta_optimizer/optimizers/de.py` | Implements Differential Evolution algorithm | Used for physiological expert training; Managed by `MetaOptimizer` |
| `EvolutionStrategy` | `meta_optimizer/optimizers/es.py` | Implements Evolution Strategy algorithm | Used for environmental expert training; Managed by `MetaOptimizer` |
| `AntColonyOptimization` | `meta_optimizer/optimizers/aco.py` | Implements Ant Colony Optimization algorithm | Used for feature selection in behavioral expert; Managed by `MetaOptimizer` |
| `HybridEvolutionaryOptimizer` | `meta_optimizer/optimizers/hybrid.py` | Implements a hybrid approach combining multiple evolutionary algorithms | Used for medication history expert training; Combines GA, DE, and local search techniques |
| `GreyWolfOptimizer` | `meta_optimizer/optimizers/gwo.py` | Implements Grey Wolf Optimizer algorithm | Used for gating network optimization; Managed by `MetaOptimizer` |
| `OptimizerFactory` | `meta_optimizer/optimizers/optimizer_factory.py` | Factory for creating optimizer instances | Used by `ExpertOptimizerIntegration` and `MetaOptimizer` to instantiate algorithms; Includes expert-specific configurations and hyperparameter spaces |

### Expert Models

#### Planned Modular Architecture

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `BaseExpert` | `moe_framework/experts/base_expert.py` | Abstract base class for all expert models | Parent class for all expert model implementations |
| `PhysiologicalExpert` | `moe_framework/experts/physiological_expert.py` | Expert model for physiological data | Trained with Differential Evolution (DE) |
| `EnvironmentalExpert` | `moe_framework/experts/environmental_expert.py` | Expert model for environmental data | Trained with Evolution Strategy (ES) |
| `BehavioralExpert` | `moe_framework/experts/behavioral_expert.py` | Expert model for behavioral data | Uses Ant Colony Optimization (ACO) for feature selection |
| `MedicationHistoryExpert` | `moe_framework/experts/medication_history_expert.py` | Expert model for medication and history data | Uses hybrid evolutionary approach |

#### Current Implementation

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `MockExpert` | `tests/moe_enhanced_validation_part1.py` | Mock implementation of expert models for testing | Used in validation tests; Implemented with RandomForestRegressor |
| `IntegratedSystemTests` | `tests/moe_validation_runner.py` | Integration tests for the MoE system | Orchestrates expert models in end-to-end tests |
| `SelectiveExpertRetraining` | `core/selective_expert_retraining.py` | Handles retraining of expert models | Integrated with drift detection system |

### Gating Network

#### Implemented Architecture

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `GatingNetwork` | `moe_framework/gating/gating_network.py` | Manages the weighting of expert models | Uses weighted average strategy; Supports dynamic selection and ensemble stacking strategies |
| `MetaLearnerGating` | `moe_framework/gating/meta_learner_gating.py` | Integrates Meta_Learner for adaptive expert weighting | Uses `Meta_Learner` for weight prediction; Tracks performance metrics |
| `QualityAwareWeighting` | `moe_framework/gating/quality_aware_weighting.py` | Adjusts expert weights based on data quality | Used by `GatingNetwork` for quality-based weighting; Supports dynamic threshold adjustment, expert-specific threshold calibration, and patient-specific adaptations |
| `GatingOptimizer` | `moe_framework/gating/gating_optimizer.py` | Optimizes gating network parameters | Integrates with Grey Wolf Optimizer (GWO) for weight optimization |

#### Testing Implementation

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `GatingNetworkTests` | `tests/test_gating_network.py` | Tests for gating network functionality | Validates weight prediction and expert integration |
| `MetaLearnerGatingTests` | `tests/test_meta_learner_gating.py` | Tests for Meta_Learner integration | Validates adaptive weighting and context handling |
| `QualityAwareIntegrationTests` | `tests/test_quality_aware_meta_learner_integration.py` | Tests for quality-aware weighting | Validates quality-based adjustments to expert weights |
| `QualityAwareWeightingTests` | `tests/test_quality_aware_weighting.py` | Tests for quality-aware weighting functionality | Validates dynamic threshold adjustment, expert-specific threshold calibration, patient adaptation mechanisms, and expert-specific quality profiles |

### Upload and Execution

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `UploadManager` | `moe_framework/upload/upload_manager.py` | Manages the upload and validation of data files | Uses `FileDataConnector` and `DataQualityAssessment` |
| `ExecutionPipeline` | `moe_framework/execution/execution_pipeline.py` | Manages the end-to-end execution workflow | Orchestrates all components for one-click execution |

### Training (Planned)

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `MoETrainingManager` | `moe_framework/training/moe_training_manager.py` | Manages the training of expert models | Uses `MetaOptimizer` to train experts |

### Validation and Evaluation

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `ValidationFramework` | `moe_framework/validation/validation_framework.py` | Validates EC algorithm performance | Uses data quality metrics; Evaluates expert models |
| `PerformanceEvaluator` | `moe_framework/validation/performance_evaluator.py` | Evaluates model performance | Used by `ValidationFramework` |
| `MetricsCollector` | `meta_optimizer/evaluation/metrics_collector.py` | Collects performance metrics for optimizers | Used by validation components to track performance |

### Preprocessing Components

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `PreprocessingOperation` | `data/preprocessing_pipeline.py` | Abstract base class for all preprocessing operations | Parent class for all preprocessing operations |
| `PreprocessingPipeline` | `data/preprocessing_pipeline.py` | Chains multiple preprocessing operations together | Uses all preprocessing operations |
| `MedicationNormalizer` | `data/domain_specific_preprocessing.py` | Normalizes medication names, dosages, and frequencies | Inherits from `PreprocessingOperation` |
| `SymptomExtractor` | `data/domain_specific_preprocessing.py` | Extracts symptoms from text data | Inherits from `PreprocessingOperation` |
| `TemporalPatternExtractor` | `data/domain_specific_preprocessing.py` | Extracts temporal features from timestamp data | Inherits from `PreprocessingOperation` |
| `PhysiologicalSignalProcessor` | `data/domain_specific_preprocessing.py` | Processes physiological signals | Inherits from `PreprocessingOperation` |
| `ComorbidityAnalyzer` | `data/domain_specific_preprocessing.py` | Analyzes comorbid conditions | Inherits from `PreprocessingOperation` |
| `EnvironmentalTriggerAnalyzer` | `data/domain_specific_preprocessing.py` | Analyzes environmental factors as potential migraine triggers | Inherits from `PreprocessingOperation` |
| `AdvancedFeatureEngineer` | `data/domain_specific_preprocessing.py` | Implements advanced feature engineering techniques | Inherits from `PreprocessingOperation` |

### Drift Detection Components

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `DriftDetector` | `meta_optimizer/drift_detection/drift_detector.py` | Detects data drift and concept drift | Used by expert models for adaptive retraining |
| `DriftAnalysis` | `meta_optimizer/visualization/drift_analysis.py` | Visualizes drift patterns and impacts | Uses `DriftDetector` output for visualization |

### Explainability Components

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `ExplainerFactory` | `meta_optimizer/explainability/explainer_factory.py` | Factory for creating explainer instances | Used by expert models for explanation generation |

### Visualization Components

| Component | File Path | Purpose | Connections |
|-----------|-----------|---------|------------|
| `LiveVisualization` | `meta_optimizer/visualization/live_visualization.py` | Provides real-time visualization of optimization | Used during training and optimization |
| `OptimizerAnalysis` | `meta_optimizer/visualization/optimizer_analysis.py` | Analyzes and visualizes optimizer performance | Used for comparing different EC algorithms |
| `PlotUtils` | `meta_optimizer/utils/plot_utils.py` | Utility functions for plotting | Used by visualization components |

# MoE Framework Architecture Updates (2025-03-25)

## MoE Visualization System

> **Update (2025-03-25)**: Added comprehensive visualization capabilities for MoE metrics and baseline comparison, enabling effective representation of confidence metrics, expert contributions, and performance comparisons.

### MoE Visualization Components

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `MoEMetricsCalculator.visualize_metrics()` | `baseline_comparison/moe_metrics.py` | Generates visualizations of MoE-specific metrics | Creates visualizations for confidence calibration, expert contributions, and error distributions. Outputs PNG files to specified directory. |
| `MoEMetricsCalculator.visualize_comparison()` | `baseline_comparison/moe_metrics.py` | Compares MoE with baseline approaches | Generates comparative visualizations showing performance differences between MoE and baseline selectors. Creates radar charts, performance comparisons, and confidence visualizations. |
| `MoEBaselineComparison.visualize_results()` | `baseline_comparison/moe_comparison.py` | Comprehensive visualization suite for comparison results | Produces standard visualizations (performance comparison, selection frequency, convergence) along with MoE-specific visualizations. |
| `MoEBaselineComparison.visualize_supervised_comparison()` | `baseline_comparison/moe_comparison.py` | Visualizes supervised learning comparison results | Creates bar charts for RMSE and MAE comparisons, expert usage distributions, and confidence histograms. |
| `visualize_moe_with_baselines()` | `baseline_comparison/moe_comparison.py` | Standalone utility for MoE vs baseline visualization | Allows visualization generation outside the comparison class framework. |

### MoE Visualization Types

#### Expert Contribution Visualizations

| Visualization Type | Description | Key Insights |
|-------------------|-------------|-------------|
| Expert Usage Pie Chart | Displays the proportion of samples where each expert dominates | Reveals which experts are most frequently selected by the gating network, indicating their importance in the ensemble |
| Expert Contribution Heatmap | Shows the contribution weights of each expert across different input regions | Identifies regions of input space where specific experts specialize |
| Expert Selection Frequency | Bar chart of how often each expert is selected | Helps identify underutilized experts that might be candidates for replacement or retraining |
| Expert Error Distribution | Box plots of error distributions for each expert | Compares the error characteristics across different experts |

#### Confidence Visualizations

| Visualization Type | Description | Key Insights |
|-------------------|-------------|-------------|
| Confidence Histogram | Distribution of confidence scores across predictions | Shows overall model confidence patterns |
| Confidence Calibration Plot | Expected vs actual accuracy across confidence bins | Reveals whether the model's confidence aligns with its actual performance |
| Confidence-Error Scatter Plot | Plots confidence scores against prediction errors | Shows relationship between confidence and accuracy |
| Expected Calibration Error (ECE) | Bar chart showing calibration error across bins | Quantifies overall calibration quality |

#### Comparison Visualizations

| Visualization Type | Description | Key Insights |
|-------------------|-------------|-------------|
| Performance Comparison Bar Chart | Side-by-side comparison of key metrics (RMSE, MAE) | Directly compares MoE against baseline approaches |
| Radar Chart | Multi-dimensional performance visualization | Shows relative strengths across multiple metrics simultaneously |
| Convergence Curve | Evolution of error metrics over iterations | Compares convergence speed and final performance |
| Selection Frequency Plot | Visualizes algorithm selection patterns | Reveals which algorithms are preferred by different selectors |

### Using MoE Visualizations

#### Basic Usage

```python
# Using MoEBaselineComparison
comparison = MoEBaselineComparison(
    simple_baseline=simple_selector,
    meta_learner=meta_selector,
    enhanced_meta=enhanced_selector,
    satzilla_selector=satzilla_selector,
    moe_adapter=moe_adapter,
    output_dir="results/my_comparison"
)

# Run comparison
results = comparison.run_supervised_comparison(X_train, y_train, X_test, y_test)

# Generate all visualizations
visualization_paths = comparison.visualize_results()

# For supervised comparison visualizations only
metrics_df = comparison.visualize_supervised_comparison(results)
```

#### Standalone Visualization

```python
# Using standalone visualization utility
from baseline_comparison.moe_comparison import visualize_moe_with_baselines

# Get MoE metrics
moe_metrics = moe_adapter.get_metrics()

# Define baseline metrics for comparison
baseline_metrics = {
    "meta": {"rmse": 0.45, "mae": 0.32},
    "satzilla": {"rmse": 0.42, "mae": 0.30}
}

# Generate comparison visualizations
paths = visualize_moe_with_baselines(
    moe_metrics=moe_metrics,
    baseline_metrics=baseline_metrics,
    output_dir="results/visualizations"
)
```

#### Customizing Visualizations

All visualization methods accept parameters to customize the output:

- **output_dir**: Directory where visualization files will be saved
- **prefix**: String prefix added to output filenames
- **save**: Boolean flag to control whether visualizations are saved to disk
- **name**: Base name for the visualization files

For example:

```python
# Customizing visualization output
comparison.visualize_supervised_comparison(
    results,
    output_dir="custom/path/to/visualizations",
    prefix="patient_123_analysis"
)
```

Specific visualization types can be customized through the metrics calculator:

```python
# Creating a metrics calculator with custom settings
from baseline_comparison.moe_metrics import MoEMetricsCalculator

metrics_calculator = MoEMetricsCalculator(
    output_dir="results/custom_visualizations",
    confidence_bins=20,  # Increase granularity of confidence visualizations
    figsize=(12, 8)      # Larger figure size for all plots
)

# Generate customized visualizations
paths = metrics_calculator.visualize_metrics(moe_metrics, name="custom_moe_analysis")
```

## Critical Path 4 Components - Testing and Validation Framework

This document details the new components implemented as part of Critical Path 4, focusing on testing, validation, and benchmarking infrastructure for the MoE framework.

### Time Series Validation Framework

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `TimeSeriesValidator` | `moe_framework/validation/time_series_validation.py` | Specialized cross-validation for time-series medical data | Provides methods for temporal validation with awareness of patient identity. Implements multiple splitting strategies specifically designed for medical time-series data. |
| `RollingWindowSplit` | `moe_framework/validation/time_series_validation.py` | Cross-validation strategy with fixed-size moving windows | Maintains temporal ordering by creating windows that move forward in time. Training set size remains consistent while incorporating new data points as they become available. |
| `ExpandingWindowSplit` | `moe_framework/validation/time_series_validation.py` | Strategy with progressively larger training windows | Ensures all historical data is utilized while maintaining temporal integrity. Particularly useful for long-term studies where historical patterns remain relevant. |
| `PatientAwareSplit` | `moe_framework/validation/time_series_validation.py` | Patient-aware data splitting strategy | Prevents data leakage between patients by ensuring all data for a specific patient is either in the training or testing set, but never split across both. |

### Performance Benchmarking Framework

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `BenchmarkRunner` | `moe_framework/benchmark/performance_benchmarks.py` | Core framework for benchmarking pipeline configurations | Provides a unified interface for running comparative benchmarks of different MoE configurations, measuring execution time, memory usage, and prediction accuracy. |
| `benchmark_pipeline_configuration()` | `moe_framework/benchmark/performance_benchmarks.py` | Method for benchmarking specific pipeline configurations | Evaluates end-to-end pipeline with different parameter settings, capturing performance metrics throughout the execution. |
| `benchmark_integration_strategies()` | `moe_framework/benchmark/performance_benchmarks.py` | Method for comparing integration approaches | Tests different integration strategies (weighted average, adaptive, context-aware) while keeping other components fixed. |
| `benchmark_gating_networks()` | `moe_framework/benchmark/performance_benchmarks.py` | Method for evaluating gating alternatives | Compares various gating network implementations to identify optimal weighting strategies for specific use cases. |
| `run_scaling_benchmark()` | `moe_framework/benchmark/performance_benchmarks.py` | Method for scaling analysis | Evaluates how different configurations handle progressively larger datasets, identifying potential bottlenecks. |
| `run_standard_benchmark_suite()` | `moe_framework/benchmark/performance_benchmarks.py` | Pre-configured benchmark suite | Runs a standardized set of benchmarks to evaluate overall framework performance across common use cases. |

### Testing Framework for Adaptive Integration

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `TestWeightedAverageIntegration` | `moe_tests/test_adaptive_integration_strategies.py` | Tests for basic weighted integration | Validates correct functioning of weighted average integration across different weight configurations and input shapes. |
| `TestAdaptiveIntegration` | `moe_tests/test_adaptive_integration_strategies.py` | Tests for dynamic integration strategies | Tests confidence-based and quality-aware integration methods that adapt weights based on runtime metrics. |
| `TestAdvancedFusionStrategies` | `moe_tests/test_adaptive_integration_strategies.py` | Tests for complex fusion approaches | Validates more sophisticated fusion methods including uncertainty-weighted integration and dynamic strategy switching. |
| `TestEdgeCases` | `moe_tests/test_adaptive_integration_strategies.py` | Edge case testing | Ensures proper handling of missing experts, empty predictions, and other boundary conditions. |

### Performance Benchmarking Tests

| Component | File Path | Purpose | Implementation Details |
|-----------|-----------|---------|------------------------|
| `TestBenchmarkRunner` | `moe_tests/test_performance_benchmarks.py` | Core benchmark runner tests | Validates the functionality of the benchmarking framework itself. |
| `TestConfigurationBenchmarks` | `moe_tests/test_performance_benchmarks.py` | Pipeline configuration benchmark tests | Tests for comparing different pipeline configurations and parameter settings. |
| `TestIntegrationStrategyBenchmarks` | `moe_tests/test_performance_benchmarks.py` | Integration strategy benchmark tests | Ensures accurate measurement of integration strategy performance differences. |
| `TestResultComparison` | `moe_tests/test_performance_benchmarks.py` | Benchmark result comparison tests | Validates the methods for comparing and analyzing benchmark results. |

## Integration with Existing Architecture

The newly implemented components integrate with the existing architecture in the following ways:

1. **Time Series Validation Framework**:
   - Used by model evaluation components to ensure proper validation of temporal predictions
   - Integrated with expert model training to prevent data leakage
   - Compatible with sklearn's cross-validation interface for easy adoption

2. **Performance Benchmarking Framework**:
   - Consumes MoEPipeline configurations to evaluate different setups
   - Produces standardized metrics for consistent comparison
   - Integrated with visualization components for performance reporting
   - Supports both local benchmarking and distributed execution

3. **Adaptive Integration Testing**:
   - Validates the integration layer components that combine expert predictions
   - Ensures robustness across different data distributions and expert configurations
   - Provides coverage for edge cases and error handling

## Design Considerations

The implementation of these components followed these key design principles:

1. **Separation of concerns**: Each component has a clear, focused responsibility
2. **Extensibility**: All components support extension through inheritance or composition
3. **Compatibility**: New functionality maintains backward compatibility with existing interfaces
4. **Configurability**: Components can be configured through the standard configuration system
5. **Testability**: All components are designed for comprehensive testing
6. **Performance awareness**: Implementation considers computation and memory efficiency

## Future Enhancements

Planned future enhancements for these components include:

1. **Distributed benchmarking**: Support for running benchmarks across multiple machines
2. **Advanced validation strategies**: More specialized cross-validation approaches for specific clinical scenarios
3. **Automated configuration optimization**: Using benchmarking results to automatically tune configurations
4. **Expanded test coverage**: Additional test cases for more complex scenarios and edge cases



## Component Relationships

### Data Flow

1. **Data Ingestion**:
   ```
   FileDataConnector → DataQualityAssessment → Meta-Optimizer
   ```

2. **Preprocessing Flow**:
   ```
   Raw Data → PreprocessingPipeline → [Domain-Specific Operations] → Processed Data
   ```

3. **Expert Model Training**:
   ```
   MoETrainingManager → MetaOptimizer → [DE, ES, ACO, GWO] → Expert Models
   ```

4. **Prediction Flow**:
   ```
   Input Data → Expert Models → GatingNetwork → Meta_Learner → Final Prediction
   ```

5. **Validation Flow**:
   ```
   TimeSeriesValidator → [Patient/Rolling/Expanding Splits] → BenchmarkRunner → MoEMetricsCalculator → Metrics
   ```

6. **Drift Detection Flow**:
   ```
   New Data → DriftDetector → SelectiveExpertRetraining → Updated Models
   ```

7. **Baseline Comparison Flow**:
   ```
   Test Data → MoEBaselineAdapter → MoEBaselineComparison → Comparison Metrics → Visualizations
   ```

### Dependency Graph

```
BaseDataConnector
└── FileDataConnector
    ├── DataQualityAssessment
    │   └── MetaOptimizer
    │       ├── BaseOptimizer
    │       │   ├── DifferentialEvolution
    │       │   ├── EvolutionStrategy
    │       │   ├── AntColonyOptimization
    │       │   └── GreyWolfOptimizer
    │       └── OptimizerFactory
    └── UploadManager

PreprocessingOperation
├── MedicationNormalizer
├── SymptomExtractor
├── TemporalPatternExtractor
├── PhysiologicalSignalProcessor
├── ComorbidityAnalyzer
├── EnvironmentalTriggerAnalyzer
└── AdvancedFeatureEngineer

PreprocessingPipeline
└── [All PreprocessingOperations]

UnifiedReportGenerator
├── BenchmarkPerformanceReport
├── ClinicalMetricsReport
├── CounterfactualExplanationsReport
├── DriftPerformanceReport
├── EnhancedDataReport
├── EvolutionaryPerformanceReport
├── ExpertPerformanceReport
├── ModelEvaluationReport
├── MoEInteractiveReport
├── PersonalizationReport
└── RealDataValidationReport

BaseExpert (Planned)
├── PhysiologicalExpert (Planned)
├── EnvironmentalExpert (Planned)
├── BehavioralExpert (Planned)
└── MedicationHistoryExpert (Planned)

GatingNetwork (Planned)
└── Meta_Learner

DriftDetector
└── SelectiveExpertRetraining

MoETrainingManager (Planned)
└── MetaOptimizer

ExecutionPipeline
├── UploadManager
│   ├── FileDataConnector
│   └── DataQualityAssessment
├── PreprocessingPipeline
│   └── [All PreprocessingOperations]
├── MoETrainingManager (Planned)
├── Expert Models (Planned)
├── GatingNetwork (Planned)
├── ValidationFramework
└── UnifiedReportGenerator
    └── ReportGeneratorComponent
```

## Implementation Status

| Component | Status | Phase | Notes |
|-----------|--------|-------|-------|
| `BaseDataConnector` | Implemented | Phase 1 | Core interface for data connectors |
| `FileDataConnector` | Implemented | Phase 1 | Supports CSV, Excel, JSON, and Parquet formats |
| `DataQualityAssessment` | Implemented | Phase 1 | Provides quality metrics for EC algorithm selection |
| `UploadManager` | Implemented | Phase 1 | Manages file uploads and validation |
| `ExecutionPipeline` | Implemented | Phase 1 | Provides one-click execution workflow |
| `phase1_demo.py` | Implemented | Phase 1 | Demonstration script for all Phase 1 components |
| `PreprocessingOperation` | Implemented | Phase 2 | Base class for all preprocessing operations |
| `PreprocessingPipeline` | Implemented | Phase 2 | Chains multiple preprocessing operations |
| `MedicationNormalizer` | Implemented | Phase 2 | Normalizes medication data |
| `SymptomExtractor` | Implemented | Phase 2 | Extracts symptoms from text data |
| `TemporalPatternExtractor` | Implemented | Phase 2 | Extracts temporal features |
| `PhysiologicalSignalProcessor` | Implemented | Phase 2 | Processes physiological signals |
| `ComorbidityAnalyzer` | Implemented | Phase 2 | Analyzes comorbid conditions |
| `EnvironmentalTriggerAnalyzer` | Implemented | Phase 2 | Analyzes environmental triggers |
| `AdvancedFeatureEngineer` | Implemented | Phase 2 | Implements advanced feature engineering |
| `DriftDetector` | Implemented | Phase 2 | Detects data and concept drift |
| `BaseOptimizer` | Implemented | Phase 2 | Base class for all optimizers |
| `DifferentialEvolution` | Implemented | Phase 2 | DE algorithm implementation |
| `EvolutionStrategy` | Implemented | Phase 2 | ES algorithm implementation |
| `AntColonyOptimization` | Implemented | Phase 2 | ACO algorithm implementation |
| `GreyWolfOptimizer` | Implemented | Phase 2 | GWO algorithm implementation |
| `OptimizerFactory` | Implemented | Phase 2 | Factory for creating optimizers |
| `MetaOptimizer` | Implemented | Phase 2 | Selects the best EC algorithm |
| `UnifiedReportGenerator` | Implemented | Phase 2 | Centralized reporting system |eport generation for CLI and dashboard |
| `ReportGeneratorComponent` | Implemented | Phase 2 | Streamlit component for generating reports |
| `MoEMetricsCalculator` | Implemented | Phase 3 | Comprehensive MoE metrics calculation |
| `MoEBaselineAdapter` | Implemented | Phase 3 | MoE integration with baseline comparison framework |
| `MoEBaselineComparison` | Implemented | Phase 3 | Extended baseline comparison for MoE |
| `MoEComparisonCommand` | Implemented | Phase 3 | CLI commands for MoE comparison |
| `TimeSeriesValidator` | Implemented | Phase 3 | Time-series cross-validation strategies |
| `BenchmarkRunner` | Implemented | Phase 3 | Performance benchmarking framework |
| Other components | Planned | Future phases | To be implemented according to the phased approach |


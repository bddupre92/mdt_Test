# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-03-05

### Fixed
- Fixed model loading issue in MigrainePredictorV2 where tuple return type was incorrectly handled as a dictionary
- Corrected the implementation of the load_model method to properly unpack the model and metadata
- Added missing required features (heart_rate, weight_kg, hydration_ml) to example prediction data
- Ensured proper scaler initialization with dummy sample during model loading
- Fixed the run_migraine_prediction function to load the default model when no model ID is provided

### Improved
- Enhanced example data generation with comprehensive feature coverage
- Added better error handling for derived feature calculations
- Updated test suite to cover all migraine prediction functionality

## [0.1.2] - 2025-03-05

### Added
- New migraine data handling capabilities for handling evolving data schemas
- Advanced `DataHandler` class to manage data imports and derived features
- Enhanced `MigrainePredictorV2` class with support for new columns and missing features
- Command-line interface in main.py for migraine data operations
- Comprehensive unit tests for migraine data handling features
- Example scripts for demonstrating new migraine data functionality
- Detailed documentation in README_MIGRAINE_DATA.md

### Improved
- Schema evolution support for migraine prediction models
- Missing value handling during prediction with default value substitution
- Model metadata to store feature information and default values
- Enhanced command-line interface with migraine data arguments

## [0.1.1] - 2025-03-05

### Fixed
- Fixed critical bug in `DriftDetector.detect_drift` method where it was returning a single float instead of the expected tuple
- Enhanced `Algorithm.update` method to properly handle scalar inputs and different array shapes
- Fixed dimensionality errors during meta-learner predictions with drift detection
- Improved error handling and input validation in the meta-learner component

### Added
- Comprehensive tests for drift detection functionality
- Enhanced logging for better debugging of drift detection issues
- New drift detection documentation guide (docs/drift_detection_guide.md)

### Improved
- Restructured test files to maintain a clean project organization
- Consolidated README files (combined main README.md with README_VISUALIZATION.md and README_EXPLAINABILITY.md)
- Updated documentation with drift detection examples and usage guidelines

## [7.0.0] - 2025-03-04

### Added
- PyPI packaging support for easy installation with pip
- Modern Python packaging with pyproject.toml
- Command-line interface accessible via `meta-optimizer` command
- Quick start example script in examples directory
- Helper script for building and publishing to PyPI
- Added PyTorch as a core dependency for neural network models

### Changed
- Refactored project structure to ensure proper imports from root directory
- Updated CLI and __main__.py to properly import from the root directory
- Improved setup.py with comprehensive dependencies and package metadata
- Restructured the meta_optimizer package for PyPI compatibility

## [6.1.1] - 2025-03-02

### Fixed
- Fixed model factory to properly handle classification tasks:
  - Added task_type parameter to determine model type (classifier vs regressor)
  - Updated ModelFactory to use RandomForestClassifier for classification tasks
  - Fixed predict_proba returning None for regression models
- Enhanced test pipeline to specify classification task type:
  - Added task_type context in meta-optimization
  - Ensured proper model type selection in test_full_pipeline.py

## [6.1.0] - 2025-03-02

### Added
- Real-time visualization for optimization processes with LiveOptimizationMonitor
- Multi-panel visualization showing optimization progress, improvement rate, convergence speed, and statistics
- Command-line interface with --live-viz and --save-plots flags
- Enhanced MetaLearner for migraine prediction with feature importance analysis
- Improved synthetic data generation for migraine temporal patterns

### Changed
- Optimized Differential Evolution to properly respect maximum evaluation limits
- Reduced population size for DE optimizer to ensure evaluation constraints
- Updated OptimizerAnalyzer to accept max_evals parameter
- Enhanced DifferentialEvolutionWrapper with better evaluation counting
- Improved benchmark tests for faster execution and consistent evaluation limits

### Fixed
- Fixed issue with DE optimizer exceeding specified max_evals limit
- Implemented wrapper function to count evaluations and enforce limits
- Adjusted maxiter parameter based on max_evals to prevent excessive evaluations
- Improved test efficiency with smaller population sizes
- Enhanced tracking of actual evaluations used during optimization

## [6.0.1] - 2025-02-25

### Fixed
- Fixed drift detection interval tracking by using absolute sample count
- Improved drift detection sensitivity with better parameter tuning:
  - Increased significance level from 0.01 to 0.05
  - Added more lenient drift conditions for KS statistic and mean shifts
  - Enhanced feature-level drift detection
  - Better trend monitoring and state management
- All drift detection tests now passing, including KS test and multiple drift scenarios

## [6.0.0] - 2025-02-25

### Added
- New FastAPI-based application structure in `app/` directory
- Comprehensive test suite with pytest fixtures and configuration
- Model factory implementation for consistent model creation
- Enhanced drift detection with severity calculation and trend analysis
- Synthetic data generation with concept drift simulation
- Development requirements in `requirements-dev.txt`
- Testing guide documentation
- Alembic database migrations
- Environment configuration example
- New example scripts for meta-optimizer usage

### Changed
- Improved meta-learner parameter ranges and optimization
- Enhanced optimizer history tracking
- Updated model wrapper implementations
- Refactored drift detection logic for better accuracy
- Restructured project layout for better organization
- Updated requirements.txt with new dependencies
- Improved documentation and code comments

### Removed
- Deprecated TensorFlow model implementation
- Old example scripts
- Unused test files

### Fixed
- DataFrame concatenation warnings in optimizer
- Multi-dimensional data handling in drift detection
- Model parameter scaling in meta-learner
- Various bug fixes in optimizers and model training

## [5.0.0] - 2025-02-24

### Added
- Comprehensive analysis framework for optimizer comparison
- LaTeX algorithm documentation with detailed pseudocode
- Statistical significance testing between optimizers
- Performance visualization tools and plots
- Progress tracking with tqdm integration
- Results analysis script with publication-ready outputs

### Enhanced
- MetaOptimizer performance across test functions
- Surrogate optimizer integration
- Problem analysis capabilities
- Optimizer selection strategies
- Convergence tracking and visualization

### Fixed
- Results visualization and data collection
- Statistical analysis methodology
- Performance metrics calculations
- Progress bar integration

## [4.0.7] - 2025-02-24

### Added
- Improved problem analysis with sophisticated feature extraction
- Enhanced sampling strategy for high-dimensional problems
- Weighted historical performance with recency bias
- Dynamic resource allocation based on problem characteristics

### Changed
- Updated optimizer selection strategy with confidence-based weighting
- Improved ruggedness estimation for better landscape analysis
- Enhanced feature correlation tracking for optimizer selection
- Optimized Latin Hypercube sampling with adaptive corner points

### Fixed
- Fixed sampling strategy for high-dimensional problems
- Resolved issue with negative dimensions in sample generation
- Fixed find_similar_problems to support k parameter
- Improved error handling in problem analysis

## [4.0.6] - 2025-02-24

### Added
- Enhanced problem analysis with sophisticated feature extraction:
  - Local and global structure analysis
  - Fitness distance correlation
  - Information content measurement
  - Basin ratio estimation
  - Gradient homogeneity analysis
- Improved sampling strategy with corner points
- Advanced optimizer selection with:
  - Success-gap based adaptation
  - Exponential decay for exploration
  - Problem-specific base rates
  - Weighted historical performance
  - Confidence-based selection

### Changed
- Upgraded gradient estimation to use central differences
- Enhanced parallel execution with dynamic resource allocation
- Improved exploration/exploitation balance
- Updated selection strategy to use complementary optimizer combinations

### Fixed
- Improved boundary exploration with corner point sampling
- Enhanced numerical stability in gradient calculations
- Fixed potential race conditions in parallel execution

## [4.0.5] - 2025-02-24

### Added
- Added verified parallel evaluation capabilities with ThreadPoolExecutor
- Added thread-safe evaluation counting with locks
- Added OptimizationResult dataclass for better result handling
- Added concurrent optimizer execution with configurable parallel runs
- Added evaluation budget management for parallel optimizers

### Changed
- Restructured optimizer execution to support parallel runs
- Improved resource management for concurrent optimization
- Enhanced MetaOptimizer to handle multiple concurrent optimizers
- Updated theoretical comparison script with parallel execution support

### Fixed
- Added thread safety for evaluation counting and history tracking
- Fixed evaluation budget distribution in parallel runs
- Ensured proper cleanup of thread resources

## [4.0.4] - 2025-02-24

### Added
- Implemented adaptive exploration/exploitation balance in MetaOptimizer
- Added SelectionTracker class for recording and analyzing optimizer selections
- Added problem-specific feature correlation analysis
- Implemented confidence-based optimizer selection mechanism
- Added early stopping criteria for efficient optimization
- Added problem type awareness and context-based optimization

### Changed
- Refactored MetaOptimizer to use dynamic exploration rates
- Updated theoretical comparison script to support selection tracking
- Improved optimization history handling with better serialization
- Enhanced problem analysis with feature extraction

### Fixed
- Improved JSON serialization for numpy types in optimization history
- Fixed exploration rate calculation to prevent premature convergence
- Added thread safety for evaluation counting and history tracking

## [4.0.3] - 2025-02-24

### Added
- Improved MetaOptimizer with learning capabilities
- Added OptimizationHistory class for tracking and analyzing optimizer performance
- Added ProblemAnalyzer class for extracting problem features
- Implemented JSON serialization for optimization history
- Added adaptive exploration/exploitation balance in optimizer selection

### Fixed
- Fixed JSON serialization issues with numpy types
- Improved error handling in optimizer selection
- Enhanced stability of meta-optimization process

### Changed
- Refactored MetaOptimizer to use problem features for selection
- Updated theoretical comparison script to use new MetaOptimizer features

## [4.0.2] - 2025-02-24

### Fixed
- Fixed function evaluation counting in MetaOptimizer and SurrogateOptimizer
- Updated history storage format for consistent performance tracking
- Fixed plotting functions to handle dictionary-based history records
- Improved performance comparison analysis and visualization

### Added
- Added actual function evaluation tracking in MetaOptimizer
- Added early stopping criteria for efficient optimization
- Enhanced logging of optimizer selection patterns

### Changed
- Modified theoretical comparison script to use actual function evaluations
- Updated optimizer history format for better analysis
- Standardized performance metrics across all optimizers

## [4.0.1] - 2025-02-24

### Fixed
- Fixed dimension mismatch error in `SurrogateOptimizer` by properly handling observation storage and GP model updates
- Fixed indexing error in `MetaOptimizer`'s optimizer selection strategy
- Improved normalization and error handling in both optimizers

### Improved
- Enhanced `MetaOptimizer`'s performance with better optimizer selection based on history
- Optimized `SurrogateOptimizer`'s GP model configuration for better efficiency
- Added early stopping conditions for faster convergence
- Improved handling of performance history and logging

### Performance
- Reduced memory usage in `SurrogateOptimizer` by limiting GP model size
- Improved convergence speed on both unimodal and multimodal functions
- Enhanced stability of optimization process across all test functions

## [4.0.0] - 2025-02-24

### Added
- Comprehensive theoretical analysis framework
  - Convergence rate analysis for all optimizers
  - Stability analysis for meta-learning decisions
  - Parameter sensitivity analysis
  - Computational complexity analysis
- Enhanced meta-learning optimization framework
  - Improved Bayesian optimization with GP
  - Multi-armed bandit strategy
  - Context-aware algorithm selection
- New unified main.py interface
  - Streamlined API for general use
  - Comprehensive benchmark suite integration
  - Automated results analysis and visualization
- Advanced performance tracking
  - Detailed convergence metrics
  - Selection stability measures
  - Cross-optimizer performance comparisons

### Changed
- Completely restructured main.py for better usability
- Enhanced optimizer implementations with theoretical guarantees
- Improved error handling and logging system
- Updated benchmark suite with more comprehensive test functions

### Fixed
- Array shape inconsistencies in high-dimensional optimization
- Convergence issues in surrogate optimization
- Meta-learner selection stability
- Performance history tracking accuracy

## [3.0.0] - 2025-02-24

### Added
- New modular benchmarking and visualization workflow:
  - `run_benchmarks.py`: Run and save optimization results
  - `visualize_results.py`: Generate visualizations from saved results
  - Test mode (`--test`) for quick validation
- Enhanced visualization suite:
  - Convergence comparison plots with standard deviation bands
  - Performance heatmaps comparing optimizers across functions
  - Parameter adaptation analysis for adaptive optimizers
  - Population diversity tracking and visualization
  - Interactive 3D landscape visualizations
- Optimizer factory for consistent optimizer creation
- Consistent file naming conventions for all outputs
- Support for selective visualization of specific functions/optimizers

### Changed
- Split benchmarking and visualization into separate scripts
- Improved plot formatting and aesthetics
- Enhanced error handling in visualization code
- Updated parameter adaptation plots to show mean trajectories
- Standardized file naming across all visualizations

### Fixed
- Fixed issue with convergence curves of different lengths
- Resolved duplicate file creation with different naming conventions
- Fixed parameter adaptation plots for non-adaptive optimizers

## [2.0.0] - 2025-02-23

### Added
- Adaptive parameter control for all optimizers
- Enhanced diversity management in ACO
- Gaussian sampling for ACO solution generation
- Success rate tracking for parameter adaptation
- Diversity history tracking for all optimizers
- Comprehensive test suite with parallel execution
- Meta-optimizer for algorithm selection

### Fixed
- Evolution Strategy (ES) sigma adaptation
- Grey Wolf Optimizer (GWO) initialization
- Base optimizer state management
- Parameter history tracking
- Meta-optimizer performance tracking

### Changed
- Improved ACO pheromone management
- Enhanced ES mutation and recombination
- Updated GWO parameter adaptation strategy
- Optimized parallel processing in test suite
- Standardized optimizer interfaces

### Performance
- ES achieves ~1e-5 precision on Sphere function
- GWO reaches ~7e-9 precision on Sphere function
- Adaptive DE shows significant improvements
- Meta-optimizer successfully selects best algorithm

## [1.0.0] - 2025-02-23

### Added

#### Optimization Framework
- Implemented base optimizer class with common functionality
- Added four optimization algorithms:
  - Grey Wolf Optimizer (GWO)
  - Differential Evolution (DE)
  - Evolution Strategy (ES)
  - Ant Colony Optimization (ACO)
- Created comprehensive parameter tuning framework
- Implemented parallel processing for parameter tuning

#### Visualization and Analysis
- Created visualization module for optimizer analysis
- Added visualization capabilities:
  - Convergence curves
  - Performance heatmaps
  - Parameter sensitivity analysis
  - 3D fitness landscapes
  - Hyperparameter correlation plots
- Implemented performance metrics tracking and reporting

#### Benchmarking
- Added classical test functions:
  - Sphere function
  - Rastrigin function
  - Rosenbrock function
- Created benchmark runner for comparative analysis
- Implemented test suite for optimizer validation

#### Testing and Documentation
- Added unit tests for all optimizers
- Created example scripts for:
  - Basic optimizer usage
  - Parameter tuning
  - Performance visualization
- Added comprehensive docstrings and type hints

### Performance Highlights
- DE achieved perfect convergence on Rosenbrock function
- GWO showed excellent performance with 10^-9 precision
- ES demonstrated robust exploration capabilities
- ACO provided fast approximate solutions

### Technical Details
- Optimized memory usage in all algorithms
- Added bounds handling and solution clipping
- Implemented parallel processing for performance
- Enhanced numerical stability in fitness calculations

### Dependencies
- Updated core dependencies:
  - NumPy ≥ 1.24.0
  - Pandas ≥ 2.0.0
  - SciPy ≥ 1.10.0
  - Matplotlib ≥ 3.7.0
  - Seaborn ≥ 0.12.0
  - Plotly ≥ 5.13.0

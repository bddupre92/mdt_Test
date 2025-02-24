# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

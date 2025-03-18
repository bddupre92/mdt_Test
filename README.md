# Migraine Prediction Optimizer Framework

A comprehensive framework for comparing optimization algorithms and benchmarking the meta-optimizer for migraine prediction models.

## Codebase Visualization

![Visualization of the codebase](./diagram.svg)

*This visualization is automatically generated and updated when the code changes.*

## Features

- **Benchmark System**: Test and compare different optimization algorithms on standard benchmark functions
- **Optimizer Adapters**: Unified interface for various optimization algorithms (DE, ES, ACO, GWO, Meta-Optimizer)
- **API Server**: REST API for running benchmarks and accessing results
- **Dashboard**: Interactive visualization of benchmark results and optimizer comparisons
- **Performance Analysis**: Tools for analyzing performance metrics and optimizer selection patterns

## Components

The framework consists of two main components:

1. **API Server**: FastAPI-based server for running benchmarks and accessing results
2. **Dashboard**: Streamlit-based visualization dashboard for exploring benchmark results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/migraine-prediction-optimizer.git
cd migraine-prediction-optimizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Framework

You can run the full framework (both API server and dashboard) with:

```bash
python app/run.py
```

This will start:
- API server on http://localhost:8000
- Dashboard on http://localhost:8501

### Running Individual Components

To run only the API server:

```bash
python app/run.py --mode api
```

To run only the dashboard:

```bash
python app/run.py --mode dashboard
```

### Command Line Options

```
usage: run.py [-h] [--mode {api,dashboard,both}] [--api_port API_PORT]
              [--dashboard_port DASHBOARD_PORT] [--results_dir RESULTS_DIR]
              [--debug]

Launch Migraine Prediction Optimizer Framework components

optional arguments:
  -h, --help            show this help message and exit
  --mode {api,dashboard,both}
                        Component to launch (api, dashboard, or both)
  --api_port API_PORT   Port for the API server
  --dashboard_port DASHBOARD_PORT
                        Port for the dashboard
  --results_dir RESULTS_DIR
                        Directory for benchmark results
  --debug               Run in debug mode with more verbose output
```

## API Documentation

When the API server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Benchmark System

The benchmark system allows you to:

1. **Run individual benchmarks** for specific optimizers
2. **Compare multiple optimizers** on the same benchmark functions
3. **Analyze the meta-optimizer** performance and selection patterns

### Available Benchmark Functions

- Sphere
- Rosenbrock
- Rastrigin
- Ackley
- Griewank
- Schwefel
- Dynamic variants with different drift types

### Available Optimizers

- Differential Evolution (DE)
- Evolution Strategy (ES)
- Ant Colony Optimization (ACO)
- Grey Wolf Optimizer (GWO)
- Meta-Optimizer

## Dashboard

The dashboard provides interactive visualizations for:

1. **Overview**: Summary of available benchmark results
2. **Benchmark Comparison**: Detailed comparison of optimizer performance
3. **Meta-Optimizer Analysis**: Analysis of meta-optimizer selection patterns and performance

## Development

### Project Structure

```
app/
├── api/                # API endpoints
├── core/               # Core components
│   ├── benchmark_repository.py  # Benchmark functions
│   ├── benchmark_service.py     # Benchmark execution service
│   └── optimizer_adapter.py     # Adapters for optimizers
├── ui/                 # Dashboard components
│   └── benchmark_dashboard.py   # Streamlit dashboard
├── visualization/      # Visualization utilities
│   └── benchmark_visualizer.py  # Plotting functions
├── __init__.py
├── main.py             # FastAPI application
├── run.py              # Unified launcher script
└── run_dashboard.py    # Dashboard launcher script
```

### Adding New Benchmark Functions

To add a new benchmark function, extend the `BenchmarkFunction` class in `app/core/benchmark_repository.py`.

### Adding New Optimizers

To add a new optimizer, create a new adapter in `app/core/optimizer_adapter.py` that extends the `OptimizerAdapter` class.

## License

[MIT License](LICENSE) 
# Optimization Framework Documentation

This directory contains comprehensive documentation for the optimization framework, including usage guides, component descriptions, and examples.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Command-Line Interface](#command-line-interface)
4. [Framework Components](#framework-components)
5. [Explainability Features](#explainability-features)
6. [Examples](#examples)

## Overview

The optimization framework provides a comprehensive suite of tools for:

- Running various optimization algorithms on test functions and real-world problems
- Evaluating and comparing optimizer performance
- Meta-learning to automatically select the best optimizer for a given problem
- Detecting and adapting to concept drift in data
- Explaining optimizer behavior and model predictions

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

Run a simple optimization:
```bash
python main.py --optimize
```

Run explainability analysis:
```bash
python main.py --explain --explainer shap --explain-plots
```

## Documentation Files

- [Command Line Interface](command_line_interface.md): Detailed documentation of all command-line arguments
- [Optimizer Explainability](optimizer_explainability.md): Guide to the optimizer explainability features
- [Model Explainability](model_explainability.md): Guide to the model explainability features
- [Framework Architecture](framework_architecture.md): Overview of the framework architecture and component relationships
- [Testing Guide](testing_guide.md): Guide to testing the framework
- [Examples](examples.md): Example usage scenarios and code snippets

# Modular Setup for Meta Optimizer Framework

This document describes the modular setup for the Meta Optimizer framework, which provides a more organized and extensible way to interact with the framework through a command-line interface.

## Overview

The modular setup uses a command-line interface (CLI) architecture that:

1. Separates concerns into distinct modules (argument parsing, command implementation, etc.)
2. Makes it easy to add new commands without modifying existing code
3. Provides a consistent interface for all framework functionality

## Directory Structure

```
.
├── main_v2.py               # Main entry point for modular CLI
├── cli/                     # CLI module
│   ├── __init__.py          # Module initialization
│   ├── main.py              # Main CLI logic
│   ├── argument_parser.py   # Argument parsing
│   └── commands.py          # Command implementations
├── baseline_comparison/     # Baseline comparison module
│   ├── ...
└── ...
```

## Available Commands

### Baseline Comparison

Compare the Meta Optimizer against baseline algorithm selection methods.

```bash
./main_v2.py baseline_comparison [OPTIONS]
```

Options:
- `--dimensions`, `-d`: Number of dimensions for benchmark functions (default: 2)
- `--max-evaluations`, `-e`: Maximum number of function evaluations per algorithm (default: 1000)
- `--num-trials`, `-t`: Number of trials to run for statistical significance (default: 3)
- `--functions`, `-f`: Specific benchmark functions to use (default: sphere, rosenbrock)
- `--all-functions`: Use all available benchmark functions
- `--output-dir`, `-o`: Output directory for results (default: results/baseline_comparison)
- `--timestamp-dir`: Create a timestamped subdirectory for results (default: True)
- `--no-visualizations`: Disable visualization generation
- `--quiet`, `-q`: Suppress progress information

Examples:

```bash
# Run with default settings
./main_v2.py baseline_comparison

# Run with 5D problems and more trials
./main_v2.py baseline_comparison --dimensions 5 --num-trials 10

# Run with specific functions
./main_v2.py baseline_comparison --functions sphere ackley rastrigin

# Run with all available functions
./main_v2.py baseline_comparison --all-functions
```

## Convenience Scripts

For ease of use, the following scripts are provided:

- `run_modular_baseline_comparison.sh`: Run the baseline comparison through the modular setup
  ```bash
  ./run_modular_baseline_comparison.sh [OPTIONS]
  ```

## Adding New Commands

To add a new command to the modular setup:

1. Add a new command class in `cli/commands.py`
2. Add the command to `COMMAND_MAP` in `cli/commands.py`
3. Add a new parser for the command in `cli/argument_parser.py`

Example:

```python
# In cli/commands.py
class NewCommand(Command):
    def execute(self) -> int:
        # Command implementation
        return 0

COMMAND_MAP = {
    "baseline_comparison": BaselineComparisonCommand,
    "new_command": NewCommand,
}

# In cli/argument_parser.py
new_command_parser = subparsers.add_parser(
    "new_command",
    help="Description of new command"
)
# Add arguments for the new command
```

## Integration with Meta Optimizer

The modular setup is designed to integrate seamlessly with the existing Meta Optimizer framework. Commands can access all functionality from the framework as needed.

## Troubleshooting

If you encounter issues with the modular setup:

1. Ensure your Python path includes the project root directory
2. Check that all required modules are installed
3. Verify that file permissions are set correctly
4. Check the CLI argument syntax

For more specific help, run:

```bash
./main_v2.py <command> --help
``` 
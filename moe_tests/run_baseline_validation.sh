#!/bin/bash
# Run the baseline validation script with default parameters
# This script provides a convenient way to run the baseline algorithm comparison validation

# Define default parameters
DIMENSIONS=10
MAX_EVALUATIONS=10000
NUM_TRIALS=30
OUTPUT_DIR=""
SELECTOR_PATH=""
FUNCTIONS_ARG=""
ALL_FUNCTIONS=false
NO_VISUALIZATIONS=false
TIMESTAMP_DIR=true
VERBOSITY=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dimensions|-d)
      DIMENSIONS="$2"
      shift 2
      ;;
    --max-evaluations|-e)
      MAX_EVALUATIONS="$2"
      shift 2
      ;;
    --num-trials|-t)
      NUM_TRIALS="$2"
      shift 2
      ;;
    --output-dir|-o)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --selector-path|-s)
      SELECTOR_PATH="$2"
      shift 2
      ;;
    --functions|-f)
      # Collect all function arguments
      FUNCTIONS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        FUNCTIONS+=("$1")
        shift
      done
      FUNCTIONS_ARG="--functions ${FUNCTIONS[@]}"
      # If we've stopped because of -v, process it now
      if [[ "$1" == "-v" || "$1" == "--verbose" ]]; then
        VERBOSITY="--verbose"
        shift
      fi
      ;;
    --all-functions)
      ALL_FUNCTIONS=true
      shift
      ;;
    --no-visualizations)
      NO_VISUALIZATIONS=true
      shift
      ;;
    --no-timestamp-dir)
      TIMESTAMP_DIR=false
      shift
      ;;
    -v|--verbose)
      VERBOSITY="--verbose"
      shift
      ;;
    -vv|--very-verbose)
      VERBOSITY="--verbose --verbose"
      shift
      ;;
    -q|--quiet)
      VERBOSITY="--quiet"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Build command arguments
CMD_ARGS="--dimensions $DIMENSIONS --max-evaluations $MAX_EVALUATIONS --num-trials $NUM_TRIALS"

if [ -n "$OUTPUT_DIR" ]; then
  CMD_ARGS="$CMD_ARGS --output-dir $OUTPUT_DIR"
else
  OUTPUT_DIR="results/baseline_validation/$(date +%Y%m%d_%H%M%S)"
  CMD_ARGS="$CMD_ARGS --output-dir $OUTPUT_DIR"
fi

if [ -n "$SELECTOR_PATH" ]; then
  CMD_ARGS="$CMD_ARGS --selector-path $SELECTOR_PATH"
fi

if [ "$ALL_FUNCTIONS" = true ]; then
  CMD_ARGS="$CMD_ARGS --all-functions"
elif [ -n "$FUNCTIONS_ARG" ]; then
  CMD_ARGS="$CMD_ARGS $FUNCTIONS_ARG"
fi

if [ "$NO_VISUALIZATIONS" = true ]; then
  CMD_ARGS="$CMD_ARGS --no-visualizations"
fi

if [ "$TIMESTAMP_DIR" = true ]; then
  CMD_ARGS="$CMD_ARGS --timestamp-dir"
fi

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Print the parameters
echo "Running baseline validation with the following parameters:"
echo "Dimensions: $DIMENSIONS"
echo "Max Evaluations: $MAX_EVALUATIONS"
echo "Number of Trials: $NUM_TRIALS"
echo "Output Directory: $OUTPUT_DIR"
echo "Selector Path: ${SELECTOR_PATH:-None}"
echo "Functions: ${FUNCTIONS_ARG:-None}"
echo "All Functions: ${ALL_FUNCTIONS}"
echo "No Visualizations: ${NO_VISUALIZATIONS}"
echo "Timestamp Directory: ${TIMESTAMP_DIR:+--timestamp-dir}"
echo "Verbosity: ${VERBOSITY}"

# Construct and run the command
CMD="python main_v2.py baseline_comparison $CMD_ARGS $VERBOSITY"
echo ""
echo "Command: $CMD"
echo ""
eval "$CMD"

echo ""
echo "Validation complete. Results saved to $OUTPUT_DIR" 
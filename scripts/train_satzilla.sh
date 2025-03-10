#!/bin/bash
# Script to train the SATzilla-inspired selector

# Default values
DIMENSIONS=2
MAX_EVALUATIONS=1000
NUM_PROBLEMS=20
FUNCTIONS="sphere rosenbrock rastrigin ackley griewank"
OUTPUT_DIR="results/satzilla_training"
VISUALIZE_FEATURES=""
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Display help
function show_help {
    echo "Usage: ./train_satzilla.sh [OPTIONS]"
    echo ""
    echo "Train the SATzilla-inspired algorithm selector."
    echo ""
    echo "Options:"
    echo "  -d, --dimensions DIMS       Number of dimensions for benchmark functions (default: 2)"
    echo "  -e, --max-evaluations EVALS Maximum number of function evaluations per algorithm (default: 1000)"
    echo "  -p, --num-problems NUM      Number of training problems to generate (default: 20)"
    echo "  -f, --functions LIST        Space-separated list of benchmark functions (default: sphere rosenbrock rastrigin ackley griewank)"
    echo "                              Available functions: sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy"
    echo "  -a, --all-functions         Use all available benchmark functions"
    echo "  -o, --output-dir DIR        Output directory for training results (default: results/satzilla_training)"
    echo "  -s, --seed SEED             Random seed for reproducibility (default: 42)"
    echo "  -v, --visualize-features    Generate feature importance visualizations"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  ./train_satzilla.sh -d 5 -p 30 -f \"sphere rastrigin ackley\""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dimensions)
            DIMENSIONS="$2"
            shift 2
            ;;
        -e|--max-evaluations)
            MAX_EVALUATIONS="$2"
            shift 2
            ;;
        -p|--num-problems)
            NUM_PROBLEMS="$2"
            shift 2
            ;;
        -f|--functions)
            # Parse quoted or space-separated list of functions
            shift
            FUNCTIONS=""
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                FUNCTIONS="$FUNCTIONS $1"
                shift
            done
            FUNCTIONS="${FUNCTIONS# }" # Remove leading space
            ;;
        -a|--all-functions)
            ALL_FUNCTIONS="--all-functions"
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -v|--visualize-features)
            VISUALIZE_FEATURES="--visualize-features"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if seed is set
if [ -z "$SEED" ]; then
    SEED="42"
    echo "Using default random seed: $SEED"
fi

# Check if all functions flag is set
if [ -n "$ALL_FUNCTIONS" ]; then
    FUNCTIONS_CMD="$ALL_FUNCTIONS"
    echo "Using all available benchmark functions"
else
    # Convert function list to command-line arguments
    FUNCTIONS_CMD="--functions ${FUNCTIONS// / }"
    echo "Using functions: $FUNCTIONS"
fi

# Create directory structure
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Run training through the modular interface
echo "Starting SATzilla training..."
echo "Dimensions: $DIMENSIONS"
echo "Max Evaluations: $MAX_EVALUATIONS"
echo "Num Problems: $NUM_PROBLEMS"
echo "Output Directory: $OUTPUT_DIR"
echo "Random Seed: $SEED"

# Create log file
LOG_FILE="$OUTPUT_DIR/logs/training_$TIMESTAMP.log"

# Run training
python main_v2.py train_satzilla \
    --dimensions "$DIMENSIONS" \
    --max-evaluations "$MAX_EVALUATIONS" \
    --num-problems "$NUM_PROBLEMS" \
    $FUNCTIONS_CMD \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    $VISUALIZE_FEATURES \
    --timestamp-dir \
    2>&1 | tee "$LOG_FILE"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo -e "\nSATzilla training completed successfully!"
    echo "Log file: $LOG_FILE"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo -e "\nSATzilla training failed. Check log file for details: $LOG_FILE"
    exit 1
fi 
#!/bin/bash
# Script to run the universal data adapter with various options

# Set the base directory to the project root
BASE_DIR=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")
cd $BASE_DIR/..

# Create necessary directories
mkdir -p data
mkdir -p models

# Color codes for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage information
function show_usage {
    echo -e "${BLUE}Universal Data Adapter CLI Script${NC}"
    echo "This script helps run the universal data adapter with different options."
    echo
    echo "Usage:"
    echo "  ./run_universal_data.sh [option]"
    echo
    echo "Options:"
    echo "  generate     Generate synthetic data and train a model"
    echo "  process      Process an existing dataset with auto-feature selection"
    echo "  meta         Use meta-optimization for feature selection"
    echo "  explain      Apply explainability analysis to a trained model"
    echo "  full         Run a complete pipeline with all features enabled"
    echo "  help         Show this help message"
    echo
}

# Function to run with synthetic data generation
function run_synthetic {
    echo -e "${GREEN}Running Universal Data Adapter with Synthetic Data${NC}"
    python main.py \
        --universal-data \
        --generate-synthetic \
        --synthetic-patients 100 \
        --synthetic-days 120 \
        --synthetic-female-pct 0.6 \
        --synthetic-missing-rate 0.1 \
        --train-model \
        --model-name "synthetic_universal_model" \
        --model-description "Model trained with synthetic data via universal adapter" \
        --evaluate-model \
        --save-synthetic \
        --summary \
        --verbose
    
    echo -e "${GREEN}Synthetic data generation and model training complete${NC}"
}

# Function to process an existing dataset
function process_existing {
    if [ -z "$1" ]; then
        echo -e "${YELLOW}No dataset specified, using default path${NC}"
        DATA_PATH="data/migraine_data.csv"
    else
        DATA_PATH=$1
    fi

    echo -e "${GREEN}Processing existing dataset: $DATA_PATH${NC}"
    python main.py \
        --universal-data \
        --data-path $DATA_PATH \
        --file-format csv \
        --train-model \
        --model-name "processed_universal_model" \
        --evaluate-model \
        --save-processed-data \
        --summary \
        --verbose
    
    echo -e "${GREEN}Dataset processing complete${NC}"
}

# Function to use meta-optimization for feature selection
function run_meta_optimization {
    if [ -z "$1" ]; then
        echo -e "${YELLOW}No dataset specified, generating synthetic data${NC}"
        
        python main.py \
            --universal-data \
            --generate-synthetic \
            --synthetic-patients 80 \
            --synthetic-days 100 \
            --use-meta-feature-selection \
            --method de \
            --surrogate rf \
            --max-features 15 \
            --train-model \
            --model-name "meta_optimized_model" \
            --evaluate-model \
            --summary \
            --verbose
    else
        echo -e "${GREEN}Using meta-optimization on dataset: $1${NC}"
        
        python main.py \
            --universal-data \
            --data-path $1 \
            --file-format csv \
            --use-meta-feature-selection \
            --method de \
            --surrogate rf \
            --max-features 15 \
            --train-model \
            --model-name "meta_optimized_model" \
            --evaluate-model \
            --summary \
            --verbose
    fi
    
    echo -e "${GREEN}Meta-optimization complete${NC}"
}

# Function to run explainability analysis
function run_explainability {
    if [ -z "$1" ]; then
        MODEL_ID="latest"
        echo -e "${YELLOW}No model ID specified, using latest model${NC}"
    else
        MODEL_ID=$1
        echo -e "${GREEN}Using model ID: $MODEL_ID${NC}"
    fi
    
    echo -e "${GREEN}Running explainability analysis${NC}"
    python main.py \
        --explain \
        --model-id $MODEL_ID \
        --explainer shap \
        --explain-plots \
        --explain-plot-types summary bar beeswarm waterfall \
        --explain-samples 50 \
        --summary \
        --verbose
    
    echo -e "${GREEN}Explainability analysis complete${NC}"
}

# Function to run a complete pipeline
function run_full_pipeline {
    echo -e "${GREEN}Running complete universal data pipeline${NC}"
    
    # Step 1: Generate synthetic data
    echo -e "${BLUE}Step 1: Generating synthetic data${NC}"
    python main.py \
        --universal-data \
        --generate-synthetic \
        --synthetic-patients 100 \
        --synthetic-days 180 \
        --synthetic-female-pct 0.6 \
        --synthetic-missing-rate 0.1 \
        --synthetic-include-severity \
        --save-synthetic \
        --summary
    
    # Step 2: Use meta-optimization for feature selection
    echo -e "${BLUE}Step 2: Applying meta-optimization for feature selection${NC}"
    python main.py \
        --universal-data \
        --data-path data/synthetic_migraine_data.csv \
        --use-meta-feature-selection \
        --method de \
        --surrogate rf \
        --max-features 15 \
        --train-model \
        --model-name "full_pipeline_model" \
        --model-description "Model trained with complete universal data pipeline" \
        --make-default \
        --evaluate-model \
        --save-processed-data \
        --summary
    
    # Step 3: Run explainability analysis
    echo -e "${BLUE}Step 3: Running explainability analysis${NC}"
    python main.py \
        --explain \
        --explainer shap \
        --explain-plots \
        --explain-plot-types summary bar beeswarm waterfall \
        --explain-samples 50 \
        --summary
    
    echo -e "${GREEN}Complete pipeline finished successfully${NC}"
}

# Parse command line arguments
case "$1" in
    generate)
        run_synthetic
        ;;
    process)
        process_existing $2
        ;;
    meta)
        run_meta_optimization $2
        ;;
    explain)
        run_explainability $2
        ;;
    full)
        run_full_pipeline
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo -e "${YELLOW}No option specified, showing usage information${NC}"
        show_usage
        ;;
esac

#!/bin/bash
# Cleanup script for the main directory
# This script organizes files into their appropriate directories and removes duplicates

# Set up fancy output formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    CLEANING UP MAIN DIRECTORY                    ${NC}"
echo -e "${BLUE}==================================================${NC}"

# Create directories if they don't exist
mkdir -p tests
mkdir -p scripts
mkdir -p examples
mkdir -p baseline_comparison

# Move test files to tests directory
echo -e "\n${YELLOW}Moving test files to tests directory...${NC}"
for file in test_*.py; do
    if [ -f "$file" ] && [ "$file" != "test_benchmark.py" ]; then
        # Check if the file already exists in tests directory
        if [ -f "tests/$file" ]; then
            echo "File $file already exists in tests directory, removing from main directory"
            rm "$file"
        else
            echo "Moving $file to tests directory"
            mv "$file" "tests/"
        fi
    fi
done

# Remove test_benchmark.py and debug_utils.py as they're just pointers to the real files
if [ -f "test_benchmark.py" ]; then
    echo "Removing test_benchmark.py (it's just a pointer to tests/test_benchmark.py)"
    rm "test_benchmark.py"
fi

if [ -f "debug_utils.py" ]; then
    echo "Removing debug_utils.py (it's just a pointer to tests/debug_utils.py)"
    rm "debug_utils.py"
fi

# Move shell scripts to scripts directory
echo -e "\n${YELLOW}Moving shell scripts to scripts directory...${NC}"
for file in *.sh; do
    if [ -f "$file" ]; then
        # Check if the file already exists in scripts directory
        if [ -f "scripts/$file" ]; then
            echo "File $file already exists in scripts directory, removing from main directory"
            rm "$file"
        else
            echo "Moving $file to scripts directory"
            mv "$file" "scripts/"
        fi
    fi
done

# Move example files to examples directory
echo -e "\n${YELLOW}Moving example files to examples directory...${NC}"
for file in *_example.py *_demo.py; do
    if [ -f "$file" ]; then
        # Check if the file already exists in examples directory
        if [ -f "examples/$file" ]; then
            echo "File $file already exists in examples directory, removing from main directory"
            rm "$file"
        else
            echo "Moving $file to examples directory"
            mv "$file" "examples/"
        fi
    fi
done

# Specific file handling
echo -e "\n${YELLOW}Handling specific files...${NC}"

# Move algorithm_selection_demo.py to examples
if [ -f "algorithm_selection_demo.py" ]; then
    if [ -f "examples/algorithm_selection_demo.py" ]; then
        echo "algorithm_selection_demo.py already exists in examples directory, removing from main directory"
        rm "algorithm_selection_demo.py"
    else
        echo "Moving algorithm_selection_demo.py to examples directory"
        mv "algorithm_selection_demo.py" "examples/"
    fi
fi

# Move run scripts to scripts directory
for file in run_enhanced_meta.py run_meta_learning.py; do
    if [ -f "$file" ]; then
        if [ -f "scripts/$file" ]; then
            echo "$file already exists in scripts directory, removing from main directory"
            rm "$file"
        else
            echo "Moving $file to scripts directory"
            mv "$file" "scripts/"
        fi
    fi
done

# Check for duplicate files in scripts directory (old versions with same name)
echo -e "\n${YELLOW}Checking for duplicate files in scripts directory...${NC}"
cd scripts
for file in *.sh; do
    if [ -f "$file" ] && [ -f "${file%.sh}_old.sh" ]; then
        echo "Found old version of $file, removing ${file%.sh}_old.sh"
        rm "${file%.sh}_old.sh"
    fi
done
cd ..

echo -e "\n${GREEN}Main directory cleanup completed!${NC}"
echo -e "${BLUE}==================================================${NC}"

# List remaining files in main directory for review
echo -e "\n${YELLOW}Remaining files in main directory:${NC}"
ls -la | grep -v "^d" | grep -v "total" | grep -v ".git"

echo -e "\n${BLUE}If any files remain in the main directory that should be moved,${NC}"
echo -e "${BLUE}please manually move them to the appropriate directory.${NC}" 
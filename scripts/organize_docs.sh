#!/bin/bash
# A script to move documentation files to the docs directory

# Create docs directory if it doesn't exist
mkdir -p docs

# Find all .md files in the root directory (excluding README.md)
for file in *.md; do
    if [ -f "$file" ] && [ "$file" != "README.md" ]; then
        # Check if the file is not a generated result file
        if [[ ! "$file" == *"result"* ]] && [[ ! "$file" == *"summary"* ]] && [[ ! "$file" == *"report"* ]]; then
            echo "Moving documentation file $file to docs directory"
            
            # If a file with the same name already exists in docs
            if [ -f "docs/$file" ]; then
                echo "File $file already exists in docs directory"
                
                # Compare files to see if they're identical
                if cmp -s "$file" "docs/$file"; then
                    echo "Files are identical, removing duplicate from main directory"
                    rm "$file"
                else
                    echo "Files differ, renaming the file in main directory before moving"
                    mv "$file" "docs/${file%.md}_main.md"
                fi
            else
                # Move the file if it doesn't exist in docs
                mv "$file" "docs/"
            fi
        else
            echo "Skipping likely result file: $file"
        fi
    fi
done

echo "Documentation files have been organized!"
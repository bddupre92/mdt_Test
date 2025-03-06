#!/usr/bin/env python3
"""
Script to fix the main.py file by moving the migraine functions before the parse_args function.
"""

import re

def fix_main_file():
    # Read the entire main.py file
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Extract the migraine functions
    migraine_funcs = re.findall(r'(def run_migraine_data_import.*?def run_migraine_prediction.*?)(?=\nif __name__ == "__main__")', content, re.DOTALL)
    
    if not migraine_funcs:
        print("Could not find migraine functions in the file")
        return
    
    migraine_code = migraine_funcs[0]
    
    # Remove the migraine functions from the end of the file
    content = re.sub(r'def run_migraine_data_import.*?def run_migraine_prediction.*?\n}.*?}\n', '', content, flags=re.DOTALL)
    
    # Insert the migraine functions before parse_args
    content = re.sub(r'def parse_args', migraine_code + '\n\ndef parse_args', content)
    
    # Write the modified content back to the file
    with open('main.py.fixed', 'w') as f:
        f.write(content)
    
    print("Fixed file written to main.py.fixed. Please review it and rename to main.py if it looks correct.")

if __name__ == "__main__":
    fix_main_file()

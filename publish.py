#!/usr/bin/env python3
"""
Helper script to build and publish the package to PyPI.
Usage:
    python publish.py [--test]
"""

import os
import sys
import subprocess
import shutil
import argparse

def run_command(command):
    """Run a shell command and return output."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False, e.stderr

def clean_build_dirs():
    """Clean up build directories."""
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for dir_name in dirs_to_clean:
        if '*' in dir_name:
            # Use glob pattern
            import glob
            for path in glob.glob(dir_name):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed {path}")
        elif os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}")

def build_package():
    """Build the package."""
    commands = [
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade build wheel twine",
        "python -m build",
    ]
    
    for cmd in commands:
        success, output = run_command(cmd)
        if not success:
            print(f"Failed to build package with command: {cmd}")
            return False
    return True

def publish_package(test=False):
    """Publish the package to PyPI or TestPyPI."""
    if test:
        cmd = "python -m twine upload --repository testpypi dist/*"
    else:
        cmd = "python -m twine upload dist/*"
    
    success, output = run_command(cmd)
    if not success:
        print("Failed to publish package")
        return False
    
    print(output)
    if test:
        print("\nPackage uploaded to TestPyPI. You can install it with:")
        print("pip install --index-url https://test.pypi.org/simple/ meta_optimizer")
    else:
        print("\nPackage uploaded to PyPI. You can install it with:")
        print("pip install meta_optimizer")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build and publish package to PyPI")
    parser.add_argument("--test", action="store_true", help="Publish to TestPyPI instead of PyPI")
    args = parser.parse_args()
    
    print("Cleaning up build directories...")
    clean_build_dirs()
    
    print("Building package...")
    if not build_package():
        sys.exit(1)
    
    print("Publishing package...")
    if not publish_package(args.test):
        sys.exit(1)
    
    print("Done!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Script to run Playwright tests for the Performance Analysis Dashboard.
This validates the dashboard's ability to handle various data formats.
"""
import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run the dashboard tests using pytest and playwright."""
    test_dir = Path(__file__).parent / "tests" / "dashboard"
    
    # Ensure we're running from the project root
    os.chdir(Path(__file__).parent)
    
    # Set environment variables for testing
    os.environ["PYTHONPATH"] = str(Path(__file__).parent)
    os.environ["MOE_CHECKPOINT_DIR"] = "checkpoints/test"
    
    print("Running Performance Analysis Dashboard flexibility tests...")
    
    # Run the tests with pytest
    cmd = [
        "python", "-m", "pytest", 
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Dashboard flexibility tests completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Dashboard flexibility tests failed with exit code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())

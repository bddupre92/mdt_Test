#!/usr/bin/env python
"""
Performance Analysis Dashboard Runner

This script runs the Performance Analysis dashboard using streamlit CLI,
which is more reliable than the bootstrap API.
"""

import os
import subprocess
import sys

if __name__ == "__main__":
    # Change to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Define the path to the performance analysis dashboard
    dashboard_path = os.path.join(script_dir, "app", "ui", "performance_analysis_dashboard.py")
    
    # Run streamlit with the dashboard file
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.port=8503"]
    
    print(f"Starting Performance Analysis Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command and display output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  universal_newlines=True, bufsize=1)
        
        # Stream the output
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running dashboard: {e}")
        sys.exit(1)

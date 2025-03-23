#!/usr/bin/env python
"""
Run API and Next.js Frontend

This script launches both the FastAPI backend and Next.js frontend.
"""

import subprocess
import os
import signal
import sys
import time
import threading

def run_api(port=8000):
    """Run the FastAPI server."""
    print(f"Starting API server on port {port}...")
    api_process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return api_process

def run_next_js(port=4001):
    """Run the Next.js development server."""
    print(f"Starting Next.js server on port {port}...")
    next_process = subprocess.Popen(
        ["cd", "v0test", "&&", "npm", "run", "dev", "--", "-p", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    return next_process

def stream_output(process, prefix):
    """Stream output from a subprocess."""
    for line in iter(process.stdout.readline, ""):
        if line:
            print(f"{prefix}: {line.strip()}")
    
    for line in iter(process.stderr.readline, ""):
        if line:
            print(f"{prefix} ERROR: {line.strip()}")

def main():
    """Main function to launch both servers."""
    try:
        # Start the API server
        api_process = run_api(port=8000)
        api_thread = threading.Thread(
            target=stream_output, 
            args=(api_process, "API"),
            daemon=True
        )
        api_thread.start()
        
        # Give the API server time to start
        time.sleep(2)
        
        # Start the Next.js server
        next_process = run_next_js(port=4001)
        next_thread = threading.Thread(
            target=stream_output,
            args=(next_process, "Next.js"),
            daemon=True
        )
        next_thread.start()
        
        print("\n====================================")
        print("Servers are running!")
        print("API: http://localhost:8000")
        print("Next.js: http://localhost:4001")
        print("====================================\n")
        print("Press Ctrl+C to stop both servers")
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        
        # Terminate processes
        if 'api_process' in locals():
            api_process.terminate()
            api_process.wait()
            
        if 'next_process' in locals():
            next_process.terminate()
            next_process.wait()
            
        print("Servers stopped")
        
if __name__ == "__main__":
    main()
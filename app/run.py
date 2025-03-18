#!/usr/bin/env python
"""
Unified Launcher Script

This script launches either the FastAPI server, the Streamlit dashboard, or both.
"""

import os
import sys
import argparse
import subprocess
import signal
import socket
import time
from pathlib import Path
import uvicorn

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is in use, False otherwise
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_free_port(start_port: int) -> int:
    """Find a free port starting from start_port.
    
    Args:
        start_port: Port to start searching from
        
    Returns:
        First free port found
    """
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

def run_api_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False):
    """Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Whether to run in debug mode
    """
    # Find free port if specified port is in use
    if is_port_in_use(port):
        new_port = find_free_port(port)
        print(f"Port {port} is in use, using port {new_port} instead")
        port = new_port
    
    print(f"Starting API server on http://{host}:{port}...")
    
    # Import and create FastAPI app
    from app.api.main import create_app
    app = create_app()
    
    # Configure Uvicorn
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=debug,
        reload_dirs=[str(Path(__file__).parent)],
        log_level="info",
        workers=1,
        loop="asyncio"
    )
    
    # Run server
    server = uvicorn.Server(config)
    server.run()

def run_dashboard(port: int = 8501):
    """Run the Streamlit dashboard.
    
    Args:
        port: Port to run on
    """
    # Find free port if specified port is in use
    if is_port_in_use(port):
        new_port = find_free_port(port)
        print(f"Port {port} is in use, using port {new_port} instead")
        port = new_port
    
    print(f"Starting dashboard on port {port}...")
    
    # Get path to dashboard script
    dashboard_path = Path(__file__).parent / "ui" / "benchmark_dashboard.py"
    
    # Run streamlit with environment variables to handle stdin
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parent.parent),
        "PYTHONUNBUFFERED": "1"
    }
    
    # Run streamlit
    try:
        subprocess.run([
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1"
        ], env=env)
    except KeyboardInterrupt:
        pass

def cleanup_processes():
    """Clean up any remaining processes."""
    try:
        # Find and kill processes on the ports
        for port in [8000, 8501]:
            if is_port_in_use(port):
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/PID", f":{port}"], capture_output=True)
                else:
                    subprocess.run(["lsof", "-ti", f":{port}", "|", "xargs", "kill", "-9"], shell=True)
    except:
        pass

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch benchmark application components")
    parser.add_argument(
        "--mode",
        choices=["api", "dashboard", "both"],
        default="both",
        help="Which component to run"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Port for dashboard"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    # Clean up any existing processes
    cleanup_processes()
    
    # Create results directory
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)
    
    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        cleanup_processes()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run components based on mode
        if args.mode in ["api", "both"]:
            if args.mode == "both":
                # Run API server in a separate process
                api_process = subprocess.Popen([
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.api_port)
                ], env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).parent.parent),
                    "PYTHONUNBUFFERED": "1"
                })
                
                # Give the API server time to start
                time.sleep(2)
            else:
                run_api_server(port=args.api_port, debug=args.debug)
        
        if args.mode in ["dashboard", "both"]:
            try:
                run_dashboard(port=args.dashboard_port)
            finally:
                if args.mode == "both" and 'api_process' in locals():
                    api_process.terminate()
                    api_process.wait()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_processes()
    
    finally:
        cleanup_processes()

if __name__ == "__main__":
    main() 
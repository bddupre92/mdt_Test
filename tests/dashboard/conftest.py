"""
Configuration and fixtures for dashboard tests.
"""
import os
import pytest
import subprocess
import time
import json
import shutil
from pathlib import Path
import signal
from typing import Generator, Dict, Any

import pytest
from playwright.sync_api import Page, expect, Playwright


@pytest.fixture(scope="session")
def dashboard_process():
    """Start the dashboard server for testing and stop it after tests complete."""
    # Use the existing checkpoint directory that contains our test data
    # We've already copied the test data into checkpoints/dev using copy_test_data_to_checkpoints.py
    
    # Set the environment variable to use the dev checkpoints
    os.environ["MOE_CHECKPOINT_DIR"] = "checkpoints/dev"
    
    # Start the dashboard on a test port
    print("\n[DEBUG] Starting dashboard with checkpoint dir:", os.environ.get("MOE_CHECKPOINT_DIR"))
    print("[DEBUG] Checking if checkpoint dir exists:", os.path.exists(os.environ.get("MOE_CHECKPOINT_DIR", "")))
    
    # List the checkpoints to confirm they're there
    checkpoint_files = list(Path(os.environ.get("MOE_CHECKPOINT_DIR", "checkpoints/dev")).glob("checkpoint_*_format_*.json"))
    print(f"[DEBUG] Found {len(checkpoint_files)} checkpoint files: {[cp.name for cp in checkpoint_files]}")
    
    # We'll use an available port
    test_port = 8506
    cmd = f"python -m streamlit run app/ui/performance_analysis_dashboard.py --server.port={test_port} --server.headless=true"
    print(f"[DEBUG] Running command: {cmd}")
    
    process = subprocess.Popen(
        cmd, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(os.environ),
        preexec_fn=os.setsid  # Use process group for clean termination
    )
    
    # Now we need to wait for the server to be ready
    server_ready = False
    start_time = time.time()
    timeout = 30  # 30 seconds timeout
    
    print("[DEBUG] Waiting for Streamlit server to start...")
    while not server_ready and time.time() - start_time < timeout:
        try:
            # Try to connect to the server
            import urllib.request
            with urllib.request.urlopen(f"http://localhost:{test_port}") as response:
                if response.status == 200:
                    server_ready = True
                    print("[DEBUG] Streamlit server is ready!")
        except Exception as e:
            # Server not ready yet
            print(f"[DEBUG] Still waiting for server... ({e.__class__.__name__})")
            time.sleep(1)
            
    if not server_ready:
        # Check if the process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("[DEBUG] Dashboard process exited prematurely with code:", process.returncode)
            print("[DEBUG] STDOUT:", stdout.decode('utf-8') if stdout else "None")
            print("[DEBUG] STDERR:", stderr.decode('utf-8') if stderr else "None")
        else:
            print("[DEBUG] Server did not become ready in time, but process is still running")
    
    yield process
    
    # Clean up - terminate the process group if it's still running
    try:
        if process.poll() is None:  # Check if process is still running
            print("[DEBUG] Terminating dashboard process...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)  # Wait for process to terminate
            print("[DEBUG] Dashboard process terminated.")
        else:
            print(f"[DEBUG] Dashboard process already terminated with exit code: {process.returncode}")
    except (ProcessLookupError, OSError) as e:
        print(f"[DEBUG] Error terminating dashboard process: {e}")


@pytest.fixture(scope="function")
def dashboard_page(playwright: Playwright, dashboard_process) -> Generator[Page, None, None]:
    """Create a page with the dashboard loaded."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()
    
    # Use the same port as in the dashboard_process fixture
    print("[DEBUG] Connecting to dashboard at http://localhost:8506")
    
    # Navigate to the dashboard with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Navigate to the dashboard
            page.goto("http://localhost:8506", timeout=10000)
            
            # Wait for Streamlit to load (look for common Streamlit elements)
            page.wait_for_selector("div.stApp", timeout=10000)
            print(f"[DEBUG] Streamlit app frame found on attempt {attempt+1}")
            
            # Now look for dashboard-specific elements
            if page.locator("text=Performance Analysis").count() > 0:
                print("[DEBUG] Dashboard title found!")
                break
                
            print(f"[DEBUG] Dashboard title not found on attempt {attempt+1}, retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"[DEBUG] Error on attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
    
    # Take a screenshot for debugging
    page.screenshot(path="dashboard_screenshot.png")
    print("[DEBUG] Took screenshot of dashboard state")
    
    yield page
    
    # Clean up
    context.close()
    browser.close()

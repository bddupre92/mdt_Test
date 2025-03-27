"""
Debug script for dashboard using Playwright Inspector.

This script allows interactive debugging of the dashboard in a browser with Playwright Inspector.
Run with: python -m pytest tests/dashboard/debug_dashboard.py --headed
"""

import os
import pytest
import time
from playwright.sync_api import Page, expect

# Set environment variables
os.environ["MOE_CHECKPOINT_DIR"] = "checkpoints/dev"

@pytest.mark.only_browser("chromium")
def test_debug_dashboard_with_inspector(page: Page):
    """
    Debug the dashboard with Playwright Inspector.
    
    Run with:
    PWDEBUG=1 python -m pytest tests/dashboard/debug_dashboard.py --headed
    """
    # Navigate to the dashboard
    base_url = "http://localhost:8501"
    page.goto(base_url)
    
    # Make sure the page is loaded
    page.wait_for_load_state("networkidle")
    
    # Add a pause to allow manual inspection with the Playwright Inspector
    page.pause()
    
    # Check if there's any text showing "No checkpoints available"
    no_checkpoints_message = page.locator("text=No checkpoints available")
    if no_checkpoints_message.count() > 0:
        print("Found 'No checkpoints available' message")
    
    # Check for the checkpoint selection dropdown
    dropdown = page.locator("select")
    if dropdown.count() > 0:
        print(f"Found {dropdown.count()} dropdown elements")
    
    # Print any error messages that might be displayed
    error_messages = page.locator("div.stException")
    if error_messages.count() > 0:
        for i in range(error_messages.count()):
            print(f"Error message {i+1}: {error_messages.nth(i).inner_text()}")
    
    # Look for any elements with "checkpoint" in them for additional clues
    checkpoint_elements = page.locator("text=/.*checkpoint.*/i")
    if checkpoint_elements.count() > 0:
        print(f"Found {checkpoint_elements.count()} elements with 'checkpoint' text")
        for i in range(checkpoint_elements.count()):
            print(f"Checkpoint element {i+1}: {checkpoint_elements.nth(i).inner_text()}")
    
    # This is just a place for the debugger to pause
    # You can interact with the page and debug manually from here
    assert page.title() != ""

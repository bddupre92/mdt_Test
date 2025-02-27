#!/usr/bin/env python3
"""
Test script to access the drift dashboard with a valid token.
"""
import sys
import os
from pathlib import Path
import logging
import argparse
import requests
import webbrowser
import json
from urllib.parse import urljoin

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:8000"
LOGIN_URL = urljoin(BASE_URL, "/auth/token")  # Updated login endpoint
DASHBOARD_URL = urljoin(BASE_URL, "/dashboard/drift")

def get_access_token(username, password):
    """Get access token by logging in"""
    # Use form data instead of JSON
    data = {
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(
            LOGIN_URL, 
            data=data,  # Use form data
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            logger.error(f"Login failed: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting token: {str(e)}")
        return None

def open_dashboard(token):
    """Open the drift dashboard in a browser with the token."""
    # First, check if the dashboard is accessible
    try:
        response = requests.get(
            DASHBOARD_URL,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            logger.info("Dashboard is accessible")
            
            # Create a simple HTML file with the token in localStorage
            html_path = Path(__file__).parent / "dashboard_access.html"
            
            with open(html_path, "w") as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Dashboard Access</title>
                    <script>
                        // Store the token
                        localStorage.setItem('access_token', '{token}');
                        
                        // Redirect to the dashboard
                        window.location.href = '{DASHBOARD_URL}';
                    </script>
                </head>
                <body>
                    <p>Redirecting to dashboard...</p>
                </body>
                </html>
                """)
            
            # Open the HTML file in a browser
            webbrowser.open(f"file://{html_path}")
            logger.info(f"Opened dashboard access page at {html_path}")
            return True
        else:
            logger.error(f"Dashboard is not accessible: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error accessing dashboard: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test access to the drift dashboard")
    parser.add_argument("--username", default="testuser", help="Username to log in with")
    parser.add_argument("--password", default="testpass123", help="Password for the user")
    
    args = parser.parse_args()
    
    # Get token
    token = get_access_token(args.username, args.password)
    
    if token:
        # Open dashboard
        open_dashboard(token)
    else:
        logger.error("Failed to get token, cannot access dashboard")

if __name__ == "__main__":
    main()

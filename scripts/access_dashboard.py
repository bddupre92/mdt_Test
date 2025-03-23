#!/usr/bin/env python3
"""
Script to access the drift detection dashboard.

This script logs in to get an access token and then opens the dashboard in a browser.
"""
import argparse
import requests
import webbrowser
import json
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def login(base_url, username, password):
    """
    Log in to get an access token.
    
    Args:
        base_url: Base URL of the API
        username: Username for login
        password: Password for login
        
    Returns:
        Access token if login is successful, None otherwise
    """
    login_url = urljoin(base_url, "/api/auth/login")
    
    try:
        response = requests.post(
            login_url,
            data={
                "username": username,
                "password": password
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )
        
        if response.status_code == 200:
            token_data = response.json()
            logger.info(f"Login successful for user {username}")
            return token_data.get("access_token")
        else:
            logger.error(f"Login failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        return None

def access_dashboard(base_url, token):
    """
    Access the drift detection dashboard with the provided token.
    
    Args:
        base_url: Base URL of the API
        token: Access token for authentication
        
    Returns:
        True if dashboard access is successful, False otherwise
    """
    dashboard_url = urljoin(base_url, "/api/dashboard/drift")
    
    try:
        response = requests.get(
            dashboard_url,
            headers={
                "Authorization": f"Bearer {token}"
            }
        )
        
        if response.status_code == 200:
            logger.info("Dashboard access successful")
            
            # Save the HTML response to a file
            with open("drift_dashboard.html", "w") as f:
                f.write(response.text)
                
            logger.info("Dashboard HTML saved to drift_dashboard.html")
            
            # Open the file in a browser
            webbrowser.open("drift_dashboard.html")
            
            return True
        else:
            logger.error(f"Dashboard access failed: {response.status_code} - {response.text}")
            
            # If we get a 401, it might be an authentication issue
            if response.status_code == 401:
                logger.error("Authentication failed. Make sure you're using a valid token.")
                logger.error("Try running the script again with a new login.")
            
            return False
            
    except Exception as e:
        logger.error(f"Error accessing dashboard: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Access the drift detection dashboard")
    parser.add_argument("--username", default="testuser", help="Username for login")
    parser.add_argument("--password", default="testpassword", help="Password for login")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the API")
    
    args = parser.parse_args()
    
    # Login to get access token
    token = login(args.url, args.username, args.password)
    
    if token:
        logger.info(f"Access token: {token}")
        # Access dashboard with token
        access_dashboard(args.url, token)
    else:
        logger.error("Failed to obtain access token. Cannot access dashboard.")

if __name__ == "__main__":
    main()

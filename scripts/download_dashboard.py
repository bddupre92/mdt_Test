#!/usr/bin/env python3
"""
Script to download the drift detection dashboard HTML and save it to a file.
"""
import argparse
import requests
import logging
import os
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

def download_dashboard(base_url, token, output_file="drift_dashboard_downloaded.html"):
    """
    Download the drift detection dashboard HTML and save it to a file.
    
    Args:
        base_url: Base URL of the API
        token: Access token for authentication
        output_file: Output file path
        
    Returns:
        True if download is successful, False otherwise
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
            logger.info("Dashboard download successful")
            
            # Save the HTML response to a file
            with open(output_file, "w") as f:
                f.write(response.text)
                
            logger.info(f"Dashboard HTML saved to {output_file}")
            
            # Make the file path absolute
            abs_path = os.path.abspath(output_file)
            logger.info(f"Absolute path: {abs_path}")
            logger.info(f"Access the dashboard by opening this file in your browser")
            
            return True
        else:
            logger.error(f"Dashboard download failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading dashboard: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download the drift detection dashboard HTML")
    parser.add_argument("--username", default="testuser", help="Username for login")
    parser.add_argument("--password", default="testpassword", help="Password for login")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of the API")
    parser.add_argument("--output", default="drift_dashboard_downloaded.html", help="Output file path")
    
    args = parser.parse_args()
    
    # Login to get access token
    token = login(args.url, args.username, args.password)
    
    if token:
        logger.info(f"Access token: {token}")
        # Download dashboard
        download_dashboard(args.url, token, args.output)
    else:
        logger.error("Failed to obtain access token. Cannot download dashboard.")

if __name__ == "__main__":
    main()

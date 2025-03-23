#!/usr/bin/env python3
"""
Script to test all dashboard endpoints and report errors
"""
import requests
import json
import time
import sys
from colorama import Fore, Style, init

# Initialize colorama
init()

BASE_URL = "http://localhost:8000"

# List of all endpoints to test
ENDPOINTS = [
    "/api/dashboard/performance",
    "/api/dashboard/sample-drift-data",
    "/api/dashboard/drift-metrics",
    "/api/dashboard/optimization-results",
    "/api/dashboard/optimization-history",
    "/api/dashboard/meta-analysis",
    "/api/dashboard/benchmarks/comparison",
    "/api/dashboard/benchmarks/convergence",
    "/api/dashboard/benchmarks/real-data-performance",
    "/api/dashboard/benchmarks/feature-importance"
]

def print_colored(message, color):
    """Print message with color"""
    print(f"{color}{message}{Style.RESET_ALL}")

def test_endpoint(endpoint):
    """Test an endpoint and return (success, response_data, error_message)"""
    url = f"{BASE_URL}{endpoint}"
    try:
        print_colored(f"Testing endpoint: {url}", Fore.CYAN)
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print_colored(f"✓ Success ({response.status_code})", Fore.GREEN)
            try:
                # Try to parse JSON response
                data = response.json()
                return True, data, None
            except json.JSONDecodeError:
                print_colored("  Warning: Response is not valid JSON", Fore.YELLOW)
                return True, response.text, None
        else:
            error_msg = f"✗ Failed with status {response.status_code}"
            try:
                # Try to get error details from response
                error_details = response.json()
                error_msg += f": {json.dumps(error_details)}"
            except:
                error_msg += f": {response.text}"
            
            print_colored(error_msg, Fore.RED)
            return False, None, error_msg
            
    except Exception as e:
        error_msg = f"✗ Error: {str(e)}"
        print_colored(error_msg, Fore.RED)
        return False, None, error_msg

def main():
    """Test all endpoints and summarize results"""
    print_colored("Dashboard Endpoint Testing Tool", Fore.MAGENTA)
    print_colored("==============================", Fore.MAGENTA)
    
    # Check if server is running
    try:
        requests.get(f"{BASE_URL}/dashboard", timeout=2)
    except:
        print_colored("❌ Error: Server does not appear to be running at " + BASE_URL, Fore.RED)
        print_colored("Please start the server with: cd app && python -m uvicorn main:app --reload", Fore.YELLOW)
        return
    
    print_colored("✓ Server is running", Fore.GREEN)
    print_colored("\nTesting all endpoints:", Fore.CYAN)
    
    results = []
    
    for endpoint in ENDPOINTS:
        success, data, error = test_endpoint(endpoint)
        results.append((endpoint, success, error))
        # Add a small delay between requests
        time.sleep(0.5)
    
    # Print summary
    print_colored("\nTest Summary:", Fore.MAGENTA)
    print_colored("============", Fore.MAGENTA)
    
    success_count = sum(1 for _, success, _ in results if success)
    
    print_colored(f"Total endpoints tested: {len(ENDPOINTS)}", Fore.CYAN)
    print_colored(f"Successful: {success_count}", Fore.GREEN)
    print_colored(f"Failed: {len(ENDPOINTS) - success_count}", Fore.RED if len(ENDPOINTS) - success_count > 0 else Fore.GREEN)
    
    if len(ENDPOINTS) - success_count > 0:
        print_colored("\nFailed endpoints:", Fore.RED)
        for endpoint, success, error in results:
            if not success:
                print_colored(f"  ✗ {endpoint}: {error}", Fore.RED)
    else:
        print_colored("\n✓ All endpoints working correctly!", Fore.GREEN)

if __name__ == "__main__":
    main()

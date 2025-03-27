#!/usr/bin/env python
"""
MoE Framework Demo Runner

This script runs a complete demo of the MoE framework:
1. Generates sample data
2. Runs the example workflow with tracking
3. Launches the dashboard to visualize the results
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def generate_sample_data(n_samples=100, output_path=None):
    """Generate sample data for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Create patient IDs and locations
    patient_ids = np.random.choice(['P001', 'P002', 'P003'], n_samples)
    locations = np.random.choice(['New York', 'Boston', 'Chicago'], n_samples)
    
    # Create physiological features
    heart_rate = np.random.normal(75, 10, n_samples)
    blood_pressure_sys = np.random.normal(120, 15, n_samples)
    blood_pressure_dia = np.random.normal(80, 10, n_samples)
    temperature = np.random.normal(37, 0.5, n_samples)
    
    # Create environmental features
    env_temperature = np.random.normal(20, 8, n_samples)
    humidity = np.random.normal(60, 15, n_samples)
    pressure = np.random.normal(1013, 10, n_samples)
    aqi = np.random.normal(50, 20, n_samples)
    
    # Create behavioral features
    sleep_duration = np.random.normal(7, 1.5, n_samples)
    sleep_quality = np.random.normal(70, 15, n_samples)
    activity_level = np.random.normal(60, 20, n_samples)
    stress_level = np.random.normal(50, 25, n_samples)
    
    # Create medication features
    medication_a = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    medication_b = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    medication_c = np.random.choice(['', 'Low', 'Medium', 'High'], n_samples)
    
    # Create target variable (migraine severity)
    # Each domain contributes to the target with some noise
    physio_effect = 0.3 * heart_rate + 0.2 * blood_pressure_sys
    env_effect = 0.25 * env_temperature + 0.15 * humidity + 0.1 * aqi
    behavior_effect = -0.2 * sleep_quality + 0.3 * stress_level
    
    # Medication effects (higher medication levels reduce severity)
    med_a_effect = np.where(medication_a == '', 0, 
                   np.where(medication_a == 'Low', -5, 
                   np.where(medication_a == 'Medium', -10, -15)))
    
    med_b_effect = np.where(medication_b == '', 0, 
                   np.where(medication_b == 'Low', -3, 
                   np.where(medication_b == 'Medium', -7, -12)))
    
    med_c_effect = np.where(medication_c == '', 0, 
                   np.where(medication_c == 'Low', -2, 
                   np.where(medication_c == 'Medium', -5, -8)))
    
    # Combine effects with noise
    migraine_severity = (
        50 +  # baseline
        physio_effect + 
        env_effect + 
        behavior_effect + 
        med_a_effect + 
        med_b_effect + 
        med_c_effect + 
        np.random.normal(0, 10, n_samples)  # random noise
    )
    
    # Ensure severity is in a reasonable range (0-100)
    migraine_severity = np.clip(migraine_severity, 0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        # Identifiers
        'patient_id': patient_ids,
        'location': locations,
        'date': timestamps,
        
        # Physiological features
        'heart_rate': heart_rate,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'temperature': temperature,
        
        # Environmental features
        'env_temperature': env_temperature,
        'humidity': humidity,
        'pressure': pressure,
        'aqi': aqi,
        
        # Behavioral features
        'sleep_duration': sleep_duration,
        'sleep_quality': sleep_quality,
        'activity_level': activity_level,
        'stress_level': stress_level,
        
        # Medication features
        'medication_a': medication_a,
        'medication_b': medication_b,
        'medication_c': medication_c,
        
        # Target
        'migraine_severity': migraine_severity
    })
    
    # Save to file if path provided
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
    
    return data

def run_command(command, env=None):
    """Run a command with the correct Python path."""
    if env is None:
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env)

def create_streamlit_dashboard():
    """Create a temporary Streamlit dashboard file."""
    dashboard_content = """
import os
import sys
import streamlit as st

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import the dashboard module
try:
    from moe_framework.event_tracking.dashboard import render_workflow_dashboard
    
    # Set page config
    st.set_page_config(
        page_title="MoE Framework Dashboard",
        layout="wide"
    )
    
    st.title("MoE Framework Dashboard")
    
    # Get tracking directory from query params or use default
    tracking_dir = st.query_params.get("tracking_dir", "./.workflow_tracking")
    
    # Render the dashboard
    render_workflow_dashboard(tracking_dir)
    
except ImportError as e:
    st.error(f"Error importing dashboard module: {str(e)}")
    st.info("Make sure the moe_framework package is in your Python path.")
    
    # Display visualizations from the visualizations directory
    st.header("MoE Framework Visualizations")
    
    # Check if visualizations directory exists
    if os.path.exists("./visualizations"):
        st.subheader("Available Visualizations")
        
        # Get all image files
        image_files = [f for f in os.listdir("./visualizations") 
                      if f.endswith((".png", ".jpg", ".jpeg"))]
        
        if image_files:
            # Display images in columns
            cols = st.columns(2)
            for i, img_file in enumerate(image_files):
                with cols[i % 2]:
                    st.image(f"./visualizations/{img_file}", caption=img_file, use_column_width=True)
        else:
            st.info("No visualizations found. Run the example workflow first.")
    else:
        st.info("Visualizations directory not found. Run the example workflow first.")
"""
    
    # Write the temporary dashboard file with UTF-8 encoding
    temp_dashboard_path = os.path.join(project_root, "temp_dashboard.py")
    with open(temp_dashboard_path, "w", encoding='utf-8') as f:
        f.write(dashboard_content)
    
    return temp_dashboard_path

def main():
    parser = argparse.ArgumentParser(description="Run MoE Framework Demo")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--tracking-dir", type=str, default="./.workflow_tracking", help="Tracking directory")
    parser.add_argument("--skip-example", action="store_true", help="Skip running the example workflow")
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip launching the dashboard")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tracking_dir, exist_ok=True)
    os.makedirs("./visualizations", exist_ok=True)
    
    # Step 1: Generate sample data
    print("\n=== Step 1: Generating Sample Data ===\n")
    sample_data_path = os.path.join(args.output_dir, "sample_data.csv")
    generate_sample_data(n_samples=args.samples, output_path=sample_data_path)
    
    # Step 2: Run the example workflow
    if not args.skip_example:
        print("\n=== Step 2: Running Example Workflow ===\n")
        
        try:
            # Try to import and run the example
            from moe_framework.event_tracking.example import run_demo
            run_demo()
            
            print("\n=== Workflow Completed Successfully! ===\n")
            print(f"Visualizations have been saved to the './visualizations' directory.")
            
        except ImportError as e:
            print(f"Error importing example module: {str(e)}")
            print("Running alternative example workflow...")
            
            # Create a simple example workflow that doesn't depend on moe_framework
            print("Creating simple example visualizations...")
            
            # Create a simple plot
            import matplotlib.pyplot as plt
            
            # Create a sample plot
            plt.figure(figsize=(10, 6))
            plt.plot(np.random.randn(100).cumsum())
            plt.title("Sample MoE Workflow Result")
            plt.xlabel("Iteration")
            plt.ylabel("Performance")
            plt.savefig("./visualizations/sample_result.png")
            plt.close()
            
            # Create another sample plot
            plt.figure(figsize=(10, 6))
            plt.bar(['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4'], 
                   [0.8, 0.6, 0.7, 0.9])
            plt.title("Expert Model Performance")
            plt.ylabel("Accuracy")
            plt.savefig("./visualizations/expert_performance.png")
            plt.close()
            
            print("Created sample visualizations in './visualizations' directory.")
    
    # Step 3: Launch the dashboard
    if not args.skip_dashboard:
        print("\n=== Step 3: Launching Dashboard ===\n")
        print("Starting Streamlit dashboard...")
        
        # Create temporary dashboard file
        temp_dashboard_path = create_streamlit_dashboard()
        
        try:
            # Run the dashboard
            run_command(["streamlit", "run", temp_dashboard_path])
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_dashboard_path):
                os.remove(temp_dashboard_path)
    else:
        print("\nSkipping dashboard launch as requested.")
        print("To launch the dashboard later, run: streamlit run temp_dashboard.py")

if __name__ == "__main__":
    main()
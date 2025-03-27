"""
Test script for the MoE Performance Analysis Dashboard.
"""

import os
import glob
from pathlib import Path

def get_latest_checkpoint():
    """Get the path to the latest checkpoint file."""
    checkpoint_dir = Path("results/moe_run/dev/checkpoints")
    checkpoints = list(checkpoint_dir.glob("moe_run_*.json"))
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found")
    return str(max(checkpoints, key=os.path.getctime))

def main():
    """Run the performance dashboard with the latest checkpoint."""
    checkpoint_path = get_latest_checkpoint()
    print(f"Using checkpoint: {checkpoint_path}")
    os.system(f"streamlit run visualization/performance_dashboard.py -- {checkpoint_path}")

if __name__ == "__main__":
    main()

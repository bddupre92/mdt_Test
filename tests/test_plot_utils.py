"""
Test script for plot_utils.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from utils.plot_utils import save_plot

def test_convergence_plot():
    """Test convergence plot saving"""
    # Create sample data
    data = {
        'optimizer': ['Optimizer1', 'Optimizer1', 'Optimizer2', 'Optimizer2'],
        'iteration': [1, 2, 1, 2],
        'score': [0.5, 0.3, 0.6, 0.2]
    }
    df = pd.DataFrame(data)
    
    try:
        # Create plot
        plt.figure(figsize=(10, 6))
        for optimizer in df['optimizer'].unique():
            subset = df[df['optimizer'] == optimizer]
            plt.plot(subset['iteration'], subset['score'], label=optimizer)
        
        plt.title('Convergence Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # Save plot using save_plot
        fig = plt.gcf()
        save_plot(fig, 'test_convergence.png', plot_type='benchmarks')
    finally:
        plt.close('all')

def test_boxplot():
    """Test boxplot saving"""
    # Create sample data
    np.random.seed(42)
    data = {
        'optimizer': ['Optimizer1'] * 30 + ['Optimizer2'] * 30,
        'score': np.concatenate([np.random.normal(0.5, 0.1, 30), np.random.normal(0.3, 0.2, 30)])
    }
    df = pd.DataFrame(data)
    
    try:
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='optimizer', y='score', data=df)
        plt.title('Performance Comparison')
        plt.xlabel('Optimizer')
        plt.ylabel('Score')
        plt.grid(True)
        
        # Save plot using save_plot
        fig = plt.gcf()
        save_plot(fig, 'test_boxplot.png', plot_type='benchmarks')
    finally:
        plt.close('all')

def main():
    """Run all tests"""
    print("Testing plot_utils.py...")
    
    # Make sure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_convergence_plot()
    test_boxplot()
    
    print("Tests completed. Check the results directory for saved plots.")

if __name__ == "__main__":
    main()

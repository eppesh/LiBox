#!/usr/bin/env python3
"""
Script to visualize overflow and underflow ratios from out.csv
X-axis: # of keys
Y-axis: overflow/underflow ratio
Each window size gets separate lines for overflow and underflow ratios
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

OVERFLOW_THRESHOLD = 0.1
UNDERFLOW_THRESHOLD = 0.5

def load_data(csv_file='out.csv'):
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {len(df)} rows from {csv_file}")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def plot_ratios(df, output_file=None, show_plot=True):
    """
    Plot overflow and underflow ratios from CSV data
    
    Args:
        df: pandas DataFrame with the data
        output_file (str): Path to save the plot (default: None)
        show_plot (bool): Whether to display the plot (default: True)
    """
    window_sizes = sorted(df['window_size'].unique())
    print(f"Found {len(window_sizes)} unique window sizes: {window_sizes}")
    
    plt.figure(figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(window_sizes)))
    
    for i, window_size in enumerate(window_sizes):
        window_data = df[df['window_size'] == window_size].sort_values('num_keys')
        
        plt.plot(window_data['num_keys'], 
                window_data['underflow_ratio'], 
                color=colors[i], 
                linestyle='-', 
                linewidth=2,
                label=f'Underflow (Window: {window_size:,})')
        
        plt.plot(window_data['num_keys'], 
                window_data['overflow_ratio'], 
                color=colors[i], 
                linestyle='--', 
                linewidth=2,
                label=f'Overflow (Window: {window_size:,})')
    
    plt.axhline(y=OVERFLOW_THRESHOLD, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Overflow Threshold ({OVERFLOW_THRESHOLD})')
    plt.axhline(y=UNDERFLOW_THRESHOLD, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Underflow Threshold ({UNDERFLOW_THRESHOLD})')
    
    plt.xlabel('# of Keys', fontsize=14)
    plt.ylabel('Ratio', fontsize=14)
    plt.title('Overflow and Underflow Ratios by # of Keys and Window Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_separate_ratios(df, output_file=None, show_plot=True):
    """
    Create separate plots for overflow and underflow ratios
    
    Args:
        df: pandas DataFrame with the data
        output_file (str): Path to save the plot (default: None)
        show_plot (bool): Whether to display the plot (default: True)
    """
    window_sizes = sorted(df['window_size'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(window_sizes)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    for i, window_size in enumerate(window_sizes):
        window_data = df[df['window_size'] == window_size].sort_values('num_keys')
        ax1.plot(window_data['num_keys'], 
                window_data['underflow_ratio'], 
                color=colors[i], 
                linewidth=2,
                label=f'Window: {window_size:,}')

    ax1.axhline(y=UNDERFLOW_THRESHOLD, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'Threshold ({UNDERFLOW_THRESHOLD})')
    
    ax1.set_xlabel('# of Keys', fontsize=12)
    ax1.set_ylabel('Underflow Ratio', fontsize=12)
    ax1.set_title('Underflow Ratios by # of Keys', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    for i, window_size in enumerate(window_sizes):
        window_data = df[df['window_size'] == window_size].sort_values('num_keys')
        ax2.plot(window_data['num_keys'], 
                window_data['overflow_ratio'], 
                color=colors[i], 
                linewidth=2,
                label=f'Window: {window_size:,}')

    ax2.axhline(y=OVERFLOW_THRESHOLD, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'Threshold ({OVERFLOW_THRESHOLD})')
    
    ax2.set_xlabel('# of Keys', fontsize=12)
    ax2.set_ylabel('Overflow Ratio', fontsize=12)
    ax2.set_title('Overflow Ratios by # of Keys', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Separate plots saved to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    csv_file = None
    save_files = False
    separate_plots = True
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print('Please provide an input csv file.')
        return
    
    if '--save' in sys.argv:
        save_files = True
    
    if '--combined' in sys.argv:
        separate_plots = False

    df = load_data(csv_file)
    if df is None:
        return
    
    if separate_plots:
        plot_separate_ratios(df, 
                           output_file='separate_ratios_plot.png' if save_files else None,
                           show_plot=not save_files)
    else:
        plot_ratios(df, 
                   output_file='ratios_plot.png' if save_files else None,
                   show_plot=not save_files)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Segment Plotter

This script reads CSV files containing segment data (start_key, end_key, window_size)
and plots them as horizontal line segments on a graph where:
- X-axis: Key space (start_key to end_key)
- Y-axis: Window size
- Each CSV file gets a different color from a color palette
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
import matplotlib.colors as mcolors

def load_segments_from_csv(csv_path):
    """
    Load segment data from a CSV file.
    
    Expected CSV format: start_key, end_key, window_size
    """
    try:
        df = pd.read_csv(csv_path, header=None, names=['start_key', 'end_key', 'window_size'])
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def plot_segments(csv_files, output_file=None, figsize=(15, 10), dpi=300):
    """
    Plot segments from multiple CSV files.
    
    Args:
        csv_files: List of CSV file paths
        output_file: Optional output file path for saving the plot
        figsize: Figure size as (width, height)
        dpi: DPI for the output image
    """
    if not csv_files:
        print("No CSV files provided!")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(csv_files) > len(colors):
        additional_colors = plt.cm.Set3(np.linspace(0, 1, len(csv_files) - len(colors)))
        colors.extend(additional_colors)
    
    all_segments = []
    file_labels = []
    
    for i, csv_file in enumerate(csv_files):
        df = load_segments_from_csv(csv_file)
        if df is None or df.empty:
            print(f"Skipping {csv_file} - no valid data")
            continue
            
        color = colors[i % len(colors)]
        file_name = Path(csv_file).stem.replace('_', ' ')
        
        for _, row in df.iterrows():
            start_key = row['start_key']
            end_key = row['end_key']
            window_size = row['window_size']
            
            # Draw horizontal line segment
            ax.plot([start_key, end_key], [window_size, window_size], 
                   color=color, linewidth=2, alpha=0.8)
            
            all_segments.append({
                'start': start_key,
                'end': end_key,
                'size': window_size,
                'file': file_name
            })
        
        file_labels.append(file_name)
        print(f"Processed {csv_file}: {len(df)} segments")
    
    if not all_segments:
        print("No valid segments found!")
        return
    
    x_min, x_max = all_segments[0]['start'], all_segments[-1]['end']
    all_sizes = [seg['size'] for seg in all_segments] 
    y_min, y_max = min(all_sizes), max(all_sizes)
    
    x_padding = (x_max - x_min) * 0.02
    y_padding = (y_max - y_min) * 0.05
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    ax.set_xlabel('Key Space', fontsize=12, fontweight='bold')
    ax.set_ylabel('Window Size', fontsize=12, fontweight='bold')
    ax.set_title('Segment Distribution by Window Size', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    legend_elements = []
    for i, file_name in enumerate(file_labels):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, label=file_name))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    total_files = len(file_labels)
    stats_text = f'Files: {total_files}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    plt.show()
    
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

def main():
    """Main function to handle command line arguments and execute plotting."""
    parser = argparse.ArgumentParser(
        description='Plot segments from CSV files with start_key, end_key, window_size columns'
    )
    parser.add_argument('csv_files', nargs='+', help='CSV files to plot')
    parser.add_argument('-o', '--output', help='Output file path for saving the plot')
    parser.add_argument('--figsize', nargs=2, type=float, default=[15, 10],
                       help='Figure size as width height (default: 15 10)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
    
    args = parser.parse_args()
    
    valid_files = []
    for csv_file in args.csv_files:
        if Path(csv_file).exists():
            valid_files.append(csv_file)
        else:
            print(f"Warning: File {csv_file} not found, skipping...")
    
    if not valid_files:
        print("No valid CSV files found!")
        sys.exit(1)
    
    plot_segments(valid_files, args.output, tuple(args.figsize), args.dpi)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Segment Plotter")
        print("==============")
        print("Usage: python plot_segments.py <csv_file1> [csv_file2] ... [options]")
        print("\nExample:")
        print("  python plot_segments.py shrunk_out.csv")
        print("  python plot_segments.py file1.csv file2.csv -o output.png")
        print("\nOptions:")
        print("  -o, --output: Save plot to file")
        print("  --figsize: Figure size (width height)")
        print("  --dpi: DPI for output image")
    else:
        main()

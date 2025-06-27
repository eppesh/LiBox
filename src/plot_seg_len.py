#!/usr/bin/env python3
"""
Script to plot CSV data with window_size on x-axis and both max_segment_length and num_keys on y-axis.
Usage: python plot_seg_len.py <csv_file> [--save]
Example: python plot_seg_len.py outputs/merge9999len1e7_p0t2e-1_r6e5t7e5.csv --save
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_csv_data(csv_file, save_plot=False):
    """
    Plot CSV data with window_size on x-axis and both max_segment_length and num_keys on y-axis.
    
    Args:
        csv_file (str): Path to the CSV file
        save_plot (bool): Whether to save the plot to a file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_columns = ['window_size', 'max_segment_length', 'num_keys']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV file must contain '{col}' column")
        
        # Create subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Max Segment Length vs Window Size
        ax1.plot(df['window_size'], df['max_segment_length'], 'o-', linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('Window Size', fontsize=12)
        ax1.set_ylabel('Max Segment Length', fontsize=12)
        ax1.set_title('Max Segment Length vs Window Size', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='x')
        
        # Plot 2: Num Keys vs Window Size
        ax2.plot(df['window_size'], df['num_keys'], 'o-', linewidth=2, markersize=6, color='red')
        ax2.set_xlabel('Window Size', fontsize=12)
        ax2.set_ylabel('Number of Keys', fontsize=12)
        ax2.set_title('Number of Keys vs Window Size', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='plain', axis='x')
        
        # Add some padding to the plot
        plt.tight_layout()
        
        # Save the plot if requested
        if save_plot:
            plt.savefig('seg_len_plot.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved as 'seg_len_plot.png'")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Plot CSV data with window_size on x-axis and both max_segment_length and num_keys on y-axis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_seg_len.py outputs/merge9999len1e7_p0t2e-1_r6e5t7e5.csv
  python plot_seg_len.py outputs/merge9999len1e7_p0t2e-1_r6e5t7e5.csv --save
        """
    )
    
    parser.add_argument('csv_file', help='Path to the CSV file to plot')
    parser.add_argument('--save', action='store_true', help='Save the plot to seg_len_plot.png')
    
    args = parser.parse_args()
    
    plot_csv_data(args.csv_file, args.save)

if __name__ == "__main__":
    main()

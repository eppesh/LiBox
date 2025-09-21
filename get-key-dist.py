#!/usr/bin/env python3
"""
Script to analyze key distribution in small.csv
- Finds min and max keys
- Partitions space into N equalized buckets (default N=500)
- Counts keys in each bucket
- Draws CDF chart
"""

import sys
import csv
import time
import os
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try different matplotlib backends
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    # Try different backends in order of preference
    backends = ['Agg', 'TkAgg', 'Qt5Agg', 'MacOSX']
    for backend in backends:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            MATPLOTLIB_AVAILABLE = True
            print(f"Using matplotlib backend: {backend}")
            break
        except:
            continue

    if not MATPLOTLIB_AVAILABLE:
        print("Warning: No suitable matplotlib backend found, will use text-based visualization")

except ImportError:
    print("Warning: matplotlib not available, will use text-based visualization")


def read_keys_from_csv(filename):
    """Read numeric keys from CSV file with progress tracking."""
    keys = []

    # Get file size for progress estimation
    try:
        file_size = os.path.getsize(filename)
        print(f"File size: {file_size:,} bytes")
    except OSError:
        file_size = 0

    try:
        start_time = time.time()
        processed_lines = 0
        last_progress_time = start_time

        with open(filename, 'r') as file:
            reader = csv.reader(file)

            for row in reader:
                processed_lines += 1

                # Show progress every 1 second
                current_time = time.time()
                if current_time - last_progress_time >= 1.0:

                    elapsed = current_time - start_time
                    if processed_lines > 0:
                        rate = processed_lines / elapsed
                        # Simple ETA based on lines processed vs file size
                        if file_size > 0:
                            estimated_total_lines = file_size / 12  # rough estimate: ~12 bytes per line
                            eta = (estimated_total_lines - processed_lines) / rate if rate > 0 else 0
                        else:
                            eta = 0

                        print(f"\rProgress: {processed_lines:,} lines processed "
                              f"({rate:.0f} lines/sec, ETA: {eta:.1f}s)", end='', flush=True)

                    last_progress_time = current_time

                if row:  # Skip empty rows
                    try:
                        key = int(row[0].strip())
                        keys.append(key)
                    except ValueError:
                        print(f"\nWarning: Skipping non-numeric value: {row[0]}")
                        continue

        # Final progress update
        total_time = time.time() - start_time
        print(f"\rCompleted: {processed_lines:,} lines processed in {total_time:.2f}s "
              f"({processed_lines/total_time:.0f} lines/sec)")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    return keys


def find_min_max(keys):
    """Find minimum and maximum key values."""
    if not keys:
        print("Error: No valid keys found in the file.")
        sys.exit(1)

    min_key = min(keys)
    max_key = max(keys)
    return min_key, max_key


def create_buckets(min_key, max_key, num_buckets):
    """Create N equalized buckets between min and max."""
    if num_buckets <= 0:
        print("Error: Number of buckets must be positive.")
        sys.exit(1)

    bucket_width = (max_key - min_key) / num_buckets
    buckets = []

    for i in range(num_buckets):
        bucket_start = min_key + i * bucket_width
        bucket_end = min_key + (i + 1) * bucket_width
        buckets.append((bucket_start, bucket_end))

    return buckets, bucket_width


def find_bucket_for_key(key, buckets):
    """Find which bucket a key belongs to."""
    for i, (bucket_start, bucket_end) in enumerate(buckets):
        # For the last bucket, include the upper bound
        if i == len(buckets) - 1:
            if bucket_start <= key <= bucket_end:
                return i
        else:
            if bucket_start <= key < bucket_end:
                return i
    return -1  # Should not happen


def count_keys_chunk(args):
    """Count keys in a chunk for multiprocessing."""
    keys_chunk, buckets, chunk_id = args
    local_counts = [0] * len(buckets)

    for key in keys_chunk:
        bucket_idx = find_bucket_for_key(key, buckets)
        if bucket_idx >= 0:
            local_counts[bucket_idx] += 1

    return chunk_id, local_counts


def count_keys_in_buckets(keys, buckets):
    """Count how many keys fall into each bucket with multiprocessing and progress tracking."""
    total_keys = len(keys)

    if total_keys == 0:
        return [0] * len(buckets)

    # Determine number of processes based on data size and CPU cores
    num_processes = min(multiprocessing.cpu_count(), max(1, total_keys // 10000))

    print(f"Counting {total_keys:,} keys into {len(buckets)} buckets using {num_processes} processes...")
    start_time = time.time()

    # Split keys into chunks for parallel processing
    chunk_size = max(1, total_keys // num_processes)
    key_chunks = [keys[i:i + chunk_size] for i in range(0, total_keys, chunk_size)]

    # Prepare arguments for multiprocessing
    chunk_args = [(chunk, buckets, i) for i, chunk in enumerate(key_chunks)]

    # Process chunks in parallel
    bucket_counts = [0] * len(buckets)
    completed_chunks = 0

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(count_keys_chunk, args): i
            for i, args in enumerate(chunk_args)
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_id, local_counts = future.result()
            completed_chunks += 1

            # Merge local counts into global counts
            for i, count in enumerate(local_counts):
                bucket_counts[i] += count

            # Show progress
            progress_pct = (completed_chunks / len(key_chunks)) * 100
            elapsed = time.time() - start_time
            rate = (completed_chunks * chunk_size) / elapsed if elapsed > 0 else 0

            print(f"\rProgress: {completed_chunks}/{len(key_chunks)} chunks ({progress_pct:.1f}%) "
                  f"({rate:.0f} keys/sec)", end='', flush=True)

    # Final progress update
    total_time = time.time() - start_time
    print(f"\rCompleted: {total_keys:,} keys processed in {total_time:.2f}s "
          f"({total_keys/total_time:.0f} keys/sec) using {num_processes} processes")

    return bucket_counts


def calculate_cdf(bucket_counts):
    """Calculate cumulative distribution function."""
    total_keys = sum(bucket_counts)
    if total_keys == 0:
        return []

    cdf = []
    cumulative = 0

    for count in bucket_counts:
        cumulative += count
        cdf.append(cumulative / total_keys)

    return cdf


def plot_cdf(buckets, bucket_counts, cdf, num_buckets):
    """Draw CDF chart."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, creating text-based visualization...")
        create_text_visualization(buckets, bucket_counts, cdf)
        create_html_visualization(buckets, bucket_counts, cdf)
        save_data_to_csv(buckets, bucket_counts, cdf)
        return

    try:
        # Calculate bucket centers for x-axis
        bucket_centers = []
        for bucket_start, bucket_end in buckets:
            center = (bucket_start + bucket_end) / 2
            bucket_centers.append(center)

        # Create a simple plot without subplots to avoid array issues
        plt.figure(figsize=(12, 6))

        # Plot CDF
        plt.plot(bucket_centers, cdf, 'b-', linewidth=2, label='CDF')
        plt.title('Cumulative Distribution Function (CDF) of Key Distribution')
        plt.xlabel('Key Value')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig('key_distribution_cdf.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create histogram plot
        plt.figure(figsize=(12, 6))

        # Only plot non-empty buckets for better visualization
        non_empty_centers = []
        non_empty_counts = []
        for i, count in enumerate(bucket_counts):
            if count > 0:
                non_empty_centers.append(bucket_centers[i])
                non_empty_counts.append(count)

        if non_empty_centers:
            if len(non_empty_centers) > 1:
                bar_width = (non_empty_centers[1] - non_empty_centers[0]) * 0.8
            else:
                bar_width = 1.0

            plt.bar(non_empty_centers, non_empty_counts, width=bar_width, alpha=0.7)
            plt.title('Key Distribution Histogram (Non-empty buckets only)')
            plt.xlabel('Key Value')
            plt.ylabel('Number of Keys')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('key_distribution_hist.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("Charts saved as 'key_distribution_cdf.png' and 'key_distribution_hist.png'")

    except Exception as e:
        print(f"Warning: Could not create chart due to matplotlib issue: {e}")
        print("Creating alternative visualizations...")
        create_text_visualization(buckets, bucket_counts, cdf)
        create_html_visualization(buckets, bucket_counts, cdf)
        save_data_to_csv(buckets, bucket_counts, cdf)


def create_text_visualization(buckets, bucket_counts, cdf):
    """Create a text-based visualization of the distribution."""
    print("\n" + "="*80)
    print("TEXT-BASED KEY DISTRIBUTION VISUALIZATION")
    print("="*80)

    # Find the maximum count for scaling
    max_count = max(bucket_counts) if bucket_counts else 1
    max_cdf = max(cdf) if cdf else 1

    # Create histogram visualization
    print("\nHISTOGRAM (scaled to 50 characters max):")
    print("-" * 60)

    # Show only non-empty buckets for readability
    non_empty_buckets = [(i, count, cdf_val) for i, (count, cdf_val) in enumerate(zip(bucket_counts, cdf)) if count > 0]

    if len(non_empty_buckets) > 20:
        # Show first 10 and last 10 non-empty buckets
        print("Showing first 10 and last 10 non-empty buckets:")
        for i, (bucket_idx, count, cdf_val) in enumerate(non_empty_buckets[:10]):
            bucket_start, bucket_end = buckets[bucket_idx]
            center = (bucket_start + bucket_end) / 2
            bar_length = int((count / max_count) * 50)
            bar = "█" * bar_length
            print(f"Bucket {bucket_idx:3d}: [{center:12.0f}] {count:3d} keys {bar}")

        print("... (showing middle buckets) ...")

        for i, (bucket_idx, count, cdf_val) in enumerate(non_empty_buckets[-10:]):
            bucket_start, bucket_end = buckets[bucket_idx]
            center = (bucket_start + bucket_end) / 2
            bar_length = int((count / max_count) * 50)
            bar = "█" * bar_length
            print(f"Bucket {bucket_idx:3d}: [{center:12.0f}] {count:3d} keys {bar}")
    else:
        # Show all non-empty buckets
        for bucket_idx, count, cdf_val in non_empty_buckets:
            bucket_start, bucket_end = buckets[bucket_idx]
            center = (bucket_start + bucket_end) / 2
            bar_length = int((count / max_count) * 50)
            bar = "█" * bar_length
            print(f"Bucket {bucket_idx:3d}: [{center:12.0f}] {count:3d} keys {bar}")

    # Create CDF visualization
    print("\nCUMULATIVE DISTRIBUTION FUNCTION (CDF):")
    print("-" * 60)

    # Show CDF at key percentiles
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    for p in percentiles:
        # Find bucket where CDF reaches this percentile
        for i, cdf_val in enumerate(cdf):
            if cdf_val >= p:
                bucket_start, bucket_end = buckets[i]
                center = (bucket_start + bucket_end) / 2
                print(f"{p*100:5.1f}%: Key value ≈ {center:12.0f}")
                break

    print("\n" + "="*80)


def create_html_visualization(buckets, bucket_counts, cdf):
    """Create an HTML-based visualization that can be opened in a browser."""
    print("Generating HTML visualization...")
    start_time = time.time()

    # Calculate bucket centers
    bucket_centers = []
    for bucket_start, bucket_end in buckets:
        center = (bucket_start + bucket_end) / 2
        bucket_centers.append(center)

    # Find max count for scaling
    max_count = max(bucket_counts) if bucket_counts else 1

    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Key Distribution Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .chart-container {{ margin: 20px 0; }}
        h1, h2 {{ color: #333; }}
        .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Key Distribution Analysis</h1>

        <div class="stats">
            <h2>Statistics</h2>
            <p><strong>Total keys:</strong> {sum(bucket_counts)}</p>
            <p><strong>Number of buckets:</strong> {len(buckets)}</p>
            <p><strong>Non-empty buckets:</strong> {sum(1 for count in bucket_counts if count > 0)}</p>
            <p><strong>Max count in any bucket:</strong> {max_count}</p>
        </div>

        <div class="chart-container">
            <h2>Cumulative Distribution Function (CDF)</h2>
            <canvas id="cdfChart" width="800" height="400"></canvas>
        </div>

        <div class="chart-container">
            <h2>Key Distribution Histogram</h2>
            <canvas id="histChart" width="800" height="400"></canvas>
        </div>
    </div>

    <script>
        // CDF Chart
        const cdfCtx = document.getElementById('cdfChart').getContext('2d');
        const cdfData = {{
            labels: {[f"{center:.0f}" for center in bucket_centers]},
            datasets: [{{
                label: 'CDF',
                data: {cdf},
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }}]
        }};

        new Chart(cdfCtx, {{
            type: 'line',
            data: cdfData,
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Key Value'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Cumulative Probability'
                        }},
                        min: 0,
                        max: 1
                    }}
                }}
            }}
        }});

        // Histogram Chart
        const histCtx = document.getElementById('histChart').getContext('2d');
        const histData = {{
            labels: {[f"{center:.0f}" for center in bucket_centers]},
            datasets: [{{
                label: 'Key Count',
                data: {bucket_counts},
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }}]
        }};

        new Chart(histCtx, {{
            type: 'bar',
            data: histData,
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Key Value'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Number of Keys'
                        }},
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    # Write HTML file
    with open('key_distribution.html', 'w') as f:
        f.write(html_content)

    html_time = time.time() - start_time
    print(f"HTML visualization saved as 'key_distribution.html' (took {html_time:.2f}s)")
    print("Open this file in your web browser to view interactive charts!")


def save_data_to_csv(buckets, bucket_counts, cdf):
    """Save distribution data to CSV files as fallback."""
    # Save bucket data
    with open('bucket_data.csv', 'w') as f:
        f.write('bucket_start,bucket_end,bucket_center,count,cdf\n')
        for i, ((start, end), count, cdf_val) in enumerate(zip(buckets, bucket_counts, cdf)):
            center = (start + end) / 2
            f.write(f'{start},{end},{center},{count},{cdf_val}\n')

    print("Data saved to 'bucket_data.csv'")
    print("You can use this data to create charts with other tools like Excel, R, or online plotting tools.")


def print_statistics(min_key, max_key, bucket_counts, num_buckets):
    """Print analysis statistics."""
    total_keys = sum(bucket_counts)
    non_empty_buckets = sum(1 for count in bucket_counts if count > 0)

    print(f"\n=== Key Distribution Analysis ===")
    print(f"Total keys: {total_keys}")
    print(f"Min key: {min_key}")
    print(f"Max key: {max_key}")
    print(f"Key range: {max_key - min_key}")
    print(f"Number of buckets: {num_buckets}")
    print(f"Non-empty buckets: {non_empty_buckets}")
    print(f"Bucket width: {(max_key - min_key) / num_buckets:.2f}")

    # Find bucket with most keys
    max_count = max(bucket_counts)
    max_bucket_idx = bucket_counts.index(max_count)
    print(f"Bucket with most keys: {max_bucket_idx} (count: {max_count})")

    # Calculate some percentiles
    sorted_counts = sorted(bucket_counts, reverse=True)
    p50_idx = int(0.5 * len(sorted_counts))
    p90_idx = int(0.9 * len(sorted_counts))
    p99_idx = int(0.99 * len(sorted_counts))

    print(f"50th percentile bucket count: {sorted_counts[p50_idx]}")
    print(f"90th percentile bucket count: {sorted_counts[p90_idx]}")
    print(f"99th percentile bucket count: {sorted_counts[p99_idx]}")


def read_bucket_data_from_csv(filename):
    """Read bucket data from CSV file generated by count-key.cpp."""
    buckets = []
    bucket_counts = []
    cdf = []

    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header

            for row in reader:
                if row:  # Skip empty rows
                    bucket_start = float(row[0])
                    bucket_end = float(row[1])
                    bucket_center = float(row[2])
                    count = int(row[3])
                    cdf_val = float(row[4])

                    buckets.append((bucket_start, bucket_end))
                    bucket_counts.append(count)
                    cdf.append(cdf_val)

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading bucket data file: {e}")
        sys.exit(1)

    return buckets, bucket_counts, cdf


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python get-key-dist.py <csv_file> <min_key> <max_key> [num_buckets] [skip_lines]")
        print("Example: python get-key-dist.py small.csv -2000000000 2000000000 500")
        print("Example: python get-key-dist.py small.csv -2000000000 2000000000 500 1")
        print("Note: This script now uses count-key.cpp for efficient processing")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        min_key = int(sys.argv[2])
        max_key = int(sys.argv[3])
    except ValueError:
        print("Error: min_key and max_key must be integers.")
        sys.exit(1)

    if min_key >= max_key:
        print("Error: min_key must be less than max_key.")
        sys.exit(1)

    num_buckets = 500  # default value
    skip_lines = 0  # default value

    if len(sys.argv) > 4:
        try:
            num_buckets = int(sys.argv[4])
        except ValueError:
            print("Error: Number of buckets must be an integer.")
            sys.exit(1)

    if len(sys.argv) > 5:
        try:
            skip_lines = int(sys.argv[5])
        except ValueError:
            print("Error: Number of lines to skip must be an integer.")
            sys.exit(1)

        if skip_lines < 0:
            print("Error: Number of lines to skip must be non-negative.")
            sys.exit(1)

    print(f"Analyzing key distribution in '{filename}' with {num_buckets} buckets...")
    print(f"Key range: [{min_key:,}, {max_key:,}]")
    if skip_lines > 0:
        print(f"Skipping first {skip_lines} lines")
    overall_start_time = time.time()

    # Check if count-key binary exists
    import subprocess
    import shutil

    count_key_binary = "./count-key"
    if not shutil.which(count_key_binary):
        print(f"Error: {count_key_binary} binary not found. Please run 'make count-key' first.")
        sys.exit(1)

    # Run count-key.cpp to generate bucket data
    print("Running count-key.cpp for efficient processing...")
    try:
        # Build command with skip_lines parameter
        cmd = [count_key_binary, filename, str(min_key), str(max_key), str(num_buckets)]
        # Always add output file parameter (use default "bucket_data.csv")
        cmd.append("bucket_data.csv")
        if skip_lines > 0:
            cmd.append(str(skip_lines))

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running count-key: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

    # Read bucket data generated by count-key.cpp
    buckets, bucket_counts, cdf = read_bucket_data_from_csv("bucket_data.csv")

    # Print statistics
    print_statistics(min_key, max_key, bucket_counts, num_buckets)

    # Plot CDF
    print("Generating CDF chart...")
    plot_cdf(buckets, bucket_counts, cdf, num_buckets)

    # Final timing summary
    total_time = time.time() - overall_start_time
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Files generated:")
    print(f"  - bucket_data.csv (raw data)")
    print(f"  - key_distribution.html (interactive charts)")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Required for multiprocessing on Windows/macOS
    multiprocessing.freeze_support()
    main()

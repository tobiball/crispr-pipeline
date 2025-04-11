#!/usr/bin/env python
"""
Module for analyzing and visualizing gRNA efficacy distributions.
Can be used as a standalone script or imported into other modules.

This version is designed to work with simplified CSV input that only contains
efficacy values, without requiring any other guide information.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy.stats import norm
from typing import Tuple, List, Dict, Optional

try:
    import polars as pl
except ImportError:
    print("Polars not installed. Falling back to pandas.")
    import pandas as pd
    HAS_POLARS = False
else:
    HAS_POLARS = True

# Set default thresholds (can be modified when calling functions)
POOR_THRESHOLD = 50.0
MODERATE_THRESHOLD = 75.0

def analyze_efficacy_distribution(
        data_source,  # Can be DataFrame, path to CSV, or numpy array
        efficacy_column: str = "efficacy",
        output_dir: str = "./efficacy_analysis",
        poor_threshold: float = POOR_THRESHOLD,
        moderate_threshold: float = MODERATE_THRESHOLD
) -> Optional[np.ndarray]:
    """
    Main function to analyze and visualize efficacy distribution.

    Args:
        data_source: DataFrame, path to CSV, or numpy array of efficacy values
        efficacy_column: Name of the column containing efficacy scores
        output_dir: Directory to save visualization outputs
        poor_threshold: Threshold below which guides are considered 'poor'
        moderate_threshold: Threshold below which guides are considered 'moderate'

    Returns:
        Array of efficacy values
    """
    # Set the global thresholds to match the function parameters
    global POOR_THRESHOLD, MODERATE_THRESHOLD
    POOR_THRESHOLD = poor_threshold
    MODERATE_THRESHOLD = moderate_threshold

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process the input data source to get efficacy values
    efficacy_vals = extract_efficacy_values(data_source, efficacy_column)

    if efficacy_vals is None or len(efficacy_vals) == 0:
        print("No valid efficacy data to analyze")
        return None

    print("\n=== EFFICACY DISTRIBUTION ANALYSIS ===\n")

    # Print summary statistics
    print_summary_statistics(efficacy_vals)

    # Create visualizations
    visualize_distribution(efficacy_vals, output_dir)
    visualize_categories(efficacy_vals, output_dir)
    visualize_skewness(efficacy_vals, output_dir)

    # Generate summary report
    generate_summary_report(efficacy_vals, output_dir)

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return efficacy_vals

def extract_efficacy_values(data_source, efficacy_column="efficacy"):
    """Extract efficacy values from various input types"""
    # If it's already a numpy array
    if isinstance(data_source, np.ndarray):
        return data_source

    # If it's a string, assume it's a file path
    if isinstance(data_source, str):
        try:
            if HAS_POLARS:
                print(f"Loading data from {data_source} using polars")
                try:
                    df = pl.read_csv(data_source)
                except:
                    df = pl.read_csv(data_source, separator='\t')

                if efficacy_column in df.columns:
                    return df[efficacy_column].to_numpy()
                else:
                    print(f"Column '{efficacy_column}' not found in {data_source}")
                    print(f"Available columns: {df.columns}")
                    return None
            else:
                print(f"Loading data from {data_source} using pandas")
                try:
                    df = pd.read_csv(data_source)
                except:
                    df = pd.read_csv(data_source, sep='\t')

                if efficacy_column in df.columns:
                    return df[efficacy_column].values
                else:
                    print(f"Column '{efficacy_column}' not found in {data_source}")
                    print(f"Available columns: {df.columns}")
                    return None
        except Exception as e:
            print(f"Error loading data from {data_source}: {e}")
            return None

    # If it's a DataFrame
    if HAS_POLARS and isinstance(data_source, pl.DataFrame):
        if efficacy_column in data_source.columns:
            return data_source[efficacy_column].to_numpy()
    elif 'pandas' in sys.modules and isinstance(data_source, pd.DataFrame):
        if efficacy_column in data_source.columns:
            return data_source[efficacy_column].values

    print(f"Unsupported data source type: {type(data_source)}")
    return None

def print_summary_statistics(efficacy_vals: np.ndarray) -> None:
    """Print summary statistics for efficacy values"""
    if len(efficacy_vals) == 0:
        print("No efficacy data available")
        return

    # Calculate statistics
    stats = {
        "count": len(efficacy_vals),
        "min": np.min(efficacy_vals),
        "max": np.max(efficacy_vals),
        "mean": np.mean(efficacy_vals),
        "median": np.median(efficacy_vals),
        "std": np.std(efficacy_vals),
        "poor_count": np.sum(efficacy_vals < POOR_THRESHOLD),
        "poor_pct": 100 * np.mean(efficacy_vals < POOR_THRESHOLD),
        "moderate_count": np.sum((efficacy_vals >= POOR_THRESHOLD) & (efficacy_vals < MODERATE_THRESHOLD)),
        "moderate_pct": 100 * np.mean((efficacy_vals >= POOR_THRESHOLD) & (efficacy_vals < MODERATE_THRESHOLD)),
        "high_count": np.sum(efficacy_vals >= MODERATE_THRESHOLD),
        "high_pct": 100 * np.mean(efficacy_vals >= MODERATE_THRESHOLD)
    }

    print("DATASET SUMMARY:")
    print(f"Total guides: {stats['count']}")
    print(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print("\nGUIDE CATEGORIES:")
    print(f"  Poor (<{POOR_THRESHOLD}): {stats['poor_count']} guides ({stats['poor_pct']:.1f}%)")
    print(f"  Moderate ({POOR_THRESHOLD}-{MODERATE_THRESHOLD}): {stats['moderate_count']} guides ({stats['moderate_pct']:.1f}%)")
    print(f"  High (>{MODERATE_THRESHOLD}): {stats['high_count']} guides ({stats['high_pct']:.1f}%)")

    # Check for skewness
    skewness = (stats['mean'] - stats['median']) / stats['std']
    if abs(skewness) > 0.2:
        print(f"\nNOTE: Distribution is {'negatively' if skewness < 0 else 'positively'} skewed ({skewness:.2f})")
        if skewness < 0:
            print("The negative skew suggests more high-efficacy guides than a normal distribution would predict.")
        else:
            print("The positive skew suggests more low-efficacy guides than a normal distribution would predict.")

def visualize_distribution(efficacy_vals: np.ndarray, output_dir: str) -> None:
    """Create histogram showing distribution of efficacy values with category boundaries"""
    plt.figure(figsize=(12, 7))

    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 14})

    # Create histogram with appropriate bins
    n_bins = min(30, int(len(efficacy_vals) / 20))  # Adjust bin count based on data size

    # Define colors for each category
    colors = ['#d32f2f', '#f57c00', '#388e3c']  # Red, Orange, Green

    # Calculate bin edges without drawing histogram
    counts, bin_edges = np.histogram(efficacy_vals, bins=n_bins)
    bin_width = bin_edges[1] - bin_edges[0]

    # Split data into categories
    poor_mask = efficacy_vals < POOR_THRESHOLD
    moderate_mask = (efficacy_vals >= POOR_THRESHOLD) & (efficacy_vals < MODERATE_THRESHOLD)
    high_mask = efficacy_vals >= MODERATE_THRESHOLD

    # Calculate counts and percentages
    poor_count = np.sum(poor_mask)
    moderate_count = np.sum(moderate_mask)
    high_count = np.sum(high_mask)

    poor_pct = 100 * poor_count / len(efficacy_vals)
    moderate_pct = 100 * moderate_count / len(efficacy_vals)
    high_pct = 100 * high_count / len(efficacy_vals)

    # Create separate histograms for each category with updated labels that include counts
    plt.hist(efficacy_vals[poor_mask], bins=bin_edges, color=colors[0],
             alpha=0.7, edgecolor='black', linewidth=0.5,
             label=f'Poor (<{POOR_THRESHOLD}): {poor_count:,d} guides ({poor_pct:.1f}%)')
    plt.hist(efficacy_vals[moderate_mask], bins=bin_edges, color=colors[1],
             alpha=0.7, edgecolor='black', linewidth=0.5,
             label=f'Moderate ({POOR_THRESHOLD}-{MODERATE_THRESHOLD}): {moderate_count:,d} guides ({moderate_pct:.1f}%)')
    plt.hist(efficacy_vals[high_mask], bins=bin_edges, color=colors[2],
             alpha=0.7, edgecolor='black', linewidth=0.5,
             label=f'High (>{MODERATE_THRESHOLD}): {high_count:,d} guides ({high_pct:.1f}%)')

    # Find max height for text positioning
    max_height = plt.gca().get_ylim()[1]

    # Add vertical lines for category boundaries
    plt.axvline(x=POOR_THRESHOLD, color='black', linestyle='--')
    plt.axvline(x=MODERATE_THRESHOLD, color='black', linestyle='--')

    # Add boundary values as text
    plt.text(POOR_THRESHOLD, max_height * 0.8, f"{POOR_THRESHOLD}",
             horizontalalignment='center', verticalalignment='center')
    plt.text(MODERATE_THRESHOLD, max_height * 0.8, f"{MODERATE_THRESHOLD}",
             horizontalalignment='center', verticalalignment='center')

    # Labels and title
    plt.xlabel('Efficacy Score', fontsize=16)
    plt.ylabel('Number of Guides', fontsize=16)
    plt.title('Distribution of gRNA Efficacy Scores with Categories', fontsize=18, fontweight='bold')

    # Add a total count to the legend
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
               title=f"Total: {len(efficacy_vals):,d} guides")

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficacy_distribution.png'), dpi=300)
    plt.close()

    print(f"Created distribution histogram: {os.path.join(output_dir, 'efficacy_distribution.png')}")
def visualize_categories(efficacy_vals: np.ndarray, output_dir: str) -> None:
    """Create a stacked bar chart showing the percentage of guides in each category"""
    plt.figure(figsize=(10, 6))

    # Set up the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 14})

    # Calculate percentages
    poor_pct = 100 * np.mean(efficacy_vals < POOR_THRESHOLD)
    moderate_pct = 100 * np.mean((efficacy_vals >= POOR_THRESHOLD) & (efficacy_vals < MODERATE_THRESHOLD))
    high_pct = 100 * np.mean(efficacy_vals >= MODERATE_THRESHOLD)

    # Define colors for each category
    colors = ['#d32f2f', '#f57c00', '#388e3c']  # Red, Orange, Green

    # Create bar chart
    categories = ['Guide Efficacy']
    plt.bar(categories, [poor_pct], color=colors[0], label=f'Poor (<{POOR_THRESHOLD})')
    plt.bar(categories, [moderate_pct], bottom=[poor_pct], color=colors[1], label=f'Moderate ({POOR_THRESHOLD}-{MODERATE_THRESHOLD})')
    plt.bar(categories, [high_pct], bottom=[poor_pct + moderate_pct], color=colors[2], label=f'High (>{MODERATE_THRESHOLD})')

    # Add percentage labels on each segment
    plt.text(0, poor_pct/2, f'{poor_pct:.1f}%', ha='center', va='center', fontsize=14, color='white', fontweight='bold')
    plt.text(0, poor_pct + moderate_pct/2, f'{moderate_pct:.1f}%', ha='center', va='center', fontsize=14, color='black', fontweight='bold')
    plt.text(0, poor_pct + moderate_pct + high_pct/2, f'{high_pct:.1f}%', ha='center', va='center', fontsize=14, color='white', fontweight='bold')

    # Add count information
    poor_count = int(np.sum(efficacy_vals < POOR_THRESHOLD))
    moderate_count = int(np.sum((efficacy_vals >= POOR_THRESHOLD) & (efficacy_vals < MODERATE_THRESHOLD)))
    high_count = int(np.sum(efficacy_vals >= MODERATE_THRESHOLD))

    # Add a table with raw counts
    table_text = f"""
    Category    | Count    | Percentage
    ------------------------------------
    Poor        | {poor_count:,d}      | {poor_pct:.1f}%
    Moderate    | {moderate_count:,d}      | {moderate_pct:.1f}%
    High        | {high_count:,d}      | {high_pct:.1f}%
    ------------------------------------
    Total       | {len(efficacy_vals):,d}      | 100.0%
    """

    plt.figtext(0.15, 0.01, table_text, fontsize=10, family='monospace')

    # Labels and title
    plt.ylabel('Percentage of Guides (%)', fontsize=16)
    plt.title('Guide Distribution by Efficacy Category', fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 10))
    plt.gca().yaxis.set_major_formatter(PercentFormatter())

    # Legend
    plt.legend(fontsize=12, loc='upper right')

    # Save the figure
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])  # Adjust layout to make room for the table
    plt.savefig(os.path.join(output_dir, 'category_distribution.png'), dpi=300)
    plt.close()

    print(f"Created category distribution chart: {os.path.join(output_dir, 'category_distribution.png')}")

def visualize_skewness(efficacy_vals: np.ndarray, output_dir: str) -> None:
    """Create visualizations showing the skewed nature of the distribution"""
    # Calculate skewness metrics
    skewness = {
        "statistical_skewness": float(np.mean(((efficacy_vals - np.mean(efficacy_vals)) / np.std(efficacy_vals)) ** 3)),
        "median_mean_diff": float(np.median(efficacy_vals) - np.mean(efficacy_vals)),
        "poor_to_high_ratio": float(np.sum(efficacy_vals < POOR_THRESHOLD) / max(1, np.sum(efficacy_vals >= MODERATE_THRESHOLD))),
        "quartile_skewness": float((np.percentile(efficacy_vals, 75) - np.percentile(efficacy_vals, 50)) /
                                   (np.percentile(efficacy_vals, 50) - np.percentile(efficacy_vals, 25)))
    }

    # Create a 1x2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Plot 1: Distribution with normal curve overlay
    sns.histplot(efficacy_vals, kde=False, bins=30, ax=ax1, color='skyblue', edgecolor='black', alpha=0.7)

    # Add normal curve for comparison
    x = np.linspace(min(efficacy_vals), max(efficacy_vals), 100)
    y = norm.pdf(x, np.mean(efficacy_vals), np.std(efficacy_vals))
    y = y * (len(efficacy_vals) * (max(efficacy_vals) - min(efficacy_vals)) / 30)  # Scale to match histogram
    ax1.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')

    # Add vertical lines for mean and median
    ax1.axvline(np.mean(efficacy_vals), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(efficacy_vals):.1f}')
    ax1.axvline(np.median(efficacy_vals), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(efficacy_vals):.1f}')

    # Add category boundaries
    ax1.axvline(POOR_THRESHOLD, color='black', linestyle='-', linewidth=1)
    ax1.axvline(MODERATE_THRESHOLD, color='black', linestyle='-', linewidth=1)

    # Labels
    ax1.set_xlabel('Efficacy Score', fontsize=14)
    ax1.set_ylabel('Number of Guides', fontsize=14)
    ax1.set_title('Efficacy Distribution vs. Normal Distribution', fontsize=16)
    ax1.legend(fontsize=12)

    # Plot 2: Q-Q Plot
    from scipy import stats

    # Calculate theoretical quantiles from a normal distribution
    theoretical_quantiles = np.linspace(0.01, 0.99, 100)
    theoretical_values = stats.norm.ppf(theoretical_quantiles, loc=np.mean(efficacy_vals), scale=np.std(efficacy_vals))

    # Calculate empirical quantiles from the data
    empirical_values = np.percentile(efficacy_vals, theoretical_quantiles * 100)

    # Create Q-Q plot
    ax2.scatter(theoretical_values, empirical_values, alpha=0.7, color='darkblue')

    # Add reference line
    min_val = min(np.min(theoretical_values), np.min(empirical_values))
    max_val = max(np.max(theoretical_values), np.max(empirical_values))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Normal Reference')

    # Labels
    ax2.set_xlabel('Theoretical Quantiles (Normal Distribution)', fontsize=14)
    ax2.set_ylabel('Empirical Quantiles (Our Data)', fontsize=14)
    ax2.set_title('Q-Q Plot: Efficacy Scores vs. Normal Distribution', fontsize=16)
    ax2.legend(fontsize=12)

    # Add skewness text annotation
    skew_text = (
        f"Skewness Metrics:\n"
        f"Statistical skewness: {skewness['statistical_skewness']:.3f}\n"
        f"Median-mean diff: {skewness['median_mean_diff']:.3f}\n"
        f"Poor:High ratio: 1:{1/skewness['poor_to_high_ratio']:.1f}\n"
        f"Data is skewed toward {'higher' if skewness['statistical_skewness'] < 0 else 'lower'} values"
    )

    plt.figtext(0.5, 0.01, skew_text, ha='center', fontsize=12, bbox=dict(facecolor='lavender', alpha=0.5))

    # Tight layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(os.path.join(output_dir, 'distribution_skewness.png'), dpi=300)
    plt.close()

    print(f"Created skewness analysis visualization: {os.path.join(output_dir, 'distribution_skewness.png')}")

def generate_summary_report(efficacy_vals: np.ndarray, output_dir: str) -> None:
    """Generate a summary report with key findings and statistics"""
    # Calculate key metrics
    poor_pct = 100 * np.mean(efficacy_vals < POOR_THRESHOLD)
    moderate_pct = 100 * np.mean((efficacy_vals >= POOR_THRESHOLD) & (efficacy_vals < MODERATE_THRESHOLD))
    high_pct = 100 * np.mean(efficacy_vals >= MODERATE_THRESHOLD)

    poor_count = int(np.sum(efficacy_vals < POOR_THRESHOLD))
    moderate_count = int(np.sum((efficacy_vals >= POOR_THRESHOLD) & (efficacy_vals < MODERATE_THRESHOLD)))
    high_count = int(np.sum(efficacy_vals >= MODERATE_THRESHOLD))

    skewness = np.mean(((efficacy_vals - np.mean(efficacy_vals)) / np.std(efficacy_vals)) ** 3)

    # Write the report
    report_path = os.path.join(output_dir, "efficacy_analysis_summary.txt")
    with open(report_path, 'w') as f:
        f.write("=== GUIDE EFFICACY DISTRIBUTION ANALYSIS ===\n\n")

        f.write("DATASET SUMMARY:\n")
        f.write(f"Total guides analyzed: {len(efficacy_vals)}\n")
        f.write(f"Efficacy range: {np.min(efficacy_vals):.2f} to {np.max(efficacy_vals):.2f}\n")
        f.write(f"Mean efficacy: {np.mean(efficacy_vals):.2f}\n")
        f.write(f"Median efficacy: {np.median(efficacy_vals):.2f}\n")
        f.write(f"Standard deviation: {np.std(efficacy_vals):.2f}\n\n")

        f.write("GUIDE CATEGORIES:\n")
        f.write(f"  Poor guides (<{POOR_THRESHOLD}): {poor_count} guides ({poor_pct:.1f}%)\n")
        f.write(f"  Moderate guides ({POOR_THRESHOLD}-{MODERATE_THRESHOLD}): {moderate_count} guides ({moderate_pct:.1f}%)\n")
        f.write(f"  High guides (>{MODERATE_THRESHOLD}): {high_count} guides ({high_pct:.1f}%)\n\n")

        f.write("DISTRIBUTION CHARACTERISTICS:\n")
        if skewness < -0.5:
            skew_desc = "strongly negatively skewed (more high efficacy guides than expected)"
        elif skewness < -0.1:
            skew_desc = "negatively skewed (slightly more high efficacy guides than expected)"
        elif skewness > 0.5:
            skew_desc = "strongly positively skewed (more low efficacy guides than expected)"
        elif skewness > 0.1:
            skew_desc = "positively skewed (slightly more low efficacy guides than expected)"
        else:
            skew_desc = "approximately symmetric"

        f.write(f"  The distribution is {skew_desc}\n")
        f.write(f"  Statistical skewness: {skewness:.3f}\n")
        f.write(f"  Mean-median difference: {np.mean(efficacy_vals) - np.median(efficacy_vals):.3f}\n\n")

        f.write("IMPLICATIONS FOR ANALYSIS:\n")
        if poor_pct < 20:
            f.write("  - Dataset has relatively few poor-performing guides, which creates an imbalance\n")
            f.write("  - Standard classification metrics may be misleading due to class imbalance\n")
            f.write("  - Consider using precision-recall curves instead of ROC curves\n")
            f.write("  - Within-gene analysis becomes especially important to identify rare poor guides\n")
        else:
            f.write("  - Dataset has a reasonable balance of guide categories\n")
            f.write("  - Standard evaluation metrics should be appropriate\n")

        f.write("\nVISUALIZATIONS CREATED:\n")
        f.write(f"  - {os.path.join(output_dir, 'efficacy_distribution.png')}\n")
        f.write(f"  - {os.path.join(output_dir, 'category_distribution.png')}\n")
        f.write(f"  - {os.path.join(output_dir, 'distribution_skewness.png')}\n")

    print(f"Generated summary report: {report_path}")

def parse_args():
    """Parse command line arguments when run as a script"""
    parser = argparse.ArgumentParser(description="Analyze gRNA efficacy distribution")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input CSV file with efficacy data")
    parser.add_argument("--output-dir", "-o", type=str, default="./efficacy_analysis",
                        help="Directory to save visualization outputs")
    parser.add_argument("--efficacy-column", type=str, default="efficacy",
                        help="Name of the column containing efficacy scores")
    parser.add_argument("--poor-threshold", type=float, default=POOR_THRESHOLD,
                        help="Threshold below which guides are considered 'poor'")
    parser.add_argument("--moderate-threshold", type=float, default=MODERATE_THRESHOLD,
                        help="Threshold below which guides are considered 'moderate'")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(f"Analyzing efficacy distribution from {args.input}")
    print(f"Output will be saved to {args.output_dir}")
    print(f"Using efficacy column: {args.efficacy_column}")
    print(f"Category thresholds: Poor <{args.poor_threshold}, Moderate <{args.moderate_threshold}, High ≥{args.moderate_threshold}")

    # Run the analysis
    result = analyze_efficacy_distribution(
        data_source=args.input,
        efficacy_column=args.efficacy_column,
        output_dir=args.output_dir,
        poor_threshold=args.poor_threshold,
        moderate_threshold=args.moderate_threshold
    )

    if result is not None:
        print("\nAnalysis completed successfully!")
        sys.exit(0)
    else:
        print("\nAnalysis failed.")
        sys.exit(1)
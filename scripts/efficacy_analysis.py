#!/usr/bin/env python
"""
Module for analyzing and visualizing gRNA efficacy distributions.
Can be used as a standalone script or imported into other modules.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy.stats import norm
from typing import Optional, Union

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    import pandas as pd


def analyze_efficacy_distribution(
        data_source: Union[str, np.ndarray],
        efficacy_column: str,
        output_dir: str,
        poor_threshold: float,
        moderate_threshold: float
) -> Optional[np.ndarray]:
    """
    Analyze and visualize efficacy distribution.
    Returns array of values or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    vals = extract_efficacy_values(data_source, efficacy_column)
    if vals is None or vals.size == 0:
        print("No valid efficacy data to analyze")
        return None

    print("\n=== EFFICACY DISTRIBUTION ANALYSIS ===\n")
    print_summary_statistics(vals, poor_threshold, moderate_threshold)
    visualize_distribution(vals, output_dir, poor_threshold, moderate_threshold)
    visualize_categories(vals, output_dir, poor_threshold, moderate_threshold)
    visualize_skewness(vals, output_dir, poor_threshold, moderate_threshold)
    generate_summary_report(vals, output_dir, poor_threshold, moderate_threshold)

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return vals


def extract_efficacy_values(data_source: Union[str, np.ndarray], efficacy_column: str) -> Optional[np.ndarray]:
    """Extract efficacy values from array, CSV path, or DataFrame"""
    if isinstance(data_source, np.ndarray):
        return data_source
    if isinstance(data_source, str):
        try:
            if HAS_POLARS:
                df = pl.read_csv(data_source)
                arr = df[efficacy_column].to_numpy()
            else:
                df = pd.read_csv(data_source)
                arr = df[efficacy_column].values
            return arr
        except Exception as e:
            print(f"Error reading '{data_source}': {e}")
            return None
    # DataFrame check
    if HAS_POLARS and isinstance(data_source, pl.DataFrame):
        return data_source[efficacy_column].to_numpy() if efficacy_column in data_source.columns else None
    if not HAS_POLARS:
        import pandas as pd
        if isinstance(data_source, pd.DataFrame) and efficacy_column in data_source.columns:
            return data_source[efficacy_column].values
    print(f"Unsupported data source: {type(data_source)}")
    return None


def print_summary_statistics(vals: np.ndarray, poor_th: float, mod_th: float) -> None:
    count = vals.size
    mn, mx = vals.min(), vals.max()
    mean, med, std = vals.mean(), np.median(vals), vals.std()
    poor_cnt = np.sum(vals < poor_th)
    mod_cnt = np.sum((vals >= poor_th) & (vals < mod_th))
    high_cnt = np.sum(vals >= mod_th)

    print(f"Total guides: {count}")
    print(f"Range: {mn:.2f} to {mx:.2f}")
    print(f"Mean: {mean:.2f}, Median: {med:.2f}, Std: {std:.2f}\n")
    print(f"Poor (<{poor_th}): {poor_cnt} ({100*poor_cnt/count:.1f}%)")
    print(f"Moderate ({poor_th}-{mod_th}): {mod_cnt} ({100*mod_cnt/count:.1f}%)")
    print(f"High (>{mod_th}): {high_cnt} ({100*high_cnt/count:.1f}%)")

    skew = (mean - med) / std if std > 0 else 0.0
    if abs(skew) > 0.2:
        direction = 'negatively' if skew < 0 else 'positively'
        print(f"\nNOTE: Distribution is {direction} skewed ({skew:.2f})")


def visualize_distribution(vals: np.ndarray, out: str, poor_th: float, mod_th: float) -> None:
    plt.figure(figsize=(10,6))
    sns.set_style('whitegrid')
    bins = np.histogram_bin_edges(vals, bins=min(30, max(10, vals.size//20)))

    masks = [(vals<poor_th), ((vals>=poor_th)&(vals<mod_th)), (vals>=mod_th)]
    colors = ['#d32f2f','#f57c00','#388e3c']
    labels = [f'Poor (<{poor_th})', f'Moderate ({poor_th}-{mod_th})', f'High (>{mod_th})']

    for mask,color,label in zip(masks,colors,labels):
        plt.hist(vals[mask], bins=bins, color=color, alpha=0.7, label=f"{label}: {np.sum(mask)} ({100*np.mean(mask):.1f}%)")

    plt.axvline(poor_th, linestyle='--', color='black')
    plt.axvline(mod_th, linestyle='--', color='black')
    plt.xlabel('Efficacy Score')
    plt.ylabel('Count')
    plt.title('Efficacy Distribution')
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out, 'efficacy_distribution.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def visualize_categories(vals: np.ndarray, out: str, poor_th: float, mod_th: float) -> None:
    """Create a stacked bar chart of guide categories"""
    poor_pct = 100 * np.mean(vals < poor_th)
    mod_pct = 100 * np.mean((vals >= poor_th) & (vals < mod_th))
    high_pct = 100 * np.mean(vals >= mod_th)

    # Define category colors
    colors = ['#d32f2f', '#f57c00', '#388e3c']

    # Plot stacked bar
    plt.figure(figsize=(6,6))
    sns.set_style('whitegrid')
    plt.bar(['Guides'], [poor_pct], color=colors[0], label=f'Poor (<{poor_th})')
    plt.bar(['Guides'], [mod_pct], bottom=[poor_pct], color=colors[1], label=f'Moderate ({poor_th}-{mod_th})')
    plt.bar(['Guides'], [high_pct], bottom=[poor_pct + mod_pct], color=colors[2], label=f'High (>{mod_th})')

    plt.ylabel('Percentage (%)')
    plt.title('Guide Category Distribution')
    plt.legend(title='Category')
    plt.tight_layout()
    path = os.path.join(out, 'category_distribution.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def visualize_skewness(vals: np.ndarray, out: str, poor_th: float, mod_th: float) -> None:
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    sns.histplot(vals, bins=30, ax=axes[0], kde=False)
    x = np.linspace(vals.min(), vals.max(), 100)
    y = norm.pdf(x, vals.mean(), vals.std())*vals.size*(x[1]-x[0])
    axes[0].plot(x, y, 'r--')
    axes[0].set_title('Distribution vs Normal')

    from scipy import stats
    theor_q = np.linspace(0.01,0.99,100)
    theor_vals = stats.norm.ppf(theor_q, loc=vals.mean(), scale=vals.std())
    emp_vals = np.percentile(vals, theor_q*100)
    axes[1].scatter(theor_vals, emp_vals)
    axes[1].plot([theor_vals.min(), theor_vals.max()],[theor_vals.min(), theor_vals.max()],'r--')
    axes[1].set_title('Q-Q Plot')

    plt.tight_layout()
    path = os.path.join(out, 'distribution_skewness.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def generate_summary_report(vals: np.ndarray, out: str, poor_th: float, mod_th: float) -> None:
    report = os.path.join(out, 'summary.txt')
    with open(report,'w') as f:
        f.write(f"Total: {vals.size}\nPoor<{poor_th}: {np.sum(vals<poor_th)}\nModerate<{mod_th}: {np.sum((vals>=poor_th)&(vals<mod_th))}\nHigh>={mod_th}: {np.sum(vals>=mod_th)}\n")
    print(f"Saved {report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gRNA efficacy distribution")
    parser.add_argument('-i','--input', required=True, help='CSV file path')
    parser.add_argument('-o','--output-dir', default='./efficacy_analysis', help='Output directory')
    parser.add_argument('--efficacy-column', default='efficacy', help='Column name')
    parser.add_argument('--poor-threshold', type=float, required=True, help='Poor threshold')
    parser.add_argument('--moderate-threshold', type=float, required=True, help='Moderate threshold')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    analyze_efficacy_distribution(
        args.input,
        args.efficacy_column,
        args.output_dir,
        args.poor_threshold,
        args.moderate_threshold
    )
    sys.exit(0)

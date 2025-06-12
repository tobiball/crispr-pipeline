#!/usr/bin/env python
"""
Module for **analyzing and visualizing gRNA efficacy distributions**.

Draws a colour‑coded stacked histogram of guide‑RNA scores (2‑point buckets)
with categories *High, Moderate, Poor* and a title of the form

    Guide Efficency Distribution – <Database>

where the database name is optional.
"""

from __future__ import annotations

import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────────
#                               PLOTTING LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def plot_efficacy_distribution(
        values: np.ndarray,
        poor_threshold: float = 60.0,
        moderate_threshold: float = 90.0,
        outfile: str = "efficacy_distribution.png",
        db_name: Optional[str] = None,
) -> None:
    """Draw and save the stacked histogram."""

    start_edge = max(0, 2 * np.floor(values.min() / 2.0))
    edges = np.arange(start_edge, 102, 2)

    poor_vals     = values[values < poor_threshold]
    moderate_vals = values[(values >= poor_threshold) & (values < moderate_threshold)]
    high_vals     = values[values >= moderate_threshold]
    subsets = [poor_vals, moderate_vals, high_vals]
    counts  = list(map(len, subsets))
    total   = len(values)

    colours = ["#d32f2f", "#f57c00", "#388e3c"]  # poor → moderate → high
    labels  = [
        f"Poor (<{poor_threshold})             : {counts[0]} ({100*counts[0]/total:.1f}%)",
        f"Moderate ({poor_threshold}–{moderate_threshold}) : {counts[1]} ({100*counts[1]/total:.1f}%)",
        f"High (≥{moderate_threshold})              : {counts[2]} ({100*counts[2]/total:.1f}%)",
    ]

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    plt.hist(subsets, bins=edges, stacked=True, color=colours,
             edgecolor="black", linewidth=0.6, label=labels)

    for th in (poor_threshold, moderate_threshold):
        plt.axvline(th, linestyle="--", color="black")

    plt.xlabel("Efficacy score")
    plt.ylabel("Count")
    title = "Guide Efficency Distribution" + (f" – {db_name}" if db_name else "")
    plt.title(title)

    # Legend High → Moderate → Poor
    handles, leg_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], leg_labels[::-1])

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"✅  Saved histogram to {outfile}")

# ──────────────────────────────────────────────────────────────────────────────
#                                 CLI ENTRY
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Plot gRNA efficacy distribution")

    # Input CSV – positional or flag
    p.add_argument("csv", nargs="?", help="CSV with efficacy scores (positional)")
    p.add_argument("-i", "--input", dest="csv_flag", help="CSV with efficacy scores (flag)")

    # Column name
    p.add_argument("--column", "--efficacy-column", dest="column", default="efficacy",
                   help="Column with efficacy scores (default: efficacy)")

    # Thresholds
    p.add_argument("--poor", "--poor-threshold", dest="poor", type=float, default=60.0,
                   help="Poor threshold (default 60)")
    p.add_argument("--moderate", "--moderate-threshold", dest="moderate", type=float, default=90.0,
                   help="Moderate threshold (default 90)")

    # Database name for title
    p.add_argument("--db", "--database", dest="db", default=None,
                   help="Database name to include in the title")

    # Output controls
    p.add_argument("--output-dir", dest="out_dir", default=None,
                   help="Directory to save the PNG (legacy Rust flag)")
    p.add_argument("-o", "--out", dest="out_file", default=None,
                   help="Exact PNG file path (overrides --output-dir)")

    args = p.parse_args(argv)

    csv_path = args.csv_flag or args.csv
    if csv_path is None:
        p.error("CSV input missing: provide positional CSV or --input")

    if args.out_file:
        out_path = args.out_file
    elif args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "efficacy_distribution.png")
    else:
        out_path = "efficacy_distribution.png"

    data = pd.read_csv(csv_path)[args.column].values
    plot_efficacy_distribution(data, args.poor, args.moderate, out_path, args.db)

if __name__ == "__main__":
    main()

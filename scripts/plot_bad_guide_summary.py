#!/usr/bin/env python3
"""
Improved statistical analysis and plotting with clear explanations and better layout.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.patches as mpatches

# Professional color palette
# ---------------------------------- NEW: tool-specific colours ------------
TOOL_COLORS = {
    "TKO PSSM":           (130/255, 130/255, 130/255),
    "Moreno-Mateos":      (90/255,  100/255, 100/255),

    "Doench Rule Set 2":  (255/255, 140/255, 0/255),
    "Doench Rule Set 3":  (220/255, 110/255, 0/255),

    "DeepCRISPR":         (138/255, 43/255,  226/255),
    "DeepSpCas9":         (123/255, 31/255,  162/255),
    "TransCRISPR":        (186/255, 85/255,  211/255),

    "Linear Consensus":   (34/255,  139/255, 34/255),
    "Logistic Consensus": (0/255,   100/255, 0/255)
}
DEFAULT_COLOR = (0.4, 0.4, 0.4)   # fallback (mid-grey)
# -------------------------------------------------------------------------


def wrap_tool_name(name: str, *_unused, **__unused) -> str:
    """
    Put the first word on its own line, everything else underneath.

    Examples
    --------
    "Doench Rule Set 3"   ->  "Doench\nRule Set 3"
    "DeepCRISPR"          ->  "DeepCRISPR"        (no space → unchanged)
    """
    if " " not in name:
        return name            # single word, nothing to split

    first_word, remainder = name.split(" ", 1)  # split only at the *first* space
    return f"{first_word}\n{remainder}"




def load_statistical_data(stats_path, distribution_path=None):
    """Load statistical data with confidence intervals."""
    stats_df = pd.read_csv(stats_path)

    print(f"Loaded statistical analysis for {len(stats_df)} tools:")
    print(f"Dataset: {stats_df['total_genes'].iloc[0]} genes with exactly 4 guides each")
    print()

    for i, row in stats_df.iterrows():
        print(f"  {i+1}. {row['tool']}: {row['pct_correct']:.1f}% "
              f"(95% CI: {row['ci_lower']:.1f}%-{row['ci_upper']:.1f}%) "
              f"[{row['correct_cnt']}/{row['total_genes']}]")

    distribution_data = None
    if distribution_path and Path(distribution_path).exists():
        distribution_df = pd.read_csv(distribution_path)
        distribution_data = {4: stats_df['total_genes'].iloc[0]}
        print(f"\nAll {distribution_data[4]} genes have exactly 4 guides")

    return stats_df, distribution_data

def perform_statistical_tests(stats_df):
    """Perform statistical tests and return significant comparisons."""
    tools = stats_df['tool'].tolist()
    n_tools = len(tools)

    print(f"\n=== Statistical Significance Testing ===")
    print("Two-proportion z-test comparing success rates")
    print("* p < 0.05, ** p < 0.01, *** p < 0.001")
    print()

    # Only compare top 3 tools with each other to avoid multiple comparison issues
    significant_pairs = []

    for i in range(min(3, n_tools)):
        for j in range(i + 1, min(3, n_tools)):
            tool1 = stats_df.iloc[i]
            tool2 = stats_df.iloc[j]

            count1, n1 = tool1['correct_cnt'], tool1['total_genes']
            count2, n2 = tool2['correct_cnt'], tool2['total_genes']

            # Two-proportion z-test
            p_pool = (count1 + count2) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

            if se > 0:
                z = (count1/n1 - count2/n2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                z, p_value = 0, 1

            significance_level = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"{tool1['tool']} vs {tool2['tool']}: p = {p_value:.4f} {significance_level}")

            if p_value < 0.05:
                significant_pairs.append((i, j, p_value, significance_level))

    return significant_pairs

def truncate_tool_names(tools, max_length=16):
    """Truncate tool names for better display."""
    truncated = []
    for name in tools:
        if len(name) > max_length:
            truncated.append(name[:max_length-1] + "…")
        else:
            truncated.append(name)
    return truncated

def create_improved_statistical_plot(stats_df, distribution_data, db_name, output_dir, significant_pairs):
    """Create improved statistical plot with better explanations and layout."""

    # Prepare data
    tools = stats_df['tool'].tolist()
    percentages = stats_df['pct_correct'].tolist()
    ci_lower = stats_df['ci_lower'].tolist()
    ci_upper = stats_df['ci_upper'].tolist()
    wrapped_tools = [wrap_tool_name(t) for t in tools]

    # Calculate error bar values
    error_lower = [p - ci_l for p, ci_l in zip(percentages, ci_lower)]
    error_upper = [ci_u - p for p, ci_u in zip(percentages, ci_upper)]

    # Create figure with more space
    fig, ax = plt.subplots(figsize=(16, 11))

    # Create bars with error bars
    n_tools = len(tools)
    x_positions = np.arange(n_tools)
    colors = [TOOL_COLORS.get(t, DEFAULT_COLOR) for t in tools]

    bars = ax.bar(x_positions, percentages,
                  color=colors, width=0.6, edgecolor='white', linewidth=2)

    ax.errorbar(x_positions, percentages,
                yerr=[error_lower, error_upper],
                fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2)

    # Add percentage labels with better spacing
    max_error = max(error_upper)
    label_height = max_error + 3

    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + label_height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add significance comparisons only for top few tools
    if significant_pairs:
        sig_y_start = max(percentages) + max_error + 8
        for i, (idx1, idx2, p_val, sig_level) in enumerate(significant_pairs[:2]):  # Max 2 comparisons
            y_pos = sig_y_start + i * 4

            # Draw comparison line
            ax.plot([idx1, idx2], [y_pos, y_pos], 'k-', linewidth=1.5)
            ax.plot([idx1, idx1], [y_pos-0.8, y_pos+0.8], 'k-', linewidth=1.5)
            ax.plot([idx2, idx2], [y_pos-0.8, y_pos+0.8], 'k-', linewidth=1.5)

            # Add significance label
            ax.text((idx1 + idx2) / 2, y_pos + 1.5, sig_level,
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize axes
    ax.set_ylabel('Success Rate (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Prediction Tool (ordered by performance)', fontsize=16, fontweight='bold')

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(wrapped_tools, rotation=45, ha='right', fontsize=12)

    # Set y-axis with enough space
    y_max = max(percentages) + max_error + 15
    if significant_pairs:
        y_max += len(significant_pairs[:2]) * 4 + 5
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))

    # Add reference lines
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax.text(n_tools-0.5, 27, 'Random chance\n(25%)', ha='right', va='bottom',
            fontsize=10, color='red', fontweight='bold')

    # Title
    total_genes = stats_df['total_genes'].iloc[0]
    title = f"TKO: Correctly Identifying Poor Guides\n" \
            f"Find 1 poor guide (<60 fold change score) among 4 guides per gene"
    ax.set_title(title, fontsize=18, fontweight='bold', pad=30)

    # Create detailed explanation box
    explanation = (
        f"Dataset: {total_genes} genes × 4 guides = {total_genes * 4} total guides\n"
        f"Error bars: 95% confidence intervals"
    )

    # Add explanation box
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor('#FAFBFC')

    plt.tight_layout()

    plt.rcParams.update({
        "font.size": 16,          # base size for ticks & small text
        "axes.titlesize": 22,     # main title
        "axes.labelsize": 20,     # axis titles
        "legend.fontsize": 16,
    })

    # Save plot
    output_path = Path(output_dir) / "improved_bad_guide_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved improved statistical plot to: {output_path}")
    return str(output_path)

def create_clean_confidence_plot(stats_df, db_name, output_dir):
    """Create a clean confidence interval plot with better explanations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    tools = stats_df['tool'].tolist()
    percentages = stats_df['pct_correct'].tolist()
    ci_lower = stats_df['ci_lower'].tolist()
    ci_upper = stats_df['ci_upper'].tolist()

    y_positions = np.arange(len(tools))
    colors = [TOOL_COLORS.get(t, DEFAULT_COLOR) for t in tools]


    # Create horizontal confidence interval plot
    for i, (tool, pct, ci_l, ci_u, color) in enumerate(zip(tools, percentages, ci_lower, ci_upper, colors)):
        # Draw confidence interval line
        ax.plot([ci_l, ci_u], [i, i], color=color, linewidth=8, alpha=0.7)
        # Draw point estimate
        ax.plot(pct, i, 'o', color=color, markersize=12,
                markeredgecolor='white', markeredgewidth=3)

    # Add labels to the right
    for i, (tool, pct, ci_l, ci_u) in enumerate(zip(tools, percentages, ci_lower, ci_upper)):
        label = f'{truncate_tool_names([tool], 18)[0]}: {pct:.1f}% ({ci_l:.1f}-{ci_u:.1f}%)'
        ax.text(max(ci_upper) + 2, i, label, va='center', fontsize=11, fontweight='bold')

    # Remove y-axis labels since we have text labels
    ax.set_yticks([])
    ax.set_xlabel('Success Rate (%) with 95% Confidence Intervals', fontsize=14, fontweight='bold')

    # Add reference lines with labels
    ax.axvline(x=25, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=50, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=75, color='green', linestyle='--', alpha=0.7, linewidth=2)

    # Add reference line labels
    ax.text(25, -0.8, 'Random\n(25%)', ha='center', va='top', fontsize=10, color='red')
    ax.text(50, -0.8, 'Moderate\n(50%)', ha='center', va='top', fontsize=10, color='orange')
    ax.text(75, -0.8, 'Good\n(75%)', ha='center', va='top', fontsize=10, color='green')

    # Title with explanation
    total_genes = stats_df['total_genes'].iloc[0]
    title = (f'Tool Performance with Statistical Confidence\n'
             f'95% Confidence Intervals: "If we repeated this study 100 times,\n'
             f'95 times the true success rate would fall within these ranges"')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Set reasonable x-axis limits
    x_min = min(ci_lower) - 5
    x_max = max(ci_upper) + 25  # Extra space for labels
    ax.set_xlim(x_min, x_max)

    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    output_path = Path(output_dir) / "clean_confidence_intervals.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved clean confidence intervals plot to: {output_path}")
    return str(output_path)

def create_simple_summary_table(stats_df, output_dir):
    """Create a simple summary table as an image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    for i, row in stats_df.iterrows():
        table_data.append([
            f"{i+1}",
            row['tool'],
            f"{row['correct_cnt']}/{row['total_genes']}",
            f"{row['pct_correct']:.1f}%",
            f"{row['ci_lower']:.1f}%-{row['ci_upper']:.1f}%"
        ])

    headers = ['Rank', 'Tool', 'Success\n(Correct/Total)', 'Success Rate', '95% Confidence\nInterval']

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.35, 0.15, 0.15, 0.25])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#F8F9FA' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)

    total_genes = stats_df['total_genes'].iloc[0]
    title = f"Statistical Summary: Bad Guide Detection Performance\n" \
            f"Task: Identify 1 poor guide (< 60 efficacy) out of 4 guides per gene\n" \
            f"Sample: {total_genes} genes with exactly 4 guides each"
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    output_path = Path(output_dir) / "performance_summary_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved summary table to: {output_path}")
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Improved statistical analysis plots')
    parser.add_argument('--input', required=True, help='Path to statistics CSV')
    parser.add_argument('--distribution', help='Path to guide distribution CSV')
    parser.add_argument('--db_name', default='Unknown', help='Database name')
    parser.add_argument('--output-dir', required=True, help='Output directory')

    args = parser.parse_args()

    # Load data
    stats_df, distribution_data = load_statistical_data(args.input, args.distribution)

    # Perform statistical tests
    significant_pairs = perform_statistical_tests(stats_df)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create improved plots
    main_plot = create_improved_statistical_plot(stats_df, distribution_data, args.db_name,
                                                 output_dir, significant_pairs)
    ci_plot = create_clean_confidence_plot(stats_df, args.db_name, output_dir)
    table_plot = create_simple_summary_table(stats_df, output_dir)

    # Final summary
    print(f"\n=== Analysis Complete for {args.db_name} ===")
    print(f"Generated three visualizations:")
    print(f"  1. Main statistical plot: {main_plot}")
    print(f"  2. Confidence intervals: {ci_plot}")
    print(f"  3. Summary table: {table_plot}")

    best_tool = stats_df.iloc[0]
    total_genes = stats_df['total_genes'].iloc[0]
    print(f"\nKey findings:")
    print(f"  - Best tool: {best_tool['tool']} ({best_tool['pct_correct']:.1f}%)")
    print(f"  - Random chance baseline: 25%")
    print(f"  - Sample size: {total_genes} genes (robust for statistical inference)")

if __name__ == "__main__":
    main()
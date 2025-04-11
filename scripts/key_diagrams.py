#!/usr/bin/env python3
"""
Key Diagrams for CRISPR Guide RNA Analysis

This script generates various visualizations for analyzing CRISPR guide RNA data,
including Spearman correlation and ROC curve analysis.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, auc

def calculate_correlations(df, efficacy_col, tool_cols):
    """
    Calculate Spearman correlation coefficients between efficacy and each tool.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with efficacy and prediction tool columns
    efficacy_col : str
        Name of the efficacy column
    tool_cols : list
        List of prediction tool column names

    Returns:
    --------
    pandas.DataFrame
        DataFrame with tools and their correlation coefficients
    """
    # Initialize results container
    results = []

    # Get efficacy values
    efficacy_values = df[efficacy_col].dropna().values

    # Calculate correlation for each tool
    for tool in tool_cols:
        if tool not in df.columns:
            print(f"Warning: Tool '{tool}' not found in the data. Skipping.")
            continue

        # Get tool values, ignoring rows with NaN in either tool or efficacy
        tool_data = df[[tool, efficacy_col]].dropna()
        if len(tool_data) < 10:  # Require at least 10 data points
            print(f"Warning: Not enough data for '{tool}' (only {len(tool_data)} valid rows). Skipping.")
            continue

        # Calculate Spearman correlation
        corr, p_value = spearmanr(tool_data[tool], tool_data[efficacy_col])

        results.append({
            'Tool': tool,
            'Correlation': corr,
            'P-value': p_value,
            'Sample_Size': len(tool_data)
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by correlation (descending)
    results_df = results_df.sort_values('Correlation', ascending=False)

    return results_df

def plot_correlation_bargraph(corr_df, output_path, title=None, figsize=(12, 8)):
    """
    Create a bar graph of correlation coefficients.

    Parameters:
    -----------
    corr_df : pandas.DataFrame
        DataFrame with correlation results
    output_path : str
        Path to save the output figure
    title : str, optional
        Custom title for the plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Set style
    sns.set_style("whitegrid")

    # Create color mapping based on correlation strength
    colors = corr_df['Correlation'].map(lambda x:
                                        'darkgreen' if x >= 0.7 else
                                        'forestgreen' if x >= 0.5 else
                                        'gold' if x >= 0.3 else
                                        'orange' if x >= 0.1 else
                                        'firebrick')

    # Create bar plot
    ax = sns.barplot(x='Tool', y='Correlation', data=corr_df, palette=colors)

    # Add title and labels
    if title:
        plt.title(title, fontsize=16, pad=20)
    else:
        plt.title('Spearman Correlation between Efficacy and Prediction Tools',
                  fontsize=16, pad=20)

    plt.xlabel('Prediction Tool', fontsize=14, labelpad=10)
    plt.ylabel('Spearman Correlation Coefficient', fontsize=14, labelpad=10)

    # Add correlation values on top of bars
    for i, v in enumerate(corr_df['Correlation']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Add horizontal lines for correlation strength reference
    plt.axhline(y=0.7, color='darkgreen', linestyle='--', alpha=0.7,
                label='Strong (0.7+)')
    plt.axhline(y=0.5, color='forestgreen', linestyle='--', alpha=0.7,
                label='Moderate (0.5+)')
    plt.axhline(y=0.3, color='gold', linestyle='--', alpha=0.7,
                label='Weak (0.3+)')
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7,
                label='Very weak (0.1+)')
    plt.axhline(y=0, color='firebrick', linestyle='-', alpha=0.7)

    # Add a legend
    plt.legend(title='Correlation Strength', title_fontsize=12, fontsize=10)

    # Adjust y-axis limits
    y_min = min(0, corr_df['Correlation'].min() - 0.1)
    y_max = max(0.8, corr_df['Correlation'].max() + 0.1)
    plt.ylim(y_min, y_max)

    # Add gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # Close the figure to free memory
    plt.close()

def plot_roc_curves(df, efficacy_col, tool_cols, output_path, cutoff=75.0, title=None, figsize=(12, 8)):
    """
    Generate ROC curves for multiple prediction tools.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with efficacy and prediction tool columns
    efficacy_col : str
        Name of the efficacy column
    tool_cols : list
        List of prediction tool column names
    output_path : str
        Path to save the output figure
    cutoff : float, optional
        Efficacy cutoff for binary classification (guides ≥ cutoff are considered "effective")
    title : str, optional
        Custom title for the plot
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns:
    --------
    pandas.DataFrame
        DataFrame with tools and their AUC values
    """
    # Create figure
    plt.figure(figsize=figsize)

    # Set style
    sns.set_style("whitegrid")

    # Convert efficacy to binary based on cutoff
    binary_efficacy = (df[efficacy_col] >= cutoff).astype(int)
    binary_count = binary_efficacy.sum()
    total_count = len(binary_efficacy)
    print(f"Using cutoff {cutoff}: {binary_count} effective guides ({binary_count/total_count:.1%}) and {total_count-binary_count} ineffective guides")

    # Get list of colors for consistency
    colors = plt.cm.tab10.colors

    # Store AUC results
    results = []

    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', alpha=0.7)

    # Calculate and plot ROC curve for each tool
    for i, tool in enumerate(tool_cols):
        if tool not in df.columns:
            print(f"Warning: Tool '{tool}' not found in data. Skipping.")
            continue

        # Get valid rows with no NaN values
        valid_data = df[[tool, efficacy_col]].dropna()
        if len(valid_data) < 10:
            print(f"Warning: Not enough data for '{tool}' (only {len(valid_data)} valid rows). Skipping.")
            continue

        y_true = (valid_data[efficacy_col] >= cutoff).astype(int)
        y_scores = valid_data[tool]

        # Calculate ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[i % len(colors)],
                     label=f'{tool} (AUC = {roc_auc:.3f})')

            # Store AUC results
            results.append({
                'Tool': tool,
                'AUC': roc_auc,
                'Sample_Size': len(valid_data)
            })
        except Exception as e:
            print(f"Error calculating ROC curve for '{tool}': {e}")
            continue

    # Add labels and title
    if title:
        plt.title(title, fontsize=16, pad=20)
    else:
        plt.title(f'ROC Curves (Efficacy cutoff ≥ {cutoff})', fontsize=16, pad=20)

    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)

    # Add AUC interpretation guidelines
    plt.annotate('Excellent: AUC ≥ 0.9', xy=(0.75, 0.2), fontsize=10, color='darkgreen')
    plt.annotate('Good: 0.8 ≤ AUC < 0.9', xy=(0.75, 0.15), fontsize=10, color='forestgreen')
    plt.annotate('Fair: 0.7 ≤ AUC < 0.8', xy=(0.75, 0.1), fontsize=10, color='gold')
    plt.annotate('Poor: 0.6 ≤ AUC < 0.7', xy=(0.75, 0.05), fontsize=10, color='orange')

    # Add legend
    plt.legend(loc='lower right', fontsize=10)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to: {output_path}")

    # Close the figure
    plt.close()

    # Convert results to DataFrame and sort by AUC (descending)
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)

    return results_df

def save_correlation_results(corr_df, output_path):
    """Save correlation results to CSV file."""
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Save to CSV
    corr_df.to_csv(output_path, index=False)
    print(f"Correlation results saved to: {output_path}")

def main():
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {plt.matplotlib.__version__}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate key diagrams for CRISPR guide RNA analysis')
    parser.add_argument('--input', '-i', required=True, help='Path to input CSV file with efficacy and prediction data')
    parser.add_argument('--output', '-o', default='./key_diagrams_results', help='Output directory for results')
    parser.add_argument('--efficacy-col', default='efficacy', help='Name of the efficacy column')
    parser.add_argument('--tools', help='Comma-separated list of tool columns to analyze (default: auto-detect)')
    parser.add_argument('--title', help='Custom title for the plots')
    parser.add_argument('--analysis', default='all',
                        help='Type of analysis to run (correlation, roc, or all)')
    parser.add_argument('--cutoffs', default='50,75,90',
                        help='Comma-separated list of efficacy cutoffs for classification')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        print(f"Available columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check if efficacy column exists
    if args.efficacy_col not in df.columns:
        print(f"Error: Efficacy column '{args.efficacy_col}' not found in data.")
        print(f"Available columns: {', '.join(df.columns)}")
        return

    # Determine tool columns
    if args.tools:
        tool_cols = [col.strip() for col in args.tools.split(',')]
    else:
        # Auto-detect tool columns (exclude common non-tool columns)
        exclude_cols = [args.efficacy_col, 'sgRNA', 'sequence', 'Gene', 'chromosome',
                        'start', 'end', 'strand', 'pam', 'sequence_with_pam', 'position', 'efficacy_category']

        tool_cols = [col for col in df.columns if col not in exclude_cols
                     and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        print(f"Auto-detected potential tool columns: {', '.join(tool_cols)}")

    # Parse cutoffs
    try:
        cutoffs = [float(c.strip()) for c in args.cutoffs.split(',')]
    except ValueError:
        print(f"Warning: Invalid cutoff values '{args.cutoffs}'. Using default cutoffs (50, 75, 90).")
        cutoffs = [50.0, 75.0, 90.0]

    # Determine which analyses to run
    analyses = args.analysis.lower().split(',')
    run_all = 'all' in analyses

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run correlation analysis
    if run_all or 'correlation' in analyses:
        print("Running correlation analysis...")
        corr_df = calculate_correlations(df, args.efficacy_col, tool_cols)

        if len(corr_df) == 0:
            print("Warning: No valid correlations could be calculated. Check your data and tool columns.")
        else:
            # Create output paths for correlation analysis
            figure_path = os.path.join(args.output, 'correlation_bargraph.png')
            csv_path = os.path.join(args.output, 'correlation_results.csv')

            # Generate visualizations
            plot_title = args.title if args.title else "Spearman Correlation: Guide Efficacy vs Prediction Tools"
            plot_correlation_bargraph(corr_df, figure_path, plot_title)

            # Save results
            save_correlation_results(corr_df, csv_path)

    # Run ROC curve analysis
    if run_all or 'roc' in analyses:
        print("Running ROC curve analysis...")
        all_roc_results = []

        # Generate ROC curves for different cutoffs
        for cutoff in cutoffs:
            title = args.title if args.title else f"ROC Curves (Efficacy cutoff ≥ {cutoff})"
            output_path = os.path.join(args.output, f'roc_curves_{int(cutoff)}.png')

            # Generate ROC curves and get results
            roc_df = plot_roc_curves(
                df,
                args.efficacy_col,
                tool_cols,
                output_path,
                cutoff=cutoff,
                title=title
            )

            if len(roc_df) > 0:
                # Add cutoff column
                roc_df['Cutoff'] = cutoff
                all_roc_results.append(roc_df)

                # Save individual cutoff results
                cutoff_csv_path = os.path.join(args.output, f'roc_auc_values_{int(cutoff)}.csv')
                roc_df.to_csv(cutoff_csv_path, index=False)

        # Combine all ROC results
        if all_roc_results:
            combined_roc_df = pd.concat(all_roc_results)
            combined_csv_path = os.path.join(args.output, 'roc_auc_all_cutoffs.csv')
            combined_roc_df.to_csv(combined_csv_path, index=False)
            print(f"Combined ROC AUC results saved to: {combined_csv_path}")

    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
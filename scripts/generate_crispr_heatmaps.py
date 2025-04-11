#!/usr/bin/env python3
"""
Generate heatmap visualizations for CRISPR gRNA prediction tools.

This script creates a set of heatmaps that help analyze and compare the
performance of different CRISPR guide RNA prediction tools:

1. Tool correlation heatmap - How tools correlate with each other and efficacy
2. Performance metrics heatmaps - Precision, recall, F1, AUC at different thresholds
3. Within-gene ranking heatmap - How well tools rank guides targeting the same gene

Usage:
    python generate_crispr_heatmaps.py --input data.csv --tools "tool1,tool2,tool3" --output ./heatmaps
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate heatmaps for CRISPR tool comparison')
    parser.add_argument('--input', required=True, help='Input CSV file with guide data')
    parser.add_argument('--tools', required=True, help='Comma-separated list of tool column names')
    parser.add_argument('--output', default='./heatmaps', help='Output directory for heatmaps')
    parser.add_argument('--efficacy', default='efficacy', help='Name of the efficacy column')
    parser.add_argument('--gene', default='Gene', help='Name of the gene column')
    return parser.parse_args()


def create_tool_correlation_heatmap(df, tool_columns, efficacy_column="efficacy", 
                                    output_path="tool_correlation_heatmap.png", 
                                    method="spearman", 
                                    figsize=(12, 10)):
    """
    Create a heatmap showing correlations between different prediction tools and efficacy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tool scores and efficacy values
    tool_columns : list
        List of column names for the prediction tools to compare
    efficacy_column : str
        Name of the column containing actual efficacy values
    output_path : str
        Where to save the heatmap image
    method : str
        Correlation method ('spearman' or 'pearson')
    figsize : tuple
        Figure size for the plot
    
    Returns:
    --------
    pandas.DataFrame
        The correlation matrix
    """
    print(f"Creating correlation heatmap...")
    
    # Extract relevant columns
    columns_to_extract = tool_columns + [efficacy_column]
    df_subset = df[columns_to_extract].copy()
    
    # Calculate correlation matrix
    if method.lower() == 'spearman':
        corr_matrix = df_subset.corr(method='spearman')
    else:
        corr_matrix = df_subset.corr(method='pearson')
    
    # Create the heatmap figure
    plt.figure(figsize=figsize)
    
    # Create a custom colormap
    colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    
    # Generate the heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    ax = sns.heatmap(
        corr_matrix, 
        annot=True,           # Show correlation values
        mask=mask,            # Show only lower triangle
        vmin=-1, vmax=1,      # Correlation range
        center=0,             # Center colormap at 0
        cmap=cmap,
        square=True,          # Make cells square
        linewidths=0.5,
        fmt=".2f",            # Format for correlation values
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} Correlation"}
    )
    
    # Set title and labels
    plt.title(f"Correlation Matrix of Prediction Tools ({method.capitalize()})", 
              fontsize=16, pad=20)
    
    # Make y-axis labels more readable by adjusting rotation
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    
    # Add a text note about interpretation
    plt.figtext(0.5, 0.01, 
                "Values closer to 1.0 indicate stronger positive correlation. "
                "Efficacy correlations show predictive power.", 
                ha="center", fontsize=10, style='italic')
    
    # Save the figure with tight layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {output_path}")
    return corr_matrix


def create_performance_metrics_heatmap(df, tool_columns, efficacy_column="efficacy", 
                                      efficacy_threshold=80.0,
                                      output_path="performance_metrics_heatmap.png",
                                      figsize=(14, 10)):
    """
    Create a heatmap showing different performance metrics for each prediction tool.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tool scores and efficacy values
    tool_columns : list
        List of column names for the prediction tools to compare
    efficacy_column : str
        Name of the column containing actual efficacy values
    efficacy_threshold : float
        Threshold to classify guides as "good" vs "bad"
    output_path : str
        Where to save the heatmap image
    figsize : tuple
        Figure size for the plot
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance metrics for each tool
    """
    print(f"Creating performance metrics heatmap (threshold: {efficacy_threshold})...")
    
    # Extract efficacy column and convert to numpy array
    efficacy_values = df[efficacy_column].dropna().values
    
    # Create binary labels based on threshold
    binary_labels = efficacy_values >= efficacy_threshold
    
    # Dictionary to store results
    results = {}
    
    # Calculate metrics for each tool
    for tool in tool_columns:
        try:
            # Get tool predictions
            tool_values = df[tool].dropna().values
            
            # Make sure we have matching data
            valid_indices = np.logical_and(~np.isnan(efficacy_values), ~np.isnan(tool_values))
            tool_valid = tool_values[valid_indices]
            efficacy_valid = efficacy_values[valid_indices]
            binary_valid = binary_labels[valid_indices]
            
            if len(tool_valid) < 10:
                print(f"Tool {tool} has too few valid data points. Skipping.")
                continue
                
            # Calculate optimal threshold using F1 score
            thresholds = np.linspace(np.min(tool_valid), np.max(tool_valid), 100)
            best_f1 = 0
            best_threshold = thresholds[0]
            
            for threshold in thresholds:
                binary_predictions = tool_valid >= threshold
                f1 = f1_score(binary_valid, binary_predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # Calculate metrics with optimal threshold
            binary_predictions = tool_valid >= best_threshold
            precision = precision_score(binary_valid, binary_predictions, zero_division=0)
            recall = recall_score(binary_valid, binary_predictions, zero_division=0)
            f1 = f1_score(binary_valid, binary_predictions, zero_division=0)
            
            # Calculate AUC
            try:
                auc = roc_auc_score(binary_valid, tool_valid)
            except:
                auc = np.nan
            
            # Calculate correlation
            corr_spearman, _ = spearmanr(tool_valid, efficacy_valid)
            
            # Store results
            results[tool] = {
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1, 
                'AUC': auc,
                'Spearman Corr': corr_spearman,
                'Optimal Threshold': best_threshold
            }
        except Exception as e:
            print(f"Error processing tool {tool}: {e}")
            results[tool] = {
                'Precision': np.nan,
                'Recall': np.nan,
                'F1 Score': np.nan,
                'AUC': np.nan,
                'Spearman Corr': np.nan,
                'Optimal Threshold': np.nan
            }
    
    # Convert results to DataFrame
    metrics_df = pd.DataFrame(results).T
    
    # Create the heatmap figure
    plt.figure(figsize=figsize)
    
    # Create a custom colormap
    colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    
    # Generate the heatmap (excluding the Optimal Threshold column)
    metrics_for_plot = metrics_df.drop(columns=['Optimal Threshold'])
    
    ax = sns.heatmap(
        metrics_for_plot, 
        annot=True,
        vmin=0, vmax=1,
        cmap=cmap,
        linewidths=0.5,
        fmt=".3f",
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8, "label": "Score (higher is better)"}
    )
    
    # Set title and labels
    plt.title(f"Performance Metrics of Prediction Tools (Efficacy Threshold: {efficacy_threshold}%)", 
              fontsize=16, pad=20)
    
    # Make axis labels more readable
    plt.ylabel("Prediction Tools", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    
    # Add a text note about thresholds
    threshold_text = "Optimal thresholds:\n" + "\n".join(
        [f"{tool}: {metrics_df.loc[tool, 'Optimal Threshold']:.2f}" for tool in metrics_df.index]
    )
    plt.figtext(0.02, 0.02, threshold_text, fontsize=9)
    
    # Add explanatory notes
    plt.figtext(0.5, 0.01, 
                "Precision: % of predicted positives that are actually positive\n"
                "Recall: % of actual positives that were correctly predicted\n"
                "F1: Harmonic mean of precision and recall\n"
                "AUC: Area under ROC curve (ranking quality)", 
                ha="center", fontsize=10, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save CSV of metrics
    csv_path = output_path.replace('.png', '.csv')
    metrics_df.to_csv(csv_path)
    
    print(f"Performance metrics heatmap saved to {output_path}")
    print(f"Performance metrics data saved to {csv_path}")
    return metrics_df


def create_within_gene_performance_heatmap(df, tool_columns, gene_column="Gene", efficacy_column="efficacy",
                                          min_guides_per_gene=3, 
                                          output_path="within_gene_performance_heatmap.png",
                                          figsize=(12, 10)):
    """
    Create a heatmap showing how well each tool ranks guides within genes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the tool scores and efficacy values
    tool_columns : list
        List of column names for the prediction tools to compare
    gene_column : str
        Column name for gene identifiers
    efficacy_column : str
        Name of the column containing actual efficacy values
    min_guides_per_gene : int
        Minimum number of guides required for a gene to be included
    output_path : str
        Where to save the heatmap image
    figsize : tuple
        Figure size for the plot
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with within-gene performance metrics
    """
    print(f"Creating within-gene performance heatmap...")
    
    # Count guides per gene and keep only genes with enough guides
    gene_counts = df[gene_column].value_counts()
    genes_to_keep = gene_counts[gene_counts >= min_guides_per_gene].index
    filtered_df = df[df[gene_column].isin(genes_to_keep)]
    
    print(f"Analyzing {len(genes_to_keep)} genes with at least {min_guides_per_gene} guides each")
    
    # Dictionary to store results
    results = {}
    
    # Function to calculate agreement in ranking
    def calculate_ranking_agreement(efficacy_ranks, tool_ranks):
        # Spearman correlation (overall ranking similarity)
        spearman_corr, _ = spearmanr(efficacy_ranks, tool_ranks)
        
        # Kendall Tau (pairwise ranking agreement)
        tau, _ = kendalltau(efficacy_ranks, tool_ranks)
        
        # Calculate if the tool correctly identifies the worst guide
        worst_efficacy_idx = efficacy_ranks.argmax()  # Reversed because higher rank = worse guide
        worst_tool_idx = tool_ranks.argmax()
        worst_guide_agreement = 1 if worst_efficacy_idx == worst_tool_idx else 0
        
        # Calculate if the tool correctly identifies the best guide
        best_efficacy_idx = efficacy_ranks.argmin()
        best_tool_idx = tool_ranks.argmin()
        best_guide_agreement = 1 if best_efficacy_idx == best_tool_idx else 0
        
        return spearman_corr, tau, worst_guide_agreement, best_guide_agreement
    
    # Process each tool
    for tool in tool_columns:
        spearman_values = []
        kendall_values = []
        worst_agreement_values = []
        best_agreement_values = []
        genes_analyzed = 0
        
        # Process each gene
        for gene in genes_to_keep:
            gene_df = filtered_df[filtered_df[gene_column] == gene]
            
            # Skip if missing data
            if gene_df[tool].isnull().any() or gene_df[efficacy_column].isnull().any():
                continue
                
            # Get efficacy and tool values
            efficacy_values = gene_df[efficacy_column].values
            tool_values = gene_df[tool].values
            
            # Skip if not enough variation in rankings
            if len(set(efficacy_values)) < 2 or len(set(tool_values)) < 2:
                continue
                
            # Calculate ranks (using scipy's rankdata)
            from scipy.stats import rankdata
            efficacy_ranks = rankdata(-efficacy_values)  # Negative so that higher efficacy = lower rank
            tool_ranks = rankdata(-tool_values)  # Negative so that higher tool score = lower rank
            
            # Calculate agreement metrics
            try:
                spearman, tau, worst_agree, best_agree = calculate_ranking_agreement(
                    efficacy_ranks, tool_ranks)
                
                spearman_values.append(spearman)
                kendall_values.append(tau)
                worst_agreement_values.append(worst_agree)
                best_agreement_values.append(best_agree)
                genes_analyzed += 1
            except Exception as e:
                print(f"Error analyzing gene {gene} for tool {tool}: {e}")
        
        # Calculate average metrics across all genes
        results[tool] = {
            'Spearman Rank Corr': np.nanmean(spearman_values) if spearman_values else np.nan,
            'Kendall Tau': np.nanmean(kendall_values) if kendall_values else np.nan,
            'Worst Guide Accuracy': np.nanmean(worst_agreement_values) if worst_agreement_values else np.nan,
            'Best Guide Accuracy': np.nanmean(best_agreement_values) if best_agreement_values else np.nan,
            'Genes Analyzed': genes_analyzed
        }
    
    # Convert results to DataFrame
    metrics_df = pd.DataFrame(results).T
    
    # Create the heatmap figure
    plt.figure(figsize=figsize)
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", 
        ["#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"], 
        N=100
    )
    
    # Generate the heatmap (excluding the Genes Analyzed column)
    metrics_for_plot = metrics_df.drop(columns=['Genes Analyzed'])
    
    ax = sns.heatmap(
        metrics_for_plot, 
        annot=True,
        vmin=-1, vmax=1,  # Correlation ranges from -1 to 1
        center=0,
        cmap=cmap,
        linewidths=0.5,
        fmt=".3f",
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8, "label": "Score"}
    )
    
    # Set title and labels
    plt.title(f"Within-Gene Ranking Performance (min {min_guides_per_gene} guides per gene)", 
              fontsize=16, pad=20)
    
    # Make axis labels more readable
    plt.ylabel("Prediction Tools", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    
    # Add a text note about gene counts
    gene_count_text = "Genes analyzed per tool:\n" + "\n".join(
        [f"{tool}: {int(metrics_df.loc[tool, 'Genes Analyzed'])}" for tool in metrics_df.index]
    )
    plt.figtext(0.02, 0.02, gene_count_text, fontsize=9)
    
    # Add explanatory notes
    plt.figtext(0.5, 0.01, 
                "Spearman: Overall ranking similarity within genes\n"
                "Kendall Tau: Agreement in pairwise comparisons\n"
                "Worst/Best Guide Accuracy: How often the tool correctly identifies extreme guides", 
                ha="center", fontsize=10, style='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save CSV of metrics
    csv_path = output_path.replace('.png', '.csv')
    metrics_df.to_csv(csv_path)
    
    print(f"Within-gene performance heatmap saved to {output_path}")
    print(f"Within-gene performance data saved to {csv_path}")
    return metrics_df


def main():
    """Main function to generate all heatmaps."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse the tools list
    tool_columns = [t.strip().strip('"\'') for t in args.tools.split(',')]
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Starting heatmap generation...")
    print(f"Input file: {args.input}")
    print(f"Tool columns: {tool_columns}")
    print(f"Output directory: {args.output}")
    
    # Read the input CSV file
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded dataframe with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Run the analysis - correlation heatmap
    create_tool_correlation_heatmap(
        df, 
        tool_columns, 
        efficacy_column=args.efficacy,
        output_path=os.path.join(args.output, "tool_correlation_heatmap.png")
    )

    # Run the analysis - performance metrics at different thresholds
    for threshold in [50, 70, 80, 90]:
        create_performance_metrics_heatmap(
            df,
            tool_columns,
            efficacy_column=args.efficacy,
            efficacy_threshold=threshold,
            output_path=os.path.join(args.output, f"performance_metrics_threshold_{threshold}.png")
        )

    # Run the analysis - within-gene ranking
    create_within_gene_performance_heatmap(
        df,
        tool_columns,
        gene_column=args.gene,
        efficacy_column=args.efficacy,
        output_path=os.path.join(args.output, "within_gene_performance_heatmap.png")
    )

    print("All heatmaps generated successfully!")


if __name__ == "__main__":
    main()

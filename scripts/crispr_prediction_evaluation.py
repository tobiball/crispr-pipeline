#!/usr/bin/env python3
"""
CRISPR Guide Prediction Tool Visualization

This script creates visualizations for evaluating CRISPR guide prediction tools.
It's designed to be called from a Rust pipeline, taking CSV data that contains:
- Efficacy scores
- Prediction scores from various tools

The script finds thresholds that identify at least 75% of good guides and
visualizes how many poor and moderate guides are incorrectly identified as good.

Usage:
    python crispr_prediction_visualization.py --input results.csv --output output_dir --efficacy-col efficacy
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")


class PredictionEvaluator:
    """Evaluate and visualize guide prediction tools."""

    def __init__(
            self,
            input_file: str,
            output_dir: str,
            efficacy_col: str = 'efficacy',
            poor_threshold: float = 50.0,
            good_threshold: float = 75.0,
            min_coverage: float = 0.75,
            tool_columns: List[str] = None,
            output_format: str = 'png'
    ):
        """Initialize with configuration parameters."""
        self.input_file = input_file
        self.output_dir = output_dir
        self.efficacy_col = efficacy_col
        self.poor_threshold = poor_threshold
        self.good_threshold = good_threshold
        self.min_coverage = min_coverage
        self.tool_columns = tool_columns
        self.output_format = output_format

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        self.df = pd.read_csv(input_file)

        # Auto-detect tool columns if not specified
        if not tool_columns:
            self.detect_tool_columns()

        # Add efficacy category
        self.df['efficacy_category'] = self.df[efficacy_col].apply(self.categorize_efficacy)

        # Count guides in each category
        self.count_categories()

    def run_evaluation(self):
        # must be indented at the same level as __init__
        self.evaluate_all_tools()
        self.create_misclassification_plot()
        self.create_stacked_bar_plot()
        self.create_bayesian_scatter_plot()  # New plot using Bayes-like calculations
        self.save_results_to_csv()

    def detect_tool_columns(self):
        """Automatically detect prediction tool columns."""
        # Assume numeric columns (except efficacy) are prediction tools
        self.tool_columns = [
            col for col in self.df.columns
            if col != self.efficacy_col
               and pd.api.types.is_numeric_dtype(self.df[col])
               and self.df[col].notna().sum() > 0
        ]

        # Filter out columns that are likely not prediction scores
        exclude_patterns = ['id', 'index', 'position', 'start', 'end', 'strand', 'chromosome']
        self.tool_columns = [
            col for col in self.tool_columns
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]

        print(f"Detected {len(self.tool_columns)} tool columns:")
        for col in self.tool_columns:
            print(f"  - {col}")

    def categorize_efficacy(self, value):
        """Categorize efficacy into 'Poor', 'Moderate', or 'Good'."""
        if pd.isna(value):
            return np.nan
        elif value < self.poor_threshold:
            return 'Poor'
        elif value >= self.good_threshold:
            return 'Good'
        else:
            return 'Moderate'

    def count_categories(self):
        """Count guides in each efficacy category."""
        counts = self.df['efficacy_category'].value_counts()
        self.poor_count = counts.get('Poor', 0)
        self.moderate_count = counts.get('Moderate', 0)
        self.good_count = counts.get('Good', 0)
        self.total_count = self.poor_count + self.moderate_count + self.good_count

        print("Guide counts by efficacy:")
        print(f"  Poor: {self.poor_count} ({self.poor_count/self.total_count:.1%})")
        print(f"  Moderate: {self.moderate_count} ({self.moderate_count/self.total_count:.1%})")
        print(f"  Good: {self.good_count} ({self.good_count/self.total_count:.1%})")

    def find_optimal_threshold(self, tool_col):
        """
        Find threshold that correctly identifies at least 75% of good guides
        while minimizing poor and moderate guide misclassification.

        Returns:
            Dict with threshold and classification metrics.
        """
        # Filter to valid data
        valid_df = self.df.dropna(subset=[tool_col, 'efficacy_category'])

        # Skip if no data
        if len(valid_df) == 0 or self.good_count == 0:
            print(f"Skipping {tool_col}: No valid data")
            return None

        # Get min/max values for this tool
        min_val = valid_df[tool_col].min()
        max_val = valid_df[tool_col].max()

        # Skip if all values are the same
        if min_val == max_val:
            print(f"Skipping {tool_col}: All prediction values are identical")
            return None

        # Try different thresholds
        steps = 100
        thresholds = np.linspace(min_val, max_val, steps)

        best_threshold = None
        best_quality_score = float('-inf')
        best_metrics = None

        for threshold in thresholds:
            # Predict 'Good' for values >= threshold
            good_predicted_good = valid_df[
                (valid_df['efficacy_category'] == 'Good')
                & (valid_df[tool_col] >= threshold)
                ].shape[0]
            poor_predicted_good = valid_df[
                (valid_df['efficacy_category'] == 'Poor')
                & (valid_df[tool_col] >= threshold)
                ].shape[0]
            moderate_predicted_good = valid_df[
                (valid_df['efficacy_category'] == 'Moderate')
                & (valid_df[tool_col] >= threshold)
                ].shape[0]

            # Calculate metrics
            good_coverage = good_predicted_good / self.good_count
            poor_misclass_rate = (poor_predicted_good / self.poor_count) if self.poor_count > 0 else 0
            moderate_misclass_rate = (moderate_predicted_good / self.moderate_count) if self.moderate_count > 0 else 0

            # Skip if coverage requirement not met
            if good_coverage < self.min_coverage:
                continue

            # Quality score: higher is better (penalize poor misclassification more)
            quality_score = good_coverage - (poor_misclass_rate + 0.5 * moderate_misclass_rate)

            if quality_score > best_quality_score:
                best_quality_score = quality_score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'good_coverage': good_coverage,
                    'poor_misclass_rate': poor_misclass_rate,
                    'moderate_misclass_rate': moderate_misclass_rate,
                    'quality_score': quality_score
                }

        if best_threshold is None:
            print(f"Skipping {tool_col}: No threshold meets {self.min_coverage:.0%} good guide coverage")
            return None

        print(f"Optimal threshold for {tool_col}: {best_threshold:.4f}")
        print(f"  Good coverage: {best_metrics['good_coverage']:.2%}")
        print(f"  Poor misclassification: {best_metrics['poor_misclass_rate']:.2%}")
        print(f"  Moderate misclassification: {best_metrics['moderate_misclass_rate']:.2%}")

        return best_metrics

    def evaluate_all_tools(self):
        """Evaluate all prediction tools."""
        results = {}

        for tool in self.tool_columns:
            metrics = self.find_optimal_threshold(tool)
            if metrics:
                results[tool] = metrics

        # Sort by quality score (higher is better)
        self.sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['quality_score'],
            reverse=True
        )

        return self.sorted_results

    def create_misclassification_plot(self):
        """
        Show '1 - moderate_misclass' on X-axis and '1 - poor_misclass' on Y-axis.
        Higher = better for both axes, so top-right corner is best.
        """
        if not hasattr(self, 'sorted_results') or not self.sorted_results:
            print("No results to plot")
            return

        plt.figure(figsize=(14, 10))

        tools = [r[0] for r in self.sorted_results]
        poor_misclass = [r[1]['poor_misclass_rate'] for r in self.sorted_results]
        moderate_misclass = [r[1]['moderate_misclass_rate'] for r in self.sorted_results]
        quality_scores = [r[1]['quality_score'] for r in self.sorted_results]

        # Transform misclass rates to "1 - misclass" so that higher is better
        x_vals = [1.0 - m for m in moderate_misclass]
        y_vals = [1.0 - p for p in poor_misclass]

        # Color points by the quality score
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(quality_scores), max(quality_scores))

        # Scale point sizes by rank (bigger = better)
        sizes = [
            150 * (1 + (len(self.sorted_results) - i) / len(self.sorted_results))
            for i in range(len(self.sorted_results))
        ]

        scatter = plt.scatter(
            x_vals,
            y_vals,
            c=quality_scores,
            cmap=cmap,
            norm=norm,
            s=sizes,
            alpha=0.8,
            edgecolors='black'
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Quality Score (higher = better)', fontsize=12)

        plt.xlabel('1 - Moderate Misclassification (higher = better)', fontsize=14)
        plt.ylabel('1 - Poor Misclassification (higher = better)', fontsize=14)
        plt.title('Tool Performance (Top-Right = Best)', fontsize=16)

        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)

        for i, tool_name in enumerate(tools):
            plt.annotate(
                tool_name,
                (x_vals[i], y_vals[i]),
                xytext=(7, 7),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )

        textstr = (
            "• X-axis: 1 - moderate misclass. (higher = fewer mod. mislabeled)\n"
            "• Y-axis: 1 - poor misclass. (higher = fewer poor mislabeled)\n"
            "• Color: overall quality score (higher = better)\n"
            "• Size: rank-based (larger = better)\n\n"
            "Best tools = top-right corner (no misclassification)."
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.annotate(
            textstr,
            xy=(0.03, 0.97),
            xycoords='axes fraction',
            fontsize=12,
            va='top',
            ha='left',
            bbox=props
        )

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'tool_misclassification_comparison.{self.output_format}')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Misclassification plot saved to {output_path}")

    def create_stacked_bar_plot(self):
        """
        Create a stacked bar plot showing classification performance
        for each tool.
        """
        if not hasattr(self, 'sorted_results') or not self.sorted_results:
            print("No results to plot")
            return

        plt.figure(figsize=(16, 12))

        tools = [r[0] for r in self.sorted_results]
        good_coverage = [r[1]['good_coverage'] for r in self.sorted_results]
        poor_rates = [r[1]['poor_misclass_rate'] for r in self.sorted_results]
        moderate_rates = [r[1]['moderate_misclass_rate'] for r in self.sorted_results]

        colors = {
            'good': '#2ecc71',
            'poor': '#e74c3c',
            'moderate': '#f39c12'
        }

        y_pos = range(len(tools))

        plt.barh(
            y_pos,
            good_coverage,
            color=colors['good'],
            alpha=0.8,
            label='Good guide coverage',
            height=0.6
        )
        plt.barh(
            y_pos,
            poor_rates,
            color=colors['poor'],
            alpha=0.8,
            label='Poor misclassification',
            height=0.6
        )
        plt.barh(
            y_pos,
            moderate_rates,
            color=colors['moderate'],
            alpha=0.8,
            label='Moderate misclassification',
            height=0.6,
            left=poor_rates
        )

        plt.yticks(y_pos, tools)
        plt.xlim(0, max(1.0, max(g + p + m for g, p, m in zip(good_coverage, poor_rates, moderate_rates))))
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])

        plt.axvline(x=self.min_coverage, color='black', linestyle='--', alpha=0.7)
        plt.text(self.min_coverage + 0.01, len(tools) - 0.8,
                 f'Min. requirement: {self.min_coverage:.0%}',
                 va='center', fontsize=10)

        plt.grid(True, linestyle='--', alpha=0.3, axis='x')
        plt.title(f'Guide Prediction Tool Evaluation\n(Min. good coverage: {self.min_coverage:.0%})', fontsize=16)
        plt.xlabel('Rate', fontsize=14)
        plt.legend(loc='lower right')

        for i, result in enumerate(self.sorted_results):
            quality = result[1]['quality_score']
            plt.text(1.02, i, f'Score: {quality:.3f}', va='center')

        textstr = '\n'.join([
            'Plot Guide:',
            f'• Green: Good coverage (≥{self.min_coverage:.0%})',
            '• Red: Poor guides mislabeled as good',
            '• Orange: Moderate guides mislabeled as good',
            '• Stacked red+orange = total misclassification',
            '• Higher score = better overall performance'
        ])
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.annotate(
            textstr,
            xy=(0.02, 0.02),
            xycoords='axes fraction',
            fontsize=12,
            va='bottom',
            ha='left',
            bbox=props
        )

        output_path = os.path.join(self.output_dir, f'tool_stacked_performance.{self.output_format}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Stacked bar plot saved to {output_path}")

    def create_bayesian_scatter_plot(self):
        """
        Create a scatter plot using adjusted PPVs with balanced class assumption:
        - x-axis: Probability a guide is truly good, adjusted for balanced classes
        - y-axis: Probability a guide is good or moderate, adjusted for balanced classes
        """
        if not hasattr(self, 'sorted_results') or not self.sorted_results:
            print("No results to plot")
            return

        tools = []
        adjusted_ppv_good = []
        adjusted_ppv_good_mod = []
        quality_scores = []

        # For each tool, compute adjusted PPVs
        for tool, metrics in self.sorted_results:
            r_G = metrics['good_coverage']             # Sensitivity for good guides
            r_M = metrics['moderate_misclass_rate']    # Rate at which moderate guides are predicted good
            r_P = metrics['poor_misclass_rate']        # Rate at which poor guides are predicted good

            # Calculate adjusted PPV for good guides, assuming equal class distribution
            balanced_ppv_g = r_G / (r_G + r_M + r_P)

            # Calculate adjusted PPV for good+moderate guides
            balanced_ppv_gm = (r_G + r_M) / (r_G + r_M + r_P)

            tools.append(tool)
            adjusted_ppv_good.append(balanced_ppv_g * 100)  # Convert to percentage
            adjusted_ppv_good_mod.append(balanced_ppv_gm * 100)  # Convert to percentage
            quality_scores.append(metrics['quality_score'])

        # Determine appropriate axis limits based on data
        max_x = max(adjusted_ppv_good)
        max_y = max(adjusted_ppv_good_mod)
        min_x = min(adjusted_ppv_good)
        min_y = min(adjusted_ppv_good_mod)

        # Add some padding (10%)
        x_padding = 0.1 * (max_x - min_x)
        y_padding = 0.1 * (max_y - min_y)

        # Create the plot
        plt.figure(figsize=(14, 10))
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(quality_scores), max(quality_scores))
        sizes = [
            150 * (1 + (len(self.sorted_results) - i) / len(self.sorted_results))
            for i in range(len(self.sorted_results))
        ]
        scatter = plt.scatter(
            adjusted_ppv_good,
            adjusted_ppv_good_mod,
            c=quality_scores,
            cmap=cmap,
            norm=norm,
            s=sizes,
            alpha=0.8,
            edgecolors='black'
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('Quality Score (G - P - 0.5M)', fontsize=12)

        plt.xlabel('Probability (%) Guide is Good (class-balanced)', fontsize=14)
        plt.ylabel('Probability (%) Guide is Good or Moderate (class-balanced)', fontsize=14)
        plt.title('gRNA Prediction Tool Performance', fontsize=16)

        # Set axis limits with padding
        plt.xlim(min_x - x_padding, max_x + x_padding)
        plt.ylim(min_y - y_padding, max_y + y_padding)

        # Add reference lines at 33.3% and 66.7% (equal chances for 3 classes)
        plt.axvline(x=33.3, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=66.7, color='gray', linestyle='--', alpha=0.5)

        # Annotate each point with the tool name - positioned to avoid colorbar
        for i, tool in enumerate(tools):
            # Check if point is close to the right edge (where colorbar is)
            x_position = adjusted_ppv_good[i]
            y_position = adjusted_ppv_good_mod[i]

            # Adjust text position to avoid colorbar overlap
            if x_position > 0.9 * max_x:
                x_offset = -50  # Place text to the left of the point
                align = 'right'
            else:
                x_offset = 7    # Default: place text to the right
                align = 'left'

            plt.annotate(
                tool,
                (x_position, y_position),
                xytext=(x_offset, 7),
                textcoords='offset points',
                fontsize=10,
                ha=align,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )

        # Add a concise annotation explaining the approach
        textstr = (
            "Analysis with balanced class assumption.\n"
            "Quality Score = Good coverage - Poor misclass. - 0.5×Moderate misclass.\n"
            "Ref. lines at 33.3% and 66.7% show random performance levels."
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.annotate(textstr, xy=(0.05, 0.05), xycoords='axes fraction',
                     fontsize=11, ha='left', va='bottom', bbox=props)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'tool_balanced_probabilities.{self.output_format}')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Balanced probability plot saved to {output_path}")

    def save_results_to_csv(self):
        """Save evaluation results to CSV and JSON."""
        if not hasattr(self, 'sorted_results') or not self.sorted_results:
            print("No results to save")
            return

        data = {
            'Tool': [],
            'Threshold': [],
            'Good Guide Coverage': [],
            'Poor Guide Misclassification': [],
            'Moderate Guide Misclassification': [],
            'Quality Score': []
        }

        for tool, metrics in self.sorted_results:
            data['Tool'].append(tool)
            data['Threshold'].append(metrics['threshold'])
            data['Good Guide Coverage'].append(metrics['good_coverage'])
            data['Poor Guide Misclassification'].append(metrics['poor_misclass_rate'])
            data['Moderate Guide Misclassification'].append(metrics['moderate_misclass_rate'])
            data['Quality Score'].append(metrics['quality_score'])

        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'tool_evaluation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Also save to JSON
        json_path = os.path.join(self.output_dir, 'tool_evaluation_results.json')
        results_dict = {
            'tools': [
                {
                    'name': tool,
                    'threshold': float(metrics['threshold']),
                    'good_guide_coverage': float(metrics['good_coverage']),
                    'poor_misclassification_rate': float(metrics['poor_misclass_rate']),
                    'moderate_misclassification_rate': float(metrics['moderate_misclass_rate']),
                    'quality_score': float(metrics['quality_score']),
                }
                for tool, metrics in self.sorted_results
            ],
            'parameters': {
                'efficacy_column': self.efficacy_col,
                'poor_threshold': float(self.poor_threshold),
                'good_threshold': float(self.good_threshold),
                'min_coverage': float(self.min_coverage)
            },
            'category_counts': {
                'poor': int(self.poor_count),
                'moderate': int(self.moderate_count),
                'good': int(self.good_count),
                'total': int(self.total_count),
            }
        }

        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {json_path}")

    def run_evaluation(self):
        """Run the complete evaluation process."""
        print(f"Starting evaluation of {len(self.tool_columns)} prediction tools")
        self.evaluate_all_tools()
        self.create_misclassification_plot()
        self.create_stacked_bar_plot()
        self.create_bayesian_scatter_plot()
        self.save_results_to_csv()
        print("Evaluation complete")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate CRISPR guide prediction tools")
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='./prediction_viz_results', help='Output directory')
    parser.add_argument('--efficacy-col', default='efficacy', help='Column name for efficacy values')
    parser.add_argument('--poor-threshold', type=float, default=50.0, help='Threshold for poor guides')
    parser.add_argument('--good-threshold', type=float, default=75.0, help='Threshold for good guides')
    parser.add_argument('--min-coverage', type=float, default=0.75, help='Minimum good guide coverage (0-1)')
    parser.add_argument('--tools', help='Comma-separated list of tool columns (default: auto-detect)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png', help='Output format for figures')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Parse tool columns if provided
    tool_columns = args.tools.split(',') if args.tools else None

    # Create evaluator
    evaluator = PredictionEvaluator(
        input_file=args.input,
        output_dir=args.output,
        efficacy_col=args.efficacy_col,
        poor_threshold=args.poor_threshold,
        good_threshold=args.good_threshold,
        min_coverage=args.min_coverage,
        tool_columns=tool_columns,
        output_format=args.format
    )

    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()

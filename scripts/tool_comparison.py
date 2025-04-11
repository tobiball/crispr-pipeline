#!/usr/bin/env python
"""
Robust CRISPR Guide RNA Tool Comparison Script

This script compares multiple gRNA prediction tools against measured guide efficacy.
It creates confusion matrices, correlation charts, poor-guide detection stats,
within-gene ranking analysis, a performance dashboard, ROC/PR curves, and distribution plots.

Key differences from simpler versions:
  - More careful checks of columns (existence, numeric, variation).
  - Debug prints for correlations.
  - Pure matplotlib usage for all charts (no seaborn).
  - Each chart is in its own figure.

Author: <Your Name>
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


def load_data(
        input_path: str,
        efficacy_col: str,
        gene_col: str,
) -> pd.DataFrame:
    """
    Load and prepare dataset from CSV.
    Verifies that 'efficacy_col' and 'gene_col' exist (though gene_col can be optional for some analyses).
    Prints debug info about columns and dtypes.
    """

    # 1) Attempt to load CSV using comma delimiter, fallback to tab.
    try:
        df = pd.read_csv(input_path)
    except:
        df = pd.read_csv(input_path, sep='\t')

    print(f"[DEBUG] Loaded dataset from {input_path}")
    print(f"[DEBUG] Columns: {df.columns.tolist()}")
    print("[DEBUG] dtypes:\n", df.dtypes)
    print("[DEBUG] Head:\n", df.head(5).to_string())

    # 2) Confirm efficacy_col is present.
    if efficacy_col not in df.columns:
        raise ValueError(f"ERROR: Efficacy column '{efficacy_col}' not found in CSV.")

    # 3) If gene_col not present, we won't fail, but we note it for the gene-based analysis step.
    if gene_col not in df.columns:
        print(f"[WARNING] Gene column '{gene_col}' not found. Within-gene ranking will be skipped.")

    # 4) Ensure the efficacy_col is numeric if possible.
    #    If it's object/string, attempt to coerce to numeric.
    if not pd.api.types.is_numeric_dtype(df[efficacy_col]):
        try:
            df[efficacy_col] = pd.to_numeric(df[efficacy_col], errors="coerce")
            num_nans = df[efficacy_col].isna().sum()
            if num_nans > 0:
                print(f"[WARNING] {num_nans} NaN values found after forcing '{efficacy_col}' to numeric.")
        except Exception as e:
            raise ValueError(
                f"ERROR: Could not convert column '{efficacy_col}' to numeric. Details: {e}"
            )

    return df


def normalize_scores(scores: np.ndarray, min_val: float = 0, max_val: float = 100) -> np.ndarray:
    """
    Normalize scores to a 0-100 range.
    If there's no variation or all NaN, return all zeros.
    """
    valid = ~np.isnan(scores)
    if not np.any(valid):
        return np.zeros_like(scores)

    valid_scores = scores[valid]
    smin = valid_scores.min()
    smax = valid_scores.max()

    if smax == smin:
        return np.zeros_like(scores)  # no variation => all zeros

    # scale
    out = np.zeros_like(scores)
    out[valid] = (valid_scores - smin) / (smax - smin) * (max_val - min_val) + min_val
    return out


def create_confusion_matrix_grid(
        df: pd.DataFrame,
        tools: list[str],
        efficacy_col: str,
        poor_threshold: float,
        moderate_threshold: float,
        output_dir: str
):
    """
    Create a grid of confusion matrices showing how each tool categorizes guides
    into Poor/Moderate/High vs the true category.
    Each confusion matrix gets its own figure.
    """

    # True categories based on efficacy
    true_cat = pd.cut(
        df[efficacy_col],
        bins=[-float("inf"), poor_threshold, moderate_threshold, float("inf")],
        labels=["Poor", "Moderate", "High"]
    )

    for tool in tools:
        if tool not in df.columns:
            print(f"[WARNING] Tool '{tool}' not found in dataset for confusion matrix.")
            continue

        # Ensure numeric
        if not pd.api.types.is_numeric_dtype(df[tool]):
            print(f"[WARNING] Tool '{tool}' is not numeric, skipping confusion matrix.")
            continue

        # Normalize
        norm_scores = normalize_scores(df[tool].values)

        # Predicted category
        pred_cat = pd.cut(
            norm_scores,
            bins=[-float("inf"), poor_threshold, moderate_threshold, float("inf")],
            labels=["Poor", "Moderate", "High"]
        )

        # Confusion matrix
        cm = confusion_matrix(true_cat, pred_cat, labels=["Poor", "Moderate", "High"])
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = np.round(cm / cm_sum * 100, 1)

        plt.figure(figsize=(4, 4))
        plt.imshow(cm_percent, interpolation='nearest', cmap="Blues")
        plt.title(f"Confusion Matrix\n{tool}", fontsize=12)
        plt.xticks([0, 1, 2], ["Poor", "Moderate", "High"], rotation=45)
        plt.yticks([0, 1, 2], ["Poor", "Moderate", "High"])
        plt.colorbar(label="Percent")

        # Label each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_percent[i, j]
                plt.text(
                    j, i, f"{val:.1f}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="black",
                    fontsize=12
                )

        plt.xlabel("Predicted Category", fontsize=10)
        plt.ylabel("True Category", fontsize=10)

        # Save each confusion matrix as separate file
        fname = os.path.join(output_dir, f"cm_{tool.replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

    print(f"[INFO] Confusion matrices saved (one per tool) to {output_dir}")


def create_rank_correlation_chart(
        df: pd.DataFrame,
        tools: list[str],
        efficacy_col: str,
        output_dir: str
):
    """
    Create a bar chart showing Spearman, Kendall, and Pearson correlations
    between each tool and the efficacy column.
    Writes these correlations to a CSV and also shows them in a bar chart.
    """

    # We'll store correlation results in a list of dicts
    corr_data = []

    for tool in tools:
        if tool not in df.columns:
            print(f"[WARNING] Tool '{tool}' not found for correlation, skipping.")
            continue

        # Must be numeric + must have variation
        coldata = df[tool]
        if not pd.api.types.is_numeric_dtype(coldata):
            print(f"[WARNING] Tool '{tool}' is not numeric, skipping correlation.")
            continue

        valid_mask = ~df[efficacy_col].isna() & ~coldata.isna()
        valid_efficacy = df.loc[valid_mask, efficacy_col]
        valid_scores = coldata.loc[valid_mask]

        if valid_scores.nunique() <= 1:
            print(f"[DEBUG] Tool '{tool}' => no variation after dropping NaNs. Correlation = 0.")
            corr_data.append(
                {
                    "Tool": tool,
                    "Spearman": 0.0,
                    "Kendall": 0.0,
                    "Pearson": 0.0,
                }
            )
            continue

        # Compute correlations
        # If any correlation is NaN, we store it as np.nan
        sp_val, _ = spearmanr(valid_efficacy, valid_scores)
        kt_val, _ = kendalltau(valid_efficacy, valid_scores)
        pr_val, _ = pearsonr(valid_efficacy, valid_scores)

        print(f"[DEBUG] {tool} => Spearman: {sp_val:.3f}, Kendall: {kt_val:.3f}, Pearson: {pr_val:.3f}")

        corr_data.append(
            {
                "Tool": tool,
                "Spearman": sp_val,
                "Kendall": kt_val,
                "Pearson": pr_val,
            }
        )

    corr_df = pd.DataFrame(corr_data)
    corr_df.to_csv(os.path.join(output_dir, "tool_correlations.csv"), index=False)

    # Make a bar chart for each correlation type in groups
    if len(corr_df) == 0:
        print("[WARNING] No valid correlation data. Skipping rank correlation chart.")
        return

    # Sort by Spearman descending
    corr_df = corr_df.sort_values("Spearman", ascending=False)

    x = np.arange(len(corr_df))
    width = 0.25

    plt.figure(figsize=(8, 6))
    plt.bar(x - width, corr_df["Spearman"], width=width, label="Spearman")
    plt.bar(x, corr_df["Kendall"], width=width, label="Kendall")
    plt.bar(x + width, corr_df["Pearson"], width=width, label="Pearson")

    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(x, corr_df["Tool"], rotation=45, ha="right")
    plt.ylabel("Correlation Coefficient")
    plt.title("Correlation with True Efficacy")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(output_dir, "tool_rank_correlations.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"[INFO] Saved rank correlation chart to {outpath}")


def create_poor_guide_detection_chart(
        df: pd.DataFrame,
        tools: list[str],
        efficacy_col: str,
        poor_threshold: float,
        output_dir: str
):
    """
    Evaluate how well each tool detects "Poor" guides (those below poor_threshold).
    We'll define a threshold for each tool by the same fraction of poor guides
    and measure precision/recall/f1/accuracy.
    """

    # Identify which rows are truly poor
    poor_mask = df[efficacy_col] < poor_threshold
    n_poor = poor_mask.sum()
    if n_poor == 0:
        print("[WARNING] No poor guides in dataset, skipping poor guide detection.")
        return

    fraction_poor = n_poor / len(df)
    results = []

    for tool in tools:
        if tool not in df.columns:
            print(f"[WARNING] Tool '{tool}' not found for poor detection, skipping.")
            continue

        coldata = df[tool]
        if not pd.api.types.is_numeric_dtype(coldata):
            print(f"[WARNING] Tool '{tool}' not numeric for poor detection, skipping.")
            continue

        # normalize
        norm_scores = normalize_scores(coldata.values)
        # choose threshold as the 'fraction_poor' percentile
        threshold = np.percentile(norm_scores[~np.isnan(norm_scores)], fraction_poor * 100)

        # predicted poor if <= threshold
        pred_poor = norm_scores <= threshold

        tp = np.sum(poor_mask & pred_poor)
        fp = np.sum(~poor_mask & pred_poor)
        fn = np.sum(poor_mask & ~pred_poor)
        tn = np.sum(~poor_mask & ~pred_poor)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        acc = (tp + tn) / len(df)

        results.append({
            "Tool": tool,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Accuracy": acc,
            "Threshold": threshold,
        })

    if len(results) == 0:
        print("[WARNING] No results for poor guide detection.")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "poor_guide_detection_metrics.csv"), index=False)

    # Sort by F1
    res_df = res_df.sort_values("F1", ascending=False)

    x = np.arange(len(res_df))
    width = 0.2

    plt.figure(figsize=(8, 6))
    plt.bar(x - width*1.5, res_df["Precision"], width=width, label="Precision")
    plt.bar(x - width/2, res_df["Recall"], width=width, label="Recall")
    plt.bar(x + width/2, res_df["F1"], width=width, label="F1")
    plt.bar(x + width*1.5, res_df["Accuracy"], width=width, label="Accuracy")

    plt.xticks(x, res_df["Tool"], rotation=45, ha="right")
    plt.ylim([0, 1])
    plt.ylabel("Score")
    plt.title(f"Poor Guide Detection (Efficacy < {poor_threshold})")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(output_dir, "poor_guide_detection.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"[INFO] Saved poor guide detection chart to {outpath}")


def create_within_gene_ranking_analysis(
        df: pd.DataFrame,
        tools: list[str],
        efficacy_col: str,
        gene_col: str,
        output_dir: str
):
    """
    Check how well each tool ranks guides within the same gene (if gene_col is present).
    We'll measure how often the tool picks the same top- (and bottom-) scoring guide as the real efficacy.
    Also measure a basic "ranking accuracy" via Spearman for each gene with >2 guides.
    """

    if gene_col not in df.columns:
        print(f"[WARNING] No gene column '{gene_col}' found, skipping within-gene ranking.")
        return

    gene_counts = df[gene_col].value_counts()
    multi_genes = gene_counts[gene_counts > 1].index.tolist()
    if len(multi_genes) == 0:
        print("[WARNING] No genes with multiple guides, skipping within-gene ranking.")
        return

    results = []

    for tool in tools:
        if tool not in df.columns:
            print(f"[WARNING] Tool '{tool}' not found for within-gene analysis, skipping.")
            continue

        coldata = df[tool]
        if not pd.api.types.is_numeric_dtype(coldata):
            print(f"[WARNING] Tool '{tool}' is not numeric, skipping within-gene analysis.")
            continue

        correct_top = 0
        correct_bot = 0
        total_genes = 0
        good_ranking_count = 0  # e.g. if Spearman > 0.5

        for gene in multi_genes:
            sub = df[df[gene_col] == gene]
            if len(sub) < 2:
                continue
            total_genes += 1

            # top guide
            true_top_idx = sub[efficacy_col].idxmax()
            pred_top_idx = sub[tool].idxmax()
            if true_top_idx == pred_top_idx:
                correct_top += 1

            # bottom guide
            true_bot_idx = sub[efficacy_col].idxmin()
            pred_bot_idx = sub[tool].idxmin()
            if true_bot_idx == pred_bot_idx:
                correct_bot += 1

            # rank correlation check
            if len(sub) > 2:
                # drop NaN
                submask = ~sub[efficacy_col].isna() & ~sub[tool].isna()
                if submask.sum() >= 2:
                    spv, _ = spearmanr(sub.loc[submask, efficacy_col], sub.loc[submask, tool])
                    if spv > 0.5:
                        good_ranking_count += 1

        top_acc = correct_top / total_genes if total_genes > 0 else 0
        bot_acc = correct_bot / total_genes if total_genes > 0 else 0
        overall_ranking = good_ranking_count / total_genes if total_genes > 0 else 0

        results.append({
            "Tool": tool,
            "Top_Guide_Accuracy": top_acc,
            "Bottom_Guide_Accuracy": bot_acc,
            "Ranking_Accuracy": overall_ranking,
            "Genes_Analyzed": total_genes
        })

    if len(results) == 0:
        print("[WARNING] No within-gene ranking results.")
        return

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(output_dir, "within_gene_ranking_metrics.csv"), index=False)

    # Sort by top_guide_accuracy
    res_df = res_df.sort_values("Top_Guide_Accuracy", ascending=False)

    x = np.arange(len(res_df))
    width = 0.25

    plt.figure(figsize=(8, 6))
    plt.bar(x - width, res_df["Top_Guide_Accuracy"], width=width, label="Top Guide")
    plt.bar(x, res_df["Bottom_Guide_Accuracy"], width=width, label="Bottom Guide")
    plt.bar(x + width, res_df["Ranking_Accuracy"], width=width, label="Spearman>0.5")

    plt.xticks(x, res_df["Tool"], rotation=45, ha="right")
    plt.ylim([0, 1])
    plt.ylabel("Accuracy")
    plt.title("Within-Gene Ranking Accuracy")
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(output_dir, "within_gene_ranking.png")
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"[INFO] Saved within-gene ranking results to {outpath}")


def create_tool_performance_dashboard(
        df: pd.DataFrame,
        tools: list[str],
        efficacy_col: str,
        gene_col: str,
        poor_threshold: float,
        moderate_threshold: float,
        output_dir: str
):
    """
    Compute key performance metrics for each tool and summarize in a table-like figure:
      - Spearman, Kendall
      - Category Agreement (3-class)
      - Poor-Guide F1
      - Top Guide Accuracy (if gene_col is present)
    Then save as CSV and also produce a 'table' figure using matplotlib.
    """

    dashboard_data = []

    # We'll re-use some partial logic from the other analyses
    # 1) pre-compute real category
    true_cat = pd.cut(
        df[efficacy_col],
        bins=[-float("inf"), poor_threshold, moderate_threshold, float("inf")],
        labels=["Poor", "Moderate", "High"]
    )
    df["true_category"] = true_cat

    fraction_poor = (true_cat == "Poor").mean()

    for tool in tools:
        if tool not in df.columns:
            print(f"[WARNING] Tool '{tool}' not found for dashboard, skipping.")
            continue

        coldata = df[tool]
        if not pd.api.types.is_numeric_dtype(coldata):
            print(f"[WARNING] Tool '{tool}' is not numeric for dashboard, skipping.")
            continue

        tool_info = {"Tool": tool}

        # Correlations
        valid_mask = ~df[efficacy_col].isna() & ~coldata.isna()
        if valid_mask.sum() > 1 and coldata[valid_mask].nunique() > 1:
            sp, _ = spearmanr(df.loc[valid_mask, efficacy_col], coldata.loc[valid_mask])
            kt, _ = kendalltau(df.loc[valid_mask, efficacy_col], coldata.loc[valid_mask])
        else:
            sp, kt = (np.nan, np.nan)

        tool_info["Spearman"] = sp
        tool_info["Kendall"] = kt

        # Category agreement
        norm_scores = normalize_scores(coldata.values)
        pred_cat = pd.cut(
            norm_scores,
            bins=[-float("inf"), poor_threshold, moderate_threshold, float("inf")],
            labels=["Poor", "Moderate", "High"]
        )
        cat_agree = (pred_cat == true_cat).mean()
        tool_info["Category_Agreement"] = cat_agree

        # Poor guide detection (precision, recall, F1)
        threshold = np.percentile(norm_scores[~np.isnan(norm_scores)], fraction_poor * 100)
        predicted_poor = norm_scores <= threshold
        tp = np.sum((true_cat == "Poor") & predicted_poor)
        fp = np.sum((true_cat != "Poor") & predicted_poor)
        fn = np.sum((true_cat == "Poor") & (~predicted_poor))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        tool_info["Poor_Guide_F1"] = f1

        # Within-gene top guide accuracy
        if gene_col in df.columns and gene_col in df:
            gene_counts = df[gene_col].value_counts()
            multi_genes = gene_counts[gene_counts > 1].index
            correct_top = 0
            total_g = 0
            for g in multi_genes:
                sub = df[df[gene_col] == g]
                if sub[efficacy_col].nunique() < 1:
                    continue
                total_g += 1
                best_true = sub[efficacy_col].idxmax()
                best_pred = sub[tool].idxmax()
                if best_true == best_pred:
                    correct_top += 1
            if total_g > 0:
                tool_info["Top_Guide_Accuracy"] = correct_top / total_g
            else:
                tool_info["Top_Guide_Accuracy"] = np.nan
        else:
            tool_info["Top_Guide_Accuracy"] = np.nan

        dashboard_data.append(tool_info)

    dash_df = pd.DataFrame(dashboard_data)
    dash_out = os.path.join(output_dir, "tool_performance_metrics.csv")
    dash_df.to_csv(dash_out, index=False)

    # Now produce a table figure
    plt.figure(figsize=(9, 0.4 * len(dash_df) + 2))
    plt.axis('off')
    plt.title("Tool Performance Dashboard", fontsize=16, pad=20)

    # Sort by Spearman descending
    dash_df = dash_df.sort_values("Spearman", ascending=False)

    columns_to_show = ["Tool", "Spearman", "Kendall", "Category_Agreement", "Poor_Guide_F1", "Top_Guide_Accuracy"]
    display_df = dash_df[columns_to_show].copy()

    # Convert numeric to short strings
    for c in ["Spearman", "Kendall", "Category_Agreement", "Poor_Guide_F1", "Top_Guide_Accuracy"]:
        display_df[c] = display_df[c].apply(
            lambda x: f"{x:.3f}" if pd.notnull(x) else "NaN"
        )

    cell_text = display_df.values.tolist()
    col_labels = display_df.columns.tolist()

    the_table = plt.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    # Adjust column widths if needed:
    # for i in range(len(col_labels)):
    #     the_table.column(i).set_width(0.1)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "tool_performance_dashboard.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved tool performance dashboard to {outpath}")


def create_roc_pr_curves(
        df: pd.DataFrame,
        tools: list[str],
        efficacy_col: str,
        moderate_threshold: float,
        output_dir: str
):
    """
    Create separate figures for ROC and PR curves (rather than subplots),
    for the binary classification: "High" vs "not High" (≥ moderate_threshold).
    Save AUCs to CSV.
    """

    # Binary label for "High"
    y_true = (df[efficacy_col] >= moderate_threshold).astype(int)

    roc_results = []
    pr_results = []

    for tool in tools:
        if tool not in df.columns:
            print(f"[WARNING] Tool '{tool}' not found for ROC/PR.")
            continue

        coldata = df[tool]
        if not pd.api.types.is_numeric_dtype(coldata):
            print(f"[WARNING] Tool '{tool}' not numeric for ROC/PR, skipping.")
            continue

        # handle NaN
        mask = ~coldata.isna()
        if mask.sum() < 10:
            print(f"[WARNING] Tool '{tool}' has <10 valid data points for ROC/PR, skipping.")
            continue

        y_ = y_true[mask]
        preds = normalize_scores(coldata[mask].values)

        # ROC
        fpr, tpr, _ = roc_curve(y_, preds)
        rauc = auc(fpr, tpr)

        # PR
        precision, recall, _ = precision_recall_curve(y_, preds)
        ap = average_precision_score(y_, preds)

        roc_results.append({"Tool": tool, "ROC_AUC": rauc})
        pr_results.append({"Tool": tool, "AP": ap})

        # Save each curve in its own figure or you can overlay them
        # 1) ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"{tool} (AUC={rauc:.3f})")
        plt.plot([0,1],[0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (High Efficacy ≥ {moderate_threshold})\n{tool}")
        plt.legend()
        outpath_roc = os.path.join(output_dir, f"roc_curve_{tool.replace(' ', '_')}.png")
        plt.savefig(outpath_roc, dpi=300)
        plt.close()

        # 2) PR curve
        plt.figure()
        plt.plot(recall, precision, label=f"{tool} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve (High Efficacy ≥ {moderate_threshold})\n{tool}")
        plt.legend()
        outpath_pr = os.path.join(output_dir, f"pr_curve_{tool.replace(' ', '_')}.png")
        plt.savefig(outpath_pr, dpi=300)
        plt.close()

    # Save combined results as CSV
    combined = pd.merge(
        pd.DataFrame(roc_results),
        pd.DataFrame(pr_results),
        on="Tool", how="outer"
    )
    combined.to_csv(os.path.join(output_dir, "roc_pr_metrics.csv"), index=False)

    print("[INFO] Saved individual ROC and PR curves for each tool.")
    print("[INFO] Also saved roc_pr_metrics.csv in output directory.")


def create_distribution_comparison(
        df: pd.DataFrame,
        tools: list[str],
        efficacy_col: str,
        poor_threshold: float,
        moderate_threshold: float,
        output_dir: str
):
    """
    Compare distribution of each tool's scores (normalized) to the distribution of actual efficacy.
    We'll do a single figure with multiple hist or KDE lines, but here we'll just do multiple
    hist overlays or lines. Using separate figure to follow the one-chart-per-figure approach.
    """

    # 1) Efficacy distribution
    plt.figure()
    plt.hist(df[efficacy_col].dropna(), bins=30, alpha=0.5, label="True Efficacy")
    plt.axvline(poor_threshold, linestyle='--')
    plt.axvline(moderate_threshold, linestyle='--')
    plt.xlabel("Efficacy")
    plt.ylabel("Count")
    plt.title("Distribution of True Efficacy")
    plt.legend()
    out_efficacy = os.path.join(output_dir, "distribution_efficacy.png")
    plt.savefig(out_efficacy, dpi=300)
    plt.close()

    # 2) For each tool, do a separate distribution figure
    for tool in tools:
        if tool not in df.columns:
            print(f"[WARNING] Tool '{tool}' missing for distribution chart, skipping.")
            continue

        coldata = df[tool]
        if not pd.api.types.is_numeric_dtype(coldata):
            print(f"[WARNING] Tool '{tool}' is not numeric, skipping distribution chart.")
            continue

        normed = normalize_scores(coldata.values)
        plt.figure()
        plt.hist(normed[~np.isnan(normed)], bins=30, alpha=0.7)
        plt.axvline(poor_threshold, linestyle='--')
        plt.axvline(moderate_threshold, linestyle='--')
        plt.title(f"Distribution of {tool} (normalized 0-100)")
        plt.xlabel("Normalized Score")
        plt.ylabel("Count")
        outpath = os.path.join(output_dir, f"distribution_{tool.replace(' ', '_')}.png")
        plt.savefig(outpath, dpi=300)
        plt.close()

    print("[INFO] Saved distribution comparisons (efficacy plus each tool) to output directory.")


def run_tool_comparison(
        input_path: str,
        tools: list[str],
        efficacy_col: str = "efficacy",
        gene_col: str = "Gene",
        output_dir: str = "./tool_comparison_results",
        poor_threshold: float = 50.0,
        moderate_threshold: float = 75.0
):
    """
    Master function to run all the analyses.
    Returns list of output files created.
    """

    os.makedirs(output_dir, exist_ok=True)
    print("\n=== ROBUST CRISPR TOOL COMPARISON ===\n")
    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Efficacy column: {efficacy_col}")
    print(f"[INFO] Gene column: {gene_col}")
    print(f"[INFO] Tools: {tools}")
    print(f"[INFO] Thresholds: Poor <{poor_threshold}, Moderate <{moderate_threshold}, High ≥{moderate_threshold}\n")

    # Load data
    df = load_data(input_path, efficacy_col, gene_col)

    # Create each analysis
    create_confusion_matrix_grid(
        df, tools, efficacy_col,
        poor_threshold, moderate_threshold,
        output_dir
    )
    create_rank_correlation_chart(df, tools, efficacy_col, output_dir)
    create_poor_guide_detection_chart(df, tools, efficacy_col, poor_threshold, output_dir)
    create_within_gene_ranking_analysis(df, tools, efficacy_col, gene_col, output_dir)
    create_tool_performance_dashboard(
        df, tools, efficacy_col, gene_col,
        poor_threshold, moderate_threshold,
        output_dir
    )
    create_roc_pr_curves(df, tools, efficacy_col, moderate_threshold, output_dir)
    create_distribution_comparison(
        df, tools, efficacy_col,
        poor_threshold, moderate_threshold,
        output_dir
    )

    print("\nAll analyses completed successfully!")
    print(f"Results saved to: {output_dir}\n")

    # Return some known output files (not an exhaustive list):
    possible_outfiles = [
        "tool_confusions", "tool_correlations.csv", "tool_rank_correlations.png",
        "poor_guide_detection.png", "poor_guide_detection_metrics.csv",
        "within_gene_ranking.png", "within_gene_ranking_metrics.csv",
        "tool_performance_metrics.csv", "tool_performance_dashboard.png",
        "roc_pr_metrics.csv", "distribution_efficacy.png"
    ]
    # Expand them with the actual full paths:
    created_files = [os.path.join(output_dir, f) for f in possible_outfiles]
    return created_files


def parse_args():
    parser = argparse.ArgumentParser(description="Robust tool comparison for CRISPR gRNA predictions")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV")
    parser.add_argument("--tools", "-t", required=True, help="Comma-separated list of tool columns")
    parser.add_argument("--efficacy-col", default="efficacy", help="Efficacy column name")
    parser.add_argument("--gene-col", default="Gene", help="Gene column name")
    parser.add_argument("--output", "-o", default="./tool_comparison_results", help="Output directory")
    parser.add_argument("--poor-threshold", type=float, default=50.0, help="Poor efficacy threshold")
    parser.add_argument("--moderate-threshold", type=float, default=75.0, help="Moderate efficacy threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tool_list = [x.strip() for x in args.tools.split(",")]
    run_tool_comparison(
        input_path=args.input,
        tools=tool_list,
        efficacy_col=args.efficacy_col,
        gene_col=args.gene_col,
        output_dir=args.output,
        poor_threshold=args.poor_threshold,
        moderate_threshold=args.moderate_threshold
    )

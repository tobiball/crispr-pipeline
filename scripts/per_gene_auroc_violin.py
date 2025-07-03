#!/usr/bin/env python3
"""
per_gene_auroc_violin.py  (rev-15 – with 6→4 guide subsampling)
───────────────────────────────────────────────────────────────────────────────
Enhanced version that subsamples 6-guide genes to 4 guides and combines them
with existing 4-guide genes for more robust statistical analysis.
"""
from __future__ import annotations
import argparse, warnings, sys, types
from pathlib import Path
from typing import Sequence, List
import itertools, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

EXPECTED_GUIDES = {2, 4, 6}  # Keep for validation, but will transform 6→4
FINAL_GUIDES = {2, 4}        # Final categories after subsampling
DEFAULT_CUTOFF = 60
COLORS_24 = {2: "#e74c3c", 4: "#3498db"}  # Updated colors for 2,4 only
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.autolayout": True})

# ── Strict AUROC (ties 0) ───────────────────────────────────────────────────

def strict_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos, neg = y_score[y_true == 1], y_score[y_true == 0]
    t = len(pos) * len(neg)
    return np.nan if t == 0 else (pos[:, None] > neg).sum() / t

# ── Data-integrity gates ────────────────────────────────────────────────────

def ensure_expected_guide_counts(df: pd.DataFrame, g: str):
    bad = df.groupby(g).size().loc[lambda s: ~s.isin(EXPECTED_GUIDES)]
    if not bad.empty:
        raise ValueError(f"❌  Unsupported guide counts:\n" + bad.to_string())

def ensure_balanced(df: pd.DataFrame, g: str, b: str):
    def _chk(x):
        if x[b].sum() * 2 != len(x):
            raise ValueError(f"❌  Gene {x.name} not balanced (+/-)")
    df.groupby(g).apply(_chk)

# ── NEW: Subsampling function ───────────────────────────────────────────────

def subsample_6_to_4_guides(df: pd.DataFrame, gene_col: str, label_col: str, cutoff: float, seed: int = 42) -> pd.DataFrame:
    """
    Subsample genes with 6 guides down to 4 guides using stratified sampling to maintain balance.

    Args:
        df: DataFrame with guide data
        gene_col: Column name for gene identifiers
        label_col: Column name for efficacy labels
        cutoff: Efficacy threshold for positive/negative classification
        seed: Random seed for reproducible subsampling

    Returns:
        DataFrame with 6-guide genes subsampled to 4 guides while maintaining balance
    """
    np.random.seed(seed)

    # Split by guide count
    guide_counts = df.groupby(gene_col).size()
    genes_2 = guide_counts[guide_counts == 2].index.tolist()
    genes_4 = guide_counts[guide_counts == 4].index.tolist()
    genes_6 = guide_counts[guide_counts == 6].index.tolist()

    print(f"📊 Guide count distribution before subsampling:")
    print(f"   2-guide genes: {len(genes_2)}")
    print(f"   4-guide genes: {len(genes_4)}")
    print(f"   6-guide genes: {len(genes_6)} (will be subsampled to 4)")

    # Keep 2-guide and 4-guide genes as-is
    df_2 = df[df[gene_col].isin(genes_2)].copy()
    df_4 = df[df[gene_col].isin(genes_4)].copy()

    # Stratified subsample 6-guide genes to 4 guides each
    df_6_subsampled = []
    skipped_genes = []

    for gene in genes_6:
        gene_data = df[df[gene_col] == gene]

        # Check if gene is balanced (3 positive, 3 negative guides)
        pos_guides = gene_data[gene_data[label_col] >= cutoff]
        neg_guides = gene_data[gene_data[label_col] < cutoff]

        if len(pos_guides) == 3 and len(neg_guides) == 3:
            # Perfect balance: randomly sample 2 from each group
            pos_sampled = pos_guides.sample(n=2, random_state=seed+hash(gene)%1000)
            neg_sampled = neg_guides.sample(n=2, random_state=seed+hash(gene)%1000+1)
            df_6_subsampled.append(pd.concat([pos_sampled, neg_sampled]))
            print(f"   ✅ {gene}: 6→4 stratified sampling (3+,3- → 2+,2-)")

        elif len(pos_guides) == 4 and len(neg_guides) == 2:
            # 4 positive, 2 negative: sample 2 positive, keep all negative
            pos_sampled = pos_guides.sample(n=2, random_state=seed+hash(gene)%1000)
            df_6_subsampled.append(pd.concat([pos_sampled, neg_guides]))
            print(f"   ✅ {gene}: 6→4 stratified sampling (4+,2- → 2+,2-)")

        elif len(pos_guides) == 2 and len(neg_guides) == 4:
            # 2 positive, 4 negative: keep all positive, sample 2 negative
            neg_sampled = neg_guides.sample(n=2, random_state=seed+hash(gene)%1000)
            df_6_subsampled.append(pd.concat([pos_guides, neg_sampled]))
            print(f"   ✅ {gene}: 6→4 stratified sampling (2+,4- → 2+,2-)")

        else:
            # Can't maintain balance with this gene - skip it
            skipped_genes.append(gene)
            print(f"   ⚠️  {gene}: Skipped (imbalanced: {len(pos_guides)}+, {len(neg_guides)}-)")

    if skipped_genes:
        print(f"   📝 Skipped {len(skipped_genes)} genes that couldn't maintain balance: {skipped_genes}")

    if df_6_subsampled:
        df_6_combined = pd.concat(df_6_subsampled, ignore_index=True)
    else:
        df_6_combined = pd.DataFrame()

    # Combine all data
    result_dfs = [df_2, df_4]
    if not df_6_combined.empty:
        result_dfs.append(df_6_combined)

    final_df = pd.concat(result_dfs, ignore_index=True)

    # Verify the result
    final_counts = final_df.groupby(gene_col).size()
    successful_6to4 = len(genes_6) - len(skipped_genes)

    print(f"📊 Guide count distribution after subsampling:")
    print(f"   2-guide genes: {(final_counts == 2).sum()}")
    print(f"   4-guide genes: {(final_counts == 4).sum()} (includes {successful_6to4} from 6→4 subsampling)")
    if skipped_genes:
        print(f"   ⚠️  Skipped genes: {len(skipped_genes)}")

    return final_df

# ── Helper: legal AUROC set ────────────────────────────────────────────────

def legal_auroc(n):
    if n % 2: return None
    h, step = n//2, 1/(n//2 * n//2)
    return {round(i*step,3) for i in range(h*h+1)}

# ── Enhanced visual functions with titles and explanations ─────────────────

def add_main_title(fig, database_name, cutoff):
    """Add consistent main title and subtitle to all plots"""
    main_title = f"{database_name}: Per-Gene AUROC Performance Analysis"
    subtitle = f"Algorithm Ranking Performance for Guide Efficacy Classification (Cutoff: {cutoff}%) | 6→4 Guide Subsampling Applied"

    fig.suptitle(main_title, fontsize=12, fontweight='bold', y=0.98)
    fig.text(0.5, 0.95, subtitle, ha='center', fontsize=9, style='italic')

def perfect_bar(au_df, order, out, database_name, cutoff):
    recs = []
    for alg, n in itertools.product(order, FINAL_GUIDES):  # Use FINAL_GUIDES
        sub = au_df[(au_df.Algorithm==alg)&(au_df.n_guides==n)]
        if not sub.empty:
            recs.append(dict(Algorithm=alg, Guides=n,
                             pct=(sub.AUROC==1).mean()*100))
    if not recs: return

    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(12,9))

    # Add main title and explanation
    add_main_title(fig, database_name, cutoff)

    w, x = 0.25, np.arange(len(order))
    for i,n in enumerate(sorted(FINAL_GUIDES)):
        vals=[df[(df.Algorithm==a)&(df.Guides==n)].pct.iloc[0] if not df[(df.Algorithm==a)&(df.Guides==n)].empty else 0 for a in order]
        ax.bar(x+i*w, vals, w, label=f"{n} guides", color=COLORS_24[n], edgecolor="black")

    ax.set_xticks(x+w/2)  # Center the x-ticks since we only have 2 categories now
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% Genes with Perfect AUROC (1.0)", fontsize=9, fontweight='bold')
    ax.set_xlabel("Algorithm", fontsize=9, fontweight='bold')
    ax.legend(frameon=False, title="Guide Count", title_fontsize=10, fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=5)

    plt.subplots_adjust(bottom=0.2, top=0.85, left=0.1, right=0.95)
    fig.savefig(out.with_suffix('.perfect_auroc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def stacked_bar(au_df, order, out, database_name, cutoff):
    gcs=sorted(FINAL_GUIDES)  # Use FINAL_GUIDES
    fig, axes = plt.subplots(1,len(gcs), figsize=(5*len(gcs),8), sharey=True)
    axes=[axes] if len(gcs)==1 else axes

    # Add main title and explanation
    add_main_title(fig, database_name, cutoff)

    for ax,n in zip(axes,gcs):
        poss=sorted(legal_auroc(n))
        sub=au_df[au_df.n_guides==n]
        if sub.empty:
            ax.text(0.5,0.5,f"No {n}-guide genes",ha='center',va='center')
            continue

        mat=pd.DataFrame(index=order, columns=[f"{v:.2f}" for v in poss]).fillna(0)
        for alg in order:
            vals=sub[sub.Algorithm==alg].AUROC.round(3)
            total=len(vals)
            for v in poss:
                mat.loc[alg,f"{v:.2f}"]= (vals==v).sum()/total*100 if total else 0

        mat.plot(kind='bar',stacked=True,ax=ax,edgecolor='black',width=0.8,
                 color=sns.color_palette('RdYlGn',len(poss)))
        ax.set_title(f"{n} Guides per Gene", fontsize=10, fontweight='bold', pad=5)
        ax.set_xlabel("Algorithm", fontsize=8, fontweight='bold')
        ax.set_ylabel("% of Genes" if ax == axes[0] else "", fontsize=8, fontweight='bold')
        ax.set_ylim(0,105)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(title="AUROC Value", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5, title_fontsize=6)

    plt.subplots_adjust(bottom=0.25, top=0.85, left=0.08, right=0.85, wspace=0.5)
    fig.savefig(out.with_suffix('.auroc_distribution_stacked.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ── put this ONCE, near your other rcParams block ───────────────────────────
import seaborn as sns
sns.set_context("talk", font_scale=1.3)     # bumps all fonts ~30 %
# ────────────────────────────────────────────────────────────────────────────

def heatmap_counts(au_df, order, out, database_name, cutoff):
    """
    Draw per-gene AUROC heat-maps with larger fonts, cleaner titles, and standard errors.

    Uses adaptive annotation format:
    - For ≤2 AUROC columns (2-guides): Shows counts + percentages ± standard errors
    - For >2 AUROC columns (4-guides): Shows only counts to save space
    """
    import numpy as np

    # colour chips for the y-axis backgrounds
    model_colors = {
        "TKO PSSM":           (0.51, 0.51, 0.51),
        "Moreno-Mateos":      (0.35, 0.40, 0.40),
        "Doench Rule Set 2":  (1.00, 0.82, 0.25),
        "Doench Rule Set 3":  (1.00, 0.73, 0.24),
        "DeepCRISPR":         (1.00, 0.39, 0.08),
        "DeepSpCas9":         (1.00, 0.35, 0.16),
        "TransCRISPR":        (0.94, 0.27, 0.39),
        "Linear Consensus":   (0.47, 0.51, 0.98),
        "Logistic Consensus": (0.24, 0.35, 0.82),
    }

    gcs  = sorted(FINAL_GUIDES)  # Use FINAL_GUIDES
    fig, axes = plt.subplots(1, len(gcs), figsize=(7 * len(gcs), 14))
    axes = [axes] if len(gcs) == 1 else axes

    # compact two-line title, lowered slightly (y=0.97)
    fig.suptitle(
        f"{database_name}: Per-Gene AUROC Performance Analysis\n"
        f"(Guide-efficacy cut-off {cutoff} fold change score)",
        fontsize=16, fontweight="bold", y=0.97
    )

    for ax, n in zip(axes, gcs):
        poss = sorted(legal_auroc(n))
        sub  = au_df[au_df.n_guides == n]

        if sub.empty:
            ax.text(0.5, 0.5, f"No {n}-guide genes",
                    ha="center", va="center", fontsize=14)
            continue

        # build count matrix
        mat = pd.DataFrame(
            0, index=order, columns=[f"{v:.2f}" for v in poss]
        )
        for alg in order:
            vals = sub[sub.Algorithm == alg].AUROC.round(3)
            for v in poss:
                mat.loc[alg, f"{v:.2f}"] = (vals == v).sum()

        # Calculate total genes for this guide count
        total_genes = len(sub[sub.Algorithm == order[0]])  # Total genes for this n

        # Create percentage matrix for coloring (but keep counts for annotations)
        mat_pct = mat / total_genes * 100

        # Calculate standard errors for each percentage
        mat_se = np.sqrt(mat_pct * (100 - mat_pct) / total_genes)

        # Decide annotation format based on number of columns (AUROC values)
        n_auroc_vals = len(poss)

        if n_auroc_vals <= 2:
            # For few columns (like 2-guides): show detailed format with counts and percentages
            annot_matrix = mat.copy().astype(str)
            for i in range(len(mat.index)):
                for j in range(len(mat.columns)):
                    count = mat.iloc[i, j]
                    pct = mat_pct.iloc[i, j]
                    se = mat_se.iloc[i, j]
                    annot_matrix.iloc[i, j] = f"{count}\n({pct:.1f}±{se:.1f}%)"
            annot_fontsize = 10
            fmt_str = ""
        else:
            # For many columns (like 4-guides): show count/percent/error in 3 lines
            annot_matrix = mat.copy().astype(str)
            for i in range(len(mat.index)):
                for j in range(len(mat.columns)):
                    count = mat.iloc[i, j]
                    pct = mat_pct.iloc[i, j]
                    se = mat_se.iloc[i, j]
                    # 3-line format: count / percentage / error
                    annot_matrix.iloc[i, j] = f"{count}\n{pct:.1f}%\n±{se:.1f}%"
            annot_fontsize = 8
            fmt_str = ""

        # heat-map with percentage coloring but count + SE annotations
        # FIXED: Added vmin=0 and vmax=100 to set colorbar range from 0% to 100%
        sns.heatmap(
            mat_pct,  # Use percentages for coloring
            annot=annot_matrix,  # Use counts with or without standard errors based on space
            fmt=fmt_str,  # Use appropriate format
            cmap="Blues",
            ax=ax,
            cbar=True,
            cbar_kws={"label": "% of Total Genes", "pad": 0.02, "shrink": 0.8},
            annot_kws={"fontsize": annot_fontsize, "weight": "bold"},
            linewidths=0.5,
            linecolor="white",
            vmin=0,    # Set minimum colorbar value to 0%
            vmax=100,  # Set maximum colorbar value to 100%
        )

        # Make colorbar tick labels consistent with annotation size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=annot_fontsize)
        cbar.set_label("% of Total Genes", fontsize=14, fontweight="bold")

        # axis styling
        ax.set_title(f"{n} Guides · Gene Counts by AUROC (n={total_genes})",
                     fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel("AUROC Value", fontsize=14, fontweight="bold")
        ax.set_ylabel("Algorithm" if ax is axes[0] else "",
                      fontsize=14, fontweight="bold", labelpad=30)

        ax.tick_params(axis="x", labelsize=annot_fontsize)
        ax.tick_params(axis="y", labelsize=14)   # Keep y-axis larger

        # colour back-plates on y-tick labels
        for tick, algo in zip(ax.get_yticklabels(), mat.index):
            tick.set_weight("bold")
            tick.set_rotation(0)
            tick.set_ha("right")
            bbox = dict(boxstyle="round,pad=0.3",
                        facecolor=model_colors.get(algo, (0.2, 0.2, 0.2)),
                        alpha=0.30, edgecolor="none")
            tick.set_bbox(bbox)

    # let tight_layout carve space, but keep 5 % at the top for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out.with_suffix(".auroc_heatmap_counts.png"), dpi=300)
    plt.close(fig)
# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Generate AUROC analysis plots with 6→4 guide subsampling")
    p.add_argument('--csv', required=True, help="Input CSV file path")
    p.add_argument('--gene-col', required=True, help="Gene column name")
    p.add_argument('--label-col', required=True, help="Label column name")
    p.add_argument('--alg-cols', required=True, help="Algorithm columns (comma-separated)")
    p.add_argument('--cutoff', type=float, default=DEFAULT_CUTOFF, help="Efficacy cutoff threshold")
    p.add_argument('--out', required=True, help="Output file prefix")
    p.add_argument('--database-name', default="Dataset", help="Name of the database/dataset for titles")
    p.add_argument('--subsample-seed', type=int, default=42, help="Random seed for 6→4 subsampling")

    args = p.parse_args()

    df = pd.read_csv(args.csv)
    algs = [a.strip() for a in args.alg_cols.split(',')]

    # Validate original data
    ensure_expected_guide_counts(df, args.gene_col)

    # Apply 6→4 subsampling BEFORE creating binary labels
    print(f"\n🔄 Applying 6→4 guide subsampling (seed={args.subsample_seed})...")
    df = subsample_6_to_4_guides(df, args.gene_col, args.label_col, args.cutoff, seed=args.subsample_seed)

    # Move binary label creation AFTER subsampling
    df['__bin__'] = (df[args.label_col] >= args.cutoff).astype(int)
    ensure_balanced(df, args.gene_col, '__bin__')

    recs = []
    for alg in algs:
        au = df.groupby(args.gene_col).apply(
            lambda g: strict_auroc(g['__bin__'].to_numpy(), g[alg].to_numpy())
        )
        recs.append(pd.DataFrame({
            args.gene_col: au.index,
            'Algorithm': alg,
            'AUROC': au.values
        }))

    au_df = pd.concat(recs, ignore_index=True)
    au_df = au_df.merge(
        df.groupby(args.gene_col).size().rename('n_guides'),
        left_on=args.gene_col,
        right_index=True
    )
    au_df = au_df[~au_df.AUROC.isna()]

    # Debug list: perfect genes per tool (now includes subsampled 6→4)
    print(f"\n=== {args.database_name} AUROC Analysis Results (with 6→4 subsampling) ===")
    for alg in algs:
        genes_4 = au_df[(au_df.Algorithm==alg)&(au_df.n_guides==4)&(au_df.AUROC==1)][args.gene_col].tolist()
        genes_2 = au_df[(au_df.Algorithm==alg)&(au_df.n_guides==2)&(au_df.AUROC==1)][args.gene_col].tolist()
        print(f"DEBUG: {alg} perfect on 4‑guides: {len(genes_4)} genes → {', '.join(genes_4[:10])}{'...' if len(genes_4)>10 else ''}")
        print(f"DEBUG: {alg} perfect on 2‑guides: {len(genes_2)} genes → {', '.join(genes_2[:10])}{'...' if len(genes_2)>10 else ''}")

    # Strength order by **mean** strict AUROC
    order = au_df.groupby('Algorithm')['AUROC'].mean().sort_values(ascending=False).index.tolist()
    print(f"\nAlgorithm ranking by mean AUROC: {order}")

    # Generate summary statistics
    print(f"\nSummary Statistics for {args.database_name} (post-subsampling):")
    summary = au_df.groupby(['Algorithm', 'n_guides'])['AUROC'].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
    print(summary)

    outp = Path(args.out)
    perfect_bar(au_df, order, outp, args.database_name, args.cutoff)
    stacked_bar(au_df, order, outp, args.database_name, args.cutoff)
    heatmap_counts(au_df, order, outp, args.database_name, args.cutoff)

    print(f"\n✅  {args.database_name} strict per‑gene AUROC analysis complete (with 6→4 subsampling).")
    print(f"    Plots saved with prefix: {outp}")

# ── Module helpers ──────────────────────────────────────────────────────────

_helpers = types.ModuleType('per_gene_auroc_violin_helpers')
_helpers.strict_auroc = strict_auroc
_helpers.legal_auroc = legal_auroc
_helpers.subsample_6_to_4_guides = subsample_6_to_4_guides
sys.modules['per_gene_auroc_violin_helpers'] = _helpers

if __name__ == '__main__':
    main()
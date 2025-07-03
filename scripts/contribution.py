#!/usr/bin/env python3
"""
Polished logistic-contribution plots (v2)
----------------------------------------

• Bold headings **and** bold feature names
• Smart label placement (inside large bar, outside small bar)
• Automatic model name in the title
• Colour-blind-safe palette
• Explanation box bottom-right
• PNG (300 dpi) and PDF export

Usage
-----
python plot_contributions_pro.py \
       contributions_Logistic_Consensus.csv \
       .logistic_logistic_consensus_model_data.json \
       [sample_index]
"""
import sys, json, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.offsetbox import AnchoredText

# --------------------------- global style
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 16,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 14,
    "xtick.labelsize"  : 12,
    "ytick.labelsize"  : 12,
    "figure.dpi"       : 120,
})

BLUE, ORANGE, GREEN, RED, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
)

# --------------------------- helpers
def savefig(fig, stem):
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf",             bbox_inches="tight")

def annotate_bars(ax, horiz=False, fmt="{:.1f}"):
    """
    Put value labels on bars.
    If a bar <10 % of axis span → print value outside (right/top),
    else print inside (left/bottom), so labels never cross zero line.
    """
    for p in ax.patches:
        val = p.get_width() if horiz else p.get_height()
        if np.isnan(val):
            continue

        if horiz:
            span = ax.get_xlim()[1] - ax.get_xlim()[0]
            outside = val < 0.10 * span
            x = p.get_x() + val + 0.01 * span if outside else p.get_x() + val - 0.01 * span
            ha = "left" if outside else "right"
            y = p.get_y() + p.get_height() / 2
            ax.text(x, y, fmt.format(val), va="center", ha=ha, fontsize=11)
        else:
            span = ax.get_ylim()[1] - ax.get_ylim()[0]
            outside = val < 0.10 * span
            y = p.get_y() + val + 0.01 * span if outside else p.get_y() + val - 0.01 * span
            va = "bottom" if outside else "top"
            x = p.get_x() + p.get_width() / 2
            ax.text(x, y, fmt.format(val), ha="center", va=va, fontsize=11)

def add_explainer(ax, text):
    box = AnchoredText(text, loc="lower right", frameon=True,
                       prop=dict(size=10), borderpad=0.6)
    box.patch.set(alpha=0.9, edgecolor=GREY, linewidth=0.8)
    box.patch.set_boxstyle("round,pad=0.3,rounding_size=0.8")
    ax.add_artist(box)

# --------------------------- I/O & basic data
csv_path  = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                         else "contributions_Logistic_Consensus.csv")
json_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                         else ".logistic_logistic_consensus_model_data.json")
sample_ix = int(sys.argv[3]) if len(sys.argv) > 3 else 0

# derive a clean model name for titles
model_name = csv_path.stem
if model_name.lower().startswith("contributions_"):
    model_name = model_name[len("contributions_"):]
model_name = model_name.replace("_", " ").strip().title()

df = pd.read_csv(csv_path)
with open(json_path) as f:
    betas = json.load(f)["betas"]

beta_sign = pd.Series({k: np.sign(v) for k, v in betas.items()})
common = df.columns.intersection(beta_sign.index)
df = df[common]
beta_sign = beta_sign[common]

pos_mask = beta_sign > 0

# =========================== 1. Feature-importance plot
abs_sum = df.abs().sum()
pct_abs = (100 * abs_sum / abs_sum.sum()).sort_values(ascending=False)

N_SHOW = min(20, len(pct_abs))
colors = [GREEN if beta_sign[c] > 0 else RED
          for c in pct_abs.head(N_SHOW).index]

fig1, ax1 = plt.subplots(figsize=(9, 5 + 0.28 * N_SHOW))
pct_abs.head(N_SHOW).iloc[::-1].plot(kind="barh", ax=ax1,
                                     color=colors[::-1])

# bold feature names
for t in ax1.get_yticklabels():
    t.set_fontweight("bold")

ax1.set_xlabel("Share of |β·z| (%)")
ax1.set_title(f"{model_name} – Feature Importance")
ax1.set_xlim(0, pct_abs.head(N_SHOW).max() * 1.10)   # 10 % headroom
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.grid(axis="x", linestyle=":", color=GREY, alpha=0.5)
annotate_bars(ax1, horiz=True)

add_explainer(
    ax1,
    "Green bar  →  β > 0 (raises score)\n"
    "Red bar    →  β < 0 (lowers score)\n"
    "Value = % of total |β·z| magnitude"
)
savefig(fig1, "global_importance")

# =========================== 2. β-balance plot
tot_pos = df.loc[:, pos_mask].abs().sum().sum()
tot_neg = df.loc[:, ~pos_mask].abs().sum().sum()
share_pos = 100 * tot_pos / (tot_pos + tot_neg)
share_neg = 100 - share_pos

fig2, ax2 = plt.subplots(figsize=(4.2, 4))
ax2.bar(["β > 0", "β < 0"], [share_pos, share_neg],
        color=[GREEN, RED])
ax2.set_ylabel("% of total |β·z|")
ax2.set_ylim(0, 100)
ax2.set_title("Positive vs Negative β share")
annotate_bars(ax2, fmt="{:.1f}%")
ax2.grid(axis="y", linestyle=":", color=GREY, alpha=0.5)

add_explainer(
    ax2,
    "Share of the overall absolute\n"
    "contribution carried by features\n"
    "with β > 0 versus β < 0."
)
savefig(fig2, "beta_balance")

# =========================== 3. Sample-level waterfall
row_abs   = df.iloc[sample_ix].abs()
row_pct   = 100 * row_abs / row_abs.sum()
row_pct_signed = row_pct * np.sign(beta_sign)
row_sorted = row_pct_signed.sort_values()
cumsum     = row_sorted.cumsum()
colors     = [RED if beta_sign[c] < 0 else GREEN for c in row_sorted.index]

fig3, ax3 = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(row_sorted))
ax3.bar(x_pos, row_sorted.values, color=colors)
ax3.plot(x_pos, cumsum.values, marker="o",
         linestyle="--", color=ORANGE, label="cumulative")

ax3.set_xticks(x_pos)
ax3.set_xticklabels(row_sorted.index, rotation=75, ha="right")
ax3.set_ylabel("% of |log-odds| (signed by β)")
ax3.set_title(f"{model_name} – Sample #{sample_ix} contributions")
ax3.axhline(0, linewidth=0.8, color=GREY)
ax3.legend()
ax3.grid(axis="y", linestyle=":", color=GREY, alpha=0.5)
annotate_bars(ax3, fmt="{:.1f}")

add_explainer(
    ax3,
    "Bar sign = β sign (direction)\n"
    "Orange dashed = cumulative\n"
    "Values are % of this sample’s\n"
    "total |β·z| magnitude."
)
plt.tight_layout()
savefig(fig3, f"sample_{sample_ix}_waterfall")

print("✔ Figures written:")
print("  – global_importance.{png,pdf}")
print("  – beta_balance.{png,pdf}")
print(f"  – sample_{sample_ix}_waterfall.{{png,pdf}}")

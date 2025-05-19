#!/usr/bin/env python3
"""
CRISPR Guide Prediction Tool – LR⁺ reliability edition
======================================================
* Keeps the original mis‑classification and stacked-bar visuals
* Replaces the class‑balanced PPV scatter with a **Positive Likelihood Ratio
  scatter**:
    • **x‑axis**  LR⁺  for Good vs (Moderate + Poor)
    • **y‑axis**  LR⁺  for (Good ∨ Moderate) vs Poor
  Top‑right = safest and most permissive predictor.
* Adds the two LR⁺ values to the CSV/JSON.

LR⁺ = TPR / FPR so a value > 1 always beats random, and the plot is prevalence‑
independent.
"""

import argparse, json, os
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("talk")

# ---------------------------------------------------------------------------
# helper --------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

# ---------------------------------------------------------------------------
# main class ----------------------------------------------------------------

class PredictionEvaluator:
    def __init__(
            self,
            input_file: str,
            output_dir: str,
            efficacy_col: str = "efficacy",
            poor_thr: float = 50,
            good_thr: float = 75,
            tool_cols: List[str] | None = None,
            fmt: str = "png",
    ):
        self.df = pd.read_csv(input_file)
        self.out = Path(output_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.efficacy_col = efficacy_col
        self.poor_thr, self.good_thr = poor_thr, good_thr
        self.tool_cols = tool_cols or self._detect_tool_cols()
        self.fmt = fmt
        self.df["eff_cat"] = self.df[efficacy_col].apply(self._cat)
        self._count_classes()

    # ------------------------------------------------------------------
    def _detect_tool_cols(self):
        numeric = [c for c in self.df.columns if c != self.efficacy_col and pd.api.types.is_numeric_dtype(self.df[c])]
        bad = ["id","index","start","end","strand","chromosome","position"]
        return [c for c in numeric if not any(b in c.lower() for b in bad)]

    def _cat(self, v: float):
        if pd.isna(v):
            return np.nan
        if v < self.poor_thr:
            return "Poor"
        if v >= self.good_thr:
            return "Good"
        return "Moderate"

    def _count_classes(self):
        vc = self.df["eff_cat"].value_counts()
        self.n_good  = int(vc.get("Good",0))
        self.n_mod   = int(vc.get("Moderate",0))
        self.n_poor  = int(vc.get("Poor",0))
        self.n_total = self.n_good + self.n_mod + self.n_poor

    # ------------------------------------------------------------------
    # ─── Replace your existing _rate_tool with this expanded version ───
    def _rate_tool(self, col: str) -> Dict:
        """
        Pick the threshold by maximizing min(sensitivity, specificity)
        (max–min criterion) for Good vs (Moderate U Poor), then record:
          • raw counts (TP/FN/FP/TN)
          • sensitivity, specificity
          • lr_g, lr_gm
          • youden = sensitivity + specificity - 1
        """
        d = self.df.dropna(subset=[col, "eff_cat"])
        if d.empty or self.n_good == 0:
            return {}

        best: Dict = {}
        best_score = -1.0
        for thr in np.linspace(d[col].min(), d[col].max(), 200):
            pred = d[col] >= thr

            # --- Good vs (Mod U Poor) ---
            tp = int(((d.eff_cat == "Good") &  pred).sum())
            fn = int(((d.eff_cat == "Good") & ~pred).sum())
            fp = int(((d.eff_cat != "Good") &  pred).sum())
            tn = int(((d.eff_cat != "Good") & ~pred).sum())

            tpr  = tp / (self.n_good or 1)
            spec = tn / ((fp + tn) or 1)

            # For threshold selection: max–min of (tpr, spec)
            worst = min(tpr, spec)
            if worst <= best_score:
                continue

            # But also compute Youden's J for coloring later
            youden = tpr + spec - 1

            # second split counts + LR+
            tp_gm = int(((d.eff_cat.isin(["Good","Moderate"])) &  pred).sum())
            fn_gm = (self.n_good + self.n_mod) - tp_gm
            fp_p  = int(((d.eff_cat == "Poor") &  pred).sum())
            tn_p  = self.n_poor - fp_p
            tpr_gm = tp_gm / ((self.n_good + self.n_mod) or 1)
            fpr_p  = fp_p  / (self.n_poor or 1)

            best_score = worst
            best = {
                "thr":      float(thr),
                # raw counts
                "tp_g":     tp,     "fn_g":   fn,
                "fp_notg":  fp,     "tn_notg":tn,
                # rates
                "tpr":      tpr,    "spec":   spec,
                # Youden's J for coloring
                "youden":   youden,
                # second split raw + lr
                "tp_gm":    tp_gm,  "fn_gm":  fn_gm,
                "fp_p":     fp_p,   "tn_p":   tn_p,
                "lr_g":     tpr  / (1 - spec) if spec  < 1 else float("inf"),
                "lr_gm":    tpr_gm/        fpr_p  if fpr_p > 0 else float("inf"),
            }

        return best




    # ─── Add this new helper to export the step‐by‐step LR table ───
    def _export_lr_steps(self):
        """
        Write out one row per tool/comparison with all counts, rates and LR+.
        Assumes your _rate_tool now returns both 'tpr' and 'spec'.
        """
        rows = []
        for tool, v in self.res.items():
            # Good vs (Moderate ∪ Poor)
            rows.append({
                "tool":        tool,
                "comparison":  "Good vs (Moderate ∪ Poor)",
                "threshold":   v["thr"],
                "TP":          v["tp_g"],
                "FN":          v["fn_g"],
                "FP":          v["fp_notg"],
                "TN":          v["tn_notg"],
                "sens.": v["tpr"],
                "spec.": v["spec"],
                "LR+":         v["lr_g"],
            })
            # (Good ∪ Moderate) vs Poor
            rows.append({
                "tool":        tool,
                "comparison":  "(Good ∪ Moderate) vs Poor",
                "threshold":   v["thr"],
                "TP":          v["tp_gm"],
                "FN":          v["fn_gm"],
                "FP":          v["fp_p"],
                "TN":          v["tn_p"],
                "sens.": v["tp_gm"] / (self.n_good + self.n_mod or 1),
                "spec.": (v["tn_p"] / self.n_poor) if self.n_poor else 0,
                "LR+":         v["lr_gm"],
            })
        df_steps = pd.DataFrame(rows)
        df_steps.to_csv(self.out / "lr_plus_steps.csv", index=False)
        df_steps.to_json(self.out / "lr_plus_steps.json", orient="records", indent=2)


    # ─── Finally, call it in evaluate() just before plotting ───

    def _plot_lr_tables(self):
        """
        Draw separate color-coded tables for each comparison showing:
          • TP, FN, FP, TN (integers)
          • sensitivity, specificity (rounded floats)
        Thresholds chosen by maximizing min(sensitivity, specificity).
        FN and FP are “bad” when high → their color scale is inverted.
        """
        # Load the detailed metrics
        df = pd.read_csv(self.out / "lr_plus_steps.csv")

        # Define columns
        count_cols = ["TP", "FN", "FP", "TN"]
        rate_cols  = ["sens.", "spec."]
        df[rate_cols] = df[rate_cols].round(2)

        comparisons = df["comparison"].unique()

        cmap = plt.colormaps["RdYlGn"]
        import matplotlib.colors as mcolors

        for comp in comparisons:
            sub = df[df["comparison"] == comp].set_index("tool")

            counts = sub[count_cols].values.astype(int)
            rates  = sub[rate_cols].values.astype(float)

            data = np.hstack([counts, rates])
            cell_text = [
                [str(x) for x in ct_row] + [""] + [f"{x:.2f}" for x in rt_row]
                for ct_row, rt_row in zip(counts, rates)
            ]

            # normalize per column
            vmin = data.min(axis=0)
            vmax = data.max(axis=0)
            vmax = np.where(vmax == vmin, vmax + 1e-6, vmax)
            norms = [mcolors.Normalize(vmin=vmin[j], vmax=vmax[j]) for j in range(data.shape[1])]

            # build colors
            cell_colors = []
            for row in data:
                cr = [cmap(norms[j](row[j])) for j in range(len(row))]
                cr[1] = cmap(1 - norms[1](row[1]))
                cr[2] = cmap(1 - norms[2](row[2]))
                cr = cr[:4] + [(1,1,1,1)] + cr[4:]
                cell_colors.append(cr)

            fig, ax = plt.subplots(figsize=(6, max(len(sub) * 0.5, 4)))
            tbl = ax.table(
                cellText=cell_text,
                cellColours=cell_colors,
                rowLabels=sub.index,
                colLabels=count_cols + [""] + rate_cols,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.3)

            # place title closer
            ax.set_title(comp.replace("∪", "U"), pad=2)
            ax.axis("off")

            plt.tight_layout(pad=0.3)
            fname = self.out / f"lr_table_{comp.replace(' ', '_').replace('∪','U')}.{self.fmt}"
            fig.savefig(fname, dpi=300, bbox_inches='tight',)
            plt.close(fig)


    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _plot_lr_scatter(self, df: pd.DataFrame):
        """LR⁺ scatter coloured & sized by Youden’s J, with non-overlapping labels."""
        fig, ax = plt.subplots(figsize=(8, 7))

        # colour/sizes ----------------------------------------------------
        df["quality"] = df["youden"]
        norm  = plt.Normalize(df["quality"].min(), df["quality"].max())
        sizes = (
                140
                + 60 * (df["quality"] - df["quality"].min())
                / (df["quality"].max() - df["quality"].min() + 1e-9)
        ).to_numpy()                          # <-- make sure it’s an array

        sc = ax.scatter(
            df["lr_g"],
            df["lr_gm"],
            s=sizes,
            c=df["quality"],
            cmap=plt.cm.viridis,
            norm=norm,
            edgecolors="k",
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Youden’s J (higher = better)")

        # smart labels ----------------------------------------------------
        annotate_conditional(ax, df)     # <- your cluster-and-stack helper

        # cosmetic --------------------------------------------------------
        ax.axvline(1, ls="--", color="grey", lw=1.1)
        ax.axhline(1, ls="--", color="grey", lw=1.1)
        ax.set_xlabel("LR+   Good vs (Moderate U Poor)")
        ax.set_ylabel("LR+   (Good U Moderate) vs Poor")
        ax.set_title("Positive Likelihood Ratio – finding a ‘good’ or 'tolerable' guide")

        fig.tight_layout()
        fig.savefig(self.out / f"lr_scatter.{self.fmt}", dpi=300)
        plt.close(fig)



    def evaluate(self):
        # Compute per-tool metrics
        self.res = {t: self._rate_tool(t) for t in self.tool_cols}
        self.res = {k: v for k, v in self.res.items() if v}
        df = pd.DataFrame.from_dict(self.res, orient="index") \
            .reset_index().rename(columns={"index": "tool"})

        # Export standard metrics
        df.to_csv(self.out / "tool_metrics.csv", index=False)
        with open(self.out / "tool_metrics.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f,
                      indent=2, default=_jsonable)

        # Export detailed LR+ steps
        self._export_lr_steps()

        # Render color-coded tables
        self._plot_lr_tables()

        # Plot the LR+ scatter
        self._plot_lr_scatter(df)

        print("Saved results to", self.out.resolve())


# ---------------------------------------------------------------------
DX_NEAR  = 0.1    # “close” in x (LR+ units)
DY_NEAR  = 0.2     # “close” in y (LR+ units)
DX_OFF   = 5       # points → left
DY_ABOVE =  8       # default: above
DY_BELOW = -8       # bumped one: below
# ---------------------

def annotate_conditional(ax, df):
    """
     Place each label left-above its marker unless the marker sits close
     to another *higher* marker – in that case move the lower label below.
     """
    # default = above for everyone
    offsets = {idx: DY_ABOVE for idx in df.index}

    xs, ys = df["lr_g"].to_numpy(), df["lr_gm"].to_numpy()
    n = len(xs)

    # pairwise search for neighbours
    for i in range(n):
        for j in range(i + 1, n):
            if abs(xs[i] - xs[j]) < DX_NEAR and abs(ys[i] - ys[j]) < DY_NEAR:
                # same little box → lower-y one goes below
                if ys[i] < ys[j]:
                    offsets[df.index[i]] = DY_BELOW
                else:
                    offsets[df.index[j]] = DY_BELOW

    # draw
    for idx, row in df.iterrows():
        ax.annotate(
            row.tool,
            (row.lr_g, row.lr_gm),
            xytext=(DX_OFF, offsets[idx]),
            textcoords="offset points",
            ha="right",
            va="bottom" if offsets[idx] > 0 else "top",
            fontsize=8,
        )




# ---------------------------------------------------------------------------
# cli -----------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',default='./prediction_viz_results')
    p.add_argument('--efficacy-col',default='efficacy')
    p.add_argument('--poor-threshold',type=float,default=50)
    p.add_argument('--good-threshold',type=float,default=75)
    p.add_argument('--min-coverage',type=float,default=0.75)
    p.add_argument('--tools')
    p.add_argument('--format',choices=['png','pdf','svg'],default='png')
    return p.parse_args()

if __name__=='__main__':
    a=_parse()
    tools = a.tools.split(',') if a.tools else None
    ev = PredictionEvaluator(a.input,a.output,a.efficacy_col,a.poor_threshold,a.good_threshold,tools,a.format)
    ev.evaluate()

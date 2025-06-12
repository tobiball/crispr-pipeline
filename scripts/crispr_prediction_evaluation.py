#!/usr/bin/env python3
"""
CRISPR Guide Prediction Tool – LR⁺ reliability edition
======================================================
* Keeps the original mis-classification and stacked-bar visuals
* Replaces the class-balanced PPV scatter with a **Positive Likelihood Ratio
  scatter**:
    • **x-axis**  LR⁺  for Good vs (Moderate + Poor)
    • **y-axis**  LR⁺  for (Good ∨ Moderate) vs Poor
  Top-right = safest and most permissive predictor.
* Adds the two LR⁺ values to the CSV/JSON.
* Adds binary confusion matrices for Good vs (Moderate ∪ Poor).
* Exports chosen thresholds per tool to CSV/JSON.

LR⁺ = TPR / FPR so a value > 1 always beats random, and the plot is prevalence-
independent.
"""

import argparse, json
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
            thresholds_df: pd.DataFrame | None = None,


    ):
        self.df = pd.read_csv(input_file)
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.efficacy_col = efficacy_col
        self.poor_thr, self.good_thr = poor_thr, good_thr
        self.tool_cols = tool_cols or self._detect_tool_cols()
        self.fmt = fmt
        self.df["eff_cat"] = self.df[efficacy_col].apply(self._cat)
        self._count_classes()
        self.thresholds_df = thresholds_df



    def _detect_tool_cols(self):
            numeric = [
                c for c in self.df.columns
                if c != self.efficacy_col and pd.api.types.is_numeric_dtype(self.df[c])
            ]
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
        self.n_good = int(vc.get("Good", 0))
        self.n_mod = int(vc.get("Moderate", 0))
        self.n_poor = int(vc.get("Poor", 0))
        self.n_total = self.n_good + self.n_mod + self.n_poor

    def _rate_tool(self, col: str) -> Dict:
        d = self.df.dropna(subset=[col, "eff_cat"])
        if d.empty or self.n_good == 0:
            return {}
        best, best_score = {}, -1.0
        # sweep thresholds
        for thr in np.linspace(d[col].min(), d[col].max(), 500):
            pred = d[col] >= thr
            # Good vs (Moderate U Poor)
            tp = int(((d.eff_cat == "Good") & pred).sum())
            fn = int(((d.eff_cat == "Good") & ~pred).sum())
            fp = int(((d.eff_cat != "Good") & pred).sum())
            tn = int(((d.eff_cat != "Good") & ~pred).sum())
            tpr = tp / (self.n_good or 1)
            spec = tn / ((fp + tn) or 1)
            worst = min(tpr, spec)
            if worst <= best_score:
                continue
            youden = tpr + spec - 1
            # (Good U Moderate) vs Poor
            tp_gm = int(((d.eff_cat.isin(["Good", "Moderate"])) & pred).sum())
            fn_gm = (self.n_good + self.n_mod) - tp_gm
            fp_p = int(((d.eff_cat == "Poor") & pred).sum())
            tn_p = self.n_poor - fp_p
            tpr_gm = tp_gm / ((self.n_good + self.n_mod) or 1)
            fpr_p = fp_p / (self.n_poor or 1)
            best_score = worst
            best = {
                "thr": float(thr),
                "tp_g": tp, "fn_g": fn,
                "fp_notg": fp, "tn_notg": tn,
                "tpr": tpr, "spec": spec,
                "youden": youden,
                "tp_gm": tp_gm, "fn_gm": fn_gm,
                "fp_p": fp_p, "tn_p": tn_p,
                "lr_g": tpr / (1 - spec) if spec < 1 else float("inf"),
                "lr_gm": tpr_gm / fpr_p if fpr_p > 0 else float("inf"),
            }
        return best


    def _rate_tool_at_threshold(self, col: str, thr: float) -> Dict:
        d = self.df.dropna(subset=[col, "eff_cat"])
        if d.empty or self.n_good == 0:
            return {}
        pred = d[col] >= thr

        # Good vs (Moderate U Poor)
        tp = int(((d.eff_cat == "Good") & pred).sum())
        fn = int(((d.eff_cat == "Good") & ~pred).sum())
        fp = int(((d.eff_cat != "Good") & pred).sum())
        tn = int(((d.eff_cat != "Good") & ~pred).sum())
        tpr = tp / (self.n_good or 1)
        spec = tn / ((fp + tn) or 1)
        youden = tpr + spec - 1

        # (Good U Moderate) vs Poor
        tp_gm = int(((d.eff_cat.isin(["Good", "Moderate"])) & pred).sum())
        fn_gm = (self.n_good + self.n_mod) - tp_gm
        fp_p = int(((d.eff_cat == "Poor") & pred).sum())
        tn_p = self.n_poor - fp_p
        tpr_gm = tp_gm / ((self.n_good + self.n_mod) or 1)
        fpr_p = fp_p / (self.n_poor or 1)

        return {
            "thr": thr,
            "tp_g": tp,
            "fn_g": fn,
            "fp_notg": fp,
            "tn_notg": tn,
            "tpr": tpr,
            "spec": spec,
            "youden": youden,
            "tp_gm": tp_gm,
            "fn_gm": fn_gm,
            "fp_p": fp_p,
            "tn_p": tn_p,
            "lr_g": tpr / (1 - spec) if spec < 1 else float("inf"),
            "lr_gm": tpr_gm / fpr_p if fpr_p > 0 else float("inf"),
        }


    def _export_lr_steps(self):
        rows = []
        for tool, v in self.res.items():
            rows.append({
                "tool": tool,
                "comparison": "Good vs (Moderate ∪ Poor)",
                "threshold": v["thr"],
                "TP": v["tp_g"], "FN": v["fn_g"],
                "FP": v["fp_notg"], "TN": v["tn_notg"],
                "sens.": v["tpr"], "spec.": v["spec"], "LR+": v["lr_g"]
            })
            rows.append({
                "tool": tool,
                "comparison": "(Good ∪ Moderate) vs Poor",
                "threshold": v["thr"],
                "TP": v["tp_gm"], "FN": v["fn_gm"],
                "FP": v["fp_p"], "TN": v["tn_p"],
                "sens.": v["tp_gm"] / ((self.n_good + self.n_mod) or 1),
                "spec.": v["tn_p"] / (self.n_poor or 1),
                "LR+": v["lr_gm"]
            })
        df_steps = pd.DataFrame(rows)
        df_steps.to_csv(self.out / "lr_plus_steps.csv", index=False)
        df_steps.to_json(self.out / "lr_plus_steps.json", orient="records", indent=2)

    def _plot_lr_tables(self):
        df = pd.read_csv(self.out / "lr_plus_steps.csv")
        count_cols = ["TP", "FN", "FP", "TN"]
        rate_cols = ["sens.", "spec."]
        df[rate_cols] = df[rate_cols].round(2)
        comparisons = df["comparison"].unique()
        cmap = plt.get_cmap("RdYlGn")
        for comp in comparisons:
            sub = df[df["comparison"] == comp].set_index("tool")
            counts = sub[count_cols].values.astype(int)
            rates = sub[rate_cols].values.astype(float)
            data = np.hstack([counts, rates])
            vmin, vmax = data.min(0), data.max(0)
            vmax = np.where(vmax == vmin, vmax + 1e-6, vmax)
            norms = [mcolors.Normalize(vmin=vmin[i], vmax=vmax[i]) for i in range(data.shape[1])]
            cell_text = [
                [str(x) for x in ct] + [""] + [f"{x:.2f}" for x in rt]
                for ct, rt in zip(counts, rates)
            ]
            cell_colors = []
            for row in data:
                row_colors = [cmap(norms[j](row[j])) for j in range(len(row))]
                # invert FN and FP
                row_colors[1] = cmap(1 - norms[1](row[1]))
                row_colors[2] = cmap(1 - norms[2](row[2]))
                row_colors = row_colors[:4] + [(1,1,1,1)] + row_colors[4:]
                cell_colors.append(row_colors)
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
            ax.set_title(comp.replace("∪", "U"), pad=2)
            ax.axis("off")
            plt.tight_layout(pad=0.3)
            fig.savefig(
                self.out / f"lr_table_{comp.replace(' ', '_').replace('∪','U')}.{self.fmt}",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close(fig)

    def _plot_binary_confusion_matrices(self):
        import seaborn as sns
        df = pd.read_csv(self.out / "lr_plus_steps.csv")
        df_bin = df[df["comparison"] == "Good vs (Moderate ∪ Poor)"].set_index("tool")
        for tool, row in df_bin.iterrows():
            cm = np.array([[row["TP"], row["FP"]], [row["FN"], row["TN"]]])
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Good", "Not Good"],
                yticklabels=["Good", "Not Good"],
                cbar=False,
                ax=ax,
            )
            ax.set_title(f"{tool}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            fig.savefig(
                self.out / f"confusion_good_vs_rest_{tool.replace(' ', '_')}.{self.fmt}",
                dpi=300,
                )
            plt.close(fig)

    def _plot_lr_scatter(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 7))
        df["quality"] = df["youden"]
        norm = plt.Normalize(df["quality"].min(), df["quality"].max())
        sizes = (
                140
                + 60 * (df["quality"] - df["quality"].min())
                / (df["quality"].max() - df["quality"].min() + 1e-9)
        ).to_numpy()
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
        annotate_conditional(ax, df)
        ax.axvline(1, ls="--", color="grey", lw=1.1)
        ax.axhline(1, ls="--", color="grey", lw=1.1)
        ax.set_xlabel("LR+   Good vs (Moderate U Poor)")
        ax.set_ylabel("LR+   (Good U Moderate) vs Poor")
        ax.set_title(
            "Positive Likelihood Ratio – finding a ‘good’ or 'tolerable' guide"
        )
        fig.tight_layout()
        fig.savefig(self.out / f"lr_scatter.{self.fmt}", dpi=300)
        plt.close(fig)

    def evaluate(self):
        # Compute per-tool metrics
        if self.thresholds_df is not None:
            self.res = {}
            for _, row in self.thresholds_df.iterrows():
                tool = row["tool"]
                thr = float(row["threshold"])
                self.res[tool] = self._rate_tool_at_threshold(tool, thr)
        else:
            self.res = {t: self._rate_tool(t) for t in self.tool_cols}
            self.res = {k: v for k, v in self.res.items() if v}
        # Prepare DataFrame for export
        df = pd.DataFrame.from_dict(self.res, orient="index").reset_index().rename(columns={"index":"tool"})

        # Export standard metrics
        df.to_csv(self.out / "tool_metrics.csv", index=False)
        with open(self.out / "tool_metrics.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, default=_jsonable)

        # Export thresholds separately
        thr_df = pd.DataFrame([{"tool": tool, "threshold": v["thr"]} for tool, v in self.res.items()])
        thr_df.to_csv(self.out / "thresholds.csv", index=False)
        thr_df.to_json(self.out / "thresholds.json", orient="records", indent=2)

        # Export detailed LR+ steps
        self._export_lr_steps()

        # Render LR+ tables
        self._plot_lr_tables()

        # Render binary confusion matrices
        self._plot_binary_confusion_matrices()

        # Plot the LR+ scatter
        self._plot_lr_scatter(df)

        print("Saved results to", self.out.resolve())

# ---------------------------------------------------------------------
DX_NEAR = 0.1
DY_NEAR = 0.2
DX_OFF = 5
DY_ABOVE = 8
DY_BELOW = -8


def annotate_conditional(ax, df):
    offsets = {idx: DY_ABOVE for idx in df.index}
    xs, ys = df["lr_g"].to_numpy(), df["lr_gm"].to_numpy()
    for i in range(len(xs)):
        for j in range(i+1, len(xs)):
            if abs(xs[i]-xs[j]) < DX_NEAR and abs(ys[i]-ys[j]) < DY_NEAR:
                if ys[i] < ys[j]:
                    offsets[df.index[i]] = DY_BELOW
                else:
                    offsets[df.index[j]] = DY_BELOW
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

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='./prediction_viz_results')
    p.add_argument('--efficacy-col', default='efficacy')
    p.add_argument('--poor-threshold', type=float, default=50)
    p.add_argument('--good-threshold', type=float, default=75)
    p.add_argument('--min-coverage', type=float, default=0.75)
    p.add_argument('--tools')
    p.add_argument('--thresholds', help='CSV with columns tool,threshold')
    args = p.parse_args()

    tools = args.tools.split(',') if args.tools else None
    preset_df = pd.read_csv(args.thresholds) if args.thresholds else None

    evaluator = PredictionEvaluator(
        args.input,
        args.output,
        efficacy_col=args.efficacy_col,
        poor_thr=args.poor_threshold,
        good_thr=args.good_threshold,
        tool_cols=tools,
        fmt='png',
        thresholds_df=preset_df,
    )
    evaluator.evaluate()
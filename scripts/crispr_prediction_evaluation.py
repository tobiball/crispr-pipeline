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

MODIFIED: Uses smooth saturation-based color gradients instead of RdYlGn
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
# Smooth color scheme functions ---------------------------------------------


import numpy as np, colorsys, matplotlib.colors as mcol

def create_sequential_red_to_blue(
        n=256,
        h_start=0.0,        #  0° = red
        h_end  =-130/360.0, # –180° → wraps to 180° = cyan
        s_fixed=0.5,       # constant saturation
        v_start=1.0,       # very bright
        v_end  =0.60# moderately dark
):
    """
    HSV ramp Red→Magenta→Blue→Cyan,
    brightness (V) linearly falls v_start→v_end.
    """
    hs = np.linspace(h_start, h_end, n) % 1.0
    vs = np.linspace(v_start,   v_end,   n)
    rgbs = [colorsys.hsv_to_rgb(h, s_fixed, v) for h, v in zip(hs, vs)]
    return mcol.ListedColormap(rgbs, name="hsv_red_to_cyan_lightdark")





def create_monotonic_red_blue_green(n=256,
                                    h_start=   0/360,   # red
                                    h_mid  = 240/360,   # blue
                                    h_end  = 120/360,   # green
                                    l_start= 0.90,      # very light at red
                                    l_end  = 0.40,      # mid-dark at green
                                    s_fixed= 0.65):     # moderate saturation
    """
    Sequential Red→Blue→Green ramp with monotonic lightness:
      • hue moves red→blue→green
      • lightness falls linearly from l_start to l_end
      • saturation held constant at s_fixed
    """
    import numpy as np, colorsys, matplotlib.colors as mcol

    # 1) build the hue trajectory: red→blue then blue→green
    hs1 = np.linspace(h_start, h_mid, n//2, endpoint=False)
    hs2 = np.linspace(h_mid,   h_end, n - n//2)
    hs  = np.concatenate([hs1, hs2])

    # 2) build a strictly decreasing lightness vector
    ls = np.linspace(l_start, l_end, n)

    # 3) constant saturation
    ss = np.full(n, s_fixed)

    # 4) convert HLS→RGB
    rgbs = [colorsys.hls_to_rgb(h, l, s) for h, l, s in zip(hs, ls, ss)]
    return mcol.ListedColormap(rgbs, name="mono_red_blue_green")


import seaborn as sns

def create_red_blue_green_colormap(n=256,
                                   h_start=  0/360,   # pure red
                                   h_mid  =240/360,   # pure blue
                                   h_end  =120/360,   # pure green
                                   l_start=0.92,      # very pale
                                   l_mid  =0.70,      # mid light
                                   l_end  =0.45,      # moderately dark
                                   s_fixed=0.65):     # medium saturation
    """
    Sequential Red→Blue→Green ramp:
      • reds → blues → greens
      • monotonically decreasing lightness
      • skips the yellow region entirely
    """
    import numpy as np, colorsys, matplotlib.colors as mcol

    # first half: red→blue, second half: blue→green
    hs1 = np.linspace(h_start, h_mid, n//2, endpoint=False)
    hs2 = np.linspace(h_mid,   h_end, n - n//2)
    hs  = np.concatenate([hs1, hs2])

    ls1 = np.linspace(l_start, l_mid, n//2, endpoint=False)
    ls2 = np.linspace(l_mid,   l_end, n - n//2)
    ls  = np.concatenate([ls1, ls2])

    rgbs = [colorsys.hls_to_rgb(h, l, s_fixed) for h, l in zip(hs, ls)]
    return mcol.ListedColormap(rgbs, name="red_blue_green")



def create_pastel_red_blue_colormap(n=256,
                                    h_start=  0/360,   # pure red
                                    h_end  =240/360,   # pure blue
                                    l_start=0.92,      # very pale red
                                    l_end  =0.55,      # mid-dark blue
                                    s_fixed=0.60):     # moderate pastel saturation
    """
    Pastel sequential Red→Blue:
     - hue from 0° (red) to 240° (blue)
     - lightness drops linearly from l_start→l_end
     - constant saturation so colours stay soft
    """
    import numpy as np, colorsys, matplotlib.colors as mcol

    hs = np.linspace(h_start, h_end, n)
    ls = np.linspace(l_start, l_end, n)
    rgbs = [colorsys.hls_to_rgb(h, l, s_fixed) for h, l in zip(hs, ls)]
    return mcol.ListedColormap(rgbs, name="pastel_red_blue")



def create_clean_pastel_rytgg(n=256):
    """
    Pastel Red→Yellow→Green ramp:
     - start at warm coral red
     - brief butter-yellow hinge
     - finish at minty green
     - all in a low-saturation, high-lightness style
    """
    # h_neg=10° (red), h_pos=150° (green)
    # s=50 (medium pastel chroma), l=75 (overall lightness)
    return sns.diverging_palette(10, 150, s=50, l=75, n=n, as_cmap=True)


def create_modern_pastel_ryg(n=256,
                             h_start=  8/360,   # a warm salmon (≈ #ff7f7f)
                             h_end  =132/360,   # a minty green  (≈ #6fe4b1)
                             l_start=0.92,      # very pale coral
                             l_end  =0.65,      # mid-light mint
                             s_fixed=0.45):     # pastel saturation
    """
    Stylish pastel R→Y→G palette with monotonic lightness.
    - no harsh reds / toxic greens
    - looks crisp on bright or dark backgrounds
    """
    import numpy as np, colorsys, matplotlib.colors as mcol

    hs = np.linspace(h_start, h_end, n)
    ls = np.linspace(l_start, l_end, n)          # brightness steps down smoothly
    rgbs = [colorsys.hls_to_rgb(h, l, s_fixed) for h, l in zip(hs, ls)]
    return mcol.ListedColormap(rgbs, name="modern_pastel_RYG")



def create_modern_red_green_colormap(n=256,
                                     h_start=  8/360,  # warm red
                                     h_mid  =  30/360, # orange hinge
                                     h_end  =120/360,  # green
                                     l_start=0.92,     # very light
                                     l_mid  =0.75,     # mid-light
                                     l_end  =0.45,     # moderately dark
                                     s_fixed=0.65):    # medium saturation
    """
    Single–hue ramp from light red → orange → green,
    with lightness stepping down monotonically: L_start > L_mid > L_end.
    """
    import numpy as np, colorsys, matplotlib.colors as mcol

    # make three segments: red→orange (0…mid), orange→green (mid…end)
    hs1 = np.linspace(h_start, h_mid, n//2, endpoint=False)
    hs2 = np.linspace(h_mid,   h_end, n - n//2)
    hs  = np.concatenate([hs1, hs2])

    ls1 = np.linspace(l_start, l_mid, n//2, endpoint=False)
    ls2 = np.linspace(l_mid,   l_end, n - n//2)
    ls  = np.concatenate([ls1, ls2])

    # constant saturation
    rgbs = [colorsys.hls_to_rgb(h, l, s_fixed) for h, l in zip(hs, ls)]
    return mcol.ListedColormap(rgbs, name="modern_red_green")



def create_monotone_red_yellow_green(n=256,
                                     h_start=  0/360,   # red   (°/360)
                                     h_end  =120/360,   # green (°/360)
                                     l_start=0.86,      # lightness at red
                                     l_end  =0.32,      # lightness at green
                                     s_fixed=0.75):     # constant saturation
    """
    Continuous R→Y→G ramp whose *lightness* steps down linearly,
    guaranteeing V/L of every red ≥ every yellow ≥ every green.
    """
    import numpy as np, colorsys, matplotlib.colors as mcol

    hs = np.linspace(h_start, h_end, n)
    ls = np.linspace(l_start, l_end, n)          # strictly decreasing
    ss = np.full(n, s_fixed)

    rgbs = [colorsys.hls_to_rgb(h, l, s_fixed) for h, l in zip(hs, ls)]
    return mcol.ListedColormap(rgbs, name="monotone_RYG")


def create_smooth_blue_colormap():
    """Create a smooth blue-based saturation gradient colormap"""
    colors = [
        '#f5f5f5',  # Very light gray
        '#e8edf5',  # Light blue-gray
        '#dbe5f1',  # Lighter blue
        '#cedced',  # Light blue
        '#bfd4e8',  # Soft blue
        '#afc9e3',  # Medium-light blue
        '#9ebddd',  # Medium blue
        '#8bb1d6',  # Deeper blue
        '#75a3cd',  # Strong blue
        '#5d94c4',  # Stronger blue
        '#4084b9',  # Deep blue
        '#2171b5',  # Rich blue
        '#08519c',  # Dark blue
        '#084081',  # Very dark blue
        '#08306b'   # Darkest blue
    ]
    return mcolors.LinearSegmentedColormap.from_list("smooth_blue", colors, N=256)

def create_smooth_purple_colormap():
    """Create a smooth purple-based saturation gradient colormap"""
    colors = [
        '#f7f7f7',  # Very light gray
        '#f0f0f5',  # Light purple-gray
        '#e3e0ed',  # Lighter purple
        '#d4d0e5',  # Light purple
        '#c4bfdd',  # Soft purple
        '#b3add4',  # Medium-light purple
        '#a099ca',  # Medium purple
        '#8c84c0',  # Deeper purple
        '#766db5',  # Strong purple
        '#5e54a8',  # Stronger purple
        '#443a9a',  # Deep purple
        '#2d1e8b'   # Dark purple
    ]
    return mcolors.LinearSegmentedColormap.from_list("smooth_purple", colors, N=256)

def create_smooth_teal_colormap():
    """Create a smooth teal-based saturation gradient colormap"""
    colors = [
        '#f0f0f0',  # Very light gray
        '#e0f3f3',  # Light teal-gray
        '#d0e9e9',  # Lighter teal
        '#c0dede',  # Light teal
        '#afd4d4',  # Soft teal
        '#9dc9c9',  # Medium-light teal
        '#8bbdbd',  # Medium teal
        '#78b1b1',  # Deeper teal
        '#64a4a4',  # Strong teal
        '#4f9696',  # Stronger teal
        '#3a8787',  # Deep teal
        '#247777',  # Dark teal
        '#0d6666'   # Very dark teal
    ]
    return mcolors.LinearSegmentedColormap.from_list("smooth_teal", colors, N=256)


def create_smooth_yellow_red_colormap():
    """
    Smooth yellow-to-red saturation gradient.
    Starts with an almost-white cream so small values stay subtle,
    then ramps through yellows / oranges to a dark crimson.
    """
    colors = [
        '#fffff5',  # 0   – almost white
        '#fff8dd',  # 1
        '#feefc5',  # 2   – very light butter-yellow
        '#fde5ad',  # 3
        '#fcd991',  # 4   – pale gold
        '#facc73',  # 5
        '#f8bd55',  # 6   – soft orange
        '#f6a93b',  # 7
        '#f39021',  # 8   – vivid orange
        '#ee7512',  # 9
        '#e55a0b',  # 10  – orange-red
        '#d53d07',  # 11
        '#bd2605',  # 12  – rich red
        '#951b04',  # 13
        '#6d1103'   # 14  – darkest red
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "smooth_yellow_red", colors, N=256
    )


def create_smooth_red_green_colormap():
    """
    Diverging palette: low = soft red, mid = warm yellow, high = relaxed green.
    Designed to avoid the ‘blood-red / neon-green’ extremes of the default RdYlGn.
    """
    colors = [
        # reds – very light ➜ medium
        "#fff6f3", "#fee8e1", "#fdd8cd", "#fcc7b8", "#f9b4a3",
        "#f69f8f", "#f3837c", "#ef6769",
        # narrow yellow hinge
        "#fadf70", "#f4e04e",
        # greens – pale ➜ rich
        "#d7e9a5", "#b9e090", "#96d57c", "#6ec568", "#45b156",
        "#2d9146", "#207942", "#14603f"
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "smooth_red_green", colors, N=256
    )

def create_bright_red_yellow_green_colormap():
    """
    Bright-red → mid-yellow → dark-green with monotonically DECREASING brightness.
    Reds ≈ 90-70 % luminance  →  Yellows ≈ 65-55 %  →  Greens ≈ 50-30 %.
    """
    colors = [
        # ----- Reds (very bright pastels) -----
        "#ffeceb", "#ffd6d3", "#ffbdb7", "#ffa29a", "#ff867d",
        # ----- Transition through warm yellows -----
        "#ffcf73", "#f8c159", "#eab543",
        # ----- Greens (progressively darker) -----
        "#c6d98c", "#9cc16d", "#6fa950", "#4a8f3d", "#2f7433"
    ]
    return mcolors.LinearSegmentedColormap.from_list(
        "bright_red_yellow_green", colors, N=256
    )



import numpy as np
import matplotlib as mpl

def create_trimmed_red_green_colormap(trim=0.05):
    base = create_smooth_red_green_colormap()       # the function we made earlier
    # keep only the central 90 % → remove darkest 5 % + brightest 5 %
    new_colors = base(np.linspace(trim, 1-trim, 256))
    return mpl.colors.ListedColormap(new_colors, name="trimmed_red_green")




def get_padded_normalizer(vmin, vmax, padding=0.15):
    """Create a normalizer with padding to avoid extreme colors"""
    value_range = vmax - vmin
    if value_range == 0:
        value_range = 1
    padded_min = vmin - padding * value_range
    padded_max = vmax + padding * value_range
    return mcolors.Normalize(vmin=padded_min, vmax=padded_max)

# ---------------------------------------------------------------------------
# main class ----------------------------------------------------------------

class PredictionEvaluator:
    def __init__(
            self,
            input_file: str,
            output_dir: str,
            efficacy_col: str = "efficacy",
            poor_thr: float = 60,
            good_thr: float = 90,
            db_name: str | None = None,
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
        self.db_name = db_name




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
        """
        Render CRISPR predictor tables with
          • pastel count columns,
          • vivid, wider SENS./SPEC. columns separated by a thick rule,
          • automatic white text on dark cells for legibility,
          • a centred caption placed 2 pt above each table.
        """
        import matplotlib.colors as mcol
        import matplotlib.pyplot as plt
        import pandas as pd

        # --- caption styling constants ----------------------------------
        CAPTION_FONT  = 11          # pt
        CAPTION_STYLE = "oblique"
        CAPTION_PAD_PT = 10          # vertical gap in *points*

        # ----------------------------------------------------------------
        df = pd.read_csv(self.out / "lr_plus_steps.csv")

        count_cols     = ["TP", "FN", "FP", "TN"]
        rate_cols_raw  = ["sens.", "spec."]
        rate_cols_disp = ["Sensitivity", "Specificity"]

        # round the two rate columns for display
        df[rate_cols_raw] = df[rate_cols_raw].round(2)
        comparisons = df["comparison"].unique()

        # fixed row-label colours ----------------------------------------
        # fixed row-label colours ----------------------------------------
        model_colors = {
            "TKO PSSM":           (130/255, 130/255, 130/255),
            "Moreno-Mateos":      (90/255,  100/255, 100/255),

            "Doench Rule Set 2":  (255/255, 140/255, 0/255),
            "Doench Rule Set 3":  (220/255, 110/255, 0/255),

            "DeepCRISPR":         (138/255, 43/255,  226/255),
            "DeepSpCas9":         (123/255, 31/255,  162/255),
            "TransCRISPR":        (186/255, 85/255,  211/255),

            "Linear Consensus":   (34/255,  139/255, 34/255),
            "Logistic Consensus": (0/255,   100/255, 0/255)        }


        # smooth yellow–blue gradient for counts/rates -------------------
        def yellow_blue_cmap():
            colors = [

                "#c3d2d9", "#9ebfd3", "#79abcc", "#5595c4",
                "#2f7db9", "#1c6db0", "#0a5aa2", "#094b8d", "#083b75"
            ]
            return mcol.LinearSegmentedColormap.from_list("yellow_blue", colors, N=256)

        cmap_yb    = yellow_blue_cmap()
        norm_rates = mcol.Normalize(vmin=0.5, vmax=1.0)

        comparison_map = {
            "Good vs (Moderate ∪ Poor)": "High efficacy (≥ 90 %) vs Sub-optimal (< 90 %)",
            "(Good ∪ Moderate) vs Poor": "Adequate (≥ 60 %) vs Inadequate (< 60 %)",
        }

        # =================================================================
        #  iterate through the two comparisons
        # =================================================================
        for comp in comparisons:
            sub     = df[df["comparison"] == comp].set_index("tool")
            counts  = sub[count_cols].values.astype(int)
            rates   = sub[rate_cols_raw].values.astype(float)

            # -------- build cell text ------------------------------------
            cell_text = [
                [*map(str, ct_row), *(f"{v:.2f}" for v in rt_row)]
                for ct_row, rt_row in zip(counts, rates)
            ]

            # -------- build cell colours ---------------------------------
            cell_colors = []
            for rt_row in rates:
                nv_s = norm_rates(rt_row[0])            # sensitivity
                nv_p = norm_rates(rt_row[1])            # specificity

                pastel_counts = [
                    (cmap_yb(nv_s)),       # TP  (sens-based)
                    (cmap_yb(nv_s)),       # FN
                    (cmap_yb(nv_p)),       # FP  (spec-based)
                    (cmap_yb(nv_p)),       # TN
                ]
                vivid_rates  = [cmap_yb(nv_s), cmap_yb(nv_p)]
                cell_colors.append(pastel_counts + vivid_rates)

            # -------- draw table -----------------------------------------
            fig_h = max(len(sub) * 0.55, 3.5)
            fig, ax = plt.subplots(figsize=(5.0, fig_h))

            # column-widths (normalised so they add up to 1.0) ------------
            raw_cw  = [0.07]*len(count_cols) + [0.10]*len(rate_cols_disp)
            factor  = 1.0 / sum(raw_cw)
            colWidths = [w * factor for w in raw_cw]

            tbl = ax.table(
                cellText    = cell_text,
                cellColours = cell_colors,
                rowLabels   = sub.index,
                colLabels   = count_cols + rate_cols_disp,
                loc         = "center",
                cellLoc     = "center",
                colWidths   = colWidths,
            )

            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1.2, 1.5)

            # -------- thick rule before SENS./SPEC. ----------------------
            for r in range(len(sub) + 1):               # header + body rows
                divider = tbl[(r, len(count_cols))]     # first rate column
                divider.set_linewidth(2.0)
                divider.set_edgecolor("#303030")

            # -------- per-cell styling ----------------------------------
            for (i, j), cell in tbl.get_celld().items():
                cell.set_edgecolor("#d0d0d0")
                cell.set_linewidth(0.8)

                # row-label background colours
                if j == -1 and i > 0:
                    mdl = cell.get_text().get_text()
                    if mdl in model_colors:
                        cell.set_facecolor(model_colors[mdl])
                        cell.set_text_props(weight="bold", color="white")

                # header cells
                if i == 0:
                    cell.set_text_props(weight="bold")

                # widen SENS./SPEC. columns
                if j >= len(count_cols):
                    cell.set_width(cell.get_width() * 1.25)

                # rate cells: bold + auto white/black text
                if j >= len(count_cols) and i > 0:
                    cell.set_text_props(weight="bold")
                    r_, g_, b_, _ = cell.get_facecolor()
                    luminance = 0.299*r_ + 0.587*g_ + 0.114*b_
                    cell.get_text().set_color("white" if luminance < 0.45 else "#333333")

            # -------- caption (figure-level text) ------------------------
            # Draw once so the renderer exists
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            # bounding box of the table (in *figure* coordinates)
            bbox_tbl = tbl.get_window_extent(renderer=renderer) \
                .transformed(fig.transFigure.inverted())

            caption = f"{self.db_name}: {comparison_map.get(comp, comp)}"

            # Convert 2 pt to figure-fraction for the small vertical gap
            pad_fig_y = CAPTION_PAD_PT / (fig_h * fig.dpi)

            fig.text(
                x       = (bbox_tbl.x0 + bbox_tbl.x1) / 2,   # centred horizontally
                y       = bbox_tbl.y1 + pad_fig_y,           # just above the table
                s       = caption,
                ha      = "center",
                va      = "bottom",
                fontsize= CAPTION_FONT,
                style   = CAPTION_STYLE,
            )

            # no axes visuals needed
            ax.axis("off")

            # ---------------------------------------------------------------
            #  tiny “desirability” legend centred *below* the table
            # ---------------------------------------------------------------
            import matplotlib as mpl

            # how tall the bar should be and how far we keep it from the table (figure-fraction units)
            bar_h   = 0.025            # 2.5 % of figure height
            bar_pad = 0.03            # 1.5 % gap

            # position of the new colour-bar axis in figure-fraction coordinates
            bar_left   = bbox_tbl.x0
            bar_width  = bbox_tbl.width
            bar_bottom = max(bbox_tbl.y0 - bar_pad - bar_h, 0)   # clamp at 0 so we stay inside the figure

            cax = fig.add_axes([bar_left, bar_bottom, bar_width, bar_h])

            # horizontal colour bar using the same cmap
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cb   = mpl.colorbar.ColorbarBase(
                cax,
                cmap=cmap_yb,
                norm=norm,
                orientation="horizontal",
                ticks=[],
            )
            cb.outline.set_visible(False)     # cleaner look

            # add labels at both ends
            cax.text(0.0, 0.5, "less desirable",
                     ha="left",  va="center", fontsize=8, transform=cax.transAxes)
            cax.text(1.0, 0.5, "more desirable",
                     ha="right", va="center", fontsize=8,color="white", transform=cax.transAxes)

            # make the background transparent so it blends in
            cax.set_facecolor("none")


# -------- save ------------------------------------------------
            safe = comp.replace(" ", "_").replace("∪", "U")
            fig.savefig(
                self.out / f"lr_table_{safe}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close(fig)



    def _plot_lr_scatter(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(9, 7))
        df["quality"] = df["youden"]
        norm = plt.Normalize(df["quality"].min(), df["quality"].max())

        sizes = (
                140
                + 80 * (df["quality"] - df["quality"].min())
                / (df["quality"].max() - df["quality"].min() + 1e-9)
        ).to_numpy()

        # Use a professional colormap
        sc = ax.scatter(
            df["lr_g"],
            df["lr_gm"],
            s=sizes,
            c=df["quality"],
            cmap=plt.cm.viridis,
            norm=norm,
            edgecolors='#333333',
            linewidth=0.8,
            alpha=0.8,
            zorder=3,
        )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Youden's J Index", fontsize=11)

        annotate_conditional(ax, df)

        # ---------- clearer grid ----------
        ax.minorticks_on()                                    # add minor ticks
        ax.grid(which='major', color='#B0B0B0', linewidth=0.8,
                linestyle='-',  alpha=0.7, zorder=0)
        ax.grid(which='minor', color='#D7D7D7', linewidth=0.5,
                linestyle='--', alpha=0.4, zorder=0)

        # Reference lines
        ax.axvline(1, ls="--", color="#888888", lw=1.0, alpha=0.6)
        ax.axhline(1, ls="--", color="#888888", lw=1.0, alpha=0.6)

        # Academic axis labels
        # Academic axis labels  – replace both original lines
        ax.set_xlabel(r"LR$^{+}$ (High Efficacy vs Sub-optimal)", fontsize=12)
        ax.set_ylabel(r"LR$^{+}$ (Adequate Efficacy vs Inadequate)", fontsize=12)


        # Academic title
        ax.set_title(
            f"{self.db_name}: Positive Likelihood Ratios for Efficacy Prediction\n"
            "Discriminative Power Analysis",
            fontsize=13,
            pad=10
        )

        # Add grid for readability
        ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)

        # Set axis limits with some padding
        x_min, x_max = df["lr_g"].min(), df["lr_g"].max()
        y_min, y_max = df["lr_gm"].min(), df["lr_gm"].max()
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        fig.tight_layout()
        fig.savefig(self.out / f"lr_scatter.{self.fmt}", dpi=300, bbox_inches='tight')
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
# -----------------------------------------------------------------
# put this next to the other helpers (outside the class definition)
def lighten(color, amount=0.35):
    """Blend a colour with white by <amount> (0-1)."""
    import colorsys
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1 - (1 - l) * (1 - amount)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2, a)
# -----------------------------------------------------------------

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
    p.add_argument('--db_name',required=True)
    p.add_argument('--thresholds', help='CSV with columns tool,threshold')
    args = p.parse_args()

    tools = args.tools.split(',') if args.tools else None
    db_name = args.db_name
    preset_df = pd.read_csv(args.thresholds) if args.thresholds else None

    evaluator = PredictionEvaluator(
        args.input,
        args.output,
        efficacy_col=args.efficacy_col,
        poor_thr=args.poor_threshold,
        good_thr=args.good_threshold,
        tool_cols=tools,
        db_name=db_name,
        fmt='png',
        thresholds_df=preset_df,
    )
    evaluator.evaluate()
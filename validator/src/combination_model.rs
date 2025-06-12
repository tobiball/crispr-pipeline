//! combination_model.rs  ─ v0.4.1
//!
//! Minimal, self‑contained helper for **learning and applying** a
//! *linear* ensemble of arbitrary CRISPR guide‑efficacy scores.
//!
//! Changes in **v0.4.1**
//! ---------------------
//! • Replaced use of now‑removed `std_as_series()` with a portable
//!   fallback: `var(1).sqrt()` (works on Polars ≥0.46).
//! • Fixed `Series::new` call (`&str → PlSmallStr`) via `.into()`.
//!
//! Add to Cargo.toml
//! ------------------
//! polars   = { version = "0.46", features = ["lazy"] }
//! ndarray  = "0.15"
//! ndarray-linalg = { version = "0.16", features = ["openblas-static"] }

use polars::prelude::*;
use std::collections::HashMap;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;

/// Mapping *tool → per‑guide score* (for ad‑hoc calls to [`linear_combo`]).
pub type ScoreMap = HashMap<String, f64>;

/// Mapping *tool → learned weight*.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WeightMap(pub HashMap<String, f64>);

impl std::ops::Deref for WeightMap {
    type Target = HashMap<String, f64>;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl std::ops::DerefMut for WeightMap {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

//───────────────────────────────── helpers ─────────────────────────────────//

/// Combine individual predictor scores using the provided `weights`.  Any
/// predictor missing for a given guide is ignored and the remaining weights
/// are renormalised on the fly.
pub fn linear_combo(scores: &ScoreMap, weights: &WeightMap) -> Result<f64, &'static str> {
    let (mut w_sum, mut wx_sum) = (0.0, 0.0);
    for (tool, &score) in scores.iter() {
        if let Some(&w) = weights.get(tool) {
            w_sum += w;
            wx_sum += w * score;
        }
    }
    if w_sum == 0.0 {
        Err("no overlapping tools between scores and weights")
    } else {
        Ok(wx_sum / w_sum)
    }
}

/// Ridge‑regularised closed‑form solution:  `w = (XᵀX + λI)⁻¹ Xᵀy`.
fn ridge_ols(
    x: Array2<f64>,
    y: Array1<f64>,
    feat: &[String],
    lambda: f64,
) -> WeightMap {
    let mut xtx = x.t().dot(&x);
    // add λI to the diagonal
    for i in 0..xtx.nrows() {
        xtx[[i, i]] += lambda;
    }
    let xtx_inv = xtx
        .inv()
        .expect("matrix inversion failed – XᵀX probably singular");
    let coeffs = xtx_inv.dot(&x.t().dot(&y));

    let mut map = WeightMap::default();
    for (i, name) in feat.iter().enumerate() {
        map.insert(name.clone(), coeffs[i]);
    }
    map
}

//───────────────────────────── Polars interface ────────────────────────────//

/// 1) Extract `tools` + `efficacy_col`  2) z‑score standardise predictors
/// 3) fit ridge weights (λ = 1.0).
fn fit_weights_from_dataframe(
    df: &DataFrame,
    tools: &[&str],
    efficacy_col: &str,
) -> PolarsResult<WeightMap> {
    // Collect predictor + target columns, drop rows with any NULL
    let mut cols: Vec<&str> = tools.to_vec();
    cols.push(efficacy_col);
    let clean = df.select(cols)?.drop_nulls::<String>(None)?;

    let n = clean.height();
    let p = tools.len();
    let mut x = Array2::<f64>::zeros((n, p));

    // Standardise each predictor → z‑score
    for (j, &tool) in tools.iter().enumerate() {
        let col = clean.column(tool)?.f64()?;
        let mean = col.mean().unwrap();
        let var = col.var(1).unwrap_or(0.0); // ddof = 1
        let std = var.sqrt().max(1e-9);
        for (i, opt) in col.into_iter().enumerate() {
            x[[i, j]] = (opt.unwrap() - mean) / std;
        }
    }

    // Target vector
    let y_ser = clean.column(efficacy_col)?.f64()?;
    let mut y = Array1::<f64>::zeros(n);
    for (i, opt) in y_ser.into_iter().enumerate() {
        y[i] = opt.unwrap();
    }

    let feat_names: Vec<String> = tools.iter().map(|s| s.to_string()).collect();
    Ok(ridge_ols(x, y, &feat_names, 1.0))
}

/// Append a `new_col` containing the ensemble score built from `weights`.
pub fn add_combined_score_column(
    df: &mut DataFrame,
    tools: &[&str],
    weights: &WeightMap,
    new_col: &str,
) -> PolarsResult<()> {
    // recompute mean & std on current df (same rows expected)
    let mut means = HashMap::new();
    let mut stds = HashMap::new();
    for &tool in tools {
        let col = df.column(tool)?.f64()?;
        let mean = col.mean().unwrap();
        let std = col.var(1).unwrap_or(0.0).sqrt().max(1e-9);
        means.insert(tool, mean);
        stds.insert(tool, std);
    }

    let n = df.height();
    let mut combined = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = ScoreMap::new();
        for &tool in tools {
            if let Some(v) = df.column(tool)?.f64()?.get(i) {
                let z = (v - means[tool]) / stds[tool];
                row.insert(tool.to_string(), z);
            }
        }
        combined.push(linear_combo(&row, weights).unwrap_or(f64::NAN));
    }

    let s = Series::new(new_col.into(), combined);
    df.with_column(s)?;
    Ok(())
}

//──────────────────────────── public API (one‑liner) ───────────────────────//

/// Learn ridge‑regularised weights (λ = 1.0), then append a `new_col`
/// with the ensemble score.  Returns the learned `WeightMap`.
pub fn fit_and_apply(
    df: &mut DataFrame,
    tools: &[&str],
    efficacy_col: &str,
    new_col: &str,
) -> PolarsResult<WeightMap> {
    let w = fit_weights_from_dataframe(df, tools, efficacy_col)?;
    add_combined_score_column(df, tools, &w, new_col)?;
    Ok(w)
}

//───────────────────────────────── tests ───────────────────────────────────//
#[cfg(test)]
mod tests {
    use super::*;
    use polars::df;

    #[test]
    fn smoke() {
        let df = df![
            "f1" => &[0.3, 0.7, 0.1],
            "f2" => &[0.4, 0.1, 0.9],
            "eff" => &[0.35, 0.65, 0.25]
        ].unwrap();
        let mut df = df.clone();
        let tools = ["f1", "f2"];
        let _w = fit_and_apply(&mut df, &tools, "eff", "combo").unwrap();
        assert!(df.column("combo").is_ok());
    }
}

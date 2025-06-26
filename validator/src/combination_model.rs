//! combination_model.rs - Easy drop-in replacement with fixed standardization
//!
//! This version maintains the same interface as before but properly stores
//! training statistics for consistent predictions.

use polars::prelude::*;
use std::collections::HashMap;
use log::info;
use ndarray::{Array1, Array2};
use ndarray_linalg::Inverse;

/// Mapping *tool → per‑guide score* (for ad‑hoc calls to [`linear_combo`]).
pub type ScoreMap = HashMap<String, f64>;

/// Mapping *tool → learned weight* (kept for compatibility).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WeightMap(pub HashMap<String, f64>);

impl std::ops::Deref for WeightMap {
    type Target = HashMap<String, f64>;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl std::ops::DerefMut for WeightMap {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

/// Complete model data stored in JSON files
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelData {
    weights: HashMap<String, f64>,
    means: HashMap<String, f64>,
    stds: HashMap<String, f64>,
}

//───────────────────────────────── helpers ─────────────────────────────────//

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

//───────────────────────────── Training ────────────────────────────//

/// Fit and apply - trains model and saves complete data to files
/// Returns WeightMap for compatibility
pub fn fit_and_apply(
    df: &mut DataFrame,
    tools: &[&str],
    efficacy_col: &str,
    new_col: &str,
) -> PolarsResult<WeightMap> {
    // Collect predictor + target columns, drop rows with any NULL
    let mut cols: Vec<&str> = tools.to_vec();
    cols.push(efficacy_col);
    let clean = df.select(cols)?.drop_nulls::<String>(None)?;

    let n = clean.height();
    let p = tools.len();
    let mut x = Array2::<f64>::zeros((n, p));

    // Store standardization parameters
    let mut means = HashMap::new();
    let mut stds = HashMap::new();

    info!("=== Training standardization parameters ===");

    // Standardise each predictor → z‑score and save parameters
    for (j, &tool) in tools.iter().enumerate() {
        let col = clean.column(tool)?.f64()?;
        let mean = col.mean().unwrap();
        let var = col.var(1).unwrap_or(0.0); // ddof = 1
        let std = var.sqrt().max(1e-9);

        // Store for later use
        means.insert(tool.to_string(), mean);
        stds.insert(tool.to_string(), std);

        info!("{:<25} μ = {:>10.6},  σ = {:>10.6}", tool, mean, std);

        for (i, opt) in col.into_iter().enumerate() {
            x[[i, j]] = (opt.unwrap() - mean) / std;
        }
    }
    info!("==========================================");

    // Target vector
    let y_ser = clean.column(efficacy_col)?.f64()?;
    let mut y = Array1::<f64>::zeros(n);
    for (i, opt) in y_ser.into_iter().enumerate() {
        y[i] = opt.unwrap();
    }

    let feat_names: Vec<String> = tools.iter().map(|s| s.to_string()).collect();
    let weights = ridge_ols(x, y, &feat_names, 1.0);

    // Save complete model data internally
    let model_data = ModelData {
        weights: weights.0.clone(),
        means: means.clone(),
        stds: stds.clone(),
    };

    // Save to a hidden file that corresponds to the weights file
    // This will be loaded automatically when add_combined_score_column is called
    let model_path = format!(".{}_model_data.json", new_col.replace(" ", "_").to_lowercase());
    std::fs::write(
        &model_path,
        serde_json::to_string_pretty(&model_data).unwrap()
    ).map_err(|e| PolarsError::ComputeError(format!("Failed to save model data: {}", e).into()))?;

    // Apply to current dataframe
    add_combined_score_column_internal(df, tools, &weights, &means, &stds, new_col)?;

    Ok(weights)
}

//───────────────────────────── Prediction ────────────────────────────//

/// Internal function that uses provided standardization params
fn add_combined_score_column_internal(
    df: &mut DataFrame,
    tools: &[&str],
    weights: &WeightMap,
    means: &HashMap<String, f64>,
    stds: &HashMap<String, f64>,
    new_col: &str,
) -> PolarsResult<()> {
    let n = df.height();
    let mut combined = Vec::with_capacity(n);

    // NEW ▸ store per-row vectors
    let mut contrib_rows: Vec<Vec<f64>> = Vec::with_capacity(n);

    info!("=== Applying model with stored statistics ===");

    for i in 0..n {
        let mut w_sum = 0.0;
        let mut wx_sum = 0.0;
        let mut row_contrib = Vec::with_capacity(tools.len());      // NEW

        for &tool in tools {
            if let Some(v) = df.column(tool)?.f64()?.get(i) {
                if let (Some(&w), Some(&mean), Some(&std)) =
                    (weights.get(tool), means.get(tool), stds.get(tool))
                {
                    let z = (v - mean) / std;
                    w_sum  += w;
                    wx_sum += w * z;
                    row_contrib.push(w * z);                        // NEW

                    if i == 0 {
                        info!(
                            "[row 0] {:<25} raw {:>10.5} z {:>9.5} w·z {:>9.5}",
                            tool, v, z, w * z
                        );
                    }
                } else {
                    row_contrib.push(f64::NAN);                     // missing weight
                }
            } else {
                row_contrib.push(f64::NAN);                         // NULL entry
            }
        }

        contrib_rows.push(row_contrib);                             // NEW
        combined.push(if w_sum > 0.0 { wx_sum / w_sum } else { f64::NAN });
    }

    // NEW ▸ dump CSV
    let csv_name = format!("contributions_{}.csv",
                           new_col.replace(' ', "_"));
    write_contributions_csv(&contrib_rows, tools, &csv_name)?;
    info!("Per-feature contributions written to {}", csv_name);

    df.with_column(Series::new(PlSmallStr::from(new_col), combined))?;
    Ok(())
}


/// Public function that loads model data from the hidden file
pub fn add_combined_score_column(
    df: &mut DataFrame,
    tools: &[&str],
    weights: &WeightMap,
    new_col: &str,
) -> PolarsResult<()> {
    // Try to load the complete model data
    let model_path = format!(".{}_model_data.json", new_col.replace(" ", "_").to_lowercase());

    match std::fs::read_to_string(&model_path) {
        Ok(json) => {
            // Use stored training statistics
            let model_data: ModelData = serde_json::from_str(&json)
                .map_err(|e| PolarsError::ComputeError(format!("Failed to parse model data: {}", e).into()))?;

            info!("Loaded training statistics from {}", model_path);
            add_combined_score_column_internal(df, tools, weights, &model_data.means, &model_data.stds, new_col)
        }
        Err(_) => {
            // Fallback: compute from current dataframe (old behavior)
            // This maintains backward compatibility but should be avoided
            info!("WARNING: No saved model data found at {}. Using current dataframe statistics (not recommended).", model_path);

            let mut means = HashMap::new();
            let mut stds = HashMap::new();

            for &tool in tools {
                let col = df.column(tool)?.f64()?;
                let mean = col.mean().unwrap();
                let std = col.var(1).unwrap_or(0.0).sqrt().max(1e-9);
                means.insert(tool.to_string(), mean);
                stds.insert(tool.to_string(), std);
            }

            add_combined_score_column_internal(df, tools, weights, &means, &stds, new_col)
        }
    }
}

// ─── Save per-feature contribution table ────────────────────────────
fn write_contributions_csv(
    contrib: &[Vec<f64>],
    header: &[&str],
    path: &str,
) -> PolarsResult<()> {
    use polars::prelude::*;
    let mut df = DataFrame::default();
    for (j, &name) in header.iter().enumerate() {
        let col: Vec<f64> = contrib.iter().map(|row| row[j]).collect();
        df.with_column(Series::new(PlSmallStr::from(name), col))?;
    }
    let mut file = std::fs::File::create(path)?;
    CsvWriter::new(&mut file).finish(&mut df)?;
    Ok(())
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
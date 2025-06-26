//! logistic_cv.rs – gene-stratified CV logistic regression (Youden J) with saved statistics

use polars::prelude::*;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, SeedableRng};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use log::info;

use crate::combination_model::WeightMap;

/// Complete logistic model data stored in JSON files
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct LogisticModelData {
    betas: HashMap<String, f64>,
    means: HashMap<String, f64>,
    stds: HashMap<String, f64>,
    threshold: f64,
}

// ───────── helpers ─────────
fn zscore_cols(df: &mut DataFrame, tools: &[&str]) -> PolarsResult<()> {
    for &tool in tools {
        let col = df.column(tool)?.f64()?;
        let mean = col.mean().unwrap();
        let std  = col.var(1).unwrap_or(0.0).sqrt().max(1e-9);
        let z: Vec<Option<f64>> = col
            .into_iter()
            .map(|opt| opt.map(|v| (v - mean) / std))
            .collect();

        df.replace(tool, Series::new(tool.into(), z))?;
    }
    Ok(())
}

fn youden_j(prob: &Array1<f64>, y: &Array1<u8>, thr: f64) -> f64 {
    let (mut tp, mut fp, mut fn_, mut tn) = (0, 0, 0, 0);
    for (&p, &lab) in prob.iter().zip(y) {
        match (p >= thr, lab == 1) {
            (true,  true)  => tp += 1,
            (true,  false) => fp += 1,
            (false, true)  => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let sens = tp as f64 / (tp + fn_) as f64;
    let spec = tn as f64 / (tn + fp) as f64;
    sens + spec - 1.0
}

// ───────── public API ─────────
pub fn fit_logistic_cv(
    df: &mut DataFrame,
    tools: &[&str],
    efficacy_col: &str,
    good_cut: f64,
    gene_col: &str,
    n_folds: usize,
    l2: f32,
) -> PolarsResult<(WeightMap, f64)> {
    // Store original statistics before standardization
    let mut means = HashMap::new();
    let mut stds = HashMap::new();

    info!("=== Training logistic standardization parameters ===");

    // Compute and store standardization parameters
    for &tool in tools {
        let col = df.column(tool)?.f64()?;
        let mean = col.mean().unwrap();
        let var = col.var(1).unwrap_or(0.0); // ddof = 1
        let std = var.sqrt().max(1e-9);

        means.insert(tool.to_string(), mean);
        stds.insert(tool.to_string(), std);

        info!("{:<25} μ = {:>10.6},  σ = {:>10.6}", tool, mean, std);
    }
    info!("================================================");

    // Now standardize the dataframe
    zscore_cols(df, tools)?;

    // 1 ▸ prepare X, y
    let y: Array1<u8> = df.column(efficacy_col)?
        .f64()?.into_no_null_iter()
        .map(|v| if v >= good_cut { 1 } else { 0 })
        .collect();

    let mut x = Array2::<f64>::zeros((df.height(), tools.len()));
    for (j, &tool) in tools.iter().enumerate() {
        for (i, v) in df.column(tool)?.f64()?.into_no_null_iter().enumerate() {
            x[[i, j]] = v;
        }
    }

    // 2 ▸ class-balanced sample weights
    let pos = y.iter().filter(|&&v| v == 1).count() as f32;
    let w_neg = if pos == 0.0 { 1.0 } else { pos / (y.len() as f32 - pos) };
    let weights: Array1<f32> = y.iter()
        .map(|&lab| if lab == 1 { 1.0 } else { w_neg })
        .collect();

    // 3 ▸ gene-stratified folds
    let mut by_gene: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, g) in df.column(gene_col)?.str()?.into_no_null_iter().enumerate() {
        by_gene.entry(g).or_default().push(idx);
    }
    let buckets: Vec<Vec<usize>> = by_gene.into_values().collect();

    let mut order: Vec<_> = (0..buckets.len()).collect();
    let mut rng = StdRng::seed_from_u64(42);
    order.shuffle(&mut rng);

    let mut fold_thr = Vec::with_capacity(n_folds);

    for f in 0..n_folds {
        let test_idx: Vec<usize> = order.iter()
            .enumerate()
            .filter(|(i, _)| i % n_folds == f)
            .flat_map(|(_, &b)| buckets[b].clone())
            .collect();

        let train_idx: Vec<usize> = order.iter()
            .enumerate()
            .filter(|(i, _)| i % n_folds != f)
            .flat_map(|(_, &b)| buckets[b].clone())
            .collect();

        let ds_train = Dataset::new(
            x.select(Axis(0), &train_idx),
            y.select(Axis(0), &train_idx)
        ).with_weights(weights.select(Axis(0), &train_idx));

        let ds_test_x = x.select(Axis(0), &test_idx);
        let ds_test_y = y.select(Axis(0), &test_idx);

        let model = LogisticRegression::default()
            .max_iterations(100)
            .gradient_tolerance(1e-6)
            .alpha(l2 as f64)
            .fit(&ds_train)
            .map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;

        let prob = model.predict_probabilities(&ds_test_x);

        let best_thr = (1..100)
            .map(|t| t as f64 / 100.0)
            .max_by(|a, b| {
                youden_j(&prob, &ds_test_y, *a)
                    .total_cmp(&youden_j(&prob, &ds_test_y, *b))
            })
            .unwrap();
        fold_thr.push(best_thr);
    }

    let thr_final = fold_thr.iter().sum::<f64>() / n_folds as f64;

    // 4 ▸ train on full data
    let final_ds = Dataset::new(x, y).with_weights(weights);

    let final_model = LogisticRegression::default()
        .max_iterations(100)
        .gradient_tolerance(1e-6)
        .alpha(l2 as f64)
        .fit(&final_ds)
        .map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;

    // 5 ▸ β-weights
    let mut betas = HashMap::new();
    for (beta, &name) in final_model.params().iter().zip(tools) {
        betas.insert(name.to_string(), *beta);
    }

    // Save complete model data
    let model_data = LogisticModelData {
        betas: betas.clone(),
        means: means.clone(),
        stds: stds.clone(),
        threshold: thr_final,
    };

    // Save to a hidden file - using a pattern that includes 'logistic' to distinguish from linear models
    let model_path = format!(".logistic_{}_model_data.json", "Logistic Consensus".replace(" ", "_").to_lowercase());
    std::fs::write(
        &model_path,
        serde_json::to_string_pretty(&model_data).unwrap()
    ).map_err(|e| PolarsError::ComputeError(format!("Failed to save logistic model data: {}", e).into()))?;

    info!("Saved logistic model data to {}", model_path);

    // Return weights in the expected format
    let mut weight_map = WeightMap::default();
    for (name, beta) in betas {
        weight_map.insert(name, beta);
    }

    Ok((weight_map, thr_final))
}

// ───────── Enhanced apply function ─────────
pub fn apply_logistic_score(
    df: &mut DataFrame,
    tools: &[&str],
    betas: &WeightMap,
    out_col: &str,
) -> PolarsResult<()> {
    // Try to load the complete model data
    let model_path = format!(".logistic_{}_model_data.json", out_col.replace(" ", "_").to_lowercase());

    match std::fs::read_to_string(&model_path) {
        Ok(json) => {
            // Use stored training statistics
            let model_data: LogisticModelData = serde_json::from_str(&json)
                .map_err(|e| PolarsError::ComputeError(format!("Failed to parse logistic model data: {}", e).into()))?;

            info!("Loaded logistic training statistics from {}", model_path);
            apply_logistic_score_internal(df, tools, &model_data.betas, &model_data.means, &model_data.stds, out_col)
        }
        Err(e) => {
            // No fallback - model data is required
            Err(PolarsError::ComputeError(
                format!("Failed to load logistic model data from {}: {}. Please ensure the model has been trained and saved.", model_path, e).into()
            ))
        }
    }
}

// Internal function that uses provided statistics
fn apply_logistic_score_internal(
    df: &mut DataFrame,
    tools: &[&str],
    betas: &HashMap<String, f64>,
    means: &HashMap<String, f64>,
    stds: &HashMap<String, f64>,
    out_col: &str,
) -> PolarsResult<()> {
    let mut prob: Vec<f64> = Vec::with_capacity(df.height());

    // Debug info for first row
    info!("=== Applying logistic model with stored statistics ===");

    let mut prob = Vec::with_capacity(df.height());
    let mut contrib_rows = Vec::with_capacity(df.height());  // NEW

    for i in 0..df.height() {
        let mut lin = 0.0;
        let mut row_contrib = Vec::with_capacity(tools.len()); // NEW

        for &tool in tools {
            let val = df.column(tool)?.f64()?.get(i).unwrap();
            let (beta, mean, std) = (
                *betas.get(tool).unwrap(),
                *means.get(tool).unwrap(),
                *stds.get(tool).unwrap(),
            );
            let z = (val - mean) / std;
            let c = z * beta;                // ← contribution
            lin += c;
            row_contrib.push(c);             // NEW
        }
        prob.push(1.0 / (1.0 + (-lin).exp()));
        contrib_rows.push(row_contrib);      // NEW
    }

    // Save contributions
    let csv_name = format!("contributions_{}.csv", out_col.replace(' ', "_"));
    write_contributions_csv(&contrib_rows, tools, &csv_name)?;
    info!("Per-feature contributions written to {}", csv_name);

    df.with_column(Series::new(PlSmallStr::from(out_col), prob))?;
    Ok(())
}

// New convenience function for getting the saved threshold
pub fn load_logistic_threshold(model_name: &str) -> Option<f64> {
    let model_path = format!(".logistic_{}_model_data.json", model_name.replace(" ", "_").to_lowercase());

    match std::fs::read_to_string(&model_path) {
        Ok(json) => {
            let model_data: LogisticModelData = serde_json::from_str(&json).ok()?;
            Some(model_data.threshold)
        }
        Err(_) => None
    }
}

// Check if a logistic model file exists
pub fn logistic_model_exists(model_name: &str) -> bool {
    let model_path = format!(".logistic_{}_model_data.json", model_name.replace(" ", "_").to_lowercase());
    std::path::Path::new(&model_path).exists()
}

// Migration helper for code using the old API
#[deprecated(note = "Use apply_logistic_score without df_training parameter")]
pub fn apply_logistic_score_legacy(
    df: &mut DataFrame,
    _df_training: &DataFrame,  // Ignored
    tools: &[&str],
    betas: &WeightMap,
    out_col: &str,
) -> PolarsResult<()> {
    apply_logistic_score(df, tools, betas, out_col)
}

// ───────── Positive-class log-likelihood (unchanged) ─────────
pub fn pos_ll(prob: &Array1<f64>, y: &Array1<u8>, thr: f64) -> f64 {
    let eps = 1e-12;
    prob.iter()
        .zip(y)
        .filter(|(_, &lab)| lab == 1)
        .map(|(&p, _)| (if p >= thr {p} else {1.0 - p}).max(eps).ln())
        .sum()
}

use polars::io::SerWriter;   // already in polars 0.39


// ---- revised helper --------------------------------------------------------
fn write_contributions_csv(
    contrib: &[Vec<f64>],
    header: &[&str],
    path: &str,
) -> PolarsResult<()> {
    // Build an empty frame and add one Series at a time.
    let mut df = DataFrame::default();

    for (j, &name) in header.iter().enumerate() {
        let col: Vec<f64> = contrib.iter().map(|row| row[j]).collect();
        df.with_column(Series::new(PlSmallStr::from(name), col))?;
    }

    // Write it
    let mut file = std::fs::File::create(path)?;
    CsvWriter::new(&mut file).finish(&mut df)?;
    Ok(())
}



// ───────── Usage Example ─────────
// Training - saves statistics automatically:
// let (weights, threshold) = fit_logistic_cv(
//     &mut df_train,
//     &tools,
//     "efficacy",
//     0.5,
//     "gene",
//     5,
//     1.0
// )?;
//
// Prediction - uses saved statistics:
// apply_logistic_score(
//     &mut df_test,
//     &tools,
//     &weights,
//     "logistic_score"
// )?;
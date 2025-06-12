//! logistic_cv.rs  – gene-stratified CV logistic regression (Youden J)

use polars::prelude::*;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, SeedableRng};
use rand::seq::SliceRandom;                 // shuffle()
use std::collections::HashMap;

use crate::combination_model::WeightMap;

// ───────── helpers ─────────
fn zscore_cols(df: &mut DataFrame, tools: &[&str]) -> PolarsResult<()> {
    for &tool in tools {
        let col = df.column(tool)?.f64()?;
        let mean = col.mean().unwrap();
        let std  = col.var(1).unwrap_or(0.0).sqrt().max(1e-9);
        let z: Vec<Option<f64>> = col
            .into_iter()
            .map(|opt| opt.map(|v| (v - mean) / std))   // keep None as None
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
    // 1 ▸ prepare X, y
    zscore_cols(df, tools)?;
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

    // 2 ▸ class-balanced sample weights  (f32!)
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
            .alpha(l2 as f64)                // L2 strength
            .fit(&ds_train)
            .map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;

        let prob = model.predict_probabilities(&ds_test_x);   // Array1<f64>

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
    let mut map = WeightMap::default();
    for (beta, &name) in final_model.params().iter().zip(tools) {
        map.insert(name.to_string(), *beta);
    }

    Ok((map, thr_final))
}

// ───────── NEW helper ─────────
pub fn apply_logistic_score(
    df: &mut DataFrame,
    tools: &[&str],
    betas: &WeightMap,
    out_col: &str,
) -> PolarsResult<()> {
    // z-score the predictor columns *exactly* like during training
    zscore_cols(df, tools)?;

    // build the probability for every row
    let prob: Vec<f64> = (0..df.height())
        .map(|i| {
            let lin = tools.iter()
                .map(|&t| df.column(t).unwrap().f64().unwrap().get(i).unwrap()
                    * betas[t])
                .sum::<f64>();
            1.0 / (1.0 + (-lin).exp())          // sigmoid
        })
        .collect();

    df.with_column(Series::new(PlSmallStr::from(out_col), prob))?;
    Ok(())
}

/// Positive-class log-likelihood at a cut-off `thr`.
fn pos_ll(prob: &Array1<f64>, y: &Array1<u8>, thr: f64) -> f64 {
    let eps = 1e-12;
    prob.iter()
        .zip(y)
        .filter(|(_, &lab)| lab == 1)      // keep only positives
        .map(|(&p, _)| (if p >= thr {p} else {1.0 - p}).max(eps).ln())
        .sum()
}

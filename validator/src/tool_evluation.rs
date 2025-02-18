use crate::helper_functions::read_csv;
use polars::prelude::{CsvReader, DataFrame, PolarsError, PolarsResult};
use plotters::prelude::*;
use std::error::Error;
use std::fs;
use std::io::Write;
use tracing::{debug, error, info};
use plotters_backend::BackendTextStyle;
// --------------------------------------------------------
//  Constants
// --------------------------------------------------------
const FRACTION_CUTOFF: f64 = 0.25;
const DATASET_CUTOFF: f64 = 90.0;
const CALIBRATION_BINS: usize = 10;

// --------------------------------------------------------
//  Entrypoint
// --------------------------------------------------------
pub fn analyze_chopchop_results(csv_file_path: &str, dataset: &str) -> PolarsResult<()> {
    // Create output directories.
    let results_directory = format!("results_{}", dataset);
    for subdir in &["", "fraction", "calibrated", "mapping"] {
        let dir = if subdir.is_empty() {
            results_directory.clone()
        } else {
            format!("{}/{}", results_directory, subdir)
        };
        fs::create_dir_all(&dir).map_err(|e| polars_err(Box::new(e)))?;
    }

    // Read CSV and extract required columns.
    let df = read_csv(csv_file_path)?;
    let differences: Vec<f64> = df
        .column("difference")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let dataset_vals: Vec<f64> = df
        .column("dataset_efficacy")?
        .f64()?
        .into_no_null_iter()
        .collect();
    let chopchop_vals: Vec<f64> = df
        .column("chopchop_efficiency")?
        .f64()?
        .into_no_null_iter()
        .collect();

    if differences.is_empty() {
        info!("No valid data. Exiting early.");
        return Ok(());
    }

    // Basic stats.
    let (min_diff, max_diff, mean_diff) = basic_stats(&differences);
    info!("--- difference stats ---");
    info!("Count: {}", differences.len());
    info!("Min:   {:.3}", min_diff);
    info!("Max:   {:.3}", max_diff);
    info!("Mean:  {:.3}", mean_diff);

    // Spearman correlation.
    if let Some(sp) = spearman_correlation(&dataset_vals, &chopchop_vals) {
        info!("Spearman Corr = {:.3}", sp);
    }

    // Fraction-based plots.
    fraction_plots(&differences, &dataset_vals, &chopchop_vals, &results_directory)?;

    // Calibrated plots.
    calibrated_plots(&dataset_vals, &chopchop_vals, &results_directory)?;

    // Mapping + Regression.
    mapping_regression_plots(&dataset_vals, &chopchop_vals, &results_directory)?;

    // Calibration table + plot.
    produce_calibration_analysis(
        &dataset_vals,
        &chopchop_vals,
        DATASET_CUTOFF,
        CALIBRATION_BINS,
        &results_directory,
    )?;

    // Now find the best threshold for chopchop to maximize F1
    let (best_pred_thr, best_f1) = find_best_predicted_cutoff(
        &dataset_vals,
        &chopchop_vals,
        DATASET_CUTOFF
    );
    println!("Best CHOPCHOP threshold = {:.3} => F1 = {:.3}", best_pred_thr, best_f1);

    // You have already computed this in your code:
    let cm = compute_confusion_matrix(&dataset_vals, &chopchop_vals, DATASET_CUTOFF, best_pred_thr);
    let (prec, rec, f1) = precision_recall_f1(&cm);
    println!("Final results: precision={:.3}, recall={:.3}, f1={:.3}", prec, rec, f1);

    // Combine a multi-line title showing both cutoffs:
    let cm_title = format!(
        "Confusion Matrix: CHOPCHOP vs Dataset\n(Actual ≥ {:.2} [manual], Pred ≥ {:.2} [auto])",
        DATASET_CUTOFF, // your manually chosen actual cutoff
        best_pred_thr   // the auto-chosen predicted threshold
    );

    // Convert [TN, FP, FN, TP] => [[TN, FP], [FN, TP]]:
    let cm_2x2 = [[cm[0], cm[1]], [cm[2], cm[3]]];

    // Call create_confusion_matrix with the multi-line title:
    create_confusion_matrix(
        &format!("{}/confusion_matrix.png", results_directory),
        cm_2x2,
        &cm_title,
    ).map_err(polars_err)?;;


    // Finally, log that we're done:
    info!("Done. See '{}' for outputs.", results_directory);
    Ok(())
}

// --------------------------------------------------------
//  Error conversion helper
// --------------------------------------------------------
fn polars_err(err: Box<dyn Error>) -> PolarsError {
    PolarsError::ComputeError(err.to_string().into())
}

// --------------------------------------------------------
//  Compute Confusion Matrix (Actual vs Predicted)
// --------------------------------------------------------
/// For each sample, if the actual value (dataset efficacy) ≥ threshold, it is positive;
/// similarly, if the predicted value (CHOPCHOP efficiency) ≥ threshold, it is predicted positive.
/// The confusion matrix is arranged as [[TN, FP], [FN, TP]].
/// Compute a confusion matrix given:
/// - `actual` values and an `actual_cutoff` to decide if actual is "positive"
/// - `predicted` values and a `predicted_cutoff` to decide if predicted is "positive"
fn compute_confusion_matrix(
    actual: &[f64],
    predicted: &[f64],
    actual_cutoff: f64,
    predicted_cutoff: f64,
) -> [usize; 4] {
    // We'll return [TN, FP, FN, TP] in a flat array for convenience
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    let mut tp = 0;
    for (a, p) in actual.iter().zip(predicted.iter()) {
        let is_actual_pos = *a >= actual_cutoff;
        let is_pred_pos = *p >= predicted_cutoff;
        match (is_actual_pos, is_pred_pos) {
            (true, true) => tp += 1,
            (true, false) => fn_ += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
        }
    }
    [tn, fp, fn_, tp]
}

/// Compute precision, recall, F1 from a confusion matrix laid out as [TN, FP, FN, TP].
fn precision_recall_f1(cm: &[usize; 4]) -> (f64, f64, f64) {
    let tn = cm[0] as f64;
    let fp = cm[1] as f64;
    let fn_ = cm[2] as f64;
    let tp = cm[3] as f64;

    let precision = if tp + fp > 0.0 {
        tp / (tp + fp)
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0.0 {
        tp / (tp + fn_)
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}

/// Search over all unique predicted values to find the threshold that gives the best F1.
/// We keep the dataset cutoff fixed (actual_cutoff).
fn find_best_predicted_cutoff(
    actual: &[f64],
    predicted: &[f64],
    actual_cutoff: f64,
) -> (f64, f64) {
    // Gather all unique predicted values
    let mut unique_preds: Vec<f64> = predicted.iter().cloned().collect();
    unique_preds.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    unique_preds.dedup();

    let mut best_threshold = unique_preds[0];
    let mut best_f1 = 0.0;

    for &thr in &unique_preds {
        let cm = compute_confusion_matrix(actual, predicted, actual_cutoff, thr);
        let (_prec, _rec, f1) = precision_recall_f1(&cm);
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = thr;
        }
    }
    (best_threshold, best_f1)
}


// --------------------------------------------------------
//  Fraction-based approach
// --------------------------------------------------------
fn fraction_plots(
    differences: &[f64],
    dataset_vals: &[f64],
    chopchop_vals: &[f64],
    results_directory: &str,
) -> PolarsResult<()> {
    let frac_dir = format!("{}/fraction", results_directory);

    // Overall.
    {
        let (pr_rec, pr_prec) = evaluate_pr_fraction(dataset_vals, chopchop_vals, FRACTION_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_overall.png", frac_dir),
            &pr_rec,
            &pr_prec,
            &format!("PR (Fraction {:.0}%) - Overall", FRACTION_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr, roc_tpr) = evaluate_roc_fraction(dataset_vals, chopchop_vals, FRACTION_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_overall.png", frac_dir),
            &roc_fpr,
            &roc_tpr,
            &format!("ROC (Fraction {:.0}%) - Overall", FRACTION_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;

        create_diff_histogram(&format!("{}/difference_histogram.png", frac_dir), differences)
            .map_err(polars_err)?;
    }

    // Dataset alone.
    {
        create_histogram(
            &format!("{}/dataset_efficacy_histogram.png", frac_dir),
            dataset_vals,
            "Dataset Efficacy (Fraction)",
            "Efficacy",
            "Count",
        )
            .map_err(polars_err)?;

        create_scatter(
            &format!("{}/dataset_scatter.png", frac_dir),
            dataset_vals,
            dataset_vals,
            "Dataset Efficacy",
            "Dataset Efficacy",
            "Dataset vs Dataset (Fraction)",
        )
            .map_err(polars_err)?;

        let (pr_rec_ds, pr_prec_ds) = evaluate_pr_fraction(dataset_vals, dataset_vals, FRACTION_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_dataset.png", frac_dir),
            &pr_rec_ds,
            &pr_prec_ds,
            &format!("PR - Dataset alone (Fraction {:.0}%)", FRACTION_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_ds, roc_tpr_ds) = evaluate_roc_fraction(dataset_vals, dataset_vals, FRACTION_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_dataset.png", frac_dir),
            &roc_fpr_ds,
            &roc_tpr_ds,
            &format!("ROC - Dataset alone (Fraction {:.0}%)", FRACTION_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;
    }

    // CHOPCHOP alone.
    {
        create_histogram(
            &format!("{}/chopchop_efficiency_histogram.png", frac_dir),
            chopchop_vals,
            "CHOPCHOP Efficiency (Fraction)",
            "CHOPCHOP Efficiency",
            "Count",
        )
            .map_err(polars_err)?;

        create_scatter(
            &format!("{}/chopchop_scatter.png", frac_dir),
            chopchop_vals,
            chopchop_vals,
            "CHOPCHOP Efficiency",
            "CHOPCHOP Efficiency",
            "CHOPCHOP vs CHOPCHOP (Fraction)",
        )
            .map_err(polars_err)?;

        let (pr_rec_cc, pr_prec_cc) =
            evaluate_pr_fraction(chopchop_vals, chopchop_vals, FRACTION_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_chopchop.png", frac_dir),
            &pr_rec_cc,
            &pr_prec_cc,
            &format!("PR - CHOPCHOP alone (Fraction {:.0}%)", FRACTION_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_cc, roc_tpr_cc) =
            evaluate_roc_fraction(chopchop_vals, chopchop_vals, FRACTION_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_chopchop.png", frac_dir),
            &roc_fpr_cc,
            &roc_tpr_cc,
            &format!("ROC - CHOPCHOP alone (Fraction {:.0}%)", FRACTION_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;
    }

    Ok(())
}

// --------------------------------------------------------
//  Calibrated approach
// --------------------------------------------------------
fn calibrated_plots(dataset_vals: &[f64], chopchop_vals: &[f64], results_directory: &str) -> PolarsResult<()> {
    let cal_dir = format!("{}/calibrated", results_directory);

    // Overall.
    {
        let (pr_rec, pr_prec) = evaluate_pr_calibrated(dataset_vals, chopchop_vals, DATASET_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_overall.png", cal_dir),
            &pr_rec,
            &pr_prec,
            &format!("PR (Calibrated ≥{:.0}%) - Overall", DATASET_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr, roc_tpr) = evaluate_roc_calibrated(dataset_vals, chopchop_vals, DATASET_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_overall.png", cal_dir),
            &roc_fpr,
            &roc_tpr,
            &format!("ROC (Calibrated ≥{:.0}%) - Overall", DATASET_CUTOFF),
        )
            .map_err(polars_err)?;
    }

    // Dataset alone.
    {
        let (pr_rec_ds, pr_prec_ds) = evaluate_pr_calibrated(dataset_vals, dataset_vals, DATASET_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_dataset.png", cal_dir),
            &pr_rec_ds,
            &pr_prec_ds,
            &format!("PR - Dataset alone (≥{:.0}%)", DATASET_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_ds, roc_tpr_ds) = evaluate_roc_calibrated(dataset_vals, dataset_vals, DATASET_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_dataset.png", cal_dir),
            &roc_fpr_ds,
            &roc_tpr_ds,
            &format!("ROC - Dataset alone (≥{:.0}%)", DATASET_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;
    }

    // CHOPCHOP alone.
    {
        let (pr_rec_cc, pr_prec_cc) = evaluate_pr_calibrated(chopchop_vals, chopchop_vals, DATASET_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_chopchop.png", cal_dir),
            &pr_rec_cc,
            &pr_prec_cc,
            &format!("PR - CHOPCHOP alone (≥{:.0}%)", DATASET_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_cc, roc_tpr_cc) = evaluate_roc_calibrated(chopchop_vals, chopchop_vals, DATASET_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_chopchop.png", cal_dir),
            &roc_fpr_cc,
            &roc_tpr_cc,
            &format!("ROC - CHOPCHOP alone (≥{:.0}%)", DATASET_CUTOFF * 100.0),
        )
            .map_err(polars_err)?;
    }

    Ok(())
}

// --------------------------------------------------------
//  Mapping + Regression
// --------------------------------------------------------
fn mapping_regression_plots(dataset_vals: &[f64], chopchop_vals: &[f64], results_directory: &str) -> PolarsResult<()> {
    let map_dir = format!("{}/mapping", results_directory);
    let pairs = analyze_value_mapping(dataset_vals, chopchop_vals); // (chopchop, dataset)
    let (slope, intercept) = compute_linear_regression(&pairs);
    info!(
        "REGRESSION => dataset ~ slope * chopchop + intercept => slope={:.3}, intercept={:.3}",
        slope, intercept
    );

    create_mapping_scatter_plot(
        &format!("{}/mapping_scatter_regression.png", map_dir),
        &pairs,
        slope,
        intercept,
    )
        .map_err(polars_err)?;

    Ok(())
}

// --------------------------------------------------------
//  Calibration table + plot
// --------------------------------------------------------
fn produce_calibration_analysis(
    dataset_vals: &[f64],
    chopchop_vals: &[f64],
    dataset_cutoff: f64,
    bin_count: usize,
    results_directory: &str,
) -> PolarsResult<()> {
    let map_dir = format!("{}/mapping", results_directory);

    let calibration_points = compute_calibration_bins(chopchop_vals, dataset_vals, dataset_cutoff, bin_count);

    // Write calibration table as CSV.
    let table_path = format!("{}/calibration_table.csv", map_dir);
    {
        let mut file = fs::File::create(&table_path).map_err(|e| polars_err(Box::new(e)))?;
        writeln!(
            file,
            "bin_index,bin_start,bin_end,mean_chopchop,fraction_good,count"
        )
            .map_err(|e| polars_err(Box::new(e)))?;
        for (i, cp) in calibration_points.iter().enumerate() {
            writeln!(
                file,
                "{},{:.3},{:.3},{:.3},{:.3},{}",
                i, cp.bin_start, cp.bin_end, cp.mean_chopchop, cp.fraction_good, cp.count
            )
                .map_err(|e| polars_err(Box::new(e)))?;
        }
        info!("Calibration table saved to {}", table_path);
    }

    // Generate calibration plot.
    let plot_path = format!("{}/calibration_curve.png", map_dir);
    create_calibration_plot(&plot_path, &calibration_points).map_err(polars_err)?;

    Ok(())
}

// --------------------------------------------------------
//  Calibration Data Structures and Computation
// --------------------------------------------------------
#[derive(Debug)]
struct CalibPoint {
    bin_start: f64,
    bin_end: f64,
    mean_chopchop: f64,
    fraction_good: f64,
    count: usize,
}

fn compute_calibration_bins(
    chopchop_vals: &[f64],
    dataset_vals: &[f64],
    cutoff: f64,
    bin_count: usize,
) -> Vec<CalibPoint> {
    let n = chopchop_vals.len();
    if n == 0 || bin_count == 0 {
        info!("Empty input data or bin_count is zero. Returning empty calibration.");
        return vec![];
    }

    // Pair and sort by CHOPCHOP values.
    let mut pairs: Vec<(f64, f64)> = chopchop_vals
        .iter()
        .cloned()
        .zip(dataset_vals.iter().cloned())
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let min_c = pairs.first().unwrap().0;
    let max_c = pairs.last().unwrap().0;
    let range = max_c - min_c;
    if range <= 0.0 {
        info!("Invalid range: {}. Returning empty calibration.", range);
        return vec![];
    }
    let bin_size = range / bin_count as f64;
    let mut bins = vec![Vec::new(); bin_count];

    for &(cval, dval) in &pairs {
        let idx = (((cval - min_c) / bin_size).floor() as isize)
            .clamp(0, bin_count as isize - 1) as usize;
        bins[idx].push((cval, dval));
    }

    for (i, bin) in bins.iter().enumerate() {
        info!("Bin {}: {} entries", i, bin.len());
    }

    let mut result = Vec::with_capacity(bin_count);
    for i in 0..bin_count {
        let bin_start = min_c + i as f64 * bin_size;
        let bin_end = bin_start + bin_size;
        let bin_vec = &bins[i];

        if bin_vec.is_empty() {
            info!("Bin {} is empty. Skipping.", i);
            continue;
        }

        let count = bin_vec.len();
        let mean_chopchop = bin_vec.iter().map(|(cc, _)| *cc).sum::<f64>() / count as f64;
        let good_count = bin_vec.iter().filter(|(_cc, dval)| *dval >= cutoff).count();
        let fraction_good = good_count as f64 / count as f64;

        result.push(CalibPoint {
            bin_start,
            bin_end,
            mean_chopchop,
            fraction_good,
            count,
        });
    }

    info!("Calibration result: {:?}", result);
    result
}

fn create_calibration_plot(output_path: &str, points: &[CalibPoint]) -> Result<(), Box<dyn Error>> {
    if points.is_empty() {
        info!("No calibration data. Skipping plot: {}", output_path);
        return Ok(());
    }

    let min_x = points.iter().map(|cp| cp.mean_chopchop).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|cp| cp.mean_chopchop).fold(f64::NEG_INFINITY, f64::max);
    let (min_y, max_y) = (0.0, 1.0);

    if max_x <= min_x {
        info!("Degenerate calibration range. Skipping plot.");
        return Ok(());
    }

    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption("Calibration Curve: CHOPCHOP vs ≥X% dataset", ("sans-serif", 20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Mean CHOPCHOP in bin")
        .y_desc("Fraction ≥X% (actual)")
        .draw()?;

    let series_points: Vec<(f64, f64)> = points
        .iter()
        .map(|cp| (cp.mean_chopchop, cp.fraction_good))
        .collect();
    chart.draw_series(LineSeries::new(series_points.clone(), &BLUE))?;
    chart.draw_series(series_points.iter().map(|&(x, y)| Circle::new((x, y), 5, BLUE.filled())))?;

    // Draw diagonal y = x.
    let diag = (0..=100).map(|i| {
        let val = min_x + (max_x - min_x) * (i as f64 / 100.0);
        (val, val)
    });
    chart.draw_series(LineSeries::new(diag, &RED))?;

    info!("Calibration curve saved: {}", output_path);
    Ok(())
}

// --------------------------------------------------------
//  Basic Statistics and Correlations
// --------------------------------------------------------
fn basic_stats(vals: &[f64]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    (min, max, mean)
}

fn spearman_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let rx = rank_data(x);
    let ry = rank_data(y);
    pearson_correlation(&rx, &ry)
}

fn rank_data(vals: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = vals.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; vals.len()];
    let mut i = 0;
    while i < indexed.len() {
        let val = indexed[i].1;
        let mut j = i + 1;
        while j < indexed.len() && (indexed[j].1 - val).abs() < std::f64::EPSILON {
            j += 1;
        }
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.is_empty() || x.len() != y.len() {
        return None;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let (mut num, mut denom_x, mut denom_y) = (0.0, 0.0, 0.0);
    for (&xx, &yy) in x.iter().zip(y.iter()) {
        let dx = xx - mean_x;
        let dy = yy - mean_y;
        num += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }
    let denom = denom_x.sqrt() * denom_y.sqrt();
    if denom == 0.0 {
        return None;
    }
    Some(num / denom)
}

// --------------------------------------------------------
//  Fraction-based Evaluation
// --------------------------------------------------------
fn evaluate_pr_fraction(dataset_vals: &[f64], chopchop_vals: &[f64], fraction: f64) -> (Vec<f64>, Vec<f64>) {
    let n = dataset_vals.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let ranks_dataset = rank_data(dataset_vals);
    let rank_cut = (1.0 - fraction) * n as f64;
    let mut actual_positives = vec![false; n];
    for i in 0..n {
        if ranks_dataset[i] >= rank_cut {
            actual_positives[i] = true;
        }
    }
    let tot_pos = actual_positives.iter().filter(|&&b| b).count();
    let mut scored: Vec<(f64, bool)> = chopchop_vals.iter().cloned().zip(actual_positives.into_iter()).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut tp = 0;
    let mut recalls = vec![0.0];
    let mut precisions = vec![1.0];
    for (i, &(_score, is_pos)) in scored.iter().enumerate() {
        if is_pos {
            tp += 1;
        }
        let precision = tp as f64 / (i as f64 + 1.0);
        let recall = if tot_pos > 0 { tp as f64 / tot_pos as f64 } else { 0.0 };
        recalls.push(recall);
        precisions.push(precision);
    }
    (recalls, precisions)
}

fn evaluate_roc_fraction(dataset_vals: &[f64], chopchop_vals: &[f64], fraction: f64) -> (Vec<f64>, Vec<f64>) {
    let n = dataset_vals.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let ranks_dataset = rank_data(dataset_vals);
    let rank_cut = (1.0 - fraction) * n as f64;
    let mut actual_positives = vec![false; n];
    for i in 0..n {
        if ranks_dataset[i] >= rank_cut {
            actual_positives[i] = true;
        }
    }
    let tot_pos = actual_positives.iter().filter(|&&b| b).count();
    let tot_neg = n - tot_pos;
    let mut scored: Vec<(f64, bool)> = chopchop_vals.iter().cloned().zip(actual_positives.into_iter()).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut tp = 0;
    let mut fp = 0;
    let mut fprs = vec![0.0];
    let mut tprs = vec![0.0];
    for &(_score, is_pos) in scored.iter() {
        if is_pos {
            tp += 1;
        } else {
            fp += 1;
        }
        let tpr = if tot_pos > 0 { tp as f64 / tot_pos as f64 } else { 0.0 };
        let fpr = if tot_neg > 0 { fp as f64 / tot_neg as f64 } else { 0.0 };
        tprs.push(tpr);
        fprs.push(fpr);
    }
    (fprs, tprs)
}

// --------------------------------------------------------
//  Calibrated-based Evaluation
// --------------------------------------------------------
fn evaluate_pr_calibrated(dataset_vals: &[f64], chopchop_vals: &[f64], dataset_cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    let n = dataset_vals.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut actual_positives = vec![false; n];
    for i in 0..n {
        if dataset_vals[i] >= dataset_cutoff {
            actual_positives[i] = true;
        }
    }
    let tot_pos = actual_positives.iter().filter(|&&b| b).count();
    let mut scored: Vec<(f64, bool)> = chopchop_vals.iter().cloned().zip(actual_positives.into_iter()).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut tp = 0;
    let mut recalls = vec![0.0];
    let mut precisions = vec![1.0];
    for (i, &(_score, is_pos)) in scored.iter().enumerate() {
        if is_pos {
            tp += 1;
        }
        let precision = tp as f64 / (i as f64 + 1.0);
        let recall = if tot_pos > 0 { tp as f64 / tot_pos as f64 } else { 0.0 };
        recalls.push(recall);
        precisions.push(precision);
    }
    (recalls, precisions)
}

fn evaluate_roc_calibrated(dataset_vals: &[f64], chopchop_vals: &[f64], dataset_cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    let n = dataset_vals.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut actual_positives = vec![false; n];
    let mut total_positives = 0;
    for i in 0..n {
        if dataset_vals[i] >= dataset_cutoff {
            actual_positives[i] = true;
            total_positives += 1;
        }
    }
    if total_positives == 0 || total_positives == n {
        info!(
            "WARNING: All examples are of the same class! Positives: {}/{}",
            total_positives, n
        );
        return (vec![0.0, 1.0], vec![0.0, 1.0]);
    }
    let total_negatives = n - total_positives;
    let mut pairs: Vec<(f64, bool)> = chopchop_vals
        .iter()
        .cloned()
        .zip(actual_positives.into_iter())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut tp = 0;
    let mut fp = 0;
    let mut tprs = vec![0.0];
    let mut fprs = vec![0.0];
    for &(_score, is_pos) in pairs.iter() {
        if is_pos {
            tp += 1;
        } else {
            fp += 1;
        }
        tprs.push(tp as f64 / total_positives as f64);
        fprs.push(fp as f64 / total_negatives as f64);
    }
    (fprs, tprs)
}

// --------------------------------------------------------
//  Plotting Utilities
// --------------------------------------------------------
fn plot_precision_recall_curve(
    output_path: &str,
    recalls: &[f64],
    precisions: &[f64],
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    if recalls.is_empty() || recalls.len() != precisions.len() {
        info!("No data or mismatch for PR curve: {}", output_path);
        return Ok(());
    }
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption(caption, ("sans-serif", 20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;
    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Recall")
        .y_desc("Precision")
        .draw()?;

    let pts: Vec<(f64, f64)> = recalls.iter().cloned().zip(precisions.iter().cloned()).collect();
    chart.draw_series(LineSeries::new(pts.clone(), &BLUE))?;
    chart.draw_series(pts.iter().map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled())))?;

    info!("Precision-Recall curve saved: {}", output_path);
    Ok(())
}

fn plot_roc_curve(
    output_path: &str,
    fprs: &[f64],
    tprs: &[f64],
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    if fprs.is_empty() || fprs.len() != tprs.len() {
        info!("No data or mismatch for ROC curve: {}", output_path);
        return Ok(());
    }
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption(caption, ("sans-serif", 20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;
    chart.configure_mesh()
        .disable_mesh()
        .x_desc("False Positive Rate")
        .y_desc("True Positive Rate")
        .draw()?;

    let pts: Vec<(f64, f64)> = fprs.iter().cloned().zip(tprs.iter().cloned()).collect();
    chart.draw_series(LineSeries::new(pts.clone(), &BLUE))?;
    chart.draw_series(pts.iter().map(|&(x, y)| Circle::new((x, y), 3, BLUE.filled())))?;
    // Diagonal line.
    chart.draw_series(LineSeries::new(
        (0..=100).map(|i| {
            let x = i as f64 / 100.0;
            (x, x)
        }),
        &RED,
    ))?;

    info!("ROC curve saved: {}", output_path);
    Ok(())
}

fn create_diff_histogram(output_path: &str, values: &[f64]) -> Result<(), Box<dyn Error>> {
    if values.is_empty() {
        info!("No difference data; skipping histogram: {}", output_path);
        return Ok(());
    }
    let (min_v, max_v, _) = basic_stats(values);
    let range = max_v - min_v;
    if range <= 0.0 {
        info!("All difference values identical; skipping histogram.");
        return Ok(());
    }
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let bin_count = 30;
    let bin_size = range / bin_count as f64;
    let mut bins = vec![0; bin_count];
    for &val in values {
        let idx = (((val - min_v) / bin_size).floor() as isize).clamp(0, bin_count as isize - 1) as usize;
        bins[idx] += 1;
    }
    let max_bin = *bins.iter().max().unwrap_or(&0);
    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption("Distribution of Difference (CHOPCHOP - Dataset)", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_v..max_v, 0..max_bin)?;
    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Difference")
        .y_desc("Count")
        .draw()?;
    chart.draw_series(bins.iter().enumerate().map(|(i, &ct)| {
        let x0 = min_v + i as f64 * bin_size;
        let x1 = x0 + bin_size;
        Rectangle::new([(x0, 0), (x1, ct)], RED.mix(0.5).filled())
    }))?;

    info!("Histogram saved: {}", output_path);
    Ok(())
}

fn create_histogram(
    output_path: &str,
    values: &[f64],
    chart_title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<(), Box<dyn Error>> {
    if values.is_empty() {
        info!("No data; skipping histogram '{}'", chart_title);
        return Ok(());
    }
    let (min_v, max_v, _) = basic_stats(values);
    let range = max_v - min_v;
    if range <= 0.0 {
        info!("All values identical; skipping histogram '{}'", chart_title);
        return Ok(());
    }
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let bin_count = 30;
    let bin_size = range / bin_count as f64;
    let mut bins = vec![0; bin_count];
    for &val in values {
        let idx = (((val - min_v) / bin_size).floor() as isize).clamp(0, bin_count as isize - 1) as usize;
        bins[idx] += 1;
    }
    let max_bin = *bins.iter().max().unwrap_or(&0);
    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption(chart_title, ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_v..max_v, 0..max_bin)?;
    chart.configure_mesh()
        .disable_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;
    chart.draw_series(bins.iter().enumerate().map(|(i, &ct)| {
        let x0 = min_v + i as f64 * bin_size;
        let x1 = x0 + bin_size;
        Rectangle::new([(x0, 0), (x1, ct)], RED.mix(0.5).filled())
    }))?;

    info!("Histogram saved: {}", output_path);
    Ok(())
}

fn create_scatter(
    output_path: &str,
    xs: &[f64],
    ys: &[f64],
    x_label: &str,
    y_label: &str,
    chart_title: &str,
) -> Result<(), Box<dyn Error>> {
    if xs.is_empty() || ys.is_empty() {
        info!("No data for scatter plot '{}'", chart_title);
        return Ok(());
    }
    let (min_x, max_x, _) = basic_stats(xs);
    let (min_y, max_y, _) = basic_stats(ys);
    if max_x <= min_x || max_y <= min_y {
        info!("Degenerate data for scatter plot '{}'", chart_title);
        return Ok(());
    }
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption(chart_title, ("sans-serif", 20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;
    chart.configure_mesh()
        .disable_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;
    chart.draw_series(xs.iter().zip(ys.iter()).map(|(&x, &y)| Circle::new((x, y), 3, BLUE.filled())))?;

    // Optionally, draw the diagonal line.
    let diag_min = min_x.min(min_y);
    let diag_max = max_x.max(max_y);
    let diag_pts = (diag_min.floor() as i64..=diag_max.ceil() as i64)
        .map(|v| (v as f64, v as f64));
    chart.draw_series(LineSeries::new(diag_pts, &GREEN))?;

    info!("Scatter plot saved: {}", output_path);
    Ok(())
}

// --------------------------------------------------------
//  Mapping and Regression Utilities
// --------------------------------------------------------
fn analyze_value_mapping(dataset_vals: &[f64], chopchop_vals: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, f64)> = dataset_vals
        .iter()
        .zip(chopchop_vals.iter())
        .map(|(&d, &c)| (c, d))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs
}

fn compute_linear_regression(pairs: &[(f64, f64)]) -> (f64, f64) {
    if pairs.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = pairs.len() as f64;
    let sum_x: f64 = pairs.iter().map(|(x, _)| *x).sum();
    let sum_y: f64 = pairs.iter().map(|(_, y)| *y).sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let (mut num, mut denom) = (0.0, 0.0);
    for &(x, y) in pairs {
        let dx = x - mean_x;
        num += dx * (y - mean_y);
        denom += dx * dx;
    }
    if denom == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let slope = num / denom;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

fn create_mapping_scatter_plot(
    output_path: &str,
    pairs: &[(f64, f64)],
    slope: f64,
    intercept: f64,
) -> Result<(), Box<dyn Error>> {
    if pairs.is_empty() {
        info!("No data for mapping scatter plot.");
        return Ok(());
    }
    let min_x = pairs.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let max_x = pairs.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
    let min_y = pairs.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let max_y = pairs.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    if max_x <= min_x || max_y <= min_y {
        info!("Degenerate data for mapping scatter plot; skipping.");
        return Ok(());
    }
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption("CHOPCHOP vs Dataset (Mapping + Regression)", ("sans-serif", 20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;
    chart.configure_mesh()
        .disable_mesh()
        .x_desc("CHOPCHOP Score")
        .y_desc("Dataset Efficacy")
        .draw()?;
    chart.draw_series(pairs.iter().map(|(x, y)| Circle::new((*x, *y), 3, BLUE.filled())))?;
    let line_pts = vec![(min_x, slope * min_x + intercept), (max_x, slope * max_x + intercept)];
    chart.draw_series(LineSeries::new(line_pts, &RED))?;

    info!("Mapping + regression plot saved: {}", output_path);
    Ok(())
}

use plotters::prelude::*;

/// Draws a 2×2 confusion matrix with multi-line "Actual" labels on the left.
use plotters::prelude::*;

fn create_confusion_matrix(
    output_path: &str,
    cm: [[usize; 2]; 2],
    title: &str,
) -> Result<(), Box<dyn Error>> {
    let area_width: i32 = 1500;
    let area_height: i32 = 1500;

    // Increase these to push the matrix down (margin_top) and to the right (margin_left).
    let margin_top: i32 = 250;
    let margin_left: i32 = 250;

    // Keep margins on the right/bottom smaller to preserve the cell size.
    let margin_right: i32 = 50;
    let margin_bottom: i32 = 50;

    // Create the drawing area.
    let root_area = BitMapBackend::new(output_path, (area_width as u32, area_height as u32))
        .into_drawing_area();
    root_area.fill(&WHITE)?;

    // --- 1) Title ---
    let title_shift_x: i32 = 200;
    let title_lines: Vec<&str> = title.split('\n').collect();
    let mut current_title_y = 40;
    let title_font = ("sans-serif", 40).into_font().color(&BLACK);

    for line in &title_lines {
        root_area.draw(&Text::new(
            line.to_string(),
            ((area_width / 2) - title_shift_x, current_title_y),
            title_font.clone(),
        ))?;
        current_title_y += 50;
    }

    // --- 2) Define the Grid (Cells) ---
    // Now the grid is (area_width - margin_left - margin_right) wide
    // and (area_height - margin_top - margin_bottom) high.
    let grid_width: i32 = area_width - margin_left - margin_right;
    let grid_height: i32 = area_height - margin_top - margin_bottom;

    let cell_width: i32 = grid_width / 2;
    let cell_height: i32 = grid_height / 2;

    let cells = vec![
        // TN
        ((margin_left, margin_top),
         (margin_left + cell_width, margin_top + cell_height)),
        // FP
        ((margin_left + cell_width, margin_top),
         (margin_left + 2*cell_width, margin_top + cell_height)),
        // FN
        ((margin_left, margin_top + cell_height),
         (margin_left + cell_width, margin_top + 2*cell_height)),
        // TP
        ((margin_left + cell_width, margin_top + cell_height),
         (margin_left + 2*cell_width, margin_top + 2*cell_height)),
    ];

    // Draw cell borders
    for &((x0, y0), (x1, y1)) in &cells {
        root_area.draw(&Rectangle::new([(x0, y0), (x1, y1)], &BLACK))?;
    }

    // --- 3) TN/FP/FN/TP + Counts ---
    let counts = vec![cm[0][0], cm[0][1], cm[1][0], cm[1][1]];
    let labels = vec!["TN", "FP", "FN", "TP"];
    let label_font = ("sans-serif", 36).into_font().color(&BLACK);
    let count_font = ("sans-serif", 48).into_font().color(&BLACK);

    for ((&((x0, y0), (x1, y1)), &count), &lbl) in cells.iter().zip(counts.iter()).zip(labels.iter()) {
        let cx = (x0 + x1) / 2;
        let cy = (y0 + y1) / 2;

        // label (TN, FP, etc.)
        root_area.draw(&Text::new(lbl.to_string(), (cx, cy - 20), label_font.clone()))?;
        // numeric count
        root_area.draw(&Text::new(count.to_string(), (cx, cy + 20), count_font.clone()))?;
    }

    // --- 4) Axis Labels ---
    let draw_multiline = |lines: &[&str], x: i32, center_y: i32, style: TextStyle, spacing: i32| -> Result<(), Box<dyn Error>> {
        let total_lines = lines.len() as i32;
        let total_height = (total_lines - 1) * spacing;
        let start_y = center_y - (total_height / 2);

        for (i, &txt) in lines.iter().enumerate() {
            let line_y = start_y + i as i32 * spacing;
            root_area.draw(&Text::new(txt.to_string(), (x, line_y), style.clone()))?;
        }
        Ok(())
    };

    let axis_font = ("sans-serif", 34).into_font().color(&BLACK);

    // top labels
    let shift_x_pred = 40;
    let top_label_y = margin_top - 40;
    let half_cell_w = cell_width / 2;

    root_area.draw(&Text::new(
        "Predicted Negative",
        (margin_left + half_cell_w - shift_x_pred, top_label_y),
        axis_font.clone(),
    ))?;
    root_area.draw(&Text::new(
        "Predicted Positive",
        (margin_left + cell_width + half_cell_w - shift_x_pred, top_label_y),
        axis_font.clone(),
    ))?;

    // left multi-line labels
    let lines_neg = ["Actual", "Negative"];
    let lines_pos = ["Actual", "Positive"];
    let half_cell_h = cell_height / 2;

    let shift_x_actual = 200;
    // top row center => margin_top + half_cell_h
    draw_multiline(
        &lines_neg,
        margin_left - shift_x_actual,
        margin_top + half_cell_h,
        axis_font.clone(),
        35,
    )?;
    // bottom row center => margin_top + cell_height + half_cell_h
    draw_multiline(
        &lines_pos,
        margin_left - shift_x_actual,
        margin_top + cell_height + half_cell_h,
        axis_font.clone(),
        35,
    )?;

    Ok(())
}

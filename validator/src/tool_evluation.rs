use polars::prelude::{PolarsError, PolarsResult};
use plotters::prelude::*;
use std::error::Error;
use std::fs;
use std::io::Write;
use crate::helper_functions::read_csv;
// If you have a real read_csv somewhere else, re-import. For a demo, define a dummy:

// --------------------------------------------------------
//  Constants
// --------------------------------------------------------
const FRACTION_CUTOFF: f64 = 0.25;
const DATASET_CUTOFF: f64 = 0.99;
const CALIBRATION_BINS: usize = 10;
panic!("Something is fishy with the curves")
// --------------------------------------------------------
//  Entrypoint
// --------------------------------------------------------
pub fn analyze_chopchop_results(csv_file_path: &str) -> PolarsResult<()> {
    // 1) Create subdirs. Fix the mismatch by using a closure to map `std::io::Error`
    fs::create_dir_all("results").map_err(|e| polars_err(Box::new(e)))?;
    fs::create_dir_all("results/fraction").map_err(|e| polars_err(Box::new(e)))?;
    fs::create_dir_all("results/calibrated").map_err(|e| polars_err(Box::new(e)))?;
    fs::create_dir_all("results/mapping").map_err(|e| polars_err(Box::new(e)))?;

    // 2) read CSV
    let df = read_csv(csv_file_path)?;
    let diff_series = df.column("difference")?.f64()?;
    let dataset_series = df.column("dataset_efficacy")?.f64()?;
    let chopchop_series = df.column("chopchop_efficiency")?.f64()?;

    let differences: Vec<f64> = diff_series.into_no_null_iter().collect();
    let dataset_vals: Vec<f64> = dataset_series.into_no_null_iter().collect();
    let chopchop_vals: Vec<f64> = chopchop_series.into_no_null_iter().collect();

    if differences.is_empty() {
        eprintln!("No valid data. Exiting early.");
        return Ok(());
    }

    // 3) Basic stats
    let (min_diff, max_diff, mean_diff) = basic_stats(&differences);
    println!("--- difference stats ---");
    println!("Count: {}", differences.len());
    println!("Min:   {:.3}", min_diff);
    println!("Max:   {:.3}", max_diff);
    println!("Mean:  {:.3}", mean_diff);

    // 4) Spearman correlation
    if let Some(sp) = spearman_correlation(&dataset_vals, &chopchop_vals) {
        println!("Spearman Corr = {:.3}", sp);
    }

    // 5) Fraction-based
    fraction_plots(&differences, &dataset_vals, &chopchop_vals)?;

    // 6) Calibrated
    calibrated_plots(&dataset_vals, &chopchop_vals)?;

    // 7) Mapping + regression
    mapping_regression_plots(&dataset_vals, &chopchop_vals)?;

    // 8) Calibration table + plot
    produce_calibration_analysis(&dataset_vals, &chopchop_vals, DATASET_CUTOFF, CALIBRATION_BINS)?;

    println!("Done. See `results/` subdirs for outputs.");
    Ok(())
}

// --------------------------------------------------------
//  polars_err to transform Box<dyn Error> -> PolarsError
// --------------------------------------------------------
fn polars_err(err: Box<dyn Error>) -> PolarsError {
    PolarsError::ComputeError(err.to_string().into())
}

// --------------------------------------------------------
//  Fraction-based approach
// --------------------------------------------------------
fn fraction_plots(
    differences: &[f64],
    dataset_vals: &[f64],
    chopchop_vals: &[f64],
) -> PolarsResult<()> {
    let frac_dir = "results/fraction";

    // Overall
    {
        let (pr_rec, pr_prec) = evaluate_pr_fraction(dataset_vals, chopchop_vals, FRACTION_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_overall.png", frac_dir),
            &pr_rec,
            &pr_prec,
            &format!("PR (Fraction {:.0}%) - Overall", FRACTION_CUTOFF*100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr, roc_tpr) = evaluate_roc_fraction(dataset_vals, chopchop_vals, FRACTION_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_overall.png", frac_dir),
            &roc_fpr,
            &roc_tpr,
            &format!("ROC (Fraction {:.0}%) - Overall", FRACTION_CUTOFF*100.0),
        )
            .map_err(polars_err)?;

        create_diff_histogram(
            &format!("{}/difference_histogram.png", frac_dir),
            differences,
        )
            .map_err(polars_err)?;
    }

    // dataset alone
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
            &format!("PR - Dataset alone (Fraction {:.0}%)", FRACTION_CUTOFF*100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_ds, roc_tpr_ds) = evaluate_roc_fraction(dataset_vals, dataset_vals, FRACTION_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_dataset.png", frac_dir),
            &roc_fpr_ds,
            &roc_tpr_ds,
            &format!("ROC - Dataset alone (Fraction {:.0}%)", FRACTION_CUTOFF*100.0),
        )
            .map_err(polars_err)?;
    }

    // chopchop alone
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
            &format!("PR - CHOPCHOP alone (Fraction {:.0}%)", FRACTION_CUTOFF*100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_cc, roc_tpr_cc) =
            evaluate_roc_fraction(chopchop_vals, chopchop_vals, FRACTION_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_chopchop.png", frac_dir),
            &roc_fpr_cc,
            &roc_tpr_cc,
            &format!("ROC - CHOPCHOP alone (Fraction {:.0}%)", FRACTION_CUTOFF*100.0),
        )
            .map_err(polars_err)?;
    }

    Ok(())
}

// --------------------------------------------------------
//  Calibrated approach
// --------------------------------------------------------
fn calibrated_plots(dataset_vals: &[f64], chopchop_vals: &[f64]) -> PolarsResult<()> {
    let cal_dir = "results/calibrated";

    // overall
    {
        let (pr_rec, pr_prec) = evaluate_pr_calibrated(dataset_vals, chopchop_vals, DATASET_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_overall.png", cal_dir),
            &pr_rec,
            &pr_prec,
            &format!("PR (Calibrated≥{:.0}%) - Overall", DATASET_CUTOFF*100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr, roc_tpr) =
            evaluate_roc_calibrated(dataset_vals, chopchop_vals, DATASET_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_overall.png", cal_dir),
            &roc_fpr,
            &roc_tpr,
            &format!("ROC (Calibrated≥{:.0}%) - Overall", DATASET_CUTOFF*100.0),
        )
            .map_err(polars_err)?;
    }

    // dataset alone
    {
        let (pr_rec_ds, pr_prec_ds) =
            evaluate_pr_calibrated(dataset_vals, dataset_vals, DATASET_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_dataset.png", cal_dir),
            &pr_rec_ds,
            &pr_prec_ds,
            &format!("PR - Dataset alone≥{:.0}%", DATASET_CUTOFF*100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_ds, roc_tpr_ds) =
            evaluate_roc_calibrated(dataset_vals, dataset_vals, DATASET_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_dataset.png", cal_dir),
            &roc_fpr_ds,
            &roc_tpr_ds,
            &format!("ROC - Dataset alone≥{:.0}%", DATASET_CUTOFF*100.0),
        )
            .map_err(polars_err)?;
    }

    // chopchop alone
    {
        let (pr_rec_cc, pr_prec_cc) =
            evaluate_pr_calibrated(chopchop_vals, chopchop_vals, DATASET_CUTOFF);
        plot_precision_recall_curve(
            &format!("{}/precision_recall_chopchop.png", cal_dir),
            &pr_rec_cc,
            &pr_prec_cc,
            &format!("PR - CHOPCHOP alone≥{:.0}%??", DATASET_CUTOFF*100.0),
        )
            .map_err(polars_err)?;

        let (roc_fpr_cc, roc_tpr_cc) =
            evaluate_roc_calibrated(chopchop_vals, chopchop_vals, DATASET_CUTOFF);
        plot_roc_curve(
            &format!("{}/roc_curve_chopchop.png", cal_dir),
            &roc_fpr_cc,
            &roc_tpr_cc,
            &format!("ROC - CHOPCHOP alone≥{:.0}%??", DATASET_CUTOFF*100.0),
        )
            .map_err(polars_err)?;
    }

    Ok(())
}

// --------------------------------------------------------
//  Mapping + Regression
// --------------------------------------------------------
fn mapping_regression_plots(dataset_vals: &[f64], chopchop_vals: &[f64]) -> PolarsResult<()> {
    let map_dir = "results/mapping";
    let pairs = analyze_value_mapping(dataset_vals, chopchop_vals); // (chopchop, dataset)
    let (slope, intercept) = compute_linear_regression(&pairs);
    println!(
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
) -> PolarsResult<()> {
    let map_dir = "results/mapping";

    let calibration_points =
        compute_calibration_bins(chopchop_vals, dataset_vals, dataset_cutoff, bin_count);

    // write a table
    let table_path = format!("{}/calibration_table.csv", map_dir);
    {
        let mut file = std::fs::File::create(&table_path).map_err(|e| polars_err(Box::new(e)))?;
        writeln!(
            file,
            "bin_index,bin_start,bin_end,mean_chopchop,fraction_good,count"
        )
            .map_err(|e| polars_err(Box::new(e)))?;

        for (i, cp) in calibration_points.iter().enumerate() {
            writeln!(
                file,
                "{},{:.3},{:.3},{:.3},{:.3},{}",
                i,
                cp.bin_start,
                cp.bin_end,
                cp.mean_chopchop,
                cp.fraction_good,
                cp.count
            )
                .map_err(|e| polars_err(Box::new(e)))?;
        }
        println!("Calibration table saved to {}", table_path);
    }

    // plot
    let plot_path = format!("{}/calibration_curve.png", map_dir);
    create_calibration_plot(&plot_path, &calibration_points).map_err(polars_err)?;

    Ok(())
}

// For each bin of chopchop, fraction of dataset≥cutoff
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
        return vec![];
    }

    // sort by chopchop ascending
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
        return vec![];
    }
    let bin_size = range / bin_count as f64;
    let mut bins = vec![Vec::new(); bin_count];

    for &(cval, dval) in pairs.iter() {
        let idx = ((cval - min_c) / bin_size).floor() as isize;
        let idx = idx.clamp(0, bin_count as isize - 1) as usize;
        bins[idx].push((cval, dval));
    }

    let mut result = Vec::with_capacity(bin_count);
    for i in 0..bin_count {
        let bin_s = min_c + i as f64 * bin_size;
        let bin_e = bin_s + bin_size;
        let bin_vec = &bins[i];
        if bin_vec.is_empty() {
            result.push(CalibPoint {
                bin_start: bin_s,
                bin_end: bin_e,
                mean_chopchop: (bin_s + bin_e) * 0.5,
                fraction_good: 0.0,
                count: 0,
            });
            continue;
        }
        let count = bin_vec.len();
        let sum_chop: f64 = bin_vec.iter().map(|(cc, _)| *cc).sum();
        let mean_c = sum_chop / count as f64;
        let mut good_count = 0;
        for &(_cx, dy) in bin_vec {
            if dy >= cutoff {
                good_count += 1;
            }
        }
        let fraction_good = good_count as f64 / count as f64;

        result.push(CalibPoint {
            bin_start: bin_s,
            bin_end: bin_e,
            mean_chopchop: mean_c,
            fraction_good,
            count,
        });
    }

    result
}

fn create_calibration_plot(
    output_path: &str,
    points: &[CalibPoint],
) -> Result<(), Box<dyn Error>> {
    if points.is_empty() {
        println!("No calibration data. Skipping: {}", output_path);
        return Ok(());
    }

    let min_x = points.iter().fold(f64::INFINITY, |acc, cp| acc.min(cp.mean_chopchop));
    let max_x = points.iter().fold(f64::NEG_INFINITY, |acc, cp| acc.max(cp.mean_chopchop));
    let min_y = 0.0;
    let max_y = 1.0;

    if max_x <= min_x {
        println!("Degenerate calibration range. Skipping plot.");
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

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Mean CHOPCHOP in bin")
        .y_desc("Fraction≥X% (actual)")
        .draw()?;

    // points
    let series_points: Vec<(f64, f64)> = points
        .iter()
        .map(|cp| (cp.mean_chopchop, cp.fraction_good))
        .collect();
    chart.draw_series(LineSeries::new(series_points.clone(), &BLUE))?;
    chart.draw_series(series_points.iter().map(|&(xx, yy)| {
        Circle::new((xx, yy), 5, BLUE.filled())
    }))?;

    // diagonal y=x
    let diag_min = min_x.min(min_y);
    let diag_max = max_x.max(max_y);
    let diag = (0..=100).map(|i| {
        let val = diag_min + (diag_max - diag_min) * (i as f64 / 100.0);
        (val, val)
    });
    chart.draw_series(LineSeries::new(diag, &RED))?;

    println!("Calibration curve saved: {}", output_path);
    Ok(())
}

// --------------------------------------------------------
//  Basic stats + Spearman
// --------------------------------------------------------
fn basic_stats(vals: &[f64]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let min_v = vals.iter().fold(f64::INFINITY, |acc, &vv| acc.min(vv));
    let max_v = vals.iter().fold(f64::NEG_INFINITY, |acc, &vv| acc.max(vv));
    let sum_v: f64 = vals.iter().sum();
    let mean_v = sum_v / (vals.len() as f64);
    (min_v, max_v, mean_v)
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
    let mut enumerated: Vec<(usize, f64)> =
        vals.iter().cloned().enumerate().collect();
    enumerated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; vals.len()];
    let mut i = 0;
    while i < enumerated.len() {
        let val = enumerated[i].1;
        let mut j = i + 1;
        while j < enumerated.len() && enumerated[j].1 == val {
            j += 1;
        }
        let avg = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            let orig_idx = enumerated[k].0;
            ranks[orig_idx] = avg;
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

    let mut numerator = 0.0;
    let mut denom_x = 0.0;
    let mut denom_y = 0.0;

    for (&xx, &yy) in x.iter().zip(y.iter()) {
        let dx = xx - mean_x;
        let dy = yy - mean_y;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }
    let denominator = denom_x.sqrt() * denom_y.sqrt();
    if denominator == 0.0 {
        return None;
    }
    Some(numerator / denominator)
}

// --------------------------------------------------------
//  fraction-based evaluate
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

    let mut scored: Vec<(f64, bool)> = chopchop_vals
        .iter()
        .cloned()
        .zip(actual_positives.into_iter())
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp = 0;
    let mut recalls = Vec::with_capacity(n + 1);
    let mut precs  = Vec::with_capacity(n + 1);

    // start (0,1)
    recalls.push(0.0);
    precs.push(1.0);

    for (i, &(_score, is_pos)) in scored.iter().enumerate() {
        if is_pos { tp += 1; }
        let pred_pos = (i+1) as f64;
        let precision = tp as f64 / pred_pos;
        let recall = if tot_pos > 0 {
            tp as f64 / tot_pos as f64
        } else { 0.0 };
        recalls.push(recall);
        precs.push(precision);
    }

    (recalls, precs)
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

    let mut scored = chopchop_vals
        .iter()
        .cloned()
        .zip(actual_positives.into_iter())
        .collect::<Vec<_>>();
    scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp=0; let mut fp=0;
    let mut fprs = Vec::with_capacity(n+1);
    let mut tprs = Vec::with_capacity(n+1);

    fprs.push(0.0);
    tprs.push(0.0);

    for &(_score, is_pos) in scored.iter() {
        if is_pos {
            tp+=1;
        } else {
            fp+=1;
        }
        let tpr = if tot_pos>0 { tp as f64 / tot_pos as f64 } else {0.0};
        let fpr = if tot_neg>0 { fp as f64 / tot_neg as f64 } else {0.0};
        tprs.push(tpr);
        fprs.push(fpr);
    }
    (fprs, tprs)
}

// --------------------------------------------------------
//  calibrated-based evaluate
// --------------------------------------------------------
fn evaluate_pr_calibrated(dataset_vals: &[f64], chopchop_vals: &[f64], dataset_cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    let n = dataset_vals.len();
    if n==0 { return (vec![], vec![]); }

    let mut actual_positives = vec![false; n];
    for i in 0..n {
        if dataset_vals[i] >= dataset_cutoff {
            actual_positives[i]= true;
        }
    }
    let tot_pos = actual_positives.iter().filter(|&&b| b).count();

    let mut scored = chopchop_vals
        .iter()
        .cloned()
        .zip(actual_positives.into_iter())
        .collect::<Vec<_>>();
    scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp=0;
    let mut recs= Vec::with_capacity(n+1);
    let mut precs= Vec::with_capacity(n+1);

    recs.push(0.0);
    precs.push(1.0);

    for (i, &(_score,is_pos)) in scored.iter().enumerate() {
        if is_pos {
            tp+=1;
        }
        let pred_pos = (i+1) as f64;
        let precision = tp as f64 / pred_pos;
        let recall = if tot_pos>0 {
            tp as f64 / tot_pos as f64
        } else { 0.0 };
        recs.push(recall);
        precs.push(precision);
    }
    (recs, precs)
}

fn evaluate_roc_calibrated(dataset_vals: &[f64], chopchop_vals: &[f64], dataset_cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    let n= dataset_vals.len();
    if n==0 {
        return (vec![], vec![]);
    }

    let mut actual_positives = vec![false; n];
    for i in 0..n {
        if dataset_vals[i]>= dataset_cutoff {
            actual_positives[i]=true;
        }
    }
    let tot_pos = actual_positives.iter().filter(|&&b| b).count();
    let tot_neg = n- tot_pos;

    let mut scored = chopchop_vals
        .iter()
        .cloned()
        .zip(actual_positives.into_iter())
        .collect::<Vec<_>>();
    scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp=0; let mut fp=0;
    let mut fprs= Vec::with_capacity(n+1);
    let mut tprs= Vec::with_capacity(n+1);
    fprs.push(0.0);
    tprs.push(0.0);

    for &(_score, is_pos) in scored.iter() {
        if is_pos {
            tp+=1;
        } else {
            fp+=1;
        }
        let tpr= if tot_pos>0 { tp as f64 / tot_pos as f64 } else {0.0};
        let fpr= if tot_neg>0 { fp as f64 / tot_neg as f64 } else {0.0};
        tprs.push(tpr);
        fprs.push(fpr);
    }
    (fprs,tprs)
}

// --------------------------------------------------------
//  PLOTTING UTILS
// --------------------------------------------------------
fn plot_precision_recall_curve(
    output_path: &str,
    recalls: &[f64],
    precs: &[f64],
    caption: &str
) -> Result<(), Box<dyn Error>> {
    if recalls.is_empty() || recalls.len()!=precs.len() {
        println!("No data or mismatch for PR curve: {}", output_path);
        return Ok(());
    }
    let root_area= BitMapBackend::new(output_path,(800,600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart= ChartBuilder::on(&root_area)
        .margin(25)
        .caption(caption,("sans-serif",20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Recall")
        .y_desc("Precision")
        .draw()?;

    let pts: Vec<(f64,f64)> = recalls.iter().cloned().zip(precs.iter().cloned()).collect();
    chart.draw_series(LineSeries::new(pts.clone(), &BLUE))?;
    chart.draw_series(pts.iter().map(|&(rx,py)| Circle::new((rx,py),3,BLUE.filled())))?;

    println!("Precision-Recall curve saved: {}", output_path);
    Ok(())
}

fn plot_roc_curve(
    output_path: &str,
    fprs: &[f64],
    tprs: &[f64],
    caption: &str
)-> Result<(),Box<dyn Error>> {
    if fprs.is_empty()|| fprs.len()!= tprs.len() {
        println!("No data or mismatch for ROC curve: {}", output_path);
        return Ok(());
    }
    let root_area= BitMapBackend::new(output_path,(800,600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart= ChartBuilder::on(&root_area)
        .margin(25)
        .caption(caption,("sans-serif",20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0.0..1.0,0.0..1.0)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("False Positive Rate")
        .y_desc("True Positive Rate")
        .draw()?;

    let pts: Vec<(f64,f64)> = fprs.iter().cloned().zip(tprs.iter().cloned()).collect();
    chart.draw_series(LineSeries::new(pts.clone(), &BLUE))?;
    chart.draw_series(pts.iter().map(|&(fx,ty)| Circle::new((fx,ty),3,BLUE.filled())))?;

    // diag
    chart.draw_series(LineSeries::new(
        (0..=100).map(|i|{
            let x = i as f64/100.0;
            (x,x)
        }),
        &RED
    ))?;

    println!("ROC curve saved: {}", output_path);
    Ok(())
}

fn create_diff_histogram(
    output_path: &str,
    values: &[f64]
)-> Result<(),Box<dyn Error>> {
    if values.is_empty() {
        println!("No difference data => skipping histogram: {}", output_path);
        return Ok(());
    }

    let (min_v,max_v,_)= basic_stats(values);
    let range= max_v-min_v;
    if range<= 0.0 {
        println!("All difference vals identical => skip hist.");
        return Ok(());
    }

    let root_area= BitMapBackend::new(output_path,(800,600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let bin_count= 30;
    let bin_size= range/bin_count as f64;
    let mut bins= vec![0; bin_count];
    for &val in values {
        let idx = ((val-min_v)/bin_size).floor() as isize;
        let idx= idx.clamp(0, bin_count as isize-1) as usize;
        bins[idx]+=1;
    }
    let max_bin= *bins.iter().max().unwrap_or(&0);

    let mut chart= ChartBuilder::on(&root_area)
        .margin(25)
        .caption("Distribution of Difference (CHOPCHOP - Dataset)",("sans-serif",20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_v..max_v,0..max_bin)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Difference")
        .y_desc("Count")
        .draw()?;

    chart.draw_series(bins.iter().enumerate().map(|(i,&ct)| {
        let x0= min_v+ i as f64* bin_size;
        let x1= x0 + bin_size;
        Rectangle::new([(x0,0),(x1,ct)], RED.mix(0.5).filled())
    }))?;

    println!("Histogram saved: {}", output_path);
    Ok(())
}

fn create_histogram(
    output_path: &str,
    values: &[f64],
    chart_title: &str,
    x_label: &str,
    y_label: &str
)-> Result<(),Box<dyn Error>> {
    if values.is_empty() {
        println!("No data => skip histogram '{}'", chart_title);
        return Ok(());
    }
    let (min_v,max_v,_)= basic_stats(values);
    let range= max_v-min_v;
    if range<=0.0 {
        println!("All vals identical => skip hist '{}'", chart_title);
        return Ok(());
    }
    let root_area= BitMapBackend::new(output_path,(800,600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let bin_count= 30;
    let bin_size= range/bin_count as f64;
    let mut bins= vec![0; bin_count];
    for &val in values {
        let idx = ((val-min_v)/bin_size).floor() as isize;
        let idx= idx.clamp(0, bin_count as isize-1) as usize;
        bins[idx]+=1;
    }
    let max_bin= *bins.iter().max().unwrap_or(&0);

    let mut chart= ChartBuilder::on(&root_area)
        .margin(25)
        .caption(chart_title,("sans-serif",20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_v..max_v,0..max_bin)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    chart.draw_series(bins.iter().enumerate().map(|(i,&ct)| {
        let x0= min_v+ i as f64* bin_size;
        let x1= x0 + bin_size;
        Rectangle::new([(x0,0),(x1,ct)], RED.mix(0.5).filled())
    }))?;

    println!("Histogram saved: {}", output_path);
    Ok(())
}

fn create_scatter(
    output_path: &str,
    xs: &[f64],
    ys: &[f64],
    x_label: &str,
    y_label: &str,
    chart_title: &str
)-> Result<(),Box<dyn Error>> {
    if xs.is_empty() || ys.is_empty() {
        println!("No scatter data => skip '{}'", chart_title);
        return Ok(());
    }
    let (min_x,max_x,_)= basic_stats(xs);
    let (min_y,max_y,_)= basic_stats(ys);
    if max_x<=min_x || max_y<=min_y {
        println!("Degenerate scatter => skip '{}'", chart_title);
        return Ok(());
    }
    let root_area= BitMapBackend::new(output_path,(800,600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart= ChartBuilder::on(&root_area)
        .margin(25)
        .caption(chart_title,("sans-serif",20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    chart.draw_series(
        xs.iter().zip(ys.iter()).map(|(&xx,&yy)| Circle::new((xx,yy),3,BLUE.filled()))
    )?;

    // optional diag
    let diag_min= min_x.min(min_y);
    let diag_max= max_x.max(max_y);
    let diag_pts = (diag_min.floor() as i64..=diag_max.ceil() as i64).map(|v| (v as f64, v as f64));
    chart.draw_series(LineSeries::new(diag_pts, &GREEN))?;

    println!("Scatter plot saved: {}", output_path);
    Ok(())
}

// --------------------------------------------------------
//  Mapping: (chopchop,dataset) => slope
// --------------------------------------------------------
fn analyze_value_mapping(dataset_vals: &[f64], chopchop_vals: &[f64]) -> Vec<(f64,f64)> {
    let mut pairs= dataset_vals
        .iter()
        .zip(chopchop_vals.iter())
        .map(|(&d,&c)| (c,d))
        .collect::<Vec<_>>();
    pairs.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
    pairs
}

fn compute_linear_regression(
    pairs: &[(f64,f64)]
)-> (f64,f64) {
    if pairs.is_empty(){
        return (f64::NAN,f64::NAN);
    }
    let n= pairs.len() as f64;
    let sum_x: f64= pairs.iter().map(|(cx,_dy)| *cx).sum();
    let sum_y: f64= pairs.iter().map(|(_cx,dy)| *dy).sum();
    let mean_x= sum_x/n;
    let mean_y= sum_y/n;

    let mut numerator=0.0;
    let mut denominator=0.0;
    for &(cx,dy) in pairs {
        let dx= cx- mean_x;
        let dyy= dy- mean_y;
        numerator+= dx*dyy;
        denominator+= dx*dx;
    }
    if denominator==0.0 {
        return (f64::NAN,f64::NAN);
    }
    let slope= numerator/denominator;
    let intercept= mean_y - slope* mean_x;
    (slope, intercept)
}

fn create_mapping_scatter_plot(
    output_path: &str,
    pairs: &[(f64,f64)],
    slope: f64,
    intercept: f64
)-> Result<(), Box<dyn Error>> {
    if pairs.is_empty() {
        println!("No data => skip mapping scatter.");
        return Ok(());
    }
    let min_x= pairs.iter().fold(f64::INFINITY,|acc,&(cx,_dy)| acc.min(cx));
    let max_x= pairs.iter().fold(f64::NEG_INFINITY,|acc,&(cx,_dy)| acc.max(cx));
    let min_y= pairs.iter().fold(f64::INFINITY,|acc,&(_cx,dy)| acc.min(dy));
    let max_y= pairs.iter().fold(f64::NEG_INFINITY,|acc,&(_cx,dy)| acc.max(dy));

    if max_x<=min_x || max_y<=min_y {
        println!("Degenerate mapping scatter => skip");
        return Ok(());
    }
    let root_area= BitMapBackend::new(output_path,(800,600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart= ChartBuilder::on(&root_area)
        .margin(25)
        .caption("CHOPCHOP vs Dataset (Mapping+Regression)",("sans-serif",20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("CHOPCHOP Score")
        .y_desc("Dataset Efficacy")
        .draw()?;

    // scatter
    chart.draw_series(
        pairs.iter().map(|&(cx,dy)| Circle::new((cx,dy),3,BLUE.filled()))
    )?;

    // regression line from min_x..max_x
    let line_pts= vec![
        (min_x, slope*min_x+ intercept),
        (max_x, slope*max_x+ intercept),
    ];
    chart.draw_series(LineSeries::new(line_pts,&RED))?;

    println!("Mapping+regression plot saved: {}", output_path);
    Ok(())
}

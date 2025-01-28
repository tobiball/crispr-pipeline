use polars::prelude::*;
use plotters::prelude::*;
use std::fs::File;
use std::path::Path;
use crate::helper_functions::read_csv;
// If you have any other local modules, import them here
// e.g., `use crate::helper_functions::read_csv;` or something else

pub fn analyze_chopchop_results(csv_file_path: &str) -> PolarsResult<()> {
    let df = read_csv(csv_file_path)?;

    let diff_series = df.column("difference")?.f64()?;
    let dataset_series = df.column("dataset_efficacy")?.f64()?;
    let chopchop_series = df.column("chopchop_efficiency")?.f64()?;

    let differences: Vec<f64> = diff_series.into_no_null_iter().collect();
    let dataset_vals: Vec<f64> = dataset_series.into_no_null_iter().collect();
    let chopchop_vals: Vec<f64> = chopchop_series.into_no_null_iter().collect();

    let count = differences.len();
    if count == 0 {
        eprintln!("No rows found in CSV. Exiting analysis early.");
        return Ok(());
    }

    let min_diff = differences
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_diff = differences
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_diff: f64 = differences.iter().sum();
    let mean_diff = sum_diff / (count as f64);

    println!("--- Summary of `difference` ---");
    println!("Count : {}", count);
    println!("Min   : {:.3}", min_diff);
    println!("Max   : {:.3}", max_diff);
    println!("Mean  : {:.3}", mean_diff);

    // 3) Compute correlation if you like
    if let Some(corr) = pearson_correlation(&dataset_vals, &chopchop_vals) {
        println!(
            "Correlation (dataset vs. chopchop) = {:.3}",
            corr
        );
    } else {
        println!("No valid correlation (could be no data).");
    }

    // 4) Generate a histogram of differences using Plotters
    create_histogram("difference_histogram.png", &differences)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    // 5) Generate a scatter plot for dataset vs chopchop
    create_scatter(
        "efficacy_scatter.png",
        &dataset_vals,
        &chopchop_vals,
        "Dataset Efficacy",
        "CHOPCHOP Efficiency",
    )
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;

    Ok(())
}

// A minimal Pearson's correlation
fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denom_x = 0.0;
    let mut denom_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
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

/// Create a histogram of the given values and save it to an image file.
fn create_histogram(
    output_path: &str,
    values: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let bin_count = 30usize;
    let range = max_val - min_val;
    let bin_size = range / bin_count as f64;
    // Edge-case if range == 0.0 => all values are the same => no meaningful bins
    if bin_size == 0.0 {
        return Ok(());
    }

    // Count how many values fall in each bin
    let mut bins = vec![0; bin_count];
    for &val in values {
        let idx = ((val - min_val) / bin_size).floor() as i64;
        let idx = idx.clamp(0, bin_count as i64 - 1) as usize;
        bins[idx] += 1;
    }
    let max_bin_count = *bins.iter().max().unwrap_or(&0);

    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption("Distribution of Difference (CHOPCHOP - Dataset)", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_val..max_val, 0..max_bin_count)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("Difference")
        .y_desc("Count")
        .draw()?;

    chart.draw_series(
        bins.iter().enumerate().map(|(i, &count)| {
            let x0 = min_val + i as f64 * bin_size;
            let x1 = x0 + bin_size;
            Rectangle::new(
                [(x0, 0), (x1, count)],
                RED.mix(0.5).filled(),
            )
        }),
    )?;

    println!("Histogram saved to {}", output_path);
    Ok(())
}

/// Create a scatter plot from x-values and y-values and save to a file.
fn create_scatter(
    output_path: &str,
    xs: &[f64],
    ys: &[f64],
    x_label: &str,
    y_label: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let min_x = xs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_x = xs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let min_y = ys.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_y = ys.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption("Dataset vs CHOPCHOP Efficiency", ("sans-serif", 20))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    // Draw data points
    chart.draw_series(
        xs.iter().zip(ys.iter()).map(|(&x_val, &y_val)| {
            Circle::new((x_val, y_val), 3, BLUE.filled())
        })
    )?;

    // Optionally draw identity line (0..100) or min to max
    let diag_min = min_x.min(min_y);
    let diag_max = max_x.max(max_y);
    chart.draw_series(LineSeries::new(
        (diag_min.floor() as i64..=diag_max.ceil() as i64).map(|v| (v as f64, v as f64)),
        &GREEN,
    ))?;

    println!("Scatter plot saved to {}", output_path);
    Ok(())
}
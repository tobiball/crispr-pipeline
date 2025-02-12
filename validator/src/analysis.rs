use polars::prelude::*;
use plotters::prelude::*;
use std::error::Error;

/// Analyzes the distribution of efficacy values from the given DataFrame.
/// The DataFrame is expected to have an "efficacy" column containing floating point numbers.
///
/// This function prints summary statistics and generates one plot: a continuous histogram
/// (using 30 equal‑width bins) with vertical dashed lines (simulated by short line segments)
/// marking each boundary that divides the distribution into 10 equal parts (i.e. 9 boundaries).
pub fn analyze_efficacy_distribution_df(df: DataFrame) -> PolarsResult<()> {
    // Extract the "efficacy" column as f64 values.
    let efficacy_series = df.column("efficacy")?.f64()?;
    let efficacies: Vec<f64> = efficacy_series.into_no_null_iter().collect();

    let count = efficacies.len();
    if count == 0 {
        eprintln!("No efficacy data found in DataFrame. Exiting analysis early.");
        return Ok(());
    }

    // Compute summary statistics.
    let min_val = efficacies.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = efficacies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_val: f64 = efficacies.iter().sum();
    let mean_val = sum_val / (count as f64);

    println!("--- Summary of `efficacy` ---");
    println!("Count : {}", count);
    println!("Min   : {:.3}", min_val);
    println!("Max   : {:.3}", max_val);
    println!("Mean  : {:.3}", mean_val);
    println!();

    // Create a continuous histogram with 10-part boundaries (i.e. 9 vertical dashed lines).
    create_histogram_with_10_part_lines("efficacy_histogram_with_10_parts.png", &efficacies)
        .map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    println!("Histogram with 10-part lines saved to efficacy_histogram_with_10_parts.png");

    Ok(())
}

/// Creates a continuous histogram (using 30 equal‑width bins) for the given efficacy values
/// and overlays vertical dashed lines (simulated by drawing short segments) at each quantile
/// boundary that divides the data into 10 equal parts (i.e. at the 1/10, 2/10, …, 9/10 quantiles).
/// No labels are drawn on these boundaries.
/// The resulting chart is saved to the specified output path.
fn create_histogram_with_10_part_lines(
    output_path: &str,
    values: &[f64],
) -> Result<(), Box<dyn Error>> {
    // Set up the drawing area.
    let root_area = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    // Determine the range and histogram parameters.
    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let bin_count = 30;
    let range = max_val - min_val;
    let bin_size = range / bin_count as f64;
    if bin_size == 0.0 {
        eprintln!("All efficacy values are identical. No histogram will be created.");
        return Ok(());
    }

    // Count values in each bin.
    let mut bins = vec![0; bin_count];
    for &val in values {
        let mut idx = ((val - min_val) / bin_size).floor() as usize;
        if idx >= bin_count {
            idx = bin_count - 1;
        }
        bins[idx] += 1;
    }
    let max_bin_count = *bins.iter().max().unwrap_or(&0);

    // Build the chart.
    let mut chart = ChartBuilder::on(&root_area)
        .margin(25)
        .caption("Efficacy Distribution with 10-Part Lines", ("sans-serif", 20))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_val..max_val, 0f64..max_bin_count as f64)?;

    chart.configure_mesh()
        .disable_mesh()
        .x_desc("Efficacy")
        .y_desc("Count")
        .draw()?;

    // Draw the histogram bars.
    chart.draw_series(
        bins.iter().enumerate().map(|(i, &count)| {
            let x0 = min_val + i as f64 * bin_size;
            let x1 = x0 + bin_size;
            Rectangle::new([(x0, 0.0), (x1, count as f64)], RED.mix(0.5).filled())
        })
    )?;

    // Compute quantile boundaries that divide the data into 10 equal parts.
    // For 10 equal parts, we want quantile probabilities for i in 1..10 (i.e. 0.1, 0.2, …, 0.9).
    let mut sorted_vals = values.to_vec();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted_vals.len();
    let quantile_probs: Vec<f64> = (1..10).map(|i| i as f64 / 10.0).collect();
    let boundaries: Vec<f64> = quantile_probs.iter().map(|&p| {
        let pos = (n as f64 - 1.0) * p;
        let idx = pos.floor() as usize;
        let frac = pos - idx as f64;
        if idx + 1 < n {
            sorted_vals[idx] * (1.0 - frac) + sorted_vals[idx + 1] * frac
        } else {
            sorted_vals[idx]
        }
    }).collect();

    // Simulate dashed vertical lines for each boundary.
    let dash_height = 5.0; // Height of each dash segment.
    let gap = 5.0;         // Gap between dash segments.
    for &b in &boundaries {
        let mut y = 0.0;
        while y < max_bin_count as f64 {
            let y_end = (y + dash_height).min(max_bin_count as f64);
            chart.draw_series(std::iter::once(
                PathElement::new(vec![(b, y), (b, y_end)], BLACK.stroke_width(2))
            ))?;
            y += dash_height + gap;
        }
    }

    println!("Histogram with 10-part lines saved to {}", output_path);
    Ok(())
}

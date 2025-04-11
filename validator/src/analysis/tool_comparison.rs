use polars::prelude::*;
use plotters::prelude::*;
use std::error::Error;
use std::fs::create_dir_all;
use tracing::{info, error};
use crate::models::polars_err;

/// Simple function to generate ROC curves for multiple prediction tools
pub fn compare_roc_curves(
    df: &DataFrame,
    tool_columns: &[&str],
    efficacy_column: &str,
    cutoff: f64,
    output_path: &str
) -> PolarsResult<()> {
    // Create output directory
    let output_dir = output_path;
    create_dir_all(&output_dir).map_err(|e| polars_err(Box::new(e)))?;

    // Extract the efficacy data (ground truth)
    let efficacy_vals = match df.column(efficacy_column) {
        Ok(series) => {
            match series.f64() {
                Ok(f64_series) => f64_series.into_no_null_iter().collect::<Vec<f64>>(),
                Err(_) => {
                    info!("Trying to cast efficacy column '{}' to f64", efficacy_column);
                    match series.cast(&DataType::Float64) {
                        Ok(casted) => casted.f64()
                            .map_err(|e| {
                                error!("Failed to interpret casted efficacy as f64: {}", e);
                                PolarsError::ComputeError("Efficacy cast failed".into())
                            })?
                            .into_no_null_iter()
                            .collect(),
                        Err(e) => {
                            error!("Casting efficacy column '{}' to f64 failed: {}", efficacy_column, e);
                            return Err(PolarsError::ComputeError("Casting efficacy column failed".into()));
                        }
                    }
                }
            }
        },
        Err(e) => {
            error!("Error extracting efficacy column: {}", e);
            return Err(e);
        }
    };


    if efficacy_vals.is_empty() {
        error!("No valid efficacy data found");
        return Err(PolarsError::ComputeError("No valid efficacy data".into()));
    }

    // Process each tool column
    let mut roc_data = Vec::new();

    for &tool in tool_columns {
        info!("Processing tool: {}", tool);

        // Extract tool prediction values
        let tool_vals = match df.column(tool) {
            Ok(series) => match series.f64() {
                Ok(f64_series) => f64_series.into_no_null_iter().collect::<Vec<f64>>(),
                Err(e) => {
                    error!("Error converting {} to f64: {}", tool, e);
                    continue;
                }
            },
            Err(e) => {
                error!("Error extracting column {}: {}", tool, e);
                continue;
            }
        };

        if tool_vals.len() != efficacy_vals.len() {
            error!("Mismatch in data lengths for {} - expected {}, got {}",
                   tool, efficacy_vals.len(), tool_vals.len());
            continue;
        }

        // Calculate ROC curve
        let (fprs, tprs) = calculate_roc(&efficacy_vals, &tool_vals, cutoff);
        let auc = calculate_auc(&fprs, &tprs);

        roc_data.push((tool, fprs, tprs, auc));
        info!("Calculated ROC for {} - AUC: {:.3}", tool, auc);
    }

    if roc_data.is_empty() {
        error!("No valid tool data to plot");
        return Err(PolarsError::ComputeError("No valid tool data".into()));
    }

    // Create the ROC plot
    let roc_path = format!("{}/roc_comparison_cutoff_{}.png", output_dir,  cutoff);
    draw_roc_plot(&roc_path, &roc_data, cutoff)?;

    info!("ROC comparison saved to: {}", roc_path);

    // Save AUC values to a CSV
    let auc_path = format!("{}/auc_values.csv", output_dir);
    save_auc_values(&auc_path, &roc_data)?;

    Ok(())
}

/// Calculate ROC curve points (FPR and TPR) at different thresholds
fn calculate_roc(
    actual: &[f64],
    predicted: &[f64],
    cutoff: f64
) -> (Vec<f64>, Vec<f64>) {
    let n = actual.len();

    // Create binary classification based on cutoff
    let actual_positives: Vec<bool> = actual.iter().map(|&x| x >= cutoff).collect();
    let positive_count = actual_positives.iter().filter(|&&x| x).count();
    let negative_count = n - positive_count;

    if positive_count == 0 || negative_count == 0 {
        info!("Warning: All instances are in one class. ROC curve will be degenerate.");
        return (vec![0.0, 1.0], vec![0.0, 1.0]);
    }

    // Pair predictions with actual labels and sort by prediction (descending)
    let mut paired_data: Vec<(f64, bool)> = predicted.iter()
        .cloned()
        .zip(actual_positives.into_iter())
        .collect();

    paired_data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate points on the ROC curve
    let mut tpr_values = vec![0.0];
    let mut fpr_values = vec![0.0];

    let mut tp = 0;
    let mut fp = 0;

    for &(_, is_positive) in &paired_data {
        if is_positive {
            tp += 1;
        } else {
            fp += 1;
        }

        let tpr = tp as f64 / positive_count as f64;
        let fpr = fp as f64 / negative_count as f64;

        tpr_values.push(tpr);
        fpr_values.push(fpr);
    }

    (fpr_values, tpr_values)
}

/// Calculate Area Under ROC Curve using trapezoidal rule
fn calculate_auc(fpr: &[f64], tpr: &[f64]) -> f64 {
    if fpr.len() != tpr.len() || fpr.len() < 2 {
        return 0.0;
    }

    let mut auc = 0.0;
    for i in 1..fpr.len() {
        let width = fpr[i] - fpr[i-1];
        let height = (tpr[i] + tpr[i-1]) / 2.0;
        auc += width * height;
    }

    auc
}

/// Draw the ROC plot with all tools
fn draw_roc_plot(
    output_path: &str,
    roc_data: &[(&str, Vec<f64>, Vec<f64>, f64)],
    cutoff: f64,
) -> PolarsResult<()> {
    // Create drawing area
    let root = BitMapBackend::new(output_path, (800, 600))
        .into_drawing_area();

    root.fill(&WHITE).map_err(|e| polars_err(Box::new(e)))?;

    // Create the chart
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("ROC Curve Comparison (guide efficacy cutoff - {})",cutoff), ("sans-serif", 22))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)
        .map_err(|e| polars_err(Box::new(e)))?;

    chart.configure_mesh()
        .x_desc("False Positive Rate")
        .y_desc("True Positive Rate")
        .draw()
        .map_err(|e| polars_err(Box::new(e)))?;

    // Draw diagonal line (random classifier)
    chart.draw_series(LineSeries::new(
        vec![(0.0, 0.0), (1.0, 1.0)],
        &BLACK.mix(0.2)
    ))
        .map_err(|e| polars_err(Box::new(e)))?
        .label("Random (AUC=0.5)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.2)));

    // Define colors
    let colors = [
        &RGBColor(0, 119, 182),    // Blue
        &RGBColor(217, 72, 1),     // Orange
        &RGBColor(0, 153, 136),    // Teal
        &RGBColor(153, 0, 153),    // Purple
        &RGBColor(230, 159, 0),    // Yellow
        &RGBColor(86, 180, 233),   // Sky Blue
        &RGBColor(213, 94, 0),     // Vermillion
        &RGBColor(0, 158, 115),    // Bluish Green
        &RGBColor(204, 121, 167),  // Reddish Purple
    ];

    // Draw each tool's ROC curve
    for (i, (tool, fprs, tprs, auc)) in roc_data.iter().enumerate() {
        let color = colors[i % colors.len()];

        // Create points for the curve
        let points: Vec<(f64, f64)> = fprs.iter()
            .zip(tprs.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        // Draw the line
        chart.draw_series(LineSeries::new(points, color.clone()))
            .map_err(|e| polars_err(Box::new(e)))?
            .label(format!("{} (AUC={:.3})", tool, auc))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
    }

    // Add legend
    chart.configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()
        .map_err(|e| polars_err(Box::new(e)))?;

    Ok(())
}

/// Save AUC values to a CSV file
fn save_auc_values(path: &str, roc_data: &[(&str, Vec<f64>, Vec<f64>, f64)]) -> PolarsResult<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)
        .map_err(|e| polars_err(Box::new(e)))?;

    // Write header
    writeln!(file, "Tool,AUC")
        .map_err(|e| polars_err(Box::new(e)))?;

    // Write data
    for (tool, _, _, auc) in roc_data {
        writeln!(file, "{},{:.5}", tool, auc)
            .map_err(|e| polars_err(Box::new(e)))?;
    }

    info!("AUC values saved to: {}", path);
    Ok(())
}
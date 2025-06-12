use polars::prelude::*;
use std::fs;
use std::path::Path;
use tracing::{info, error};
use sankey::{Sankey, SankeyStyle};
use std::process::Command;

use crate::models::polars_err;

/// Generates Sankey diagrams comparing prediction tools against actual efficacy
/// with three fixed efficacy bins (0–59, 50–90, 90–100) on the right side.
pub fn create_prediction_sankey(
    df: &DataFrame,
    efficacy_column: &str,
    tool_columns: &[&str],
    output_dir: &str,
) -> PolarsResult<()> {
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir).map_err(|e| polars_err(Box::new(e)))?;

    // Extract the efficacy column
    let efficacy = df.column(efficacy_column)?.f64()?;

    // Decide how many bins you want for the *prediction* side
    let bins_count = 6;

    // For each prediction tool, do the following:
    for &tool_name in tool_columns {
        info!("Creating Sankey diagram for prediction tool '{}'", tool_name);

        // Extract this tool’s prediction column
        let predictions = match df.column(tool_name) {
            Ok(series) => series.f64()?,
            Err(e) => {
                error!("Failed to extract prediction column '{}': {}", tool_name, e);
                continue; // Skip this tool, but don't fail everything
            }
        };

        // Generate bins for the predictions of this specific tool
        let Ok((pred_bin_labels, pred_bin_edges)) = generate_bins(&predictions, bins_count) else { todo!() };

        // Initialize Sankey
        let mut diagram = Sankey::new();

        //---------------------------------------------
        // LEFT SIDE: PREDICTION BINS (auto-generated)
        // (Created in reverse so the top bin is the highest range)
        //---------------------------------------------
        let mut pred_nodes = Vec::new();
        let mut pred_node_colors = Vec::new();

        // A color palette for left-side bins
        let pred_colors = vec![
            "#1abc9c",  // turquoise
            "#3498db",  // blue
            "#9b59b6",  // purple
            "#e67e22",  // orange
            "#f1c40f",  // yellow
        ];

        for (i, lbl) in pred_bin_labels.iter().rev().enumerate() {
            let color = pred_colors[i % pred_colors.len()].to_string();
            let node = diagram.node(
                None,
                Some(format!("Pred {}", lbl)),
                Some(color.clone()),
            );
            pred_nodes.push(node);
            pred_node_colors.push(color);
        }

        //---------------------------------------------
        // RIGHT SIDE: EFFICACY BINS (fixed)
        // Hard-code three bins in descending order
        // ---------------------------------------------
        // RIGHT SIDE: EFFICACY BINS (HIGHEST TO LOWEST)
        // ---------------------------------------------
        let efficacy_bin_labels = ["90–100", "60–90", "0–59"];
        let efficacy_bin_colors = ["#e74c3c", "#e67e22", "#f1c40f"];

        // Create nodes in that same top-to-bottom order:
        let mut act_nodes = Vec::new();
        for (i, lbl) in efficacy_bin_labels.iter().enumerate() {
            let color = efficacy_bin_colors[i].to_string();
            let node = diagram.node(
                None,
                Some(format!("Act {}", lbl)),
                Some(color),
            );
            act_nodes.push(node);
        }

        // Make sure our indexing matches this order:
        //   0 => "90–100"
        //   1 => "50–90"
        //   2 => "0–59"
        fn efficacy_bin_index(value: f64) -> usize {
            if value >= 90.0 {
                0 // goes to node for "90–100"
            } else if value >= 60.0 {
                1 // goes to node for "50–90"
            } else {
                2 // goes to node for "0–50"
            }
        }



        // A small helper to return the *index* of the efficacy bin
        // 0 => 90–100
        // 1 => 50–90


        //---------------------------------------------
        // Count flows from each pred bin into each of
        // the three efficacy bins
        //---------------------------------------------
        // We have 'bins_count' pred bins (rows) × 3 efficacy bins (columns)
        let mut connections = vec![vec![0_u64; 3]; bins_count];

        for (pred_val, eff_val) in predictions.iter().zip(efficacy.iter()) {
            if let (Some(p), Some(a)) = (pred_val, eff_val) {
                // 1) Find which bin the *prediction* belongs to
                let pred_idx_normal = bin_index(p, &pred_bin_edges);
                let reversed_pred_idx = (bins_count - 1) - pred_idx_normal;

                // 2) Find which bin the *actual efficacy* belongs to
                let eff_idx = efficacy_bin_index(a);

                // Accumulate
                connections[reversed_pred_idx][eff_idx] += 1;
            }
        }

        //---------------------------------------------
        // Add edges
        //---------------------------------------------
        for pred_idx in 0..bins_count {
            for eff_idx in 0..3 {
                let count = connections[pred_idx][eff_idx];
                if count > 0 {
                    let edge_label = None;  // or Some(count.to_string()) if you prefer
                    let edge_color = &pred_node_colors[pred_idx];
                    diagram.edge(
                        pred_nodes[pred_idx],
                        act_nodes[eff_idx],
                        count as f64,
                        edge_label,
                        Some(edge_color.to_string()),
                    );
                }
            }
        }

        //---------------------------------------------
        // Style & render
        //---------------------------------------------
        let style = SankeyStyle {
            number_format: Some(|x| format!("{:.0}", x)), // integer flows
            font_size: Some(12.0),
            node_width: Some(20.0),
            font_family: Some("Arial, sans-serif".to_string()),
            font_color: Some("#333333".to_string()),
            node_separation: Some(80.0), // increase from 50.0
            border: Some(150.0),        // increase from 100.0
        };

        let svg = diagram.draw(1600.0, 1000.0, style);

        // Save to file
        let safe_filename = tool_name
            .replace(" ", "_")
            .replace("'", "")
            .replace(".", "")
            .replace("-", "_");
        let svg_path = Path::new(output_dir).join(format!("sankey_{}.svg", safe_filename));

        fs::write(&svg_path, svg.to_string()).map_err(|e| polars_err(Box::new(e)))?;
        info!("Sankey diagram for '{}' saved to {}", tool_name, svg_path.display());

        // Optionally attempt to convert to PNG
        convert_svg_to_png(&svg_path);
    }

    info!("All Sankey diagrams completed successfully");
    Ok(())
}

/// Helper function (unchanged) for the prediction side
/// to find which bin a value belongs to based on the
/// automatically generated bin_edges.
fn bin_index(value: f64, bin_edges: &[f64]) -> usize {
    let last_bin = bin_edges.len() - 1;
    for i in 0..last_bin {
        if value >= bin_edges[i] && value < bin_edges[i + 1] {
            return i;
        }
    }
    last_bin - 1
}

/// Unchanged: tries to convert an SVG file to PNG using ImageMagick
fn convert_svg_to_png(svg_path: &Path) {
    let png_path = svg_path.with_extension("png");
    info!("Attempting to convert SVG to PNG: {}", png_path.display());
    let result = Command::new("convert")
        .arg("-density")
        .arg("300")  // high resolution
        .arg(svg_path)
        .arg(&png_path)
        .output();

    match result {
        Ok(output) => {
            if output.status.success() {
                info!("Successfully converted to PNG: {}", png_path.display());
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                error!("Failed to convert to PNG. ImageMagick error: {}", stderr);
            }
        },
        Err(e) => {
            error!("Could not run ImageMagick 'convert': {e}. Make sure ImageMagick is installed.");
        }
    }
}

/// Unchanged: used for auto-binning predictions
/// Generates `bins_count` quantile-based bins so that each bin
/// contains roughly the same number of data points.
///
/// Returns `(labels, edges)`:
///   - `labels[i]` is a string label for bin `i` (e.g. "12.3–25.7")
///   - `edges[i..=i+1]` define that bin’s numeric range
fn generate_bins(
    values: &polars::prelude::Float64Chunked,
    bins_count: usize,
) -> PolarsResult<(Vec<String>, Vec<f64>)> {
    // Collect non-null values into a Vec
    let mut sorted: Vec<f64> = values.into_iter().flatten().collect();
    // Sort ascending
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Edge cases: if everything is identical or empty, just return dummy bins
    if sorted.is_empty() {
        return Ok((vec!["Empty".to_string()], vec![0.0, 1.0]));
    }

    // Build bin_edges using quantiles
    let n = sorted.len();
    let mut bin_edges = Vec::with_capacity(bins_count + 1);
    // For i in [0..bins_count], compute the i/(bins_count) quantile
    // and push that value as a bin edge
    for i in 0..=bins_count {
        let q = i as f64 / bins_count as f64; // fraction of the way through
        // position in the sorted array
        let pos = (q * (n - 1) as f64).round() as usize;
        bin_edges.push(sorted[pos]);
    }

    // Build bin labels
    let mut labels = Vec::with_capacity(bins_count);
    for i in 0..bins_count {
        let left  = bin_edges[i];
        let right = bin_edges[i + 1];
        labels.push(format!("{:.1}–{:.1}", left, right));
    }

    Ok((labels, bin_edges))
}


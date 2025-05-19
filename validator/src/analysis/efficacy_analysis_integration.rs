// Add this module to your project

use std::process::Command;
use std::path::Path;
use polars::prelude::*;
use tracing::{info, error, debug, warn};
use crate::helper_functions::project_root; // Import your helper function

/// Run the efficacy distribution analysis using the Python module
pub fn analyze_efficacy_distribution(
    df: &DataFrame,
    output_dir: &str,
    poor_threshold: f64,
    moderate_threshold: f64,
) -> PolarsResult<()> {
    info!("Starting efficacy distribution analysis");

    // Create the output directory if it doesn't exist
    std::fs::create_dir_all(output_dir).map_err(|e| {
        error!("Failed to create output directory {}: {}", output_dir, e);
        PolarsError::ComputeError(format!("Failed to create directory: {}", e).into())
    })?;

    // Extract the efficacy column
    let efficacy_column = match df.column("efficacy") {
        Ok(col) => col.clone(),
        Err(_) => {
            error!("Efficacy column not found in DataFrame");
            return Err(PolarsError::ComputeError("Efficacy column not found".into()));
        }
    };

    // Generate a simple index column
    let len = df.height();
    let idx_col = Series::new(
        PlSmallStr::from("id"),
        (0..len).map(|i| i as i32).collect::<Vec<i32>>(),
    );

    // Create a simplified DataFrame with just ID and efficacy
    let simple_df = DataFrame::new(vec![
        Column::from(idx_col),
        efficacy_column,
    ])?;

    // Save DataFrame to a temporary CSV
    let temp_dir = "./temp_analysis";
    std::fs::create_dir_all(temp_dir).map_err(|e| {
        error!("Failed to create temp directory {}: {}", temp_dir, e);
        PolarsError::ComputeError(format!("Failed to create temp directory: {}", e).into())
    })?;

    let input_path = format!("{}/efficacy_values.csv", temp_dir);
    info!("Saving simplified DataFrame to temporary CSV at {}", input_path);

    let mut file = std::fs::File::create(&input_path).map_err(|e| {
        error!("Failed to create temporary CSV file: {}", e);
        PolarsError::ComputeError(format!("Failed to create file: {}", e).into())
    })?;

    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut simple_df.clone())?;

    // Build paths using project_root
    let root_path = project_root();
    let script_path = root_path.join("scripts/efficacy_analysis.py");
    let python_path = root_path.join("scripts/efficacy_analysis_env/bin/python");

    info!("Using Python interpreter at: {}", python_path.display());
    info!("Using script at: {}", script_path.display());

    // Execute the Python script with configurable thresholds
    let output = Command::new(python_path)
        .arg(&script_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--efficacy-column")
        .arg("efficacy")
        .arg("--poor-threshold")
        .arg(poor_threshold.to_string())
        .arg("--moderate-threshold")
        .arg(moderate_threshold.to_string())
        .output()
        .map_err(|e| {
            error!("Failed to execute Python script: {}", e);
            PolarsError::ComputeError(format!("Failed to execute script: {}", e).into())
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("Python script failed: {}", stderr);
        return Err(PolarsError::ComputeError(
            format!("Python script error: {}", stderr).into(),
        ));
    }

    // Log stdout for debugging
    let stdout = String::from_utf8_lossy(&output.stdout);
    debug!("Python script output: {}", stdout);

    // Verify outputs
    let dist_plot = Path::new(output_dir).join("efficacy_distribution.png");
    let cat_plot = Path::new(output_dir).join("category_distribution.png");
    if dist_plot.exists() && cat_plot.exists() {
        info!("Successfully created plots: {}, {}", dist_plot.display(), cat_plot.display());
    } else {
        error!("Expected output files not found in {}", output_dir);
    }

    // Clean up temporary CSV
    if let Err(e) = std::fs::remove_file(&input_path) {
        warn!("Failed to remove temporary file {}: {}", input_path, e);
    }

    info!("Efficacy distribution analysis complete. Results in {}", output_dir);
    Ok(())
}

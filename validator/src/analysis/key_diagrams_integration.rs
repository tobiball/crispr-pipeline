// File: key_diagrams_integration.rs

use std::process::Command;
use std::fs::create_dir_all;
use std::path::Path;
use polars::prelude::*;
use tracing::{debug, error, info, warn};
use crate::helper_functions::{dataframe_to_csv, project_root};
use crate::models::polars_err;

/// Run the key_diagrams.py script to generate visualizations for CRISPR guide RNA data
///
/// # Arguments
///
/// * `df` - DataFrame containing efficacy and prediction tool columns
/// * `output_dir` - Directory where results should be saved
/// * `tool_columns` - List of tool column names to analyze
/// * `efficacy_column` - Name of the efficacy column (default is "efficacy")
/// * `title` - Custom title for the plots
///
/// # Returns
///
/// * `PolarsResult<()>` - Ok(()) on success, or an error
pub fn run_key_diagrams(
    df: &mut DataFrame,
    output_dir: &str,
    tool_columns: Vec<&str>,
    efficacy_column: &str,
    title: &str
) -> PolarsResult<()> {
    // Create output directory
    create_dir_all(output_dir).map_err(|e| {
        error!("Failed to create output directory {}: {}", output_dir, e);
        polars_err(Box::new(e))
    })?;

    // Create a temporary directory for the input data
    let temp_dir = format!("{}/temp", output_dir);
    create_dir_all(&temp_dir).map_err(|e| {
        error!("Failed to create temp directory: {}", e);
        polars_err(Box::new(e))
    })?;

    // Save DataFrame to CSV
    let input_csv_path = format!("{}/input_data.csv", temp_dir);
    dataframe_to_csv(df, &input_csv_path, true)?;
    info!("Data exported to CSV: {}", input_csv_path);

    // Path to the wrapper script
    let wrapper_script = project_root()
        .join("scripts/run_key_diagrams.sh");

    // Prepare command arguments
    let tools_str = tool_columns.join(",");

    info!("Running key diagrams analysis using wrapper script: {}", wrapper_script.display());

    // Build the command
    let args = vec![
        "--input".to_string(), input_csv_path,
        "--output".to_string(), output_dir.to_string(),
        "--efficacy-col".to_string(), efficacy_column.to_string(),
        "--tools".to_string(), tools_str,
        "--title".to_string(), title.to_string()
    ];

    // Log the command
    debug!("Running command: {} {}", wrapper_script.display(), args.join(" "));

    // Execute the command
    let output = Command::new(wrapper_script)
        .args(&args)
        .output()
        .map_err(|e| {
            error!("Failed to execute wrapper script: {}", e);
            polars_err(Box::new(e))
        })?;

    // Check if the command was successful
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("Key diagrams script failed: {}", stderr);
        return Err(PolarsError::ComputeError(format!("Key diagrams script failed: {}", stderr).into()));
    }

    // Log success
    let stdout = String::from_utf8_lossy(&output.stdout);
    info!("Key diagrams output: {}", stdout);
    info!("Key diagrams generated successfully in: {}", output_dir);

    // Check if any output files were actually created
    let output_files = std::fs::read_dir(output_dir)
        .map(|entries| {
            entries
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "png" || ext == "csv") {
                        Some(path.to_string_lossy().to_string())
                    } else {
                        None
                    }
                })
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();

    if output_files.is_empty() {
        warn!("No output files were found. The script may have completed without producing visualizations.");
    } else {
        info!("Generated {} files:", output_files.len());
        for file in &output_files {
            info!("  - {}", file);
        }
    }

    Ok(())
}
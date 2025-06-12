//! src/analysis/efficacy_distribution.rs
//!
//! Thin wrapper that writes a minimal CSV with just an `id` and `efficacy`
//! column, then calls the Python plotting utility.
//!
//! Requires `db_name` so the plot title shows
//!     “Guide Efficency Distribution – <Database>”

use std::process::Command;
use std::path::Path;

use polars::prelude::*;
use tracing::{info, error, debug, warn};

use crate::helper_functions::project_root;       // helper that returns repo root

/// Run the efficacy–distribution analysis using the Python script.
///
/// * `df`  – DataFrame with an `efficacy` column (float or numeric string)
/// * `output_dir` – directory for the PNG
/// * `poor_threshold`, `moderate_threshold` – 60 / 90 by convention
/// * `db_name` – database label for the plot title ( *mandatory* )
pub fn analyze_efficacy_distribution(
    df: &DataFrame,
    output_dir: &str,
    poor_threshold: f64,
    moderate_threshold: f64,
    db_name: &str,                    // ← NEW
) -> PolarsResult<()> {
    info!("Starting efficacy distribution analysis");

    // ── ensure output & temp directories exist ───────────────────────────────
    std::fs::create_dir_all(output_dir).map_err(|e| {
        error!("Failed to create output directory {output_dir}: {e}");
        PolarsError::ComputeError(format!("Failed to create directory: {e}").into())
    })?;

    let temp_dir = "./temp_analysis";
    std::fs::create_dir_all(temp_dir).map_err(|e| {
        error!("Failed to create temp directory {temp_dir}: {e}");
        PolarsError::ComputeError(format!("Failed to create temp dir: {e}").into())
    })?;

    // ── build a minimal DataFrame (id, efficacy) ─────────────────────────────
    let len = df.height();
    let idx_series = Series::new(PlSmallStr::from("id"), (0..len as i64).collect::<Vec<i64>>());

    let eff_series = df
        .column("efficacy")
        .map_err(|_| PolarsError::ComputeError("Efficacy column not found".into()))?
        .clone();

    let simple_df = DataFrame::new(vec![Column::from(idx_series), eff_series])?;

    // ── write temporary CSV ───────────────────────────────────────────────────
    let input_path = format!("{temp_dir}/efficacy_values.csv");
    info!("Saving simplified DataFrame to {input_path}");

    let mut file = std::fs::File::create(&input_path)
        .map_err(|e| PolarsError::ComputeError(format!("Failed to create file: {e}").into()))?;

    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut simple_df.clone())?;

    // ── construct paths to the Python venv interpreter & script ──────────────
    let root = project_root();
    let py = root.join("scripts/efficacy_analysis_env/bin/python");
    let script = root.join("scripts/efficacy_analysis.py");

    info!("Using Python interpreter : {}", py.display());
    info!("Using plotting script    : {}", script.display());

    // ── invoke the script ────────────────────────────────────────────────────
    let output = Command::new(&py)
        .arg(&script)
        // legacy pipeline flags ↓
        .arg("--input").arg(&input_path)
        .arg("--output-dir").arg(output_dir)
        .arg("--efficacy-column").arg("efficacy")
        // thresholds
        .arg("--poor").arg(poor_threshold.to_string())
        .arg("--moderate").arg(moderate_threshold.to_string())
        // NEW: database name for the title
        .arg("--db").arg(db_name)
        .output()
        .map_err(|e| PolarsError::ComputeError(format!("Failed to run script: {e}").into()))?;

    // ── handle Python exit status / logging ──────────────────────────────────
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("Python script failed:\n{stderr}");
        return Err(PolarsError::ComputeError(format!("Python error: {stderr}").into()));
    }
    debug!("Python stdout:\n{}", String::from_utf8_lossy(&output.stdout));

    // ── verify expected outputs ──────────────────────────────────────────────
    for file in ["efficacy_distribution.png", "category_distribution.png"] {
        let f = Path::new(output_dir).join(file);
        if f.exists() {
            info!("Created {file}");
        } else {
            warn!("Expected {file} not found in {output_dir}");
        }
    }

    // ── clean up temp CSV ────────────────────────────────────────────────────
    if let Err(e) = std::fs::remove_file(&input_path) {
        warn!("Could not delete temp file {input_path}: {e}");
    }

    info!("Efficacy distribution analysis complete.");
    Ok(())
}

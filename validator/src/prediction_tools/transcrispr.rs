use polars::prelude::*;
use std::fs::File;
use std::process::Command;
use std::path::Path;

use tracing::{info, error, debug, warn};
use crate::helper_functions::{project_root, read_csv};
use crate::models::polars_err;

/// Run TransCRISPR on the provided DataFrame by writing to its expected input path
pub fn run_transcrispr_meta(df: DataFrame, database_name: &str) -> PolarsResult<DataFrame> {
    // Create a mutable copy of the dataframe for return
    let mut result_df = df.clone();

    // Find the sequence column
    let seq_column = if df.schema().contains("sequence_with_pam") {
        "sequence_with_pam"
    } else if df.schema().contains("sgRNA") {
        "sgRNA"
    } else {
        error!("Could not find sequence column in DataFrame");
        return Err(PolarsError::ComputeError("Missing sequence column".into()));
    };

    // TransCRISPR expected paths
    let transcrispr_dir = project_root().join("TransCrispr");
    let input_path = transcrispr_dir.join("input/SpCas9.csv");

    // Create input directory if it doesn't exist
    if let Some(parent) = input_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Create output directory
    let processed_dir = "./processed_data";
    std::fs::create_dir_all(processed_dir)?;
    let processed_output_path = format!("{}/transcrispr_{}.csv", processed_dir, database_name);

    // Create a dataframe with just the sequences in the expected format
    info!("Creating input file for TransCRISPR at {:?}", input_path);

    // Get the sequence column and clone it
    let seq_series = df.column(seq_column)?.clone();
    let mut renamed_series = seq_series.clone();
    renamed_series.rename("Input_Sequence".into());

    // Create a new DataFrame with just this column
    let mut input_df = DataFrame::new(vec![renamed_series])?;

    // Save to CSV at the expected input path
    let file = File::create(&input_path)
        .map_err(|e| polars_err(Box::new(e)))?;
    CsvWriter::new(file).finish(&mut input_df)?;

    // Path to the TransCRISPR script
    let script_path = transcrispr_dir.join("PureNet.py");

    // Run TransCRISPR
    info!("Running TransCRISPR script at {:?}", script_path);
    let python_path = transcrispr_dir
        .join("transcrispr_env/bin/python")
        .to_str()
        .unwrap_or("python")
        .to_string();

    let output = Command::new(&python_path)
        .arg(script_path.to_str().unwrap())
        .current_dir(&transcrispr_dir)  // Set working directory to TransCRISPR folder
        .output()
        .map_err(|e| polars_err(Box::new(e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("TransCRISPR failed: {}", stderr);
        return Err(PolarsError::ComputeError(format!("TransCRISPR failed: {}", stderr).into()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    debug!("TransCRISPR output: {}", stdout);

    // Look for the predictions file
    let predictions_path = transcrispr_dir.join("output/predictions.csv");
    info!("Looking for TransCRISPR predictions at {:?}", predictions_path);

    if predictions_path.exists() {
        // Read the predictions
        info!("Reading TransCRISPR predictions");
        let mut transcrispr_df = read_csv(predictions_path.to_str().unwrap())?;

        // Check if transcrispr_prediction column exists
        if transcrispr_df.schema().contains("transcrispr_prediction") {
            // Get the predictions column
            let pred_col = transcrispr_df.column("transcrispr_prediction")?;
            let pred_f64 = pred_col.f64()?;

            // Create a new scaled column
            let scaled_preds: Vec<f64> = pred_f64
                .into_iter()
                .map(|opt_val| opt_val.map(|v| v * 100.0).unwrap_or(0.0))
                .collect();

            // Create a Series with the scaled values
            let scaled_series = Series::new("transcrispr_prediction".into(), scaled_preds);

            // Drop the old column and add the new one
            transcrispr_df = transcrispr_df.drop("transcrispr_prediction")?;
            transcrispr_df.with_column(scaled_series)?;
        }

        // Save the scaled predictions to processed_data
        let file = File::create(&processed_output_path)?;
        CsvWriter::new(file).finish(&mut transcrispr_df.clone())?;

        // Join with original data
        info!("Joining TransCRISPR predictions with original data");
        let joined_df = df.join(
            &transcrispr_df,
            [seq_column],
            ["sequence"],  // Column name in the predictions file
            JoinArgs::new(JoinType::Left),
            None
        )?;

        info!("TransCRISPR analysis completed, results joined with DataFrame");
        return Ok(joined_df);
    } else {
        warn!("TransCRISPR predictions file not found at {:?}", predictions_path);
        // Return the original DataFrame if we couldn't find predictions
        Ok(result_df)
    }
}
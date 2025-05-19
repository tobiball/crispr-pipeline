use std::fs::File;
use std::process::Command;
use polars::prelude::*;
use tracing::{error, info};
use crate::helper_functions::read_csv;

/// Runs DeepCRISPR on the input DataFrame and returns a new DataFrame
/// with predictions joined back to the original.
pub fn run_deepcrispr_meta(
    df: DataFrame,
    database_name: &str,
) -> PolarsResult<DataFrame> {
    // Prepare directories and paths
    // Prepare directories and paths
    let output_dir = "./processed_data";
    std::fs::create_dir_all(output_dir)?;
    let input_path  = format!("{}/deepcrispr_input_{}.csv",  output_dir, database_name);
    let output_path = format!("{}/deepcrispr_output_{}.csv", output_dir, database_name);

    let mut input_file = File::create(&input_path)?;
    CsvWriter::new(&mut input_file)
        .include_header(true)
        .finish(&mut df.clone())?;

    // Then call the script with the *input* CSV first, then the output path:
    let cmd = format!(
        "cd ~/crispr_pipeline && \
     source ./DeepCRISPR/deepcrispr_env/bin/activate && \
     python ./DeepCRISPR/run_genome_crispr_regression.py {} {}",
        input_path, output_path
    );

    let result = Command::new("bash").arg("-c").arg(&cmd).output()?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        error!("DeepCRISPR failed: {}", stderr);
        return Err(PolarsError::ComputeError(
            format!("DeepCRISPR failed: {}", stderr).into(),
        ));
    }
    info!("DeepCRISPR completed: output at {}", output_path);

    // Read predictions CSV
    let mut preds = read_csv(&output_path)?;

    // Assume the prediction column is named "prediction" (adjust if different)
    // If the rows align one-to-one, simply hstack predictions
    let pred_col = preds
        .column("deepcrispr_prediction")?
        .clone();
    let mut joined = df.clone();
    joined.hstack_mut(&[pred_col])?;

    Ok(joined)
}

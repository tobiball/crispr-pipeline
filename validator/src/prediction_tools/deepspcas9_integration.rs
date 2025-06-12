use std::fs::File;
use polars::prelude::*;
use std::process::Command;
use tracing::{error, info};
use crate::helper_functions::{read_csv, project_root};

/// Run DeepSpCas9 on a DataFrame of guides
pub fn run_deepspcas9_meta(
    mut df: DataFrame,
    database_name: &str
) -> PolarsResult<DataFrame> {
    let output_dir = "./processed_data";
    std::fs::create_dir_all(output_dir)?;
    let input_path  = format!("{}/deepspcas9_input_{}.csv",  output_dir, database_name);
    let output_path = format!("{}/deepspcas9_output_{}.csv", output_dir, database_name);

    // 1) dump the full DF (it already has `sequence_deepspcas9`)
    {
        let mut f = File::create(&input_path)?;
        CsvWriter::new(&mut f).include_header(true).finish(&mut df.clone())?;
    }

    // 2) call into Python
    let status = Command::new(project_root().join("DeepSpCas9/env/bin/python"))
        .arg(project_root().join("DeepSpCas9/run_deepspcas9.py"))
        .arg("--input").arg(&input_path)
        .arg("--output").arg(&output_path)
        .status()?;
    if !status.success() {
        return Err(PolarsError::ComputeError("DeepSpCas9 execution failed".into()));
    }

    // 3) read back what Python wrote
    let mut result_df = read_csv(&output_path)?;
    // Python always emits a column called "deepspcas9_prediction"
    let python_col = "deepspcas9_prediction";
    // we want to call it e.g. "deepspcas9_avana-depmap"
    let our_col = format!("deepspcas9_{}", database_name);

    // 4) pull out that series, rename _the series_, then stick it onto our original df
    let mut series = result_df.column(python_col)?.clone();
    series.rename(PlSmallStr::from(&our_col));
    let df = df.with_column(series)?;

    Ok(df.clone())
}

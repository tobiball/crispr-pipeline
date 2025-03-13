use std::process::Command;
use csv;
use polars::prelude::*;
use tracing::{error, info};

pub fn run_deepcrispr_meta(df: &str, database_name: &str) -> PolarsResult<()> {
    // Create minimal directories and file paths
    let output_dir = "./deepcrispr_processed_data";
    std::fs::create_dir_all(output_dir)?;

    // let input_path = format!("{}/deepcrispr_input_{}.csv", processing_dir, database_name);
    let output_path = format!("{}/deepcrispr_output_{}.csv", output_dir, database_name);

    // // Write minimal CSV for DeepCRISPR
    // {
    //     let mut wtr = csv::Writer::from_path(&input_path)
    //         .map_err(|e| PolarsError::ComputeError(format!("Cannot create input: {e}").into()))?;
    //
    //     wtr.write_record(&["sequence", "chr", "start", "end", "strand", "efficacy"])
    //         .map_err(|e| PolarsError::ComputeError(format!("Write error: {e}").into()))?;
    //
    //     for i in 0..df.height() {
    //         let sequence = df.column("sgRNA")?.get(i)?.to_string().trim_matches('"').to_string();
    //         let chr = df.column("chromosome")?.get(i)?.to_string().trim_matches('"').replace("chr", "");
    //
    //         let start = match df.column("start")?.get(i)? {
    //             AnyValue::Int64(v) => v.to_string(),
    //             _ => continue,
    //         };
    //
    //         let end = match df.column("end")?.get(i)? {
    //             AnyValue::Int64(v) => v.to_string(),
    //             _ => continue,
    //         };
    //
    //         let efficacy = match df.column("efficacy")?.get(i)? {
    //             AnyValue::Float64(v) => v.to_string(),
    //             _ => continue,
    //         };
    //
    //         wtr.write_record(&[&sequence, &chr, &start, &end, "+", &efficacy])
    //             .map_err(|e| PolarsError::ComputeError(format!("Write error: {e}").into()))?;
    //     }
    //
    //     wtr.flush().map_err(|e| PolarsError::ComputeError(format!("Flush error: {e}").into()))?;
    // }

    // Run DeepCRISPR
    let cmd = format!(
        "cd ~/crispr_pipeline && \
         source ./DeepCRISPR/deepcrispr_env/bin/activate && \
         python ./DeepCRISPR/run_genome_crispr_regression.py {} {}",
        df, output_path
    );

    let result = Command::new("bash")
        .arg("-c")
        .arg(&cmd)
        .output()
        .map_err(|e| PolarsError::ComputeError(format!("Failed to run: {e}").into()))?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        error!("DeepCRISPR failed: {stderr}");
        return Err(PolarsError::ComputeError(format!("DeepCRISPR failed: {stderr}").into()));
    }

    info!("DeepCRISPR completed: output at {output_path}");
    Ok(())
}
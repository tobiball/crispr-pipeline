use polars::prelude::{DataFrame, CsvWriter};
use polars::prelude::*;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::helper_functions::{project_root, read_txt};
use crate::models::polars_err; // or your error helpers

const GENOME: &str = "hg38";

use std::time::Instant;
use log::{error, info};
// ...

pub fn run_crispor_meta(df: DataFrame, database_name: &str) -> PolarsResult<()> {
    // (1) Limit the DataFrame to 450 guides for testing:
    //     (If your df has <= 450 rows, you can skip or adjust.)

    // (2) Start timing
    let start_time = Instant::now();

    // ----------------------------
    // Build the multi‐record FASTA
    // ----------------------------
    let guide_column = df.column("guide")?;
    let guides_str = guide_column.str()?;

    let mut fasta_content = String::new();
    for (i, guide_opt) in guides_str.into_iter().enumerate() {
        if let Some(guide_str) = guide_opt {
            fasta_content.push_str(&format!(">guide{}\n{}\n", i, guide_str));
        }
    }

    // Write FASTA to file
    let fasta_file = format!("{}/crispor/all_guides.fasta", project_root().display());
    std::fs::write(&fasta_file, fasta_content.as_bytes())?;

    // ----------------------------
    // Run CRISPOR once
    // ----------------------------
    let output_file = format!("{}/temp/crispor/all_guides.tsv", project_root().display());
    let python_executable = format!("{}/crispor/crispor_env/bin/python", project_root().display());
    let crispor_script = format!("{}/crispor/crispor.py", project_root().display());
    let status = Command::new(&python_executable)
        .arg(&crispor_script)
        .arg(GENOME)
        .arg(&fasta_file)
        .arg(&output_file)
        .status()
        .expect("Failed to execute CRISPOR command");

    if !status.success() {
        error!("CRISPOR command failed with status: {:?}", status);
        // handle error accordingly
    }

    // ----------------------------
    // Read CRISPOR output & join
    // ----------------------------
    let crispor_results = read_txt(&output_file)?;
    let mut df_joined = df.left_join(&crispor_results, ["guide"], ["targetSeq"])?;

    // ----------------------------
    // Write final results once
    // ----------------------------
    let results_csv_path = format!("{}/processed_data/crispor_{}.csv", project_root().display(), database_name);
    let file = File::create(&results_csv_path)?;
    CsvWriter::new(file).finish(&mut df_joined)?;

    // (3) Print timing results
    let elapsed = start_time.elapsed();
    info!("CRISPOR completed in: {:?}", elapsed);

    Ok(())
}

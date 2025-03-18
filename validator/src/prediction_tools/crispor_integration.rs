use std::env;
use std::fs;
use std::fs::{create_dir, create_dir_all, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use log::{debug, error};
use polars::error::PolarsResult;
use polars::frame::DataFrame;
use polars_ops::prelude::DataFrameJoinOps;
use crate::helper_functions::{project_root, read_csv, read_txt};

const GENOME: &str = "hg38";

pub fn run_crispor_meta(df: DataFrame) -> PolarsResult<()> {


    // Set up paths – adjust these to your environment
    let python_executable = format!("{}/crispor/crispor_env/bin/python", project_root().display());
    let crispor_script = format!("{}/crispor/crispor.py", project_root().display());
    let crispor_output = format!("{}/temp/crispor", project_root().display());
    create_dir_all(crispor_output.clone())?;

    // Check that the Python executable exists
    if !Path::new(&python_executable).exists() {
        error!("Error: Python executable not found at '{}'", python_executable);
        std::process::exit(1);
    }
    // Check that the CRISPOR script exists
    if !Path::new(&crispor_script).exists() {
        error!("Error: CRISPOR script not found at '{}'", crispor_script);
        std::process::exit(1);
    }

    // Prepare a FASTA file for the guide RNA.

    for i in 0..df.height() {
        let guide_rna_raw = df.column("guide").unwrap().get(i).unwrap().to_string();
        let guide_rna = guide_rna_raw.trim_matches('"').trim();
        // let eff: f64 = df.column("dataset_efficacy").unwrap().get(i).unwrap().try_extract().unwrap();

        // Or with more descriptive error messages
        let fasta_file = format!("{}/crispor/guide_input.fasta", project_root().display());
        let fasta_content = format!(">guide\n{}\n", guide_rna);
        fs::write(&fasta_file, fasta_content.as_bytes())
            .expect("Failed to write the FASTA file");

        // Define output file for CRISPOR results
        let output_file = format!("{}/guide_output.tsv", crispor_output);

        // Print the command being executed for debugging
        debug!("Executing command:");
        debug!("{} {} {} {} {}",
                 python_executable, crispor_script, GENOME, fasta_file, output_file);

        // Run CRISPOR
        let status = Command::new(&python_executable)
            .arg(&crispor_script)
            .arg(GENOME)
            .arg(&fasta_file)
            .arg(&output_file)
            .current_dir(project_root().as_path())
            .env("PYTHONPATH", project_root().as_path())
            .env("PATH", format!("{}/bin/Linux:{}", project_root().display(), env::var("PATH").unwrap_or_default()))
            .status()
            .expect("Failed to execute CRISPOR command");

        if status.success() {
            debug!("CRISPOR finished successfully. Results in: {}", output_file);
        } else {
            error!("CRISPOR command failed with status: {}", status);
        }
        
        
        let crispor_result = read_txt(&output_file)?;

        debug!("CRISPOR result was: {:?}", crispor_result);
        df.left_join(
            &crispor_result,
            ["guide"],
            ["target_seq"]
        )?;


        debug!("Output: {}", df);
    }
    Ok(())
}

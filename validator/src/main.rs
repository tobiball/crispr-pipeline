use std::fs;
use polars::prelude::*;
use std::path::PathBuf;
use sequence_retriever::chopchop_integration::{run_chopchop, parse_chopchop_results, ChopchopOptions};


// Helper function to read CSV files
fn read_csv(file_path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(file_path)))?
        .finish()
}

fn main() -> PolarsResult<()> {
    // Read the CSV files into DataFrames using the helper function
    let efficacy = read_csv("CRISPRInferredGuideEfficacy.csv")?;
    let guide_map = read_csv("AvanaGuideMap.csv")?;

    // Merge the DataFrames on the "sgRNA" column
    let df_gene_guide_efficiencies = efficacy.join(
        &guide_map,
        ["sgRNA"],
        ["sgRNA"],
        JoinArgs::from(JoinType::Inner),
    )?;

    // Print the first 5 rows of the merged DataFrame
    println!("{}", df_gene_guide_efficiencies.head(Some(5)));


    // Define the main output directory with absolute path
    let main_output_dir = format!("/home/mrcrispr/crispr_pipeline/output/{}", gene_symbol);

    // Ensure the main output directory exists
    fs::create_dir_all(&main_output_dir)?;

    // Define the dedicated CHOPCHOP output subdirectory
    let chopchop_output_dir = format!("{}/chopchop_output", main_output_dir);

    // Ensure the CHOPCHOP output directory exists
    fs::create_dir_all(&chopchop_output_dir)?;

    // Set up CHOPCHOP options
    let chopchop_options = ChopchopOptions {
        python_executable: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop_env/bin/python2.7".to_string(),
        chopchop_script: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop.py".to_string(),
        genome: "hg38".to_string(),
        target_type: "GENE".to_string(), // Changed from 'GENE_NAME' to 'GENE'
        target: gene_symbol.to_string(), // Use the gene symbol as the target
        output_dir: chopchop_output_dir.clone(),
        pam_sequence: "NGG".to_string(),
        guide_length: 20,
        scoring_method: "DOENCH_2016".to_string(),
        max_mismatches: 3,
    };

    println!("Running CHOPCHOP for target: {}", gene_symbol);
    if let Err(e) = run_chopchop(&chopchop_options) {
        eprintln!("Error running CHOPCHOP for target {}: {}", gene_symbol, e);
        return Ok(());
    }

    let mut guides = parse_chopchop_results(&chopchop_output_dir)?;



    Ok(())
}

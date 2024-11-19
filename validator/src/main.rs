use std::fs;
use polars::prelude::*;
use std::path::PathBuf;
use crate::chopchop_integration::{parse_chopchop_results, run_chopchop, ChopchopOptions};

pub mod chopchop_integration;

// At the top with other constants
const SEARCH_MARGIN: i32 = 1;  // Small margin of error around expected position
const PLUS_STRAND_OFFSET: i32 = 16;
const MINUS_STRAND_OFFSET: i32 = 5;

// Helper function to read CSV files
fn read_csv(file_path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(file_path)))?
        .finish()
}

fn main() -> PolarsResult<()> {
    // Read the CSV files into DataFrames using the helper function
    let efficacy = read_csv("/home/mrcrispr/data/depmap/data/CRISPRInferredGuideEfficacy.csv")?;
    let guide_map = read_csv("/home/mrcrispr/data/depmap/data/AvanaGuideMap.csv")?;

    // Merge the DataFrames on the "sgRNA" column
    let df_gene_guide_efficiencies = efficacy.join(
        &guide_map,
        ["sgRNA"],
        ["sgRNA"],
        JoinArgs::from(JoinType::Inner),
    )?;

    // Print the first 5 rows of the merged DataFrame
    println!("{}", df_gene_guide_efficiencies.height());

    // Split the 'GenomeAlignment' column into parts
    // At the top with other constants
    const SEARCH_MARGIN: i32 = 5;  // Small margin of error around expected position
    const PLUS_STRAND_OFFSET: i32 = 16;
    const MINUS_STRAND_OFFSET: i32 = 5;

    // Then modify the DataFrame processing part to use two potential windows
    let df = df_gene_guide_efficiencies.lazy()
        .with_column(
            col("GenomeAlignment")
                .str()
                .split(lit("_"))
                .alias("split_parts"),
        )
        .with_columns(vec![
            col("split_parts")
                .list()
                .get(lit(0), false)
                .alias("chromosome"),
            col("split_parts")
                .list()
                .get(lit(1), false)
                .alias("position"),
            col("split_parts")
                .list()
                .get(lit(2), false)
                .alias("strand"),
        ])
        // Cast 'position' to Int64 for arithmetic operations
        .with_column(col("position").cast(DataType::Int64))
        // Calculate start and end based on strand
        .with_columns(vec![
            (when(col("strand").eq(lit("+")))
                .then(col("position") - lit(PLUS_STRAND_OFFSET))
                .otherwise(col("position") - lit(MINUS_STRAND_OFFSET))
            ).alias("start"),
            (when(col("strand").eq(lit("+")))
                .then(col("position") - lit(PLUS_STRAND_OFFSET - 1))
                .otherwise(col("position") - lit(MINUS_STRAND_OFFSET -1))
            ).alias("end"),
        ])
        .select(&[col("*")])
        .collect()?;



    // Retrieve each column as a Series
    let chromosome_series = df.column("chromosome")?;
    let start_series = df.column("start")?;
    let end_series = df.column("end")?;
    let position_series = df.column("position")?;
    let guides_series = df.column("sgRNA")?;
    let mut counter = 0;


    // Iterate over each row by index
    for i in 0..df.height() {
        // Retrieve the `i`th element from each Series
        let chromosome = chromosome_series.get(i)?.to_string().replace('"', "");
        let start = start_series.get(i)?;
        let end = end_series.get(i)?;
        let position = match position_series.get(i)? {
                AnyValue::Int64(p) => p,
                _ => panic!("Expected Int64 value for position")
            };
        let raw_guide = guides_series.get(i)?.to_string();
        let guide = raw_guide.trim_matches('"');

        // Build the target region in the format "chr:start-end"
        let target_region = format!("{}:{}-{}", chromosome, start, end);

        // Define the output directory for this target
        let output_dir = format!("/home/mrcrispr/crispr_pipeline/validator/output/{}/{}", chromosome, i);

        // Ensure the output directory exists
        fs::create_dir_all(&output_dir)?;

        // Set up CHOPCHOP options
        let chopchop_options = ChopchopOptions {
            python_executable: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop_env/bin/python2.7".to_string(),
            chopchop_script: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop.py".to_string(),
            genome: "hg38".to_string(),
            target_type: "REGION".to_string(), // Changed to 'REGION' to specify genomic coordinates
            target: target_region.clone(),
            output_dir: output_dir.clone(),
            pam_sequence: "NGG".to_string(),
            guide_length: 20,
            scoring_method: "DOENCH_2016".to_string(),
            max_mismatches: 3,
        };

        // println!("Running CHOPCHOP for target region: {}", target_region);

        if let Err(e) = run_chopchop(&chopchop_options) {
            eprintln!("Error running CHOPCHOP for target {}: {}", target_region, e);
            continue; // Continue with the next iteration
        }

        let guides = parse_chopchop_results(&output_dir);

        let sequence_comparisons = guides.unwrap().iter().enumerate().map(|(i, g)| {
            let guide_seq = &g.sequence[..g.sequence.len()-3]; // Remove PAM



            if guide_seq == guide {
                let distance = match position {
                    (s) => (s- g.start as i64),
                    _ => panic!("Expected Int64 value for start position")
                };
                // println!("{} vs {}", guide_seq, guide);
                // println!("Matching sequence: {}", guide_seq);
                // println!("CHOPCHOP coords - Chr: {}, Start: {}, End: {}",
                //          g.chromosome, g.start, g.end);
                // println!("DB coords - Chr: {}, Start: {}, End: {}",
                //          chromosome, position, position + 23);
                println!("Distance: {}", distance);

                if distance != 16 && g.strand == '+' { panic!(); };
                if distance != 5 && g.strand == '-' { panic!(); };

            }
            guide_seq == guide

            //
            // Create chopchop query that will only query the exact position of the guide in question
            // Compare predicted against actual efficency
            // Think about next steps
            //

        }).collect::<Vec<bool>>();

        if !sequence_comparisons.iter().any(|&x| x){
            println!("No sequence comparison found for {}", guide);
        }

        counter = counter + 1;
        // if counter == 100 {
        //     break
        // }
        println!("Count: {}",counter)

    }


    Ok(())
}

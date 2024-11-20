use std::fs;
use polars::prelude::*;
use std::path::PathBuf;
use prediction_tools::chopchop_integration::{parse_chopchop_results, run_chopchop, ChopchopOptions};
use tracing::{debug, error, info};
use tracing_subscriber::{fmt, EnvFilter};

mod tool_evluation;
mod models;
mod data_handling;
mod prediction_tools;

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
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting the CRISPR pipeline");

    // Read the CSV files into DataFrames using the helper function
    let efficacy_path = "/home/mrcrispr/data/depmap/data/CRISPRInferredGuideEfficacy.csv";
    let guide_map_path = "/home/mrcrispr/data/depmap/data/AvanaGuideMap.csv";

    info!("Reading efficacy data from {}", efficacy_path);
    let efficacy = match read_csv(efficacy_path) {
        Ok(df) => df,
        Err(e) => {
            error!("Failed to read efficacy CSV: {}", e);
            return Err(e);
        }
    };

    info!("Reading guide map data from {}", guide_map_path);
    let guide_map = match read_csv(guide_map_path) {
        Ok(df) => df,
        Err(e) => {
            error!("Failed to read guide map CSV: {}", e);
            return Err(e);
        }
    };

    // Merge the DataFrames on the "sgRNA" column
    info!("Merging efficacy and guide map DataFrames on 'sgRNA' column");
    let df_gene_guide_efficiencies = match efficacy.join(
        &guide_map,
        ["sgRNA"],
        ["sgRNA"],
        JoinArgs::from(JoinType::Inner),
    ) {
        Ok(df) => df,
        Err(e) => {
            error!("Failed to join DataFrames: {}", e);
            return Err(e);
        }
    };

    let total_guides =        df_gene_guide_efficiencies.height();

    info!(
        "Number of sgRNAs to check: {}", total_guides
    );


    // Modify the DataFrame processing part to use two potential windows
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

    debug!("DataFrame after processing: {:?}", df);

    // Retrieve each column as a Series
    let chromosome_series = df.column("chromosome")?;
    let start_series = df.column("start")?;
    let end_series = df.column("end")?;
    let position_series = df.column("position")?;
    let guides_series = df.column("sgRNA")?;
    let efficacy_series =  df.column("Efficacy")?;
    let mut counter: f64 = 0.0;

    // Iterate over each row by index
    for i in 0..df.height() {
        // Retrieve the `i`th element from each Series
        let chromosome = chromosome_series.get(i)?.to_string().replace('"', "");
        let start = start_series.get(i)?;
        let end = end_series.get(i)?;
        let position = match position_series.get(i)? {
            AnyValue::Int64(p) => p,
            _ => {
                error!("Expected Int64 value for position at row {}", i);
                panic!("Invalid position type");
            }
        };
        let raw_guide = guides_series.get(i)?.to_string();
        let guide = raw_guide.trim_matches('"');
        let efficacy: f64 = efficacy_series.get(i)?.try_extract()?;
        let efficacy_scaled = efficacy * 100.0;

        // Build the target region in the format "chr:start-end"
        let target_region = format!("{}:{}-{}", chromosome, start, end);

        // Define the output directory for this target
        let output_dir = format!("/home/mrcrispr/crispr_pipeline/validator/output/{}/{}", chromosome, i);

        // Ensure the output directory exists
        if let Err(e) = fs::create_dir_all(&output_dir) {
            error!("Failed to create directory {}: {}", output_dir, e);
            continue;
        }

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

        debug!("Running CHOPCHOP with options: {:?}", chopchop_options);

        if let Err(e) = run_chopchop(&chopchop_options) {
            error!("Error running CHOPCHOP for target {}: {}", target_region, e);
            continue; // Continue with the next iteration
        }

        let guides = match parse_chopchop_results(&output_dir) {
            Ok(guides) => guides,
            Err(e) => {
                error!("Failed to parse CHOPCHOP results in {}: {}", output_dir, e);
                continue;
            }
        };

        let sequence_comparisons = guides.iter().enumerate().map(|(i, g)| {
            let guide_seq = &g.sequence[..g.sequence.len()-3]; // Remove PAM

            if guide_seq == guide {
                let distance = position - g.start as i64;
                debug!("Matching guide found: {} with distance {}", guide_seq, distance);

                if (distance != 16 && g.strand == '+') || (distance != 5 && g.strand == '-') {
                    error!(
                        "Distance mismatch for guide {}: expected {}, got {}",
                        guide,
                        if g.strand == '+' { 16 } else { 5 },
                        distance
                    );
                    panic!("Distance does not match expected offset");
                }
                debug!("Tool: {:}", g.efficiency);
                debug!("Dataset: {:}", efficacy_scaled);
                debug!("Difference Tool vs Dataset {:}", g.efficiency - efficacy_scaled);
            }

            guide_seq == guide
        }).collect::<Vec<bool>>();

        if !sequence_comparisons.iter().any(|&x| x){
            error!("No sequence comparison found for {}", guide);
        }
        counter = counter + 1.0;
        debug!("Count {:}",counter);
        if  counter % 50.0 == 0.0  {
            info!("Progress {:.2}%", 100.0 * counter / total_guides as f64 );
        };

    }

    info!("CRISPR pipeline completed successfully");
    Ok(())
}

use polars::prelude::*;

use tracing::{debug, error, info};
use crate::helper_functions::read_csv;
use crate::models::Dataset;

// Constants used in calculations
const PLUS_STRAND_OFFSET: i32 = 16;
const MINUS_STRAND_OFFSET: i32 = 5;

/// Trait representing a dataset in the CRISPR pipeline.

/// Struct for the Avana dataset.
pub struct AvanaDataset {
    pub(crate) efficacy_path: String,
    pub(crate) guide_map_path: String,
}

impl Dataset for AvanaDataset {


    fn load(&self) -> PolarsResult<DataFrame> {
        // Read the CSV files into DataFrames using the helper function
        let efficacy_path = &self.efficacy_path;
        let guide_map_path = &self.guide_map_path;

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
        let mut df_gene_guide_efficiencies = match efficacy.join(&guide_map, ["sgRNA"], ["sgRNA"], JoinArgs::from(JoinType::Inner), None) {
            Ok(mut df) => {
                df.rename("Efficacy", "efficacy".into())?;
                df
            },
            Err(e) => {
                error!("Failed to join DataFrames: {}", e);
                return Err(e);
            }
        };


        let total_guides = df_gene_guide_efficiencies.height();

        info!("Number of sgRNAs to check: {}", total_guides);




        let pattern = r"\s*\([^)]*\)$";
        let df_annot_clean = df_gene_guide_efficiencies
            .lazy()
            .with_column(
                col("Gene")
                    .str()
                    .replace(lit(pattern), lit(""), false)  // false => pattern is a regex
                    .alias("Gene")
            )
            .collect()?;

        // Modify the DataFrame processing part to use two potential windows
        let df = df_annot_clean.clone()
            .lazy()
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
                    .otherwise(col("position") - lit(MINUS_STRAND_OFFSET)))
                    .alias("start"),
                (when(col("strand").eq(lit("+")))
                    .then(col("position") - lit(PLUS_STRAND_OFFSET - 1))
                    .otherwise(col("position") - lit(MINUS_STRAND_OFFSET - 1)))
                    .alias("end"),
            ])
            .with_column(
                (col("efficacy") * lit(100.0)).alias("efficacy")
            )
            .select(&[col("*")])
            .collect()?;

        debug!("DataFrame after processing: {:?}", df);



        Ok(df)
    }

    fn augment_guides(df: DataFrame) -> PolarsResult<DataFrame> {
        Ok(augment_guides_with_pam(df)?)
    }

    fn mageck_efficency_scoring(df: DataFrame) -> PolarsResult<DataFrame> {
        Ok(df)
    }
}

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::Command;
use std::path::Path;

use polars::prelude::*;

use crate::helper_functions::project_root;
use crate::models::polars_err;

const PAM_LENGTH: usize = 3; // NGG PAM length
const GENOMIC_CONTEXT_PADDING: i64 = 50; // Get additional sequence for PAM extraction

/// Add PAM sequences to guides in the DataFrame
pub fn augment_guides_with_pam(mut df: DataFrame) -> PolarsResult<DataFrame> {
    info!("Augmenting guides with PAM sequences with genomic context for debugging purposes");

    let chromosome = df.column("chromosome")?.str()?;
    let start = df.column("start")?.i64()?;
    let end = df.column("end")?.i64()?;
    let strand = df.column("strand")?.str()?;
    let guide_seq = df.column("sgRNA")?.str()?;

    let mut pam_sequences = Vec::new();
    let mut sequences_with_pam = Vec::new();


    for i in 0..df.height() {
        // Use pattern matching for safer unwrapping
        let chrom = match chromosome.get(i) {
            Some(c) => c,
            None => {
                debug!("Missing chromosome for row {}, using empty string", i);
                ""
            }
        };

        let start_pos = match start.get(i) {
            Some(s) => s,
            None => {
                debug!("Missing start position for row {}, skipping PAM extraction", i);
                continue;
            }
        };

        let guide = match guide_seq.get(i) {
            Some(g) => g,
            None => {
                debug!("Missing guide sequence for row {}, using empty string", i);
                ""
            }
        };

        let strand_val = match strand.get(i) {
            Some(s) => s,
            None => {
                debug!("Missing strand for row {}, assuming '+'", i);
                "+"
            }
        };

        // Extract genomic sequence with error handling
        let region_seq = match extract_genomic_sequence(chrom, start_pos - 1, start_pos + 22) {
            Ok(seq) => seq,
            Err(e) => {
                debug!("Failed to extract genomic sequence for row {}: {}", i, e);
                String::new()
            }
        };

        // Process sequence based on strand
        let sequence_to_use = if strand_val == "+" {
            region_seq.clone()
        } else {
            reverse_complement(&region_seq)
        };

        // PAM position with bounds checking
        let pam = sequence_to_use[20..23].to_string();

        let full_seq = format!("{}{}", guide, pam);

        // debug!("Extracted genomic context: {}", sequence_to_use);
        // debug!("Debugging guide #{}: chrom={}, pos={}, strand={}", i, chrom, start_pos, strand_val);
        // debug!("Guide seq: {}, PAM seq: {}", guide, pam);

        pam_sequences.push(pam);
        sequences_with_pam.push(full_seq);
    }

    df.with_column(Series::new(PlSmallStr::from("pam"), pam_sequences))?;
    df.with_column(Series::new(PlSmallStr::from("sequence_with_pam"), sequences_with_pam))?;

    Ok(df)
}

fn extract_genomic_sequence(chromosome: &str, start: i64, end: i64) -> Result<String, Box<dyn std::error::Error>> {
    let twobit_path = project_root().join("chopchop/hg38.2bit");
    let twobit_to_fa = project_root().join("chopchop/twoBitToFa").to_string_lossy().to_string();

    let temp_output = format!("/tmp/sequence_{}_{}_{}.fa", chromosome, start, end);

    Command::new(&twobit_to_fa)
        .arg(twobit_path)
        .arg(&temp_output)
        .arg(format!("-seq={}", chromosome))
        .arg(format!("-start={}", start))
        .arg(format!("-end={}", end))
        .output()?;

    let content = std::fs::read_to_string(&temp_output)?;
    std::fs::remove_file(&temp_output)?;

    Ok(content.lines().skip(1).collect::<String>())
}


/// Reverse complements a DNA sequence
fn reverse_complement(seq: &str) -> String {
    seq.chars()
        .rev()
        .map(|c| match c {
            'A' | 'a' => 'T',
            'T' | 't' => 'A',
            'G' | 'g' => 'C',
            'C' | 'c' => 'G',
            'N' | 'n' => 'N',
            other => other, // Keep other characters as is
        })
        .collect()
}
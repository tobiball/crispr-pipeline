use polars::prelude::*;

use tracing::{debug, error, info, warn};
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

        debug!("After Gene-clean: {:>7} rows", df_annot_clean.height()); // ← NEW


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



        let uppercase_cols: Vec<Expr> = df
            .get_column_names()
            .iter()
            .filter(|name| {
                let lower = name.to_lowercase();
                lower.contains("sgrna") || lower.contains("sequence") || lower.contains("pam")
            })
            .map(|name| {
                let name_str: &str = name.as_ref();   // now unambiguous

                col(name_str)                         // S = &str implements Into<PlSmallStr>
                    .str()
                    .to_uppercase()
                    .alias(name_str)                  // same `&str` again
            })
            .collect();



        let df = df.lazy()
            .with_columns(uppercase_cols)
            .collect()?;

        debug!("After all filters: {:>7} rows", df.height());            // ← NEW



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


pub fn augment_guides_with_pam(mut df: DataFrame) -> PolarsResult<DataFrame> {
    info!("Augmenting guides with PAM sequences with genomic context for debugging purposes");

    // grab columns
    let chr_col    = df.column("chromosome")?.str()?;
    let start_col  = df.column("start")?.i64()?;
    let strand_col = df.column("strand")?.str()?;
    let guide_col  = df.column("sgRNA")?.str()?;

    // prepare output vectors
    let mut sequences_deepspcas9 = Vec::with_capacity(df.height());
    let mut pam_sequences        = Vec::with_capacity(df.height());
    let mut sequences_with_pam   = Vec::with_capacity(df.height());

    for i in 0..df.height() {
        // pull row values, or skip with empties
        let (chrom, pos, strand_val, guide) = match (
            chr_col.get(i),
            start_col.get(i),
            strand_col.get(i),
            guide_col.get(i),
        ) {
            (Some(c), Some(p), Some(s), Some(g)) => (c, p, s, g),
            _ => {
                continue;
            }
        };

        // — 1) build 30 nt window for DeepSpCas9: [pos-4 .. pos+26)
        let raw30 = if strand_val == "+" {
            extract_genomic_sequence(chrom, pos - 5, pos + 25).unwrap()  // ← Shift by 1 for positive
        } else {
            extract_genomic_sequence(chrom, pos - 4, pos + 26).unwrap()  // ← Keep original for negative
        };
        let seq30 = if strand_val == "+" {
            raw30.clone()
        } else {
            reverse_complement(&raw30)
        };

        // — 2) run your *old* PAM‐extraction on a 23 nt slice [pos-1 .. pos+22)
        let region_seq = extract_genomic_sequence(chrom, pos - 1, pos + 22)
            .unwrap();
        let sequence_to_use = if strand_val == "+" {
            region_seq.clone()
        } else {
            reverse_complement(&region_seq)
        };
        // slice out bases 20..23 of that 23 nt
        let pam = if sequence_to_use.len() >= 23 {
            sequence_to_use[20..23].to_string()
        } else {
            String::new()
        };

        let guide_fixed = extend_to_20nt(guide, &sequence_to_use).unwrap();


        let full23 = format!("{}{}", guide_fixed, pam);

        // — 3) push into our columns
        sequences_deepspcas9.push(seq30);
        pam_sequences.push(pam);
        sequences_with_pam.push(full23);
    }

    df = df
        .filter(&(&chr_col.is_not_null() & &start_col.is_not_null()
            & strand_col.is_not_null() & guide_col.is_not_null()))?;

    use polars::prelude::*;

    // helper: turn Vec<String> → Series (use an empty name; we’ll alias later)
    let seq30  = Series::new(PlSmallStr::from(""), sequences_deepspcas9);
    let pams   = Series::new(PlSmallStr::from(""), pam_sequences);
    let seq23  = Series::new(PlSmallStr::from(""), sequences_with_pam);

    df = df.lazy()
        .with_columns([
            lit(seq30.clone()).alias("sequence_deepspcas9"),
            lit(pams.clone()).alias("pam"),
            lit(seq23.clone()).alias("sequence_with_pam"),
        ])
        .with_columns([                                   // now upper-case them
            col("pam").str().to_uppercase().alias("pam"),
            col("sequence_deepspcas9").str().to_uppercase().alias("sequence_deepspcas9"),
            col("sequence_with_pam").str().to_uppercase().alias("sequence_with_pam"),
        ])
        .collect()?;



    // drop your helper if you like
    Ok(df.drop("split_parts")?)
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

/// Extend a 19-nt guide to a full 20-nt guide by **prepending** the first
/// genomic base (index 0 of the 23-nt context = 5′ end of protospacer).
///
/// Returns `Some(fixed)` on success, or `None` when the context is too
/// short / ambiguous (‘N’), so the caller can skip that row.
#[inline]
fn extend_to_20nt(guide: &str, context23: &str) -> Option<String> {
    if guide.len() == 19 {
        match context23.chars().nth(0) {
            Some(b) if !matches!(b, 'N' | 'n') => {
                let mut g = String::with_capacity(20);
                g.push(b);        // prepend the missing 5′ base
                g.push_str(guide);
                Some(g)
            }
            _ => {
                warn!("Could not derive 1ˢᵗ base for 19-nt guide {}", guide);
                None
            }
        }
    } else {
        Some(guide.to_owned())
    }
}



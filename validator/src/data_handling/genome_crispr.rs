use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use polars::prelude::*;
use polars::lazy::dsl::*;
use tracing::{debug, error, info, warn};

// Suppose you have a `Dataset` trait defined somewhere in your code:
use crate::models::{polars_err, Dataset};

// Some helper for reading CSV:
use crate::helper_functions::read_csv;

/// A struct representing the GenomeCRISPR dataset
pub struct GenomeCrisprDatasets {
    pub path: String,
}

/// Rename columns from whatever the CSV actually has
/// to what our code internally expects.
/// Adjust these to match your real CSV headers!
fn rename_columns(mut df: DataFrame) -> PolarsResult<DataFrame> {
    df.rename("chr", PlSmallStr::from("chromosome"))?;
    df.rename("sequence", PlSmallStr::from("sgRNA"))?;
    df.rename("symbol", PlSmallStr::from("Gene"))?;


    Ok(df)
}

/// Filter down to viability screens (dropout / viability after X days)
fn create_sub_dataframe(df_original: DataFrame) -> PolarsResult<DataFrame> {
    let df_sub = df_original
        .lazy()
        .filter(
            col("condition").eq(lit("viability"))
                .or(col("condition").eq(lit("viability after 36 days")))
        )
        .filter(
            col("rc_final").is_not_null()
        )
        .collect()?;

    Ok(df_sub)
}

use polars::prelude::*;
use polars::prelude::LiteralValue::Int;
use crate::mageck_processing::run_mageck_pipeline;

fn rescale_depletion_column(df: DataFrame) -> PolarsResult<DataFrame> {
    let df_rescaled = df
        .lazy()
        .with_column(
            when(col("effect").lt(lit(0.0)))  // For negative values
                .then(
                    (col("effect") / lit(-10.0)) * lit(100.0)
                )
                .otherwise(lit(0.0))  // For zero and positive values
                .alias("effect"),
        )
        .collect()?;

    Ok(df_rescaled)
}

fn ensure_chr_prefix(df: DataFrame) -> PolarsResult<DataFrame> {
    let df_fixed = df
        .lazy()
        .with_column(
            when(col("chromosome").str().starts_with(lit("chr")))
                .then(col("chromosome"))
                .otherwise(lit("chr") + col("chromosome"))
                .alias("chromosome")
        )
        .collect()?;

    Ok(df_fixed)
}

impl Dataset for GenomeCrisprDatasets {
    fn load(&self) -> PolarsResult<DataFrame> {
        info!("Reading GenomeCRISPR data from: {}", &self.path);

        // Use the specialized reader instead of generic read_csv
        let df_original = match read_csv(&self.path) {
            Ok(df) => df,
            Err(e) => {
                error!("Failed to read data CSV: {}", e);
                return Err(e);
            }
        };
        debug!("Loaded {} rows", df_original.shape().0);

        let df_renamed = rename_columns(df_original)?;

        let df_filtered = create_sub_dataframe(df_renamed)?;
        debug!("After filter, {} rows", df_filtered.shape().0);

        let df_chr_fixed = ensure_chr_prefix(df_filtered)?;

        let df_final = rescale_depletion_column(df_chr_fixed)?;
        debug!(
            "After rescaling depletion, final shape: {} rows, {} cols",
            df_final.shape().0,
            df_final.shape().1
        );

        debug!("df after reading = {:?}", df_final.head(Some(5)));

        Ok(df_final)
    }


    fn mageck_efficency_scoring(df: DataFrame) -> PolarsResult<DataFrame> {
        debug!("Starting MAGeCK processing for GenomeCRISPR dataset");
        debug!("Original dataframe shape: {:?}", df.shape());

        // Get the list of unique experiment IDs
        // Assuming there's a column that identifies different screens, like "pubmed"
        // If there's no explicit column, we might need to use a combination of columns
        let binding = df.column("pubmed")?
            .unique()?
            .cast(&DataType::String)?;
        let pubmeds = binding
            .str()?
            .into_iter()
            .filter_map(|x| x)
            .collect::<Vec<_>>();

        info!("Found {} unique experiments/screens to process", pubmeds.len());

        // Process each experiment separately and collect results
        let mut results_dfs = Vec::with_capacity(pubmeds.len());

        for (idx, exp_id) in pubmeds.iter().enumerate() {
            info!("Processing pubmed_id {}/{}: {}", idx + 1, pubmeds.len(), exp_id);

            // Filter dataframe to just this experiment
            let exp_df = df.clone().lazy()
                .filter(col("pubmed").eq(lit(exp_id.parse::<i64>().unwrap())))
                .collect()?;


            // Create a unique output prefix for this experiment
            let output_prefix = format!("./mageck_processing_artifacts/genomecrispr_pubmed_{}", exp_id.replace(" ", "_"));

            // Run MAGeCK for this experiment
            let result_df = match run_mageck_pipeline(
                exp_df,
                "/home/mrcrispr/crispr_pipeline/mageck/mageck_venv/bin/mageck",
                &output_prefix,
                &["rc_final".to_string()],
                &["rc_initial".to_string()],
                "sgRNA",
                "Gene",
                &["rc_initial", "rc_final"]
            ) {
                Ok(df) => df,
                Err(e) => {
                    error!("Error processing pubmed_id {}: {}", exp_id, e);
                    continue; // Skip this experiment and continue with the next
                }
            };

            results_dfs.push(result_df);
            info!("Completed MAGeCK analysis for experiment: {}", exp_id);
        }

        // Combine all results
        if results_dfs.is_empty() {
            error!("No experiment results were successfully processed");
            return Err(PolarsError::ComputeError("No experiment results to combine".into()));
        }

        // Start with the first dataframe
        let mut combined_df = results_dfs.remove(0);

        // Append the rest
        for df in results_dfs {
            combined_df = match combined_df.vstack(&df) {
                Ok(df) => df,
                Err(e) => {
                    error!("Error combining result dataframes: {}", e);
                    return Err(e);
                }
            };
        }

        info!("Successfully combined {} experiment results", pubmeds.len());
        debug!("Final combined dataframe shape: {:?}", combined_df.shape());

        Ok(combined_df)
    }
}


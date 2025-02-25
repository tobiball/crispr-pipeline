use polars::prelude::*;
use polars::lazy::dsl::*;
use tracing::{debug, error, info};

// Suppose you have a `Dataset` trait defined somewhere in your code:
use crate::models::Dataset;

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
                .alias("efficacy"),
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


        debug!("{:?}", df);

        run_mageck_pipeline(
            df,
            "/home/mrcrispr/crispr_pipeline/mageck/mageck_venv/bin/mageck",
            "./validator/my_experiment",
            &["rc_final".to_string()],
            &["rc_initial".to_string()],
            "sgRNA",
            "Gene",
            &["rc_initial", "rc_final"]
        )

    }
}

// fn remove_quotes_from_column_names(mut df: DataFrame) -> PolarsResult<DataFrame> {
//     // Current names
//     let binding = df.clone();
//     let old_names: Vec<_> = binding.get_column_names_str().iter().map(|s| *s).collect();
//
//     // Calculate the new names (removing quotes, trimming)
//     let new_names: Vec<String> = old_names
//         .iter()
//         .map(|name| name.replace('"', "").trim().to_owned())
//         .collect();
//
//     // Rename columns one by one
//     for (old, new) in old_names.iter().zip(new_names.iter()) {
//         df.rename(old, PlSmallStr::from(new))?;
//     }
//     Ok(df)
// }















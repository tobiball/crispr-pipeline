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
    // Example renames; adapt to your CSV's columns:
    // - `chr`        -> `chromosome`
    // - `Name`       -> `sgRNA`
    // - `start`      -> `position`
    // - `effect`     -> `effect`
    //   (If your CSV already calls it `effect`, no rename needed)
    df.rename("chr", PlSmallStr::from("chromosome"))?;
    df.rename("sequence", PlSmallStr::from("sgRNA"))?;
    // df.rename("start", PlSmallStr::from("position"))?;
    // If your CSV uses "effect" as-is, you can skip or rename:
    // df.rename("effect", "effect")?;

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
        .collect()?;

    Ok(df_sub)
}

/// Rescale only *negative* effect values from [−9..0] to [1..100].
/// Positive values (≥0) remain unchanged.
fn rescale_depletion_column(df: DataFrame) -> PolarsResult<DataFrame> {
    let df_rescaled = df
        .lazy()
        .with_column(
            when(col("effect").lt(lit(0.0)))
                .then(
                    // scaled = 1 - 11 * effect
                    lit(1.0) - (lit(11.0) * col("effect"))
                )
                .otherwise(col("effect"))    // or you might choose a different fallback
                .alias("effect_depletion_1to100")
        )
        .collect()?;

    Ok(df_rescaled)
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

        // 2) Rename columns for consistency
        let df_renamed = rename_columns(df_original)?;

        // 3) Filter down to viability screens only
        let df_filtered = create_sub_dataframe(df_renamed)?;
        debug!("After filter, {} rows", df_filtered.shape().0);

        // 4) Rescale negative (depletion) effect values from [−9..0] to [1..100]
        let df_final = rescale_depletion_column(df_filtered)?;
        debug!(
            "After rescaling depletion, final shape: {} rows, {} cols",
            df_final.shape().0,
            df_final.shape().1
        );

        // Optionally, do more analysis (group_by, aggregations, etc.)
        // let grouped = df_final.lazy() ...
        // or just return df_final

        Ok(df_final)
    }
}

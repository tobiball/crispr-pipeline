use polars::prelude::*;
use tracing::{debug, error, info};
use crate::helper_functions::read_csv;

// Constants used in calculations
const PLUS_STRAND_OFFSET: i32 = 16;
const MINUS_STRAND_OFFSET: i32 = 5;

/// Trait representing a dataset in the CRISPR pipeline.
pub trait Dataset {
    /// Loads and preprocesses the dataset, returning a DataFrame.
    fn load(&self) -> PolarsResult<DataFrame>;
}

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

        let total_guides = df_gene_guide_efficiencies.height();

        info!("Number of sgRNAs to check: {}", total_guides);

        // Modify the DataFrame processing part to use two potential windows
        let df = df_gene_guide_efficiencies
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
            .select(&[col("*")])
            .collect()?;

        debug!("DataFrame after processing: {:?}", df);

        Ok(df)
    }
}


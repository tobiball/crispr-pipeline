use polars::prelude::*;
use tracing::{error, info};
use crate::helper_functions::read_csv;
use crate::models::Dataset;


pub struct GenomeCrisprDatsets {
    pub(crate) path: String,
}

impl Dataset for GenomeCrisprDatsets {
    fn load(&self) -> PolarsResult<DataFrame> {


        info!("Reading data from {}", &self.path);
        let df = match read_csv(&self.path) {
            Ok(df) => df,
            Err(e) => {
                error!("Failed to read data CSV: {}", e);
                return Err(e);
            }
        };


        let total_guides = df.height();

        info!("Number of sgRNAs to check: {}", total_guides);

        // Get the "name" column as a Series
        let name_series = df.column("condition")?;

        // Get the unique values in the "name" column
        let unique_names = name_series.unique()?;


        // If you need them as a Vec<String>, for example:
        let unique_name_vec: Vec<String> = unique_names
            .str()?
            .into_no_null_iter()
            .map(|s| s.to_string())
            .collect();

        println!("Unique name vector: {:?}", unique_name_vec);

        // // Modify the DataFrame processing part to use two potential windows
        // let df = df_gene_guide_efficiencies
        //     .lazy()
        //     .with_column(
        //         col("GenomeAlignment")
        //             .str()
        //             .split(lit("_"))
        //             .alias("split_parts"),
        //     )
        //     .with_columns(vec![
        //         col("split_parts")
        //             .list()
        //             .get(lit(0), false)
        //             .alias("chromosome"),
        //         col("split_parts")
        //             .list()
        //             .get(lit(1), false)
        //             .alias("position"),
        //         col("split_parts")
        //             .list()
        //             .get(lit(2), false)
        //             .alias("strand"),
        //     ])
        //     // Cast 'position' to Int64 for arithmetic operations
        //     .with_column(col("position").cast(DataType::Int64))
        //     // Calculate start and end based on strand
        //     .with_columns(vec![
        //         (when(col("strand").eq(lit("+")))
        //             .then(col("position") - lit(PLUS_STRAND_OFFSET))
        //             .otherwise(col("position") - lit(MINUS_STRAND_OFFSET)))
        //             .alias("start"),
        //         (when(col("strand").eq(lit("+")))
        //             .then(col("position") - lit(PLUS_STRAND_OFFSET - 1))
        //             .otherwise(col("position") - lit(MINUS_STRAND_OFFSET - 1)))
        //             .alias("end"),
        //     ])
        //     .select(&[col("*")])
        //     .collect()?;
        //
        // debug!("DataFrame after processing: {:?}", df);

        Ok(df)
    }
}


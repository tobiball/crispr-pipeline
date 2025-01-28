use polars::prelude::*;
use tracing::{debug, error, info};
use crate::helper_functions::read_csv;
use anyhow::Error;


pub struct GenomeCrisprDatsets {
    pub(crate) path: String,
}

use polars::prelude::*;
use crate::models::Dataset;
use polars::prelude::*;
use polars::lazy::dsl::*;

fn create_sub_dataframe(df_original: DataFrame) -> PolarsResult<DataFrame> {
    let df_sub = df_original
        .lazy()
        .filter(
            col("condition").eq(lit("viability"))
                .or(col("condition").eq(lit("viability after 36 days")))
                .or(col("condition").eq(lit("viability after 21 days")))
                .or(col("condition").eq(lit("viability after 25 days")))
        )
        .collect()?;

    Ok(df_sub)
}


impl Dataset for GenomeCrisprDatsets {
    fn load(&self) -> PolarsResult<DataFrame> {
        info!("Reading data from {}", &self.path);

        // Read the CSV once
        let df_original = match read_csv(&self.path) {
            Ok(df) => df,
            Err(e) => {
                error!("Failed to read data CSV: {}", e);
                return Err(e);
            }
        };
        debug!("{}", &df_original);
        // Work with `df_original`
        let total_guides = df_original.height();
        info!("Number of sgRNAs to check: {}", total_guides);



        let df_filtered = create_sub_dataframe(df_original)?;

        debug!("{:?}", df_filtered);

        // let name_series = df_filtered.column("condition")?;
        // let unique_names = name_series.unique()?;
        // let unique_name_vec: Vec<String> = unique_names
        //     .str()?
        //     .into_no_null_iter()
        //     .map(|s| s.to_string())
        //     .collect();
        // debug!("Unique name vector: {:?}", unique_name_vec);

        let df_result = df_filtered
            .clone()
            .lazy()
            .group_by([col("pubmed")])
            .agg([col("condition").unique().alias("unique_conditions")])
            .collect()?;

        for i in df_result["unique_conditions"].phys_iter() {
            debug!("{:?}", i);
        }

        // Return the original DataFrame, or lazy_result, or whichever you need
        Ok(df_filtered)
    }
}



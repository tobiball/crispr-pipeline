use polars::prelude::*;
use tracing::{error, info};
use crate::helper_functions::read_csv;
use anyhow::Error;


pub struct GenomeCrisprDatsets {
    pub(crate) path: String,
}

use polars::prelude::*;
use crate::models::Dataset;

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
    println!("{}", &df_original);
        // Work with `df_original`
        let total_guides = df_original.height();
        info!("Number of sgRNAs to check: {}", total_guides);

        // Get the "condition" column
        let name_series = df_original.column("condition")?;
        let unique_names = name_series.unique()?;
        let unique_name_vec: Vec<String> = unique_names
            .str()?
            .into_no_null_iter()
            .map(|s| s.to_string())
            .collect();
        println!("Unique name vector: {:?}", unique_name_vec);

        let df_result = df_original
            .clone()
            .lazy()
            .group_by([col("pubmed")])
            .agg([col("condition").unique().alias("unique_conditions")])
            .collect()?;

        println!("{:?}", df_result);

        for i in df_result["unique_conditions"].phys_iter() {
            println!("{:?}", i);
        }

        // Return the original DataFrame, or lazy_result, or whichever you need
        Ok(df_original)
    }
}


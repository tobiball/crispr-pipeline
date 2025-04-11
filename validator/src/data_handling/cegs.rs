use polars::prelude::*;
use std::fs::File;
use polars::prelude::{CsvReadOptions, SerReader};
use tracing::{error, info};
use crate::helper_functions::{read_csv, read_txt};
use crate::models::Dataset;

pub struct Cegs {
    pub(crate) path: String,
}


impl Dataset for Cegs {

    fn load(&self) -> PolarsResult<DataFrame> {


        info!("Reading data from {}", &self.path);
        let df = match read_txt(&self.path) {
            Ok(df) => df,
            Err(e) => {
                error!("Failed to read ceg CSV: {}", e);
                return Err(e);
            }
        };


        Ok(df)}

    fn augment_guides(df: DataFrame) -> PolarsResult<DataFrame> {
        Ok((df))
    }

    fn validate_columns(df: &DataFrame, dataset_name: &str) -> PolarsResult<()> {
        Ok(())
    }

    fn mageck_efficency_scoring(df: DataFrame) -> PolarsResult<DataFrame> {
        Ok((df))
    }
}
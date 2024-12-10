use polars::prelude::*;
use std::fs::File;
use polars::prelude::{CsvReadOptions, SerReader};
use tracing::{error, info};
use crate::data_handling::genome_crispr::GenomeCrisprDatsets;
use crate::helper_functions::read_csv;
use crate::models::Dataset;

pub struct Cegs {
    pub(crate) path: String,
}


impl Dataset for Cegs {
    fn load(&self) -> PolarsResult<DataFrame> {


        info!("Reading data from {}", &self.path);
        let df = match read_csv(&self.path) {
            Ok(df) => df,
            Err(e) => {
                error!("Failed to read ceg CSV: {}", e);
                return Err(e);
            }
        };


        Ok(df)}
}
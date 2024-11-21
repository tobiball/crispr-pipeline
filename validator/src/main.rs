use std::fs;
use polars::prelude::*;
use std::path::PathBuf;
use prediction_tools::chopchop_integration::{parse_chopchop_results, run_chopchop, ChopchopOptions};
use data_handling::avana_depmap::get_df;
use tracing::{debug, error, info};
use tracing_subscriber::{fmt, EnvFilter};
use crate::helper_functions::read_csv;
use crate::prediction_tools::chopchop_integration::run_chopchop_meta;

mod tool_evluation;
mod models;
mod data_handling;
mod prediction_tools;
mod helper_functions;



// Helper function to read CSV files


fn main() -> PolarsResult<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting the CRISPR pipeline");


    let df = get_df()?.extract_dataframe();

    run_chopchop_meta(df);





    info!("CRISPR pipeline completed successfully");
    Ok(())
}

use crate::data_handling::cegs::Cegs;
use polars::prelude::*;
use tracing::info;
use tracing_subscriber::EnvFilter;
use crate::data_handling::avana_depmap::AvanaDataset;
use crate::data_handling::genome_crispr::GenomeCrisprDatsets;

use crate::models::Dataset;
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
    let cegs = Cegs {
        path: "/home/mrcrispr/crispr_pipeline/data/cegv2.txt".to_string()
    };
    let genomecrispr_datasets = GenomeCrisprDatsets {
        path: "/home/mrcrispr/crispr_pipeline/data/genomecrispr/GenomeCRISPR_full05112017_brackets.csv".to_string(),
    };
    let avana_dataset = AvanaDataset {
        efficacy_path: "/home/mrcrispr/crispr_pipeline/data/depmap/CRISPRInferredGuideEfficacy_23Q4.csv".to_string(),
        guide_map_path: "/home/mrcrispr/crispr_pipeline/data/depmap/AvanaGuideMap_23Q4.csv".to_string(),
    };

    let cegs = cegs.load();
    let _df_gc = genomecrispr_datasets.load()?;
    let df = avana_dataset.load()?;


    // run_chopchop_meta(df).expect("TODO: panic message");





    info!("CRISPR pipeline completed successfully");
    Ok(())
}

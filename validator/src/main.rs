use crate::data_handling::cegs::Cegs;
use polars::prelude::*;
use tracing::info;
use tracing_subscriber::EnvFilter;
use crate::data_handling::genome_crispr::GenomeCrisprDatasets;

use crate::data_handling::avana_depmap::{AvanaDataset};
use crate::helper_functions::{project_root, read_csv, write_config_json};
use crate::mageck_processing::{run_mageck_test, write_mageck_input, MageckOptions};
use crate::models::{polars_err, Dataset};
use crate::prediction_tools::chopchop_integration::run_chopchop_meta;
use crate::prediction_tools::crispor_integration::run_crispor_meta;
use crate::prediction_tools::deepcrispr_integration::run_deepcrispr_meta;
use crate::prediction_tools::terst;

mod models;
mod data_handling;
mod prediction_tools;
mod helper_functions;
mod mageck_processing;
mod analysis;
// Helper function to read CSV files


fn main() -> PolarsResult<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting the CRISPR pipeline");

    let binding = project_root();
    let project_root = binding.to_str().unwrap();
    write_config_json(project_root).map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;



    let cegs = Cegs {
        path: "./data/cegv2.txt".to_string()
    };
    let genomecrispr_datasets = GenomeCrisprDatasets {
        path: "./data/genomecrispr/GenomeCRISPR_full05112017_brackets.csv".to_string(),

    };
    let avana_dataset = AvanaDataset {
        efficacy_path: "./data/depmap/CRISPRInferredGuideEfficacy_23Q4.csv".to_string(),
        guide_map_path: "./data/depmap/AvanaGuideMap_23Q4.csv".to_string(),
    };

    let cegs = cegs.load()?;
    // let df = genomecrispr_datasets.load_validated("genome_crispr_short", cegs)?;
    let df = avana_dataset.load_validated("depmap", cegs)?;

    // analysis::umap::run_kmer_analysis(&df, "sgRNA", "efficacy", "kmer_analysis").map_err(polars_err)?;

    let df_n = read_csv("./processed_data/chopchop_avana-depmap_results.csv")?;

    // run_crispor_meta(df_n,"avana-depmap");







    //
    // run_chopchop_meta(df.clone(), "genome_cirspr_short")?;


    // Run DeepCRISPR
    // run_deepcrispr_meta("./processed_data/chopchop_avana-depmap_results.csv", "avana-depmap")?;

    // let tools = vec!["chopchop_efficiency","deepcrispr_prediction"];
    //
    // tool_evluation::analyze_tool_results("./DeepCRISPR/genomecrispr_predictions_ttest.csv", "avana-depmap",tools)?;
    //
    //

    terst::terst(df_n);




    info!("CRISPR pipeline completed successfully");
    Ok(())
}

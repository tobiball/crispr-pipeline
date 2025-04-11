#![allow(unused)]



use crate::analysis::prediction_evaluation::evaluate_prediction_tools;
use crate::prediction_tools::tko_pssm;
use crate::analysis::guide_analysis::generate_high_efficacy_low_prediction_df;
use crate::analysis::within_gene::{plot_stripplot_for_tool, plot_tool_vs_efficacy};
use crate::analysis::tool_evaluation::analyze_tool_results;
use crate::analysis::tool_comparison;
use crate::data_handling::cegs::Cegs;
use polars::prelude::*;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use crate::analysis::efficacy_analysis_integration::analyze_efficacy_distribution;
use crate::analysis::key_diagrams_integration::run_key_diagrams;
use crate::analysis::sankey::create_prediction_sankey;
use crate::analysis::tool_comparison_integration::{compare_tools, get_tool_rankings};
use crate::analysis::within_gene::{evaluate_scores, within_gene_analysis};
use crate::data_handling::any_dataset::AnyDataset;
use crate::data_handling::genome_crispr::GenomeCrisprDatasets;

use crate::data_handling::avana_depmap::{AvanaDataset};
use crate::helper_functions::{dataframe_to_csv, project_root, read_csv, write_config_json};
use crate::mageck_processing::{run_mageck_test, write_mageck_input, MageckOptions};
use crate::models::{polars_err, Dataset};
use crate::prediction_tools::chopchop_integration::run_chopchop_meta;
use crate::prediction_tools::deepcrispr_integration::run_deepcrispr_meta;
use crate::prediction_tools::crispor_integration;
use crate::prediction_tools::transcrispr::run_transcrispr_meta;

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
    // let df = avana_dataset.load_validated("avana-depmap", cegs)?;
    // run_chopchop_meta(df.clone(), "avana-depmap")?;



    // let df = AnyDataset{
    //     path: "./processed_data/crispor_avana-depmap.csv".to_string(),
    // }.load()?;
    //
    // run_deepcrispr_meta("./processed_data/crispor_avana-depmap.csv", "avana-depmap");
    //


    let mut df = AnyDataset{
        path: "./processed_df.csv".to_string(),
    }.load()?;
    //
    //
    // analyze_efficacy_distribution(&mut df, "./general_results/efficacy_analysis_results")?;
    //

    // let df = tko_pssm::run_pssm_meta(df.clone(), "sgRNA", "avana-depmap")?;
    //
    //
    // let mut df = run_transcrispr_meta(
    //     AnyDataset{
    //         path: "./processed_data/deepcrispr_output_avana-depmap.csv".to_string(),
    //     }.load()?,
    //     "avana-depmap"
    // )?;
    // info!("TransCRISPR analysis completed");

    // dataframe_to_csv(&mut df, "processed_df.csv", true);

    // Define the prediction tool columns to compare
    // crispor_integration::run_crispor_meta(df, "avana-depmap")?;


    // Select the prediction tools to evaluate
    let tools_to_evaluate: Vec<&str> = vec![
        "deepcrispr_prediction",     // DeepCRISPR score
        "transcrispr_prediction",    // TransCRISPR score
        "mitSpecScore",              // MIT specificity score
        "cfdSpecScore",              // CFD specificity score
        "Doench '16-Score",          // Doench 2016 score
        "Moreno-Mateos-Score",       // Moreno-Mateos score
        "Doench-RuleSet3-Score",     // Doench RuleSet3 score
        "Out-of-Frame-Score",        // Out-of-Frame score
        "Lindel-Score",              // Lindel score
        // "pssm_score",                // PSSM model score
        // "chopchop_efficiency",       // CHOPCHOP score
    ];

    read_csv("processed_df.csv");



    // // Generate all key diagrams at once
    // info!("Generating key diagrams for prediction tools...");
    // let diagram_paths = run_key_diagrams(&mut df, "./visualization_results", tools_to_evaluate.clone(), "efficacy", "title")?;


    // Define evaluation parameters
    let efficacy_column = "efficacy";
    let poor_threshold = 50.0;
    let good_threshold = 75.0;
    let min_good_coverage = 0.75; // Require tools to identify at least 75% of good guides
    let output_dir = "./prediction_evaluation_results";

    // Run the evaluation
    evaluate_prediction_tools(
        &df,
        &tools_to_evaluate,
        efficacy_column,
        poor_threshold,
        good_threshold,
        min_good_coverage,
        output_dir
    )?;



    // // Run the comparison analysis
    //
    // println!("Tools to evaluate: {:?}", tools_to_compare);
    //
    // println!("DataFrame columns: {:?}", df.get_column_names());
    //
    //
    //
    // analyze_tool_results(df.clone(), "avana-depmap", tools_to_compare.clone());
    //
    //
    // let df_an = within_gene_analysis(df.clone(), tools_to_compare.clone())?;
    //
    //
    //
    // info!("Running comparative analysis of prediction tools...");
    //
    // for cutoff in vec![50,60,70,80,90,95] {
    //     tool_comparison::compare_roc_curves(&df, &tools_to_compare.clone(), "efficacy", cutoff as f64, "./general_results") ?;
    // }
    //
    // info!("Comparative analysis completed");
    //
    //
    // // Evaluate each score's ROC/AUC against the “guide_quality” boolean column
    // let roc_results = evaluate_scores(&df_an, &tools_to_compare).map_err(|e| polars_err(Box::new(e)))?;
    //
    //
    // // Print out AUC for each tool
    // for (tool, roc) in roc_results {
    //     println!("Tool: {:<25}  AUC: {:.4}", tool, roc.auc);
    // }
    //
    // for tool in &tools_to_compare {
    //     let out_path = format!("./results_{}/", tool);
    //     // plot_tool_vs_efficacy(&df_an, tool, format!("{}tool_vs_efficacy.png", &out_path))?;
    //     plot_stripplot_for_tool(&df_an, tool,format!("{}stripplot_within_gene.png", &out_path))?;
    //
    //     println!("Wrote {}", out_path);
    // }
    //
    // generate_high_efficacy_low_prediction_df(
    //     &df,
    //     tools_to_compare.clone(),
    //     "./general_results_guide_discrepancies.csv",
    //     Some("./filtered_guide_discrepancies.csv"), // Optional filtered output path
    //     Some(2) // Minimum number of mismatches
    // )?;

    // info!("Analyzing how prediction tools categorize guides into efficacy bins");





    // // Create output directory for evaluation results
    // let prediction_eval_dir = "./prediction_evaluation_results";
    // std::fs::create_dir_all(prediction_eval_dir).expect("Failed to create directory");
    //
    // let efficacy_thresholds = PredictionThresholds {
    //     poor_threshold: 50.0,  // < 50 = poor guides
    //     good_threshold: 75.0,  // >= 75 = good guides
    // };
    //
    //
    // let poor_threshold = 50.0; // Guides with efficacy < 50 are considered poor
    // let good_threshold = 75.0; // Guides with efficacy >= 75 are considered good
    //
    // // Run comparison of all tools
    // info!("Running categorical prediction evaluation for all tools...");
    // compare_prediction_tools(
    //     &df,
    //     tools_to_evaluate,
    //     "efficacy",
    //     efficacy_thresholds,
    //     prediction_eval_dir
    // )?;
    //
    //
    // info!("Prediction tool categorization analysis complete");
    //
    //
    // // info!("Running comparative analysis with heatmaps...");
    // let heatmap_dir = "./general_results/heatmaps";
    // analysis::heatmap_comparison::generate_tool_comparison_heatmaps(&df, tools_to_compare.clone(), heatmap_dir)?;

    // ---- Add these lines to your main.rs file after your existing analysis code ----

    // Generate Sankey diagram to visualize prediction accuracy
    // info!("Generating Sankey diagram for prediction tools...");

    // Use the same tools list you defined earlier in your code for consistency
    // This ensures the Sankey diagram includes the same tools as your other analyses
    // NEW: Add Sankey diagram analysis
    // info!("Running Sankey diagram analysis...");
    // create_prediction_sankey(&df, "efficacy", &tools_to_compare, "general_results")?;
    //
    // info!("Sankey visualization completed");



    info!("CRISPR pipeline completed successfully");
    Ok(())
}

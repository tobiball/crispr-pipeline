#![allow(unused)]

use serde_json::from_reader;
use crate::analysis::roc;
use crate::analysis::within_gene;
use std::fs::File;
use crate::prediction_tools::deepspcas9_integration;
use crate::analysis::prediction_evaluation::evaluate_prediction_tools;
use crate::prediction_tools::tko_pssm;
use crate::analysis::within_gene::{plot_stripplot_for_tool, plot_tool_vs_efficacy, run_bad_guide_summary};
use crate::data_handling::cegs::Cegs;
use polars::prelude::*;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::analysis::within_gene::{evaluate_scores, within_gene_analysis};
use crate::data_handling::any_dataset::AnyDataset;
use crate::data_handling::genome_crispr::GenomeCrisprDatasets;

use crate::data_handling::avana_depmap::{AvanaDataset};
use crate::helper_functions::{dataframe_to_csv, drop_mono_class_genes, project_root, read_csv, stratified_within_gene_balance, undersample_equal_classes, write_config_json};
use crate::mageck_processing::{run_mageck_test, write_mageck_input, MageckOptions};
use crate::models::{polars_err, Dataset};
use crate::prediction_tools::chopchop_integration::run_chopchop_meta;
use crate::prediction_tools::deepcrispr_integration::run_deepcrispr_meta;
use crate::prediction_tools::crispor_integration;
use crate::prediction_tools::transcrispr::run_transcrispr_meta;
use combination_model::fit_and_apply;
use combination_model::WeightMap;
use crate::analysis::efficacy_analysis_integration::analyze_efficacy_distribution;
use crate::combination_model::add_combined_score_column;
use crate::data_handling::tko_one::TkoScreensDataset;
use crate::logistic_cv::{apply_logistic_score, fit_logistic_cv};

mod models;
mod data_handling;
mod prediction_tools;
mod helper_functions;
mod mageck_processing;
mod analysis;
mod combination_model;
mod logistic_cv;
// Helper function to read CSV files


const DB_NAME: &str = "Avana";


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
    let avana = AvanaDataset {
        efficacy_path: "./data/depmap/CRISPRInferredGuideEfficacy_23Q4.csv".to_string(),
        guide_map_path: "./data/depmap/AvanaGuideMap_23Q4.csv".to_string(),
    };

    let tko_dataset = TkoScreensDataset {
        screen_path: "./data/tko/tko_one.xlsx".to_string().into(),
        guide_map_path:"./data/tko/tko_hg38_annotation_for_rust.txt".into()
    };

    let cegs = cegs.load()?;
    // let df = genomecrispr_datasets.load_validated("genome_crispr_short", cegs)?;
    let mut df = avana.load_validated(DB_NAME, cegs)?;
    // let mut df = tko_dataset.load_validated("tko_one", cegs)?;











    // Define evaluation parameters
    let efficacy_column = "efficacy";
    let poor_threshold = 60.0;
    let good_threshold = 90.0;
    let min_good_coverage = 0.75;
    let output_dir = "./prediction_evaluation_results";


    // let (mut df, stats) = stratified_within_gene_balance(&df, "efficacy", 90.0, Some(10.0), Some(42))?;
    // println!("{stats:#?}");


    // let mut df = undersample_equal_classes(&drop_mono_class_genes(df)?,"efficiency", good_threshold,Some(42))?;



    analyze_efficacy_distribution(&df, output_dir, poor_threshold, good_threshold, "Avana");



    df = run_deepcrispr_meta(df, DB_NAME)?;
    //
    df = run_transcrispr_meta(
            df,
        DB_NAME
    )?;

    df = deepspcas9_integration::run_deepspcas9_meta(df.clone(), "prediction")?;

    df = tko_pssm::run_pssm_meta(df.clone(), "sequence_with_pam", DB_NAME)?;

    df = crispor_integration::run_crispor_meta(df.clone(), DB_NAME)?;

    dataframe_to_csv(&mut df, "processed_avana.csv", true);
    //
    //
    //
    //







    //
    // let mut df = AnyDataset {
    //     path: "./processed_df.csv".to_string(),
    // }.load()?;

//
//
//
//
//
//
//
//
//
//
//     let rename_map = [
//         ("Doench '16-Score",       "Doench 16"),
//         ("Moreno-Mateos-Score",    "Moreno-Mateos"),
//         ("Doench-RuleSet3-Score",  "Doench RuleSet 3"),
//         ("deepcrispr_prediction",  "DeepCRISPR"),
//         // ("transcrispr_prediction", "TransCRISPR"),
//         ("deepspcas9_prediction",  "DeepSpCas9"),
//         ("pssm_score",             "TKO PSSM"),
//     ];
//
//     for &(old, new) in &rename_map {
//         if df.get_column_names().iter().any(|c| *c == old) {
//             df.rename(old, new.into())?;
//             info!("    • Renamed `{}` → `{}`", old, new);
//         }
//     }
//
//     // 5) Before cast
//     for &col_name in &["Doench 16", "Moreno-Mateos", "Doench RuleSet 3"] {
//         // compare as_str() → &str
//         if df.get_column_names().iter().any(|c| c.as_str() == col_name) {
//             let s = df
//                 .column(col_name)?
//                 .cast(&DataType::Float64)?;         // cast the Series
//             df = df.with_column(s)?.clone();                // put it back
//             info!("▶ Casted `{}` to Float64", col_name);
//         } else {
//             info!("⚠ `{}` not found, skipping cast", col_name);
//         }
//     }
//
//     // … continue with your tool list …
//     let mut tools_to_evaluate: Vec<&str> = vec![
//         "DeepCRISPR",
//         // "TransCRISPR",
//         "Doench 16",
//         "Moreno-Mateos",
//         "Doench RuleSet 3",
//         "TKO PSSM",
//         "DeepSpCas9",
//     ];
//
//     //___________________________________________________________________________
//     // let learned_weights = fit_and_apply(
//     //     &mut df,
//     //     &tools_to_evaluate,
//     //     "efficacy",
//     //     "Linear Consensus",
//     // )?;
//     // serde_json::to_writer_pretty(
//     //     File::create("weights_linear_avana.json")?, &learned_weights).unwrap();
//     //
//     // let subset_cols: Vec<String> = tools_to_evaluate
//     //     .iter()
//     //     .map(|&s| s.to_string())
//     //     .collect();
//     //
//     // df = df.drop_nulls(Some(&subset_cols))?; //todo (populate 0 values)
//     //
//     //
//     // let mut df_log = df.clone();
//     //
//     //
//     //
//     //
//     //
//     // let (log_weights, log_thr) = fit_logistic_cv(
//     //     &mut df_log,
//     //     &tools_to_evaluate,
//     //     "efficacy",
//     //     good_threshold as f64,
//     //     "Gene",                  // ← gene-ID column in your DataFrame
//     //     5,                       // 5-fold CV
//     //     0.01,                    // L2 strength
//     // )?;
//     //
//     // apply_logistic_score(
//     //     &mut df_log,
//     //     &tools_to_evaluate,
//     //     &log_weights,
//     //     "Logistic Consensus",
//     // )?;
//     // let log_col = df_log.column("Logistic Consensus")?.clone();
//     // df = df.with_column(log_col)?.clone();
//     //
//     // serde_json::to_writer_pretty(
//     //     File::create("weights_logistic_avana.json")?, &log_weights).unwrap();
//     // println!("Logistic threshold (Youden-J): {:.3}", log_thr);
//     //___________________________________________________________________________
//
//
//
//     let linear_weights: WeightMap =
//         from_reader(File::open("weights_linear_avana.json")?).unwrap();
//
//     // 2) Apply them (this appends a "Linear Consensus" column to your df)
//     add_combined_score_column(
//         &mut df,
//         &tools_to_evaluate,
//         &linear_weights,
//         "Linear Consensus",
//     )?;
//
//     // 3) Load logistic consensus weights
//     let logistic_weights: WeightMap =
//         from_reader(File::open("weights_logistic_avana.json")?).unwrap();
//
//
//     let mut df_log = df.clone();
//
//
//
//     // 4) Apply them analogously (appends "Logistic Consensus")
//     apply_logistic_score(
//         &mut df_log,
//         &tools_to_evaluate,
//         &logistic_weights,
//         "Logistic Consensus",
//     )?;
//
//     let log_col = df_log.column("Logistic Consensus")?.clone();
//     df = df.with_column(log_col)?.clone();
//
//     //___________________________________________________________________________
//
//
//     tools_to_evaluate.push("Linear Consensus");
//     tools_to_evaluate.push("Logistic Consensus");
//
//
//
//
//     let mut tools_to_evaluate = tools_to_evaluate;
//
//
//
//
//     // Run the evaluation
//     evaluate_prediction_tools(
//         &df,
//         &tools_to_evaluate,
//         efficacy_column,
//         // Some("prediction_evaluation_results/thresholds_full.csv"),
//         None,
//         poor_threshold,
//         good_threshold,
//         min_good_coverage,
//         output_dir
//     )?;
//
//
//
//     for cutoff in vec![80] {
//         roc::compare_roc_curves(&df, &tools_to_evaluate.clone(), "efficacy", cutoff as f64, "./general_results") ?;
//     }
     let df_filtered = within_gene_analysis(df.clone(), tools_to_evaluate.clone())?;
//     // for tool in &tools_to_evaluate {
//     //     let out_path = format!("./general_results/results_{}/", tool);
//     //     // plot_tool_vs_efficacy(&df_an, tool, format!("{}tool_vs_efficacy.png", &out_path))?;
//     //     plot_stripplot_for_tool(&df_filtered, tool,format!("{}stripplot_within_gene.png", &out_path))?;
//     //
//     //     println!("Wrote {}", out_path);
//     // }
//     let summary_df = within_gene::bad_guide_detection_summary(&df_filtered, tools_to_evaluate.clone())?;
//     run_bad_guide_summary(&df_filtered, tools_to_evaluate.clone(), "./results/bad_guides")?;
//
//
//
//
//
//     //_________________________________________________________________________________________________________________________________________
//     // generate_high_efficacy_low_prediction_df(
//     //     &df,
//     //     tools_to_compare.clone(),
//     //     "./general_results_guide_discrepancies.csv",
//     //     Some("./filtered_guide_discrepancies.csv"), // Optional filtered output path
//     //     Some(2) // Minimum number of mismatches
//     // )?;
//
//
//     // info!("Running categorical prediction evaluation for all tools...");
//     // compare_prediction_tools(
//     //     &df,
//     //     tools_to_evaluate,
//     //     "efficacy",
//     //     efficacy_thresholds,
//     //     prediction_eval_dir
//     // )?;
//
//     // let heatmap_dir = "./general_results/heatmaps";
//     // analysis::heatmap_comparison::generate_tool_comparison_heatmaps(&df, tools_to_compare.clone(), heatmap_dir)?;
//
//     // create_prediction_sankey(&df, "efficacy", &tools_to_compare, "general_results")?;
//
//
//
//     // let corr_df = df
//     //     .lazy()
//     //     .select(
//     //         tools_to_evaluate
//     //             .iter()
//     //             .enumerate()
//     //             .flat_map(|(i, &a)| {
//     //                 tools_to_evaluate[i..].iter().map(move |&b| {
//     //                     pearson_corr(col(a), col(b)).alias(&format!("{a}_vs_{b}"))
//     //                 })
//     //             })
//     //             .collect::<Vec<_>>(),
//     //     )
//     //     .collect()?;
//     //
//     // println!("{corr_df}");
//     // println!("{corr_df}");
//
//     info!("CRISPR pipeline completed successfully");
    Ok(())
 }

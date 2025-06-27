#![allow(unused)]

use serde_json::from_reader;
use crate::analysis::roc;
use crate::analysis::within_gene;
use std::fs::File;
use crate::prediction_tools::deepspcas9_integration;
use crate::analysis::prediction_evaluation::evaluate_prediction_tools;
use crate::prediction_tools::tko_pssm;
use crate::analysis::within_gene::{plot_stripplot_for_tool, plot_tool_vs_efficacy, run_statistical_bad_guide_analysis};
use crate::data_handling::cegs::Cegs;
use polars::prelude::*;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::analysis::within_gene::{evaluate_scores};
use crate::data_handling::any_dataset::AnyDataset;
use crate::data_handling::genome_crispr::GenomeCrisprDatasets;
use crate::data_handling::avana_depmap::{AvanaDataset};
use crate::helper_functions::{dataframe_to_csv, drop_mono_class_genes, project_root, read_csv, stratified_within_gene_balance, stratified_within_gene_balance_with_margin, undersample_equal_classes, write_config_json, ModelColors};
use crate::mageck_processing::{run_mageck_test, write_mageck_input, MageckOptions};
use crate::models::{polars_err, Dataset};
use crate::prediction_tools::chopchop_integration::run_chopchop_meta;
use crate::prediction_tools::deepcrispr_integration::run_deepcrispr_meta;
use crate::prediction_tools::crispor_integration;
use crate::prediction_tools::transcrispr::run_transcrispr_meta;
use combination_model::fit_and_apply;
use combination_model::WeightMap;
use crate::analysis::efficacy_analysis_integration::analyze_efficacy_distribution;
use crate::analysis::gene_balanced::plot_auroc_violin;
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

const DB_NAME: &str = "TKO (Full dataset)";

fn main() -> PolarsResult<()> {
    // Setup logging and project configuration
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("Starting the CRISPR pipeline");

    let binding = project_root();
    let project_root = binding.to_str().unwrap();
    write_config_json(project_root).map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;
    let model_colors = ModelColors::new();

    // Initialize dataset loaders
    let cegs = Cegs {
        path: "./data/cegv2.txt".to_string()
    };
    let genomecrispr_datasets = GenomeCrisprDatasets {
        path: "./data/genomecrispr/GenomeCRISPR_full05112017_brackets.csv".to_string(),
    };
    let ky = AvanaDataset {
        efficacy_path: "./data/depmap/CRISPRInferredGuideEfficacy_23Q4.csv".to_string(),
        guide_map_path: "./data/depmap/KYGuideMap.csv".to_string(),
    };
    let tko_dataset = TkoScreensDataset {
        screen_path: "./data/tko/tko_one.xlsx".to_string().into(),
        guide_map_path:"./data/tko/tko_hg38_annotation_for_rust.txt".into(),
    };

    // Load essential gene annotations
    let cegs = cegs.load()?;

    // Load and process dataset - uncomment desired dataset
    // let df = genomecrispr_datasets.load_validated("genome_crispr_short", cegs)?;
    // let mut df = ky.load_validated(DB_NAME, cegs)?;
    // let mut df = tko_dataset.load_validated("tko_one", cegs)?;

    // Define evaluation thresholds
    let efficacy_column = "efficacy";
    let poor_threshold = 60.0;
    let good_threshold = 90.0;
    let min_good_coverage = 0.0;
    let output_dir = "./prediction_evaluation_results";

    // Load preprocessed dataset
    let mut df = AnyDataset {
        path: "./processed_tkou.csv".to_string(),
    }.load()?;

    // Balance dataset with stratified sampling within genes
    let (mut df, stats) = stratified_within_gene_balance_with_margin(&df, "efficacy", 85.0, 95.0, Some(42))?;

    // Analyze efficacy distribution
    analyze_efficacy_distribution(&df, output_dir, poor_threshold, good_threshold, DB_NAME);

    // Run prediction tools (commented out - enable as needed)
    // df = run_deepcrispr_meta(df, DB_NAME)?;
    // df = run_transcrispr_meta(df, DB_NAME)?;
    // df = deepspcas9_integration::run_deepspcas9_meta(df.clone(), "prediction")?;
    // df = tko_pssm::run_pssm_meta(df.clone(), "sequence_with_pam", DB_NAME)?;
    // df = crispor_integration::run_crispor_meta(df.clone(), DB_NAME)?;

    // Standardize prediction tool column names
    let rename_map = [
        ("Doench '16-Score",       "Doench Rule Set 2"),
        ("Moreno-Mateos-Score",    "Moreno-Mateos"),
        ("Doench-RuleSet3-Score",  "Doench Rule Set 3"),
        ("deepcrispr_prediction",  "DeepCRISPR"),
        ("transcrispr_prediction", "TransCRISPR"),
        ("deepspcas9_prediction",  "DeepSpCas9"),
        ("pssm_score",             "TKO PSSM"),
    ];

    for &(old, new) in &rename_map {
        if df.get_column_names().iter().any(|c| *c == old) {
            df.rename(old, new.into())?;
        }
    }

    // Cast specific columns to Float64
    for &col_name in &["Doench Rule Set 2", "Moreno-Mateos", "Doench Rule Set 3"] {
        if df.get_column_names().iter().any(|c| c.as_str() == col_name) {
            let s = df.column(col_name)?.cast(&DataType::Float64)?;
            df = df.with_column(s)?.clone();
        } else {
            info!("⚠ `{}` not found, skipping cast", col_name);
        }
    }

    // Define tools for evaluation
    let mut tools_to_evaluate: Vec<&str> = vec![
        "TKO PSSM",
        "Moreno-Mateos",
        "Doench Rule Set 2",
        "Doench Rule Set 3",
        "DeepCRISPR",
        "DeepSpCas9",
        "TransCRISPR",
    ];

    // Train linear consensus model
    let learned_weights = fit_and_apply(
        &mut df,
        &tools_to_evaluate,
        "efficacy",
        "Linear Consensus",
    )?;

    // Save linear weights
    serde_json::to_writer_pretty(
        File::create("weights_linear_avana.json")?, &learned_weights).unwrap();

    let subset_cols: Vec<String> = tools_to_evaluate
        .iter()
        .map(|&s| s.to_string())
        .collect();

    let mut df_log = df.clone();
    let mut df_aug = df.clone();

    // Train logistic consensus model with cross-validation
    let (log_weights, log_thr) = fit_logistic_cv(
        &mut df_log,
        &tools_to_evaluate,
        "efficacy",
        good_threshold as f64,
        "Gene",     // gene-ID column
        5,          // 5-fold CV
        0.01,       // L2 regularization strength
    )?;

    // Apply logistic consensus predictions
    apply_logistic_score(
        &mut df_aug,
        &tools_to_evaluate,
        &log_weights,
        "Logistic Consensus",
    )?;
    let log_col = df_aug.column("Logistic Consensus")?.clone();
    df = df.with_column(log_col)?.clone();

    // Save logistic weights and threshold
    serde_json::to_writer_pretty(
        File::create("weights_logistic_avana.json")?, &log_weights).unwrap();
    println!("Logistic threshold (Youden-J): {:.3}", log_thr);

    // Alternative: Load pre-trained consensus weights (commented out)
    // let linear_weights: WeightMap = from_reader(File::open("weights_linear_avana.json")?).unwrap();
    // add_combined_score_column(&mut df, &tools_to_evaluate, &linear_weights, "Linear Consensus")?;

    // Add consensus models to evaluation list
    tools_to_evaluate.push("Linear Consensus");
    tools_to_evaluate.push("Logistic Consensus");

    // Save final dataset with all predictions
    dataframe_to_csv(&mut df, "weights.csv", true);

    // Generate violin plots (commented out)
    // plot_auroc_violin(&df, "Gene", tools_to_evaluate.clone(), good_threshold, DB_NAME, "./figures/ba_violin_60.png")?;

    // Evaluate all prediction tools
    evaluate_prediction_tools(
        &df,
        &tools_to_evaluate,
        efficacy_column,
        None,  // threshold CSV path (optional)
        poor_threshold,
        good_threshold,
        min_good_coverage,
        output_dir,
        DB_NAME
    )?;

    // Generate ROC curves for different cutoffs
    for cutoff in vec![90, 60] {
        roc::compare_roc_curves(&df, &tools_to_evaluate.clone(), "efficacy", cutoff as f64, "./general_results", DB_NAME)?;
    }

    // Run statistical analysis for bad guide identification
    run_statistical_bad_guide_analysis(&df, tools_to_evaluate.clone(), "./results/statistical_bad_guides", "TKO")?;

    Ok(())
}
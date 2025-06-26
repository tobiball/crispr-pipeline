#![allow(unused)]

use crate::test::{smoking_gun_test, verify_only_raw_values_used};
use crate::test::comprehensive_model_isolation_test;
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
use crate::data_handling::tko_two::{ScreenKind, TkoEssentialomeDataset};
use crate::logistic_cv::{apply_logistic_score, fit_logistic_cv};

mod models;
mod data_handling;
mod prediction_tools;
mod helper_functions;
mod mageck_processing;
mod analysis;
mod combination_model;
mod logistic_cv;
mod test;
// Helper function to read CSV files


// const DB_NAME: &str = "TKO2";
// const DB_NAME: &str = "KY";
const DB_NAME: &str = "TKO (Full dataset)";



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
    let model_colors = ModelColors::new();




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

    let tko_dataset = TkoEssentialomeDataset {
        screen_path: "./data/tko/tko_two.xlsx".to_string().into(),
        guide_map_path:"./data/tko/tko_hg38_annotation_for_rust.txt".into(),
        screen_kind: ScreenKind::Essentialome_WT,
    };

    let cegs = cegs.load()?;
    // let df = genomecrispr_datasets.load_validated("genome_crispr_short", cegs)?;
    // let mut df = ky.load_validated(DB_NAME, cegs)?;
    // let mut df = tko_dataset.load_validated("tko_one", cegs)?;











    // Define evaluation parameters
    let efficacy_column = "efficacy";
    let poor_threshold = 60.0;
    let good_threshold = 90.0;
    let min_good_coverage = 0.0;
    let output_dir = "./prediction_evaluation_results";





    // let (mut df, stats) = stratified_within_gene_balance(&df, "efficacy", 90.0,None, Some(42))?;
    // println!("{stats:#?}");


    // let mut df = undersample_equal_classes(&drop_mono_class_genes(df)?,"efficiency", good_threshold,Some(42))?;

    //
    //

    // let mut df = AnyDataset {
    //     path: "./processed_df.csv".to_string(),
    // }.load()?;

    let mut df = AnyDataset {
        path: "./processed_tkou.csv".to_string(),
    }.load()?;

    let mut df_training = AnyDataset {
        path: "./processed_df.csv".to_string(),
    }.load()?;


    let (mut df, stats) = stratified_within_gene_balance_with_margin(&df, "efficacy", 85.0, 95.0, Some(42))?;


    // let mut df = AnyDataset {
    //     path: "./processed_tko.csv".to_string(),
    // }.load()?;
    //
    analyze_efficacy_distribution(&df, output_dir, poor_threshold, good_threshold, DB_NAME);
    //
    //
    //
    //
    // df = run_deepcrispr_meta(df, DB_NAME)?;
    // //
    // df = run_transcrispr_meta(
    //     df,
    //     DB_NAME
    // )?;
    //
    // df = deepspcas9_integration::run_deepspcas9_meta(df.clone(), "prediction")?;
    //
    // df = tko_pssm::run_pssm_meta(df.clone(), "sequence_with_pam", DB_NAME)?;
    //
    // df = crispor_integration::run_crispor_meta(df.clone(), DB_NAME)?;
    //
    //
    // dataframe_to_csv(&mut df, "processed_ky_final.csv", true);







    let rename_map = [
        ("Doench '16-Score",       "Doench Rule Set 2"),
        ("Moreno-Mateos-Score",    "Moreno-Mateos"),
        ("Doench-RuleSet3-Score",  "Doench Rule Set 3"),
        ("deepcrispr_prediction",  "DeepCRISPR"),
        ("transcrispr_prediction", "TransCRISPR"),
        ("deepspcas9_prediction",  "DeepSpCas9"),
        ("pssm_score",             "TKO PSSM"),
    ];
//
    for &(old, new) in &rename_map {
        if df.get_column_names().iter().any(|c| *c == old) {
            df.rename(old, new.into())?;
        }
    }
//
    // 5) Before cast
    for &col_name in &["Doench Rule Set 2", "Moreno-Mateos", "Doench Rule Set 3"] {
        // compare as_str() → &str
        if df.get_column_names().iter().any(|c| c.as_str() == col_name) {
            let s = df
                .column(col_name)?
                .cast(&DataType::Float64)?;         // cast the Series
            df = df.with_column(s)?.clone();                // put it back
        } else {
            info!("⚠ `{}` not found, skipping cast", col_name);
        }
    }

    let rename_map = [
        ("Doench '16-Score",       "Doench Rule Set 2"),
        ("Moreno-Mateos-Score",    "Moreno-Mateos"),
        ("Doench-RuleSet3-Score",  "Doench Rule Set 3"),
        ("deepcrispr_prediction",  "DeepCRISPR"),
        ("transcrispr_prediction", "TransCRISPR"),
        ("deepspcas9_prediction",  "DeepSpCas9"),
        ("pssm_score",             "TKO PSSM"),
    ];
    //
    for &(old, new) in &rename_map {
        if df_training.get_column_names().iter().any(|c| *c == old) {
            df_training.rename(old, new.into())?;
        }
    }
    //
    // 5) Before cast
    for &col_name in &["Doench Rule Set 2", "Moreno-Mateos", "Doench Rule Set 3"] {
        // compare as_str() → &str
        if df_training.get_column_names().iter().any(|c| c.as_str() == col_name) {
            let s = df_training
                .column(col_name)?
                .cast(&DataType::Float64)?;         // cast the Series
            df_training = df_training.with_column(s)?.clone();                // put it back
        } else {
            info!("⚠ `{}` not found, skipping cast", col_name);
        }
    }
//
    // … continue with your tool list …
    let mut tools_to_evaluate: Vec<&str> = vec![
        "TKO PSSM",
        "Moreno-Mateos",
        "Doench Rule Set 2",
        "Doench Rule Set 3",
        "DeepCRISPR",
        "DeepSpCas9",
        "TransCRISPR",
    ];

//     ___________________________________________________________________________
    let learned_weights = fit_and_apply(
        &mut df,
        &tools_to_evaluate,
        "efficacy",
        "Linear Consensus",
    )?;
    serde_json::to_writer_pretty(
        File::create("weights_linear_avana.json")?, &learned_weights).unwrap();

    let subset_cols: Vec<String> = tools_to_evaluate
        .iter()
        .map(|&s| s.to_string())
        .collect();

    let mut df_log = df.clone();
    let mut df_aug = df.clone();
    //
    //
    //
    //

    let (log_weights, log_thr) = fit_logistic_cv(
        &mut df_log,
        &tools_to_evaluate,
        "efficacy",
        good_threshold as f64,
        "Gene",                  // ← gene-ID column in your DataFrame
        5,                       // 5-fold CV
        0.01,                    // L2 strength
    )?;

    apply_logistic_score(
        &mut df_aug,
        &tools_to_evaluate,
        &log_weights,
        "Logistic Consensus",
    )?;
    let log_col = df_aug.column("Logistic Consensus")?.clone();
    df = df.with_column(log_col)?.clone();

    serde_json::to_writer_pretty(
        File::create("weights_logistic_avana.json")?, &log_weights).unwrap();
    println!("Logistic threshold (Youden-J): {:.3}", log_thr);
    // ___________________________________________________________________________




    //

    // let linear_weights: WeightMap =
    //     from_reader(File::open("weights_linear_avana.json")?).unwrap();
    //
    // // 2) Apply them (this appends a "Linear Consensus" column to your df)
    // add_combined_score_column(
    //     &mut df,
    //     &tools_to_evaluate,
    //     &linear_weights,
    //     "Linear Consensus",
    // )?;
    //
    // // 3) Load logistic consensus weights
    // let logistic_weights: WeightMap =
    //     from_reader(File::open("weights_logistic_avana.json")?).unwrap();
    //
    //
    // let mut df_log = df.clone();
    //
    //
    //
    // // 4) Apply them analogously (appends "Logistic Consensus")
    // apply_logistic_score(
    //     &mut df_log,
    //     &mut df_training,
    //     &tools_to_evaluate,
    //     &logistic_weights,
    //     "Logistic Consensus",
    // )?;
    //
    // let log_col = df_log.column("Logistic Consensus")?.clone();
    // df = df.with_column(log_col)?.clone();

    //___________________________________________________________________________





    let mut tools_to_evaluate = tools_to_evaluate;

    tools_to_evaluate.push("Linear Consensus");
    tools_to_evaluate.push("Logistic Consensus");

    dataframe_to_csv(&mut df, "weights.csv", true);


    // plot_auroc_violin(&df, "Gene", tools_to_evaluate.clone(), good_threshold, DB_NAME, "./figures/ba_violin_60.png")?;




    // Run the evaluation
    evaluate_prediction_tools(
        &df,
        &tools_to_evaluate,
        efficacy_column,
        // Some("prediction_evaluation_results/thresholdsfull.csv"),
        None,
        poor_threshold,
        good_threshold,
        min_good_coverage,
        output_dir,
        DB_NAME

    )?;

    // let mut tools_to_evaluate: Vec<&str> = vec![];
    //
    //
    // tools_to_evaluate.push("Linear Consensus");
    // tools_to_evaluate.push("Logistic Consensus");
    //
    // // Run the evaluation
    // evaluate_prediction_tools(
    //     &df,
    //     &tools_to_evaluate,
    //     efficacy_column,
    //     // Some("prediction_evaluation_results/thresholds_full.csv"),
    //     None,
    //     poor_threshold,
    //     good_threshold,
    //     min_good_coverage,
    //     output_dir,
    //     DB_NAME
    //
    // )?;
    //
    // comprehensive_model_isolation_test(&df)?;
    // smoking_gun_test()?;
    // verify_only_raw_values_used(&df)?;



    for cutoff in vec![98,90,80,70,60,50] {
        roc::compare_roc_curves(&df, &tools_to_evaluate.clone(), "efficacy", cutoff as f64, "./general_results", DB_NAME) ?;
    }
     // let df_filtered = within_gene_analysis(df.clone(), tools_to_evaluate.clone(), DB_NAME)?;
    // for tool in &tools_to_evaluate {
    //     let out_path = format!("./general_results/results_{}/", tool);
    //     // plot_tool_vs_efficacy(&df_an, tool, format!("{}tool_vs_efficacy.png", &out_path))?;
    //     plot_stripplot_for_tool(&df_filtered, tool,format!("{}stripplot_within_gene.png", &out_path))?;
    //
    //     println!("Wrote {}", out_path);
    // }
    // let summary_df = within_gene::bad_guide_detection_summary(&df_filtered, tools_to_evaluate.clone(), DB_NAME)?;
    // run_bad_guide_summary(&df, tools_to_evaluate.clone(), "./results/bad_guides", DB_NAME)?;

    run_statistical_bad_guide_analysis(&df, tools_to_evaluate.clone(), "./results/statistical_bad_guides", "TKO")?;



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


























// _______________________________________________

// // src/fix_missing.rs ── Polars-0.46 safe back-fill helper
// use polars::lazy::dsl::*;
// use polars::prelude::*;
// use regex::Regex;
// use std::collections::HashSet;
// use std::ops::Not;      // for mask.not()
//
// // ─── sequence columns that might still carry lower-case chars
// const SEQ_COLS: &[&str] = &[
//     "sgRNA",
//     "sequence_deepspcas9",
//     "sequence_with_pam",
//     "pam",
// ];
//
// // ─── every column produced by the prediction stack
// const TOOL_COLS: &[&str] = &[
//     "deepcrispr_prediction",
//     "transcrispr_prediction",
//     "deepspcas9_prediction",
//     "pssm_score",
//     "pssm_raw_score",
//     "#seqId",
//     "guideId",
//     "mitSpecScore",
//     "cfdSpecScore",
//     "offtargetCount",
//     "Doench '16-Score",
//     "Moreno-Mateos-Score",
//     "Doench-RuleSet3-Score",
// ];
//
// /// Patch rows whose prediction columns are **null or 0.0**, re-run the
// /// predictors on that slice, and splice the fresh scores back.
// pub fn backfill_missing_predictions(
//     mut df: DataFrame,
//     db_name: &str,
// ) -> PolarsResult<DataFrame> {
//     // 0) stable row index (Polars 0.46 has no with_row_count)
//     let row_idx = UInt32Chunked::from_iter_values("row_idx".into(), 0..df.height() as u32)
//         .into_series();
//     df = df.hstack(&[Column::from(row_idx)])?;
//
//     // 1) Boolean mask: needs_fix[i] = true if any tool col is null OR 0.0
//     let mut needs_fix = Vec::with_capacity(df.height());
//     for i in 0..df.height() {
//         let mut bad = false;
//         for &pred in TOOL_COLS {
//             if let Ok(col) = df.column(pred) {
//                 match col.get(i) {
//                     Ok(AnyValue::Null)                         => { bad = true; break }
//                     Ok(AnyValue::Float64(v)) if v == 0.0        => { bad = true; break }
//                     _ => {}
//                 }
//             }
//         }
//         needs_fix.push(bad);
//     }
//     if !needs_fix.iter().any(|&b| b) {
//         info!("All prediction columns populated – nothing to back-fill");
//         return Ok(df.drop("row_idx")?);
//     }
//
//     let mask = BooleanChunked::from_slice("mask".into(), &needs_fix);
//     let mut df_fix  = df.filter(&mask)?;          // rows to repair
//     let mut df_keep = df.filter(&mask.not())?;    // already fine
//
//     info!("Back-filling {} rows", df_fix.height());
//
//     // 2) drop every tool column from *both* slices so they’ll be recreated once
//     for &c in TOOL_COLS {
//         if df_fix.get_column_names().iter().any(|n| n.as_str() == c) {
//             df_fix = df_fix.drop(c)?;      // keep this
//         }
//     }
//
//     // 3) upper-case residual lower-case bases in df_fix
//     let lc_re = Regex::new(r"[acgt]")?;
//     let mut lf = df_fix.clone().lazy();
//     for &seq in SEQ_COLS {
//         if df_fix.get_column_names().iter().any(|c| c.as_str() == seq) {
//             let s = df_fix.column(seq)?.str()?;
//             if s.into_iter().any(|o| o.map(|v| lc_re.is_match(v)).unwrap_or(false)) {
//                 lf = lf.with_column(col(seq).str().to_uppercase().alias(seq));
//             }
//         }
//     }
//     df_fix = lf.collect()?;
//
//     // 4) re-run the prediction stack *only* on df_fix
//     df_fix = run_deepcrispr_meta(df_fix,        db_name)?;
//     df_fix = run_transcrispr_meta(df_fix,       db_name)?;
//     df_fix = deepspcas9_integration::run_deepspcas9_meta(df_fix.clone(), "prediction")?;
//     df_fix = tko_pssm::run_pssm_meta(df_fix.clone(), "sequence_with_pam", db_name)?;
//     df_fix = crispor_integration::run_crispor_meta(df_fix.clone(), db_name)?;
//
//     // --- 5) ensure both frames have identical schemas before vstack ------------
//     let cols_fix:  HashSet<_> = df_fix .get_column_names().iter().map(|s| s.as_str()).collect();
//     let cols_keep: HashSet<_> = df_keep.get_column_names().iter().map(|s| s.as_str()).collect();
//
//     // collect first – no borrows alive while we mutate
//     let missing_in_keep: Vec<String> = cols_fix .difference(&cols_keep).map(|s| (*s).to_string()).collect();
//     let missing_in_fix : Vec<String> = cols_keep.difference(&cols_fix ).map(|s| (*s).to_string()).collect();
//
//     // add null columns to df_keep so it matches df_fix
//     for col in &missing_in_keep {
//         let dtype = df_fix.column(col)?.dtype().clone();           // ← actual dtype
//         let nulls = Series::full_null(col.as_str().into(), df_keep.height(), &dtype);
//         df_keep = df_keep.hstack(&[nulls.into()])?;
//     }
//
//     // add null columns to df_fix so it matches df_keep
//     for col in &missing_in_fix {
//         let dtype = df_keep.column(col)?.dtype().clone();          // ← actual dtype
//         let nulls = Series::full_null(col.as_str().into(), df_fix.height(), &dtype);
//         df_fix = df_fix.hstack(&[nulls.into()])?;
//     }
//
//
//     // 5c) put both DataFrames in identical column order
//     let col_order: Vec<String> = df_keep
//         .get_column_names()
//         .iter()
//         .map(|s| s.to_string())   // <-- own the String
//         .collect();
//
//     df_keep = df_keep.select(&col_order)?;   // now &[String]
//     df_fix  = df_fix .select(&col_order)?;
//
//     // 6) merge back and restore original order
//     let mut df_out = df_keep.vstack(&df_fix)?;
//     df_out = df_out.sort(vec!["row_idx"], Default::default())?;
//     Ok(df_out.drop("row_idx")?)
//
// }

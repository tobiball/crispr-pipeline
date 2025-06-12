use std::process::Command;
use std::collections::HashMap;
use std::path::Path;
use polars::prelude::*;
use tracing::{info, error, debug};
use crate::helper_functions::{project_root, read_csv};
use crate::models::polars_err;

/// Struct to hold tool comparison configuration
pub struct ToolComparisonConfig {
    pub input_path: String,
    pub output_dir: String,
    pub efficacy_column: String,
    pub gene_column: String,
    pub tool_columns: Vec<String>,
    pub poor_threshold: f64,
    pub moderate_threshold: f64,
    pub python_path: Option<String>,
}

impl Default for ToolComparisonConfig {
    fn default() -> Self {
        Self {
            input_path: String::from("./processed_data/all_tools_output.csv"),
            output_dir: String::from("./general_results/tool_comparison"),
            efficacy_column: String::from("efficacy"),
            gene_column: String::from("Gene"),
            tool_columns: vec![],
            poor_threshold: 50.0,
            moderate_threshold: 75.0,
            python_path: None,
        }
    }
}

/// Run the tool comparison analysis
pub fn run_tool_comparison(config: ToolComparisonConfig) -> PolarsResult<Vec<String>> {
    // Ensure the tool columns are valid
    if config.tool_columns.is_empty() {
        return Err(PolarsError::ComputeError("No tool columns specified for comparison".into()));
    }

    // Find the script path - should be in the same directory as the executable
    let current_dir = std::env::current_dir()
        .map_err(|e| polars_err(Box::new(e)))?;

    // In tool_comparison_integration.rs:
    let tool_comparison_script = project_root()  // Use your helper function
        .join("scripts/tool_comparison.py")
        .to_string_lossy()
        .to_string();


    // Ensure output directory exists
    std::fs::create_dir_all(&config.output_dir)
        .map_err(|e| polars_err(Box::new(e)))?;

    // Format the tool columns as a comma-separated string
    let tools_str = config.tool_columns.join(",");

    info!("Running tool comparison analysis with {} tools", config.tool_columns.len());
    debug!("Using script at: {}", tool_comparison_script);
    debug!("Input path: {}", config.input_path);
    debug!("Output directory: {}", config.output_dir);
    debug!("Tools to compare: {}", tools_str);

    // In your tool_comparison_integration.rs file:
    let venv_python = crate::helper_functions::project_root()
        .join("scripts/tool_comparison_venv/bin/python")
        .to_string_lossy()
        .to_string();


    // Build the command
    let tool_output = Command::new(venv_python)
        .arg(&tool_comparison_script)
        .arg("--input").arg(&config.input_path)
        .arg("--tools").arg(&tools_str)
        .arg("--efficacy-col").arg(&config.efficacy_column)
        .arg("--gene-col").arg(&config.gene_column)
        .arg("--output").arg(&config.output_dir)
        .arg("--poor-threshold").arg(config.poor_threshold.to_string())
        .arg("--moderate-threshold").arg(config.moderate_threshold.to_string())
        .output()
        .map_err(|e| polars_err(Box::new(e)))?;

    // Check if the command was successful
    if !tool_output.status.success() {
        let stderr = String::from_utf8_lossy(&tool_output.stderr);
        error!("Tool comparison analysis failed: {}", stderr);
        return Err(PolarsError::ComputeError(format!("Tool comparison analysis failed: {}", stderr).into()));
    }

    // Log the output
    let stdout = String::from_utf8_lossy(&tool_output.stdout);
    info!("Tool comparison analysis completed successfully");
    debug!("Tool comparison output: {}", stdout);

    // Return a list of expected output files
    let expected_files = vec![
        format!("{}/tool_confusion_matrices.png", config.output_dir),
        format!("{}/tool_rank_correlations.png", config.output_dir),
        format!("{}/tool_correlations.csv", config.output_dir),
        format!("{}/poor_guide_detection.png", config.output_dir),
        format!("{}/poor_guide_detection_metrics.csv", config.output_dir),
        format!("{}/within_gene_ranking.png", config.output_dir),
        format!("{}/within_gene_ranking_metrics.csv", config.output_dir),
        format!("{}/tool_performance_dashboard.png", config.output_dir),
        format!("{}/tool_performance_metrics.csv", config.output_dir),
        format!("{}/roc_pr_curves.png", config.output_dir),
        format!("{}/roc_pr_metrics.csv", config.output_dir),
        format!("{}/distribution_comparison.png", config.output_dir)
    ];

    Ok(expected_files)
}

/// Wrapper for a simplified API to run tool comparison with minimal configuration
pub fn compare_tools(df: &DataFrame, tools: Vec<&str>, output_dir: &str) -> PolarsResult<Vec<String>> {
    // Create a temporary CSV file
    let temp_dir = format!("{}/temp", output_dir);
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| polars_err(Box::new(e)))?;

    let temp_csv = format!("{}/tool_comparison_input.csv", temp_dir);

    // Write the DataFrame to CSV
    CsvWriter::new(std::fs::File::create(&temp_csv)
        .map_err(|e| polars_err(Box::new(e)))?)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df.clone())?;

    info!("Saved temporary input file to {}", temp_csv);

    // Convert tools to String
    let tool_columns = tools.iter()
        .map(|&s| s.to_string())
        .collect::<Vec<String>>();

    // Build the config
    let config = ToolComparisonConfig {
        input_path: temp_csv,
        output_dir: output_dir.to_string(),
        tool_columns,
        ..Default::default()
    };

    // Run the comparison
    run_tool_comparison(config)
}

/// Parse tool performance metrics CSV to get performance rankings
pub fn get_tool_rankings(output_dir: &str) -> PolarsResult<DataFrame> {
    let metrics_path = format!("{}/tool_performance_metrics.csv", output_dir);

    if !Path::new(&metrics_path).exists() {
        return Err(PolarsError::ComputeError(
            format!("Tool performance metrics file not found at {}", metrics_path).into()
        ));
    }

    read_csv(&metrics_path as &str)
}
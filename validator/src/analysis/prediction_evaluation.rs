use std::fs::{self, create_dir_all};
use std::path::Path;
use std::process::Command;
use polars::prelude::*;
use tracing::{info, error, debug, warn};
use crate::models::polars_err;

/// Evaluates prediction tools with visualization using an external Python script
///
/// This function processes the prediction data in the DataFrame and then calls
/// a Python script to generate visualizations that show how well each prediction
/// tool can identify good guides while minimizing misclassification of poor
/// and moderate guides.
///
/// # Arguments
///
/// * `df` - DataFrame containing efficacy and prediction tool columns
/// * `tool_columns` - List of column names for the prediction tools to evaluate
/// * `efficacy_column` - Name of the column containing true efficacy values
/// * `efficacy_poor_threshold` - Efficacy threshold below which guides are considered poor
/// * `efficacy_good_threshold` - Efficacy threshold above which guides are considered good
/// * `min_good_coverage` - Minimum fraction of good guides that must be identified (e.g., 0.75)
/// * `output_dir` - Directory where output files will be saved
///
/// # Returns
///
/// * `PolarsResult<()>` - Ok if successful, Err otherwise
pub fn evaluate_prediction_tools(
    df: &DataFrame,
    tool_columns: &[&str],
    efficacy_column: &str,
    efficacy_poor_threshold: f64,
    efficacy_good_threshold: f64,
    min_good_coverage: f64,
    output_dir: &str,
) -> PolarsResult<()> {
    info!("Evaluating prediction tools with visualization");

    // Create output directory if it doesn't exist
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        create_dir_all(output_path).map_err(|e| {
            error!("Failed to create output directory: {}", e);
            polars_err(Box::new(e))
        })?;
    }

    // Create a temporary CSV file with the data
    let temp_csv_path = format!("{}/prediction_data.csv", output_dir);
    info!("Saving data to temporary CSV file: {}", temp_csv_path);

    // Make a copy of the DataFrame to avoid modifying the original
    let mut df_clone = df.clone();

    // Write to CSV
    let mut file = std::fs::File::create(&temp_csv_path).map_err(|e| polars_err(Box::new(e)))?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_clone)
        .map_err(|e| {
            error!("Failed to write data to CSV: {}", e);
            e
        })?;

    // Find the path to the Python visualization script
    let script_path = "./scripts/crispr_prediction_evaluation.py";

    let venv_python = crate::helper_functions::project_root()
        .join("scripts/tool_comparison_venv/bin/python")
        .to_string_lossy()
        .to_string();


    // Build the command
    let mut cmd = Command::new(venv_python);
    cmd.arg(&script_path)
        .arg("--input").arg(&temp_csv_path)
        .arg("--output").arg(output_dir)
        .arg("--efficacy-col").arg(efficacy_column)
        .arg("--poor-threshold").arg(efficacy_poor_threshold.to_string())
        .arg("--good-threshold").arg(efficacy_good_threshold.to_string())
        .arg("--min-coverage").arg(min_good_coverage.to_string());

    // Add tool columns if specified
    if !tool_columns.is_empty() {
        let tools_arg = tool_columns.join(",");
        cmd.arg("--tools").arg(tools_arg);
    }

    // Run the Python script
    info!("Running Python visualization script: {}", script_path);
    debug!("Command: {:?}", cmd);

    match cmd.output() {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                info!("Python script output: {}", stdout);

                // Check if the expected output files were created
                let expected_files = [
                    "tool_misclassification_comparison.png",
                    "tool_stacked_performance.png",
                    "tool_evaluation_results.csv",
                    "tool_evaluation_results.json"
                ];

                for file in &expected_files {
                    let file_path = output_path.join(file);
                    if !file_path.exists() {
                        warn!("Expected output file not found: {}", file_path.display());
                    } else {
                        info!("Generated output file: {}", file_path.display());
                    }
                }

                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                error!("Python script failed: {}", stderr);
                Err(PolarsError::ComputeError(format!("Python script failed: {}", stderr).into()))
            }
        },
        Err(e) => {
            error!("Failed to run Python script: {}", e);
            Err(PolarsError::ComputeError(format!("Failed to run Python script: {}", e).into()))
        }
    }
}


/// A struct to hold the evaluation results for a single tool
#[derive(Debug, Clone)]
pub struct ToolEvaluationResult {
    /// Name of the prediction tool
    pub name: String,
    /// Threshold value used for classifying guides as good
    pub threshold: f64,
    /// Percentage of good guides correctly identified (should be >= min_good_coverage)
    pub good_guide_coverage: f64,
    /// Percentage of poor guides incorrectly classified as good
    pub poor_misclassification: f64,
    /// Percentage of moderate guides incorrectly classified as good
    pub moderate_misclassification: f64,
    /// Combined quality score (higher is better)
    pub quality_score: f64,
}

impl ToolEvaluationResult {
    /// Format the result as a string
    pub fn to_string(&self) -> String {
        format!(
            "{}: threshold={:.2}, good_coverage={:.1}%, poor_misclass={:.1}%, moderate_misclass={:.1}%, score={:.3}",
            self.name,
            self.threshold,
            self.good_guide_coverage * 100.0,
            self.poor_misclassification * 100.0,
            self.moderate_misclassification * 100.0,
            self.quality_score
        )
    }
}

/// Parse the evaluation results from the JSON file produced by the Python script
pub fn parse_evaluation_results(json_path: &str) -> Result<Vec<ToolEvaluationResult>, Box<dyn std::error::Error>> {
    // Read the JSON file
    let json_str = std::fs::read_to_string(json_path)?;
    let json: serde_json::Value = serde_json::from_str(&json_str)?;

    // Extract the tools array
    let tools = json["tools"].as_array().ok_or("No tools array found in JSON")?;

    // Parse each tool
    let mut results = Vec::new();
    for tool in tools {
        results.push(ToolEvaluationResult {
            name: tool["name"].as_str().unwrap_or("Unknown").to_string(),
            threshold: tool["threshold"].as_f64().unwrap_or(0.0),
            good_guide_coverage: tool["good_guide_coverage"].as_f64().unwrap_or(0.0),
            poor_misclassification: tool["poor_misclassification_rate"].as_f64().unwrap_or(0.0),
            moderate_misclassification: tool["moderate_misclassification_rate"].as_f64().unwrap_or(0.0),
            quality_score: tool["quality_score"].as_f64().unwrap_or(0.0),
        });
    }

    Ok(results)
}

/// Helper function to get the top N performing tools based on quality score
pub fn get_top_tools(results: &[ToolEvaluationResult], n: usize) -> Vec<&ToolEvaluationResult> {
    let mut sorted = results.iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(n);
    sorted
}
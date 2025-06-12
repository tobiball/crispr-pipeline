use polars::prelude::*;
use std::error::Error;
use tracing::{info, error};
use crate::models::polars_err;

use polars::prelude::*;
use polars::datatypes::BooleanChunked;



/// Generate mismatch columns and optionally filter for guides with frequent mismatches
pub fn generate_high_efficacy_low_prediction_df(
    df: &DataFrame,
    tool_columns: Vec<&str>,
    output_csv_path: &str,
    filtered_output_csv_path: Option<&str>,
    min_mismatches: Option<u32>,
) -> PolarsResult<DataFrame> {
    // 1. Compute efficacy 90th percentile
    let efficacy_series = df.column("efficacy")?.f64().map_err(|e| polars_err(Box::new(e)))?;
    let efficacy_90 = efficacy_series
        .quantile(0.50, QuantileMethod::Nearest)
        .map_err(|e| polars_err(Box::new(e)))?
        .ok_or_else(|| PolarsError::ComputeError("Failed to get 90th percentile".into()))?;

    let efficacy_90 = efficacy_90 as f64;
    let top_efficacy_mask: Vec<bool> = efficacy_series
        .into_no_null_iter()
        .map(|val| val >= efficacy_90)
        .collect();

    // 2. Generate mismatch columns
    let mut new_df = df.clone();
    for tool in &tool_columns {
        let tool_series = df.column(tool)?.f64().map_err(|e| polars_err(Box::new(e)))?;

        let tool_10 = tool_series
            .quantile(0.10, QuantileMethod::Nearest)
            .map_err(|e| polars_err(Box::new(e)))?
            .ok_or_else(|| PolarsError::ComputeError("Failed to get 10th percentile".into()))? as f64;

        let mismatch_mask: Vec<bool> = tool_series
            .into_no_null_iter()
            .zip(top_efficacy_mask.iter())
            .map(|(tool_val, &is_top)| is_top && (tool_val <= tool_10))
            .collect();

        let mismatch_col_name = format!("{}_mismatch", tool);
        let mismatch_series = Series::new(PlSmallStr::from(&mismatch_col_name), mismatch_mask);
        new_df.with_column(mismatch_series)?;
    }

    // 3. Select desired columns
    let mut desired_order: Vec<String> = vec!["sequence_with_pam".to_string(), "efficacy".to_string()];
    for tool in &tool_columns {
        desired_order.push(format!("{}_mismatch", tool));
    }
    let column_refs: Vec<&str> = desired_order.iter().map(|s| s.as_str()).collect();
    let mut new_df = new_df.select(column_refs)?;

    // 4. Save mismatch DataFrame
    {
        let mut file = std::fs::File::create(output_csv_path).map_err(|e| polars_err(Box::new(e)))?;
        CsvWriter::new(&mut file)
            .include_header(true)
            .finish(&mut new_df.clone()) // clone to keep `new_df` usable
            .map_err(|e| polars_err(Box::new(e)))?;
        info!("Mismatch DataFrame saved to: {}", output_csv_path);
    }

    // 5. Filter and save guides with multiple mismatches if requested
    if let (Some(filtered_path), Some(min_mismatches_count)) = (filtered_output_csv_path, min_mismatches) {
        // Find guides with multiple mismatches
        let mut filtered_df = find_guides_with_multiple_mismatches(
            &new_df,
            &tool_columns,
            min_mismatches_count
        )?;

        // Save filtered DataFrame
        let mut file = std::fs::File::create(filtered_path).map_err(|e| polars_err(Box::new(e)))?;
        CsvWriter::new(&mut file)
            .include_header(true)
            .finish(&mut filtered_df)?;
        info!("Filtered mismatch DataFrame saved to: {}", filtered_path);
    }

    Ok(new_df)
}

// Fixed find_guides_with_multiple_mismatches function
// Fixed find_guides_with_multiple_mismatches function
pub fn find_guides_with_multiple_mismatches(
    df: &DataFrame,
    tool_columns: &[&str],
    min_mismatches: u32,
) -> PolarsResult<DataFrame> {
    if tool_columns.is_empty() {
        return Err(PolarsError::ComputeError("No tool columns provided".into()));
    }

    // Get all the mismatch columns
    let mismatch_columns: Vec<String> = tool_columns
        .iter()
        .map(|tool| format!("{}_mismatch", tool))
        .collect();

    // Create a new column that counts the number of mismatches for each guide
    let mut accum_expr = lit(0);
    for col_name in &mismatch_columns {
        accum_expr = accum_expr + col(col_name).cast(DataType::UInt32);
    }

    // Add the count column to the DataFrame
    let with_count = df.clone().lazy()
        .with_column(accum_expr.alias("mismatch_count"))
        .collect()?;

    // Filter for rows that have at least min_mismatches
    // Use a different approach to filter the DataFrame
    let mismatch_count_series = with_count.column("mismatch_count")?;
    let filter_mask = mismatch_count_series.u32()?
        .into_iter()
        .map(|opt_val| opt_val.map(|val| val >= min_mismatches).unwrap_or(false))
        .collect::<Vec<bool>>();

    // Create a proper boolean mask series
    let mask = BooleanChunked::new("".into(), filter_mask.as_slice());
    let filtered = with_count.filter(&mask)?;

    // Sort by mismatch_count in descending order to prioritize guides with more mismatches
    let sorted = filtered.sort(["mismatch_count"], Default::default())?;  // false for descending

    info!("Found {} guides with {} or more tool mismatches", sorted.height(), min_mismatches);

    Ok(sorted)
}
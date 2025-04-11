use std::collections::HashMap;
use polars::prelude::*;
use tracing::{debug, info, error};
use crate::models::polars_err;

/// PSSM (Position-Specific Scoring Matrix) based gRNA prediction functions

/// Scoring matrix for nucleotides at each position (1-indexed)
/// Values directly transcribed from the provided schema
/// Format: [nucleotide][position] where nucleotide: 0=A, 1=C, 2=G, 3=T
pub const NUCLEOTIDE_SCORES: [[f64; 21]; 4] = [
    // A (position 1-20, plus a dummy 0 index since we're 1-indexed)
    [0.0, 0.322, 0.409, 0.324, 0.072, 0.039, 0.143, 0.178, -0.013, 0.439, 0.458, 0.318, 0.254, -0.242, 0.291, 0.106, 0.151, -0.191, -0.673, -0.523, 0.048],
    // C
    [0.0, -0.776, -0.131, -0.5, -0.143, -0.059, -0.079, 0.017, 0.03, -0.245, -0.092, -0.107, 0.174, 0.509, -0.163, -0.108, 0.366, 0.177, 1.0, 0.075, -0.631],
    // G
    [0.0, 0.281, -0.103, 0.088, 0.437, 0.11, 0.344, 0.169, 0.003, 0.013, 0.103, 0.052, -0.431, -0.056, -0.585, -0.223, -0.377, 0.012, -0.326, 0.442, 0.584],
    // T
    [0.0, 0.172, -0.174, 0.087, -0.367, -0.207, -0.402, -0.365, -0.014, -0.206, -0.468, -0.258, 0.006, -0.209, 0.461, 0.227, -0.144, -1.0, -1.0, -1.0, -1.0],
];

/// Maps nucleotide characters to their index in the NUCLEOTIDE_SCORES array
pub fn get_nucleotide_index(nucleotide: char) -> Option<usize> {
    match nucleotide {
        'A' | 'a' => Some(0),
        'C' | 'c' => Some(1),
        'G' | 'g' => Some(2),
        'T' | 't' => Some(3),
        _ => None,
    }
}

/// Calculate a guide score based on the position-specific scoring matrix
///
/// # Arguments
///
/// * `guide_sequence` - The 20nt guide RNA sequence (without PAM)
///
/// # Returns
///
/// * A score indicating predicted efficiency (higher is better)
///
/// # Errors
///
/// * Returns `None` if the sequence is not exactly 20nt or contains invalid nucleotides
pub fn calculate_pssm_score(guide_sequence: &str) -> Option<f64> {
    // Validate sequence length
    if guide_sequence.len() != 20 {
        debug!("Guide sequence must be 20nt long, got {}nt", guide_sequence.len());
        return None;
    }

    let mut score = 0.0;

    // Calculate score by summing position-specific values
    for (pos, nucleotide) in guide_sequence.chars().enumerate() {
        let position = pos + 1; // Convert to 1-indexed

        if let Some(nucleotide_idx) = get_nucleotide_index(nucleotide) {
            score += NUCLEOTIDE_SCORES[nucleotide_idx][position];
        } else {
            debug!("Invalid nucleotide '{}' at position {}", nucleotide, position);
            return None; // Invalid nucleotide
        }
    }

    Some(score)
}

/// Normalize a raw PSSM score to 0-100 range for consistency with other tools
///
/// # Arguments
///
/// * `raw_score` - The raw score from calculate_pssm_score
///
/// # Returns
///
/// * A normalized score between 0 and 100
pub fn normalize_score(raw_score: f64) -> f64 {
    // These min/max values are based on the theoretical range of the PSSM
    // Calculated by summing worst/best possible scores at each position
    let min_possible_score = -6.798;
    let max_possible_score = 6.534;

    let normalized = (raw_score - min_possible_score) / (max_possible_score - min_possible_score) * 100.0;
    normalized.clamp(0.0, 100.0)
}

/// Apply the PSSM prediction to a DataFrame
///
/// # Arguments
///
/// * `df` - DataFrame containing guide sequences
/// * `sequence_column` - Name of column containing guide sequences
///
/// # Returns
///
/// * A DataFrame with an additional 'pssm_score' column
pub fn process_dataframe(df: &DataFrame, sequence_column: &str) -> PolarsResult<DataFrame> {
    // Extract the sequence column
    let seq_series = df.column(sequence_column)?;
    let seq_strs = seq_series.str()?;

    // Calculate scores for each sequence
    let mut scores: Vec<Option<f64>> = Vec::with_capacity(df.height());
    let mut normalized_scores: Vec<Option<f64>> = Vec::with_capacity(df.height());

    for i in 0..df.height() {
        if let Some(seq) = seq_strs.get(i) {
            // Trim the sequence to ensure clean input (no whitespace)
            let clean_seq = seq.trim();

            // Calculate raw score
            if let Some(raw_score) = calculate_pssm_score(clean_seq) {
                scores.push(Some(raw_score));
                normalized_scores.push(Some(normalize_score(raw_score)));
            } else {
                scores.push(None);
                normalized_scores.push(None);
            }
        } else {
            scores.push(None);
            normalized_scores.push(None);
        }
    }

    // Create new Series for raw and normalized scores
    let raw_score_series = Series::new(PlSmallStr::from("pssm_raw_score"), scores);
    let norm_score_series = Series::new(PlSmallStr::from("pssm_score"), normalized_scores);

    // Add columns to DataFrame
    let mut result_df = df.clone();
    result_df.with_column(raw_score_series)?;
    result_df.with_column(norm_score_series)?;

    Ok(result_df)
}

/// Run the PSSM prediction on a DataFrame and save results
///
/// # Arguments
///
/// * `df` - Input DataFrame
/// * `sequence_column` - Column containing guide sequences
/// * `output_path` - Where to save the results
///
/// # Returns
///
/// * Result indicating success or failure
pub fn run_pssm_meta(df: DataFrame, sequence_column: &str, database_name: &str) -> PolarsResult<DataFrame> {
    info!("Running PSSM prediction analysis on {} guides", df.height());

    // Process the DataFrame
    let result_df = process_dataframe(&df, sequence_column)?;

    // Save the results
    let output_dir = "./processed_data";
    std::fs::create_dir_all(output_dir).map_err(|e| {
        error!("Failed to create output directory: {}", e);
        polars_err(Box::new(e))
    })?;

    let output_path = format!("{}/pssm_{}.csv", output_dir, database_name);
    let mut file = std::fs::File::create(&output_path).map_err(|e| polars_err(Box::new(e)))?;

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut result_df.clone())?;

    info!("PSSM prediction results saved to {}", output_path);

    Ok(result_df)
}
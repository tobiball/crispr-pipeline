use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufRead};
use std::process::Command;


// use polars::prelude::*;
use tracing::{debug, error, info};

/// Holds user-defined parameters to run MAGeCK.
#[derive(Debug)]
pub struct MageckOptions {
    /// Path to the `mageck` binary (e.g. "/usr/local/bin/mageck").
    pub mageck_path: String,

    /// Output prefix. MAGeCK will produce e.g. `{output_prefix}.sgrna_summary.txt`.
    pub output_prefix: String,

    /// The sample columns to treat as "treatment" in `-t`.
    pub treat_labels: Vec<String>,

    /// The sample columns to treat as "control" in `-c`.
    pub ctrl_labels: Vec<String>,

    /// Optional: full path to the RRA executable.
    pub rra_path: Option<String>,
}


/// Example structure representing a single row from MAGeCK’s .sgrna_summary.txt
#[derive(Debug)]
pub struct MageckGuideResult {
    pub sgrna: String,
    pub log2fc: f64,
    pub pval: f64,
    pub fdr: f64,  // adjusted p-value
}


use polars::prelude::*;
use polars::prelude::*;

/// Constants controlling the transformation of MAGeCK LFC + FDR into a 0–100 guide efficacy score
const FDR_THRESHOLD: f64 = 0.05;       // FDR cutoff for "statistically significant" depletion
const EXPONENT: f64 = 0.8;            // Exponent to shape the LFC distribution (0.8 "boosts" mid-range values slightly)
const NON_SIGNIFICANT_CAP: f64 = 0.2; // Guides that are not significant get capped at 20% max
const CLIP_FACTOR: f64 = 0.9;         // Clip the absolute min LFC to 90% for outlier resistance
const DEFAULT_MIN_LFC: f64 = -3.0;    // Fallback if the LFC column is empty or cannot be computed


//https://pubmed.ncbi.nlm.nih.gov/29083409/
pub fn calculate_efficacy_scores(mut df: DataFrame) -> PolarsResult<DataFrame> {
    info!("Calculating biologically-interpretable efficacy scores based on CERES methodology");

    // If the DataFrame is empty, no work to do
    if df.is_empty() {
        info!("DataFrame is empty; returning unchanged.");
        return Ok(df);
    }

    // Check if the required column exists
    let columns = df.get_column_names();
    if !columns.contains(&&PlSmallStr::from_str("mageck_log2fc")) {
        return Err(PolarsError::ComputeError(
            "Missing required column 'mageck_log2fc'".into()
        ));
    }

    // Define biologically meaningful log2FC thresholds based on CERES and related literature
    // These represent fold-change magnitudes commonly associated with significant effects
    let severe_depletion: f64 = -3.0;    // >8x depletion (2^3)
    let strong_depletion: f64 = -2.0;    // >4x depletion (2^2)
    let moderate_depletion: f64 = -1.0;  // >2x depletion (2^1)
    let weak_depletion: f64 = -0.585;    // >1.5x depletion (2^0.585)

    info!("Using biologically meaningful LFC thresholds from https://pubmed.ncbi.nlm.nih.gov/29083409/:");
    info!("  Severe depletion (90-100): LFC ≤ {:.2} (>8x depletion)", severe_depletion);
    info!("  Strong depletion (70-89): LFC ≤ {:.2} (>4x depletion)", strong_depletion);
    info!("  Moderate depletion (40-69): LFC ≤ {:.2} (>2x depletion)", moderate_depletion);
    info!("  Weak depletion (20-39): LFC ≤ {:.2} (>1.5x depletion)", weak_depletion);
    info!("  Minimal effect (1-19): LFC between {:.2} and 0", weak_depletion);
    info!("  No effect (0): LFC ≥ 0");

    // Extract the log2FC series
    let lfc_series = df.column("mageck_log2fc")?.f64()?;

    // LFC-only approach
    let efficacy_ca: Float64Chunked = lfc_series
        .into_iter()
        .map(|lfc_opt| {
            lfc_opt.map(|lfc| {
                if lfc >= 0.0 {
                    0.0
                } else {
                    let abs_lfc = lfc.abs();

                    if abs_lfc >= severe_depletion.abs() {
                        let beyond_severe = ((abs_lfc - severe_depletion.abs()) / 3.0).min(1.0);
                        90.0 + (beyond_severe * 10.0)
                    } else if abs_lfc >= strong_depletion.abs() {
                        let range_position = (abs_lfc - strong_depletion.abs()) /
                            (severe_depletion.abs() - strong_depletion.abs());
                        70.0 + (range_position * 20.0)
                    } else if abs_lfc >= moderate_depletion.abs() {
                        let range_position = (abs_lfc - moderate_depletion.abs()) /
                            (strong_depletion.abs() - moderate_depletion.abs());
                        40.0 + (range_position * 30.0)
                    } else if abs_lfc >= weak_depletion.abs() {
                        let range_position = (abs_lfc - weak_depletion.abs()) /
                            (moderate_depletion.abs() - weak_depletion.abs());
                        20.0 + (range_position * 20.0)
                    } else {
                        1.0 + ((abs_lfc / weak_depletion.abs()) * 19.0)
                    }
                }
            })
        })
        .collect();

    // Create the new efficacy column
    let efficacy_s = Series::new("efficacy".into(), efficacy_ca);

    // Replace existing efficacy column or add new one
    if columns.contains(&&PlSmallStr::from_str("efficacy")) {
        df.drop_in_place("efficacy")?;
    }
    df.with_column(efficacy_s)?;

    // Add a category column based on the efficacy score
    let cat_s = df.column("efficacy")?
        .f64()?
        .into_iter()
        .map(|eff_opt| {
            match eff_opt {
                Some(eff) if eff >= 90.0 => "Severe",
                Some(eff) if eff >= 70.0 => "Strong",
                Some(eff) if eff >= 40.0 => "Moderate",
                Some(eff) if eff >= 20.0 => "Weak",
                Some(eff) if eff > 0.0 => "Minimal",
                _ => "None"
            }
        });

    // Create a categorical column
    let category_s = Series::new("efficacy_category".into(), cat_s.collect::<Vec<&str>>());
    df.with_column(category_s)?;

    // Log distribution of efficacy scores
    let new_efficacy = df.column("efficacy")?.f64()?;

    // Log statistics
    info!("Efficacy score distribution:");
    info!("  Min: {:.2}, Max: {:.2}",
          new_efficacy.min().unwrap_or(0.0),
          new_efficacy.max().unwrap_or(0.0));
    info!("  Mean: {:.2}, Median: {:.2}",
          new_efficacy.mean().unwrap_or(0.0),
          new_efficacy.median().unwrap_or(0.0));

    // Log distribution by category
    let mut bins = vec![0; 10];
    for eff in new_efficacy.into_iter().flatten() {
        let bin_index = (eff / 10.0).floor().min(9.0) as usize;
        bins[bin_index] += 1;
    }

    info!("Efficacy distribution by decile:");
    for (i, count) in bins.iter().enumerate() {
        let lower = i * 10;
        let upper = (i + 1) * 10;
        info!("  {}-{}: {} guides", lower, upper, count);
    }

    Ok(df)
}


// Update the run_mageck_pipeline function to calculate efficacy scores
pub fn run_mageck_pipeline(
    mut df_in: DataFrame,
    mageck_path: &str,
    output_prefix: &str,
    treat_labels: &[String],
    ctrl_labels: &[String],
    sgrna_col: &str,
    gene_col: &str,
    count_cols: &[&str],
) -> PolarsResult<DataFrame> {


    // 2) Prepare the MAGeck options
    let mageck_opts = MageckOptions {
        mageck_path: mageck_path.to_string(),
        output_prefix: output_prefix.to_string(),
        treat_labels: treat_labels.to_vec(),
        ctrl_labels: ctrl_labels.to_vec(),
        rra_path: Some("/home/mrcrispr/crispr_pipeline/mageck/mageck_venv/bin/RRA".to_string()),
    };

    // In your mageck_processing.rs, before running MAGeCK
    let output_dir = std::path::Path::new(&mageck_opts.output_prefix).parent().unwrap_or_else(|| std::path::Path::new("."));
    // Debug log the path
    debug!("Creating directory: {:?}", output_dir);
    std::fs::create_dir_all(output_dir)?;
    let exists = std::path::Path::new(output_dir).exists();
    debug!("Directory exists after creation: {}", exists);

    // 1) Write a tab-delimited file for MAGeck
    let mageck_input_path = format!("{}.input.txt", output_prefix);

    write_mageck_input(&df_in, &mageck_input_path, sgrna_col, gene_col, count_cols)?;


    // 3) Run mageck test
    run_mageck_test(&mageck_opts, &mageck_input_path)
        .map_err(|e| PolarsError::ComputeError(format!("Mageck error: {}", e).into()))?;

    // 4) Parse the .sgrna_summary.txt
    let sgrna_summary_path = format!("{}.sgrna_summary.txt", output_prefix);
    let mageck_results = parse_mageck_sgrna_summary(&sgrna_summary_path)?;

    // 5) Merge results into the original DataFrame
    let df_merged = merge_mageck_results(df_in, &mageck_results, sgrna_col)?;


    // 6) Calculate efficacy scores based on the MAGeCK results
    let mut df_with_efficacy = calculate_efficacy_scores(df_merged)?;


    let mut file = File::create("processing_artifacts/mageckinput_log.csv").expect("could not create file");

    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_with_efficacy)?;






    Ok(df_with_efficacy)
}


pub fn write_mageck_input(
    df: &DataFrame,
    output_path: &str,
    sgrna_col: &str,
    gene_col: &str,
    sample_cols: &[&str],
) -> PolarsResult<()> {
    debug!("[write_mageck_input] DataFrame shape: {:?}", df.shape());
    debug!("[write_mageck_input] Columns: {:?}", df.get_column_names());

    // Let's examine a few rows of input data to see what we're dealing with
    let sample_size = std::cmp::min(5, df.height());
    for i in 0..sample_size {
        let sgrna = df.column(sgrna_col)?.get(i)?.to_string();
        let gene = df.column(gene_col)?.get(i)?.to_string();
        debug!("Sample row {}: sgRNA={}, Gene={}", i, sgrna, gene);

        // Check the count columns too
        for &col in sample_cols {
            let val = df.column(col)?.get(i)?.to_string();
            debug!("  {} = {}", col, val);
        }
    }

    // Count unique sgRNAs and genes to check for duplicate issues
    let unique_sgrnas = df.column(sgrna_col)?.unique()?;
    let unique_genes = df.column(gene_col)?.unique()?;
    debug!("Unique sgRNAs: {}, Unique genes: {}", unique_sgrnas.len(), unique_genes.len());

    // Continue with the original function...
    let mut df_out = df.select([sgrna_col, gene_col])?;
    debug!("[write_mageck_input] df_out shape after select: {:?}", df_out.shape());

    for &colname in sample_cols {
        debug!("[write_mageck_input] Adding column '{}'.", colname);
        df_out = df_out.hstack(&[df.column(colname)?.clone()])?;
    }

    // Add a check for NaN or empty values in the count columns
    for &col in sample_cols {
        let null_count = df_out.column(col)?
            .null_count();

        debug!("Column {} has {} null values", col, null_count);

        // Sample some non-null values to see their format
        if df_out.height() > 0 {
            for i in 0..std::cmp::min(3, df_out.height()) {
                let val = df_out.column(col)?.get(i)?.to_string();
                debug!("Sample value for {}, row {}: {}", col, i, val);
            }
        }
    }

    // Create the output file with modified row writing
    let mut f = File::create(output_path)?;
    // Write header
    write!(f, "sgrna\tgene")?;
    for col in sample_cols {
        write!(f, "\t{}", col)?;
    }
    writeln!(f)?;

    // Set to track already processed guides to avoid duplication
    use std::collections::HashSet;
    let mut processed_guides = HashSet::new();

    // Write rows with deduplication check
    for row_idx in 0..df_out.height() {
        // Remove quotes from string values
        let sgrna_val = df_out.column(sgrna_col)?.get(row_idx)?.to_string().trim_matches('"').to_string();
        let gene_val = df_out.column(gene_col)?.get(row_idx)?.to_string().trim_matches('"').to_string();

        // Skip if we've already processed this guide
        if processed_guides.contains(&sgrna_val) {
            if row_idx < 10 {  // Limit logging
                debug!("Skipping duplicate guide: {} (gene: {})", sgrna_val, gene_val);
            }
            continue;
        }

        // Add to processed set
        processed_guides.insert(sgrna_val.clone());

        write!(f, "{}\t{}", sgrna_val, gene_val)?;

        let mut all_columns_valid = true;

        for &col in sample_cols {
            // Process array values, calculate average
            let array_str = df_out.column(col)?.get(row_idx)?.to_string();

            // Parse array format and extract numeric values
            let values: Vec<f64> = array_str
                .trim_matches('"')
                .trim_start_matches('{')
                .trim_end_matches('}')
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();

            // Check if we have valid values
            if values.is_empty() {
                if row_idx < 10 {
                    debug!("No valid values for guide: {}, gene: {}, column: {}, raw: {}",
                           sgrna_val, gene_val, col, array_str);
                }
                all_columns_valid = false;
                write!(f, "\t0")?;  // Write a default value
            } else {
                let final_value = values.iter().sum::<f64>() / values.len() as f64;
                write!(f, "\t{}", final_value)?;
            }
        }

        if !all_columns_valid && row_idx < 10 {
            debug!("Row {} had some invalid columns", row_idx);
        }

        writeln!(f)?;
    }

    debug!("Wrote {} unique guides to file (from {} input rows)",
           processed_guides.len(), df_out.height());

    Ok(())
}



/// Run `mageck test` with your chosen arguments, returning standard output/err on success.
pub fn run_mageck_test(opts: &MageckOptions, input_counts_path: &str) -> Result<(), Box<dyn Error>> {
    info!("Running MAGeCK test with options: {:?}", opts);

    // Join the treat and control labels
    let treat_str = opts.treat_labels.join(",");
    let ctrl_str  = opts.ctrl_labels.join(",");

    let mut cmd = Command::new(&opts.mageck_path);
    cmd.arg("test")
        .arg("-k").arg(input_counts_path)
        .arg("-t").arg(treat_str)
        .arg("-c").arg(ctrl_str)
        .arg("-n").arg(&opts.output_prefix)
        .arg("--additional-rra-parameters")
        .arg("-p 0");  // effectively skip the rank aggregation

    // If rra_path is provided, update PATH so RRA can be found.
    if let Some(rra_full_path) = &opts.rra_path {
        if let Some(rra_dir) = std::path::Path::new(rra_full_path).parent() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{}:{}", rra_dir.to_string_lossy(), current_path);
            cmd.env("PATH", new_path);
        }
    }

    debug!("About to spawn: {:?}", cmd);
    let output = cmd.output()?;
    if !output.status.success() {
        error!("MAGeCK failed. Stderr:\n{}", String::from_utf8_lossy(&output.stderr));
        return Err(format!("MAGeCK exited with status {:?}", output.status).into());
    }

    debug!("MAGeck test completed successfully. Stdout:\n{}",
           String::from_utf8_lossy(&output.stdout));
    Ok(())
}


pub fn parse_mageck_sgrna_summary(file_path: &str) -> PolarsResult<Vec<MageckGuideResult>> {
    let f = File::open(file_path)?;
    let reader = std::io::BufReader::new(f);

    let mut results = Vec::new();
    let mut lines = reader.lines();

    // Read and parse header to find correct column indices
    let header_line = lines.next().ok_or_else(|| PolarsError::ComputeError("Missing header in MAGeCK output".into()))??;
    let header_fields: Vec<&str> = header_line.split('\t').collect();

    // Find the indices for important columns
    let log2fc_idx = header_fields.iter().position(|&h| h.contains("neg.lfc"))
        .unwrap_or(6); // Fallback to your original guess
    let pval_idx = header_fields.iter().position(|&h| h == "neg.p-value")
        .unwrap_or(7);
    let fdr_idx = header_fields.iter().position(|&h| h == "neg.fdr")
        .unwrap_or(9);

    debug!("Column indices: log2fc={}, pval={}, fdr={}", log2fc_idx, pval_idx, fdr_idx);

    for line in lines {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() <= fdr_idx {
            debug!("Skipping short line: {}", line);
            continue;
        }

        let sgrna = fields.get(0).unwrap_or(&"").to_string();
        let neg_lfc = fields.get(log2fc_idx).unwrap_or(&"0").parse::<f64>().unwrap_or(0.0);
        let neg_pval = fields.get(pval_idx).unwrap_or(&"1").parse::<f64>().unwrap_or(1.0);
        let fdr = fields.get(fdr_idx).unwrap_or(&"1").parse::<f64>().unwrap_or(1.0);

        results.push(MageckGuideResult {
            sgrna,
            log2fc: neg_lfc,
            pval: neg_pval,
            fdr,
        });
    }

    Ok(results)
}

/// Merge the MAGeCK results back into your original DataFrame.
/// This is optional; you could keep them separate if you want.
use polars::prelude::{
    DataFrame, Series, PolarsResult,
    JoinArgs, JoinType,
};
use crate::models::polars_err;

pub fn merge_mageck_results(
    df_original: DataFrame,
    mageck_results: &[MageckGuideResult],
    sgrna_col: &str,
) -> PolarsResult<DataFrame> {
    // 1) Convert mageck_results into separate Series
    let mut s_sgrna = Series::from_iter(mageck_results.iter().map(|r| r.sgrna.clone()));
    s_sgrna.rename(sgrna_col.into());

    let mut s_lfc = Series::from_iter(mageck_results.iter().map(|r| r.log2fc));
    s_lfc.rename("mageck_log2fc".into());

    let mut s_pval = Series::from_iter(mageck_results.iter().map(|r| r.pval));
    s_pval.rename("mageck_pval".into());

    let mut s_fdr = Series::from_iter(mageck_results.iter().map(|r| r.fdr));
    s_fdr.rename("mageck_fdr".into());

    // 2) Construct a DataFrame from these Series
    let df_mageck = DataFrame::new(vec![
        Column::from(s_sgrna), Column::from(s_lfc), Column::from(s_pval), Column::from(s_fdr)
    ])?;

    // 3) Join df_original with df_mageck on `sgrna_col`
    let df_joined = df_original.join(&df_mageck, [sgrna_col], [sgrna_col], JoinArgs::new(JoinType::Left), None)?;

    Ok(df_joined)
}

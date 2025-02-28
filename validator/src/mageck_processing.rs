use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::{Write, BufRead};
use std::process::Command;

use polars::prelude::*;
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

/// A convenience function that:
/// 1) Writes a tab-delimited input file for MAGeck
/// 2) Invokes mageck test
/// 3) Parses the resulting .sgrna_summary.txt
/// 4) Merges the guide-level log2FC/p-values into the original DataFrame
///
/// # Arguments
/// * `df_in` - The Polars DataFrame containing sgRNA, gene, and count columns
/// * `mageck_path` - Path to the `mageck` binary
/// * `output_prefix` - Output prefix (will produce e.g. `{prefix}.sgrna_summary.txt`)
/// * `treat_labels` / `ctrl_labels` - Label(s) for columns used by MAGeck test
/// * `sgrna_col` / `gene_col` - The columns used for sgRNA IDs and gene IDs
/// * `count_cols` - The columns with read counts (e.g. `["rc_initial", "rc_final"]`)
///
/// Returns a PolarsResult<DataFrame> with new columns:
///     - mageck_log2fc
///     - mageck_pval
///     - mageck_fdr
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

    // 1) Write a tab-delimited file for MAGeck
    let mageck_input_path = format!("{}.input.txt", output_prefix);
    write_mageck_input(&df_in, &mageck_input_path, sgrna_col, gene_col, count_cols)?;


    // 2) Prepare the MAGeck options
    let mageck_opts = MageckOptions {
        mageck_path: mageck_path.to_string(),
        output_prefix: output_prefix.to_string(),
        treat_labels: treat_labels.to_vec(),
        ctrl_labels: ctrl_labels.to_vec(),
        // If you have a path to RRA, place it here; otherwise None
        rra_path: None,
    };

    // 3) Run mageck test
    run_mageck_test(&mageck_opts, &mageck_input_path)
        .map_err(|e| PolarsError::ComputeError(format!("Mageck error: {}", e).into()))?;

    // 4) Parse the .sgrna_summary.txt
    let sgrna_summary_path = format!("{}.sgrna_summary.txt", output_prefix);
    let mageck_results = parse_mageck_sgrna_summary(&sgrna_summary_path)?;

    // 5) Merge results into the original DataFrame
    let df_merged = merge_mageck_results(df_in, &mageck_results, sgrna_col)?;

    Ok(df_merged)
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

    // Check for the specific repeating pattern
    let btf3_count = df.clone().lazy()
        .filter(col(gene_col).eq(lit("BTF3")))
        .collect()?
        .height();

    let specific_guide_count = df.clone().lazy()
        .filter(col(sgrna_col).eq(lit("AGGAGAGGAAGGCGATGCGACGG")))
        .collect()?
        .height();

    debug!("BTF3 gene count: {}, AGGAGAGGAAGGCGATGCGACGG guide count: {}",
           btf3_count, specific_guide_count);

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


/// Parse MAGeCK’s {prefix}.sgrna_summary.txt to get guide-level LFC, p-values, etc.
pub fn parse_mageck_sgrna_summary(file_path: &str) -> PolarsResult<Vec<MageckGuideResult>> {
    let f = File::open(file_path)?;
    let reader = std::io::BufReader::new(f);

    let mut results = Vec::new();
    // The .sgrna_summary.txt typically has a header line like:
    // sgRNA  gene  score  p-value neg|pos.lfc  neg|pos.p-value ...
    //
    // In negative selection, we often look at the `neg.lfc` column for the log2FC.
    // The naming may differ by MAGeCK version, so you might parse carefully.
    let mut lines = reader.lines();
    // skip the header
    if let Some(Ok(header_line)) = lines.next() {
        debug!("MAGeCK .sgrna_summary header: {}", header_line);
    }

    for line in lines {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        // Adjust indices to match actual columns in .sgrna_summary
        let sgrna  = fields.get(0).unwrap_or(&"").to_string();
        // Example: "neg.lfc" might be at column 5 or 6, etc. Inspect your output to find it.
        let neg_lfc_str = fields.get(5).unwrap_or(&"0");
        let neg_lfc = neg_lfc_str.parse::<f64>().unwrap_or(0.0);

        let neg_pval_str = fields.get(6).unwrap_or(&"1");
        let neg_pval = neg_pval_str.parse::<f64>().unwrap_or(1.0);

        // Alternatively, or in addition, there's an FDR column at maybe index 7 or 8.
        let fdr_str = fields.get(8).unwrap_or(&"1");
        let fdr = fdr_str.parse::<f64>().unwrap_or(1.0);

        results.push(MageckGuideResult {
            sgrna,
            log2fc: neg_lfc,
            pval:   neg_pval,
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

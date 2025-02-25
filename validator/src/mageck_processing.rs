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

    panic!();

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

/// Writes a Polars DataFrame into the MAGeCK-friendly count table format.
/// E.g.:
/// sgrna  gene    initial  final
/// g1     A1CF    100      50
/// g2     A1CF    75       10
pub fn write_mageck_input(
    df: &DataFrame,
    output_path: &str,
    sgrna_col: &str,
    gene_col: &str,
    sample_cols: &[&str],
) -> PolarsResult<()> {
    debug!("[write_mageck_input] DataFrame shape: {:?}", df.shape());
    debug!("[write_mageck_input] Columns: {:?}", df.get_column_names());
    debug!("[write_mageck_input] Head(5): {:?}", df.head(Some(5)));

    let mut df_out = df.select([sgrna_col, gene_col])?;
    debug!("[write_mageck_input] df_out shape after select: {:?}", df_out.shape());

    for &colname in sample_cols {
        debug!("[write_mageck_input] Adding column '{}'.", colname);
        df_out = df_out.hstack(&[df.column(colname)?.clone()])?;
    }

    debug!("[write_mageck_input] df_out shape after hstack: {:?}", df_out.shape());

    //
    // >>>> ADD THE SNIPPET HERE <<<<
    //
    // Let's log the first row that we intend to write, printing each column name,
    // its Polars data type, and its value in row 0.
    if df_out.height() > 0 {
        debug!("[write_mageck_input] First row sample:");
        // We can iterate over all columns in df_out:
        for (field_idx, field) in df_out.schema().iter_fields().enumerate() {
            let col_name = field.name();
            // Safely fetch row 0
            let val_str = df_out
                .column(col_name)?
                .get(0)
                .map(|v| v.to_string())
                .unwrap_or_else(|_| "N/A".to_string());

            debug!("    Column '{}' : {}", col_name, val_str);
        }
    } else {
        debug!("[write_mageck_input] df_out is empty! No rows to print.");
    }

    //
    // Now proceed to actually create/write the file:
    //
    let mut f = File::create(output_path)?;
    // Write header
    write!(f, "sgrna\tgene")?;
    for col in sample_cols {
        write!(f, "\t{}", col)?;
    }
    writeln!(f)?;

    // Write rows
    for row_idx in 0..df_out.height() {
        let sgrna_val = df_out.column(sgrna_col)?.get(row_idx)?.to_string();
        let gene_val  = df_out.column(gene_col)?.get(row_idx)?.to_string();
        write!(f, "{}\t{}", sgrna_val, gene_val)?;

        for &col in sample_cols {
            let val = df_out.column(col)?.get(row_idx)?.to_string();
            write!(f, "\t{}", val)?;
        }
        writeln!(f)?;
    }

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
    let df_joined = df_original.join(
        &df_mageck,
        [sgrna_col],
        [sgrna_col],
        JoinArgs::new(JoinType::Left) // or JoinType::Inner, Right, etc.
    )?;

    Ok(df_joined)
}

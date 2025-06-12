use tracing::{debug, error, info};
use std::path::PathBuf;
use polars::error::PolarsResult;
use polars::frame::DataFrame;
use polars::prelude::{col, lit, CsvParseOptions, CsvReadOptions, CsvWriter, IntoLazy, NamedFrom, PlSmallStr, SerReader, SerWriter, Series};

use std::env;
pub fn project_root() -> PathBuf {
    match env::var_os("PROJECT_ROOT") {
        Some(val) => PathBuf::from(val),
        None => {
            // Fall back to current directory if PROJECT_ROOT not set
            env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        }
    }
}

use std::fs;
use std::path::Path;
use plotters::style::RGBColor;
use serde_json::json;
use crate::models::polars_err;

pub fn write_config_json(project_root: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Define the config.json content
    let config = json!({
        "PATH": {
            "PRIMER3": format!("{}/chopchop/primer3_core", project_root),
            "BOWTIE": format!("{}/chopchop/bowtie/bowtie", project_root),
            "TWOBITTOFA": format!("{}/chopchop/twoBitToFa", project_root),
            "TWOBIT_INDEX_DIR": format!("{}/chopchop", project_root),
            "BOWTIE_INDEX_DIR": format!("{}/chopchop", project_root),
            "ISOFORMS_INDEX_DIR": format!("{}/chopchop", project_root),
            "ISOFORMS_MT_DIR": format!("{}/chopchop", project_root),
            "GENE_TABLE_INDEX_DIR": format!("{}/chopchop", project_root)
        },
        "THREADS": 1
    });

    // Define the path to the config.json file
    let config_path = Path::new(project_root).join("chopchop/config.json");

    // Write the config.json file
    fs::write(config_path, serde_json::to_string_pretty(&config)?)?;

    Ok(())
}



pub fn read_csv(file_path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(10000))  // Look at more rows to detect mixed types
        .try_into_reader_with_file_path(Some(PathBuf::from(file_path)))?
        .finish()
}

pub fn read_txt(file_path: &str) -> PolarsResult<DataFrame> {
    info!("Reading txt file: {}", file_path);

    let df_result = CsvReadOptions::default()
        .with_has_header(true)
        .map_parse_options(|opts| CsvParseOptions {
            separator: b'\t', // Set delimiter to tab
            ..opts
        })
        .try_into_reader_with_file_path(Some(PathBuf::from(file_path)))?
        .finish();

    match &df_result {
        Ok(df) => debug!("Loaded txt with columns: {:?}", df.get_column_names()),
        Err(e) => error!("Failed to read txt file: {}", e),
    }

    df_result
}


pub fn dataframe_to_csv(
    df: &mut DataFrame,
    path: &str,
    include_header: bool
) -> PolarsResult<()> {
    // Ensure the directory exists
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            error!("Failed to create directory for {}: {}", path, e);
            polars_err(Box::new(e))
        })?;
    }

    // Open the file for writing
    let file = std::fs::File::create(path).map_err(|e| {
        error!("Failed to create file {}: {}", path, e);
        polars_err(Box::new(e))
    })?;

    // Write the DataFrame to CSV
    CsvWriter::new(file)
        .include_header(include_header)
        .with_separator(b',')
        .finish(df)?;

    info!("DataFrame successfully written to {}", path);
    Ok(())
}

fn name_to_hue(name: &str) -> f64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME:    u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for b in name.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    (hash % 360) as f64
}

/// HSV ➜ RGB in 0‥=255  (fixed S & V give nice vibrant colours)
pub fn colour_for_tool(name: &str) -> RGBColor {
    let h = name_to_hue(name);
    let s = 0.65;  // saturation
    let v = 0.85;  // brightness

    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = match h as u32 {
        0..=59   => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179=> (0.0, c, x),
        180..=239=> (0.0, x, c),
        240..=299=> (x, 0.0, c),
        _        => (c, 0.0, x),            // 300–359
    };

    RGBColor(
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}

/// Attach a `quality` column with values "good" | "moderate" | "bad".

use polars::prelude::*;


/// Remove genes that have guides in only one quality class
pub fn drop_mono_class_genes(df: DataFrame) -> PolarsResult<DataFrame> {
    // ── 1) find the genes we want to keep (lazy is shortest) ──────────────
    let keep = df
        .clone()
        .lazy()
        .group_by([col("Gene")])
        .agg([col("quality").n_unique().alias("nq")])
        .filter(col("nq").gt(lit(1)))
        .select([col("Gene")])
        .collect()?;                         // eager frame with the IDs

    // ── 2) semi-join back to the big table ────────────────────────────────
    df.join(
        &keep,
        ["Gene"],
        ["Gene"],
        JoinArgs::from(JoinType::Inner),
        None,
    )
}
/// --------------------------------------------------------------------
/// 2.  Undersample the two classes so they contain the same number of rows
///     (keeps min(class_size) rows from each bucket)
/// --------------------------------------------------------------------
pub fn undersample_equal_classes(
    df: &DataFrame,
    efficacy_col: &str,
    cutoff: f64,
    seed: Option<u64>,
) -> PolarsResult<DataFrame> {
    // ── 1) attach the bucket column (lazy) --------------------------------
    let with_bucket = df
        .clone()
        .lazy()
        .with_column(
            when(col(efficacy_col).gt_eq(lit(cutoff)))
                .then(lit("good"))
                .otherwise(lit("other"))
                .alias("__bucket__"),
        )
        .collect()?;

    // ── 2) minority bucket size ------------------------------------------
    let min_n = with_bucket
        .clone()
        .lazy()
        .group_by([col("__bucket__")])
        .agg([col("__bucket__").count().alias("n")])
        .select([col("n").min().alias("min_n")])
        .collect()?
        .column("min_n")?
        .u32()?
        .get(0)
        .unwrap_or(0) as usize;

    if min_n == 0 {
        return with_bucket.drop("__bucket__");      // nothing to balance
    }

    // ── 3) sample each bucket down to min_n --------------------------------
    let balanced = with_bucket
        .group_by(["__bucket__"])?
        .apply(|g| g.sample_n_literal(min_n, false, true, seed))?;

    // ── 4) clean up helper column & return --------------------------------
    balanced.drop("__bucket__")
}


use polars::prelude::*;
use std::collections::HashSet;

/// -------------------------------------------------------------------------
///  Stats returned by `stratified_within_gene_balance`
/// -------------------------------------------------------------------------
#[derive(Debug)]
pub struct StratifiedSampleStats {
    pub total_genes: usize,            // genes in the original frame
    pub kept_genes: usize,             // genes that passed the ≥1-good ≥1-poor filter
    pub dropped_genes: Vec<String>,    // list of genes that were removed
    pub guides_per_gene: DataFrame,    // columns: Gene | n_guides_kept (after sampling)
}

/// -------------------------------------------------------------------------
///  Balance good : poor within each gene
///     – `good_cutoff`  … efficacy ≥ cutoff  ⇒  "good"
///     – efficacy  < cutoff                  ⇒  "poor"
///  Returns `(balanced_df, stats)`
/// -------------------------------------------------------------------------

pub fn stratified_within_gene_balance(
    df: &DataFrame,
    efficacy_col: &str,
    good_cutoff: f64,
    min_gap: Option<f64>,          // ← already supported
    seed: Option<u64>,
) -> PolarsResult<(DataFrame, StratifiedSampleStats)> {
    // ── 0) bucket “good” vs “poor” ───────────────────────────────────────
    let with_bucket = df
        .clone()
        .lazy()
        .with_column(
            when(col(efficacy_col).gt_eq(lit(good_cutoff)))
                .then(lit("good"))
                .otherwise(lit("poor"))
                .alias("__bucket"),
        )
        .collect()?;

    // ── 1) genes that pass the gap / both-bucket test ────────────────────
    let genes_keep = with_bucket
        .clone()
        .lazy()
        .group_by([col("Gene")])
        .agg([
            col("__bucket").n_unique().alias("nq"),
            col(efficacy_col)
                .filter(col("__bucket").eq(lit("good")))
                .max()
                .alias("max_good"),
            col(efficacy_col)
                .filter(col("__bucket").eq(lit("poor")))
                .min()
                .alias("min_poor"),
        ])
        .filter(
            col("nq").eq(lit(2)).and(match min_gap {
                Some(g) => (col("max_good") - col("min_poor")).gt(lit(g)),
                None => lit(true),
            }),
        )
        .select([col("Gene")])
        .collect()?;   // eager for joins later

    // ── 2) bookkeeping sets ─────────────────────────────────────────────
    let all_genes: HashSet<_> = df
        .column("Gene")?
        .str()?
        .into_no_null_iter()
        .map(String::from)
        .collect();

    let kept_gene_set: HashSet<_> = genes_keep
        .column("Gene")?
        .str()?
        .into_no_null_iter()
        .map(String::from)
        .collect();

    let dropped_genes: Vec<_> = all_genes.difference(&kept_gene_set).cloned().collect();

    // ── 3) restrict to eligible genes ────────────────────────────────────
    let eligible = with_bucket.join(
        &genes_keep,
        ["Gene"],
        ["Gene"],
        JoinArgs::from(JoinType::Inner),
        None,
    )?;

    // ── 4) balance within each gene ──────────────────────────────────────
    let balanced = eligible
        .group_by_stable(["Gene"])?
        .apply(|g| {
            let good = g.filter(&g.column("__bucket")?.str()?.equal("good"))?;
            let poor = g.filter(&g.column("__bucket")?.str()?.equal("poor"))?;
            let k = good.height().min(poor.height());
            if k == 0 {
                return Ok(DataFrame::empty());
            }
            let sample = |df: DataFrame| df.sample_n_literal(k, false, true, seed);
            let mut keep = sample(good)?;
            keep.vstack_mut(&sample(poor)?)?;
            Ok(keep)
        })?;

    // ── 5a) guides kept per gene (already used downstream) ───────────────
    let guides_per_gene = balanced
        .clone()
        .lazy()
        .group_by([col("Gene")])
        .agg([col("Gene").count().alias("n_guides_kept")])
        .collect()?;

    // ── 5b) NEW: total + dropped guides per gene + status ────────────────
    let guides_total = with_bucket
        .clone()
        .lazy()
        .group_by([col("Gene")])
        .agg([col("Gene").count().alias("n_guides_total")])
        .collect()?;

    let mut per_gene_stats = guides_total
        .join(
            &guides_per_gene,
            ["Gene"],
            ["Gene"],
            JoinArgs::from(JoinType::Left),
            None,
        )?
        .lazy()
        .with_column(
            col("n_guides_kept").fill_null(lit(0)).alias("n_guides_kept"),
        )
        .with_column(
            (col("n_guides_total") - col("n_guides_kept")).alias("n_guides_dropped"),
        )
        .with_column(
            when(col("n_guides_kept").eq(lit(0)))
                .then(lit("dropped"))
                .otherwise(lit("kept"))
                .alias("status"),
        )
        .collect()?;

    // ── 5c) NEW: persist stats → CSV (helper handles dirs & headers) ────
    let csv_out = "./stats/stratified_balance_stats.csv";
    dataframe_to_csv(&mut per_gene_stats, csv_out, true)?;

    // ── 6) global stats & return ─────────────────────────────────────────
    let stats = StratifiedSampleStats {
        total_genes: all_genes.len(),
        kept_genes: kept_gene_set.len(),
        dropped_genes,
        guides_per_gene,
    };

    Ok((balanced.drop("__bucket")?, stats))
}

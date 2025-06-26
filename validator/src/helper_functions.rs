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


pub fn stratified_within_gene_balance_with_margin(
    df: &DataFrame,
    efficacy_col: &str,
    margin_low: f64,   // e.g., 75.0
    margin_high: f64,  // e.g., 85.0
    seed: Option<u64>,
) -> PolarsResult<(DataFrame, StratifiedSampleStats)> {
    // ── 0) Create buckets: "poor" < 75, "ignore" 75-85, "good" > 85 ─────
    let with_bucket = df
        .clone()
        .lazy()
        .with_column(
            when(col(efficacy_col).lt(lit(margin_low)))
                .then(lit("poor"))
                .when(col(efficacy_col).gt(lit(margin_high)))
                .then(lit("good"))
                .otherwise(lit("ignore"))  // This is our safety zone!
                .alias("__bucket"),
        )
        .collect()?;

    // ── 1) Filter out the "ignore" zone entirely ─────────────────────────
    let usable_data = with_bucket
        .clone()
        .lazy()
        .filter(col("__bucket").neq(lit("ignore")))
        .collect()?;

    // ── 2) Find genes that have BOTH good AND poor guides ───────────────
    let genes_keep = usable_data
        .clone()
        .lazy()
        .group_by([col("Gene")])
        .agg([
            col("__bucket").n_unique().alias("nq"),
            col("Gene").count().alias("n_usable"),
        ])
        .filter(col("nq").eq(lit(2)))  // Must have both "good" and "poor"
        .select([col("Gene")])
        .collect()?;

    // ── 3) Track which genes we're keeping/dropping ─────────────────────
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

    // ── 4) Keep only eligible genes and usable guides ───────────────────
    let eligible = usable_data.join(
        &genes_keep,
        ["Gene"],
        ["Gene"],
        JoinArgs::from(JoinType::Inner),
        None,
    )?;

    // ── 5) Balance within each gene (equal good/poor) ───────────────────
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

    // ── 6) Calculate statistics ─────────────────────────────────────────
    let guides_per_gene = balanced
        .clone()
        .lazy()
        .group_by([col("Gene")])
        .agg([col("Gene").count().alias("n_guides_kept")])
        .collect()?;

    // Include margin zone stats
    let margin_stats = with_bucket
        .clone()
        .lazy()
        .group_by([col("Gene")])
        .agg([
            col("Gene").count().alias("n_total"),
            col("__bucket").filter(col("__bucket").eq(lit("ignore")))
                .count().alias("n_in_margin"),
            col("__bucket").filter(col("__bucket").eq(lit("good")))
                .count().alias("n_good_eligible"),
            col("__bucket").filter(col("__bucket").eq(lit("poor")))
                .count().alias("n_poor_eligible"),
        ])
        .collect()?;

    let mut per_gene_stats = margin_stats
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
            when(col("n_guides_kept").eq(lit(0)))
                .then(lit("dropped"))
                .otherwise(lit("kept"))
                .alias("status"),
        )
        .with_column(
            (col("n_in_margin").cast(DataType::Float64) / col("n_total").cast(DataType::Float64))
                .alias("pct_in_margin"),
        )
        .collect()?;

    // Save detailed stats
    let csv_out = "./stats/stratified_margin_balance_stats.csv";
    dataframe_to_csv(&mut per_gene_stats, csv_out, true)?;

    // Log summary info
    info!(
        "Margin zone [{}-{}]: Kept {} of {} genes",
        margin_low, margin_high, kept_gene_set.len(), all_genes.len()
    );

    let stats = StratifiedSampleStats {
        total_genes: all_genes.len(),
        kept_genes: kept_gene_set.len(),
        dropped_genes,
        guides_per_gene,
    };

    Ok((balanced.drop("__bucket")?, stats))
}


// In your Rust helper_functions.rs or main.rs

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// RGB color representation that can be serialized
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to normalized RGB tuple for Python
    pub fn to_normalized(&self) -> (f32, f32, f32) {
        (
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
        )
    }
}

/// Model color mappings that can be passed to Python
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelColors {
    colors: HashMap<String, Color>,
}

impl ModelColors {
    pub fn new() -> Self {
        let mut colors = HashMap::new();

        // Foundational models - Different gray shades
        colors.insert("TKO PSSM".to_string(), Color::new(80, 80, 80));          // Dark gray
        colors.insert("Moreno-Mateos".to_string(), Color::new(130, 130, 130));  // Medium gray

        // Classical ML - Different blue shades
        colors.insert("Doench Rule Set 2".to_string(), Color::new(31, 119, 180)); // Standard blue
        colors.insert("Doench Rule Set 3".to_string(), Color::new(70, 130, 200)); // Lighter blue

        // CNN models - Different orange/red shades
        colors.insert("DeepCRISPR".to_string(), Color::new(255, 127, 14));       // Orange
        colors.insert("DeepSpCas9".to_string(), Color::new(255, 87, 51));        // Red-orange

        // Transformer models - Purple shades
        colors.insert("TransCRISPR".to_string(), Color::new(200, 120, 189));     // Standard purple

        // Ensemble/Consensus models - Different green shades
        colors.insert("Linear Consensus".to_string(), Color::new(44, 160, 44));   // Standard green
        colors.insert("Logistic Consensus".to_string(), Color::new(60, 180, 60)); // Lighter green

        Self { colors }
    }

    pub fn get_color(&self, model_name: &str) -> Option<Color> {
        self.colors.get(model_name).copied()
    }

    pub fn add_model(&mut self, model_name: String, color: Color) {
        self.colors.insert(model_name, color);
    }

    /// Export colors as JSON string for passing to Python
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.colors)
    }

    /// Save colors to a JSON file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.colors)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
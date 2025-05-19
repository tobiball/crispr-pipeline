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

pub fn label_by_efficacy(df: DataFrame) -> PolarsResult<DataFrame> {
    df.lazy()
        .with_column(
            when(col("efficacy").gt_eq(lit(80.0)))
                .then(lit("good"))
                .when(col("efficacy").gt_eq(lit(40.0)))
                .then(lit("moderate"))
                .otherwise(lit("bad"))
                .alias("quality"),
        )
        .collect()        // back to an eager DataFrame
}


use polars::prelude::*;

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



use polars::prelude::*;

/// Down-sample every class (good/moderate/bad) to the size of the smallest one
pub fn undersample_equal_classes(
    df: &DataFrame,
    class_col: &str,
    seed: Option<u64>,
) -> PolarsResult<DataFrame> {
    // size of the smallest bucket
    let min_n = df
        .group_by([class_col])?
        .count()?
        .column("count")?
        .u32()?
        .min()
        .unwrap() as usize;

    // sample `min_n` rows without replacement inside each class
    df.group_by([class_col])?
        .apply(|g| g.sample_n_literal(min_n, /*replace*/ false, /*shuffle*/ true, seed))?
        // .apply already concatenates the groups; no explode needed
        .pipe(Ok)
}

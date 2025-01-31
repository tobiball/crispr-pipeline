use tracing::{debug, error, info};
use std::path::PathBuf;
use polars::error::PolarsResult;
use polars::frame::DataFrame;
use polars::prelude::{col, lit, CsvParseOptions, CsvReadOptions, IntoLazy, SerReader, Series};

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
use serde_json::json;

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


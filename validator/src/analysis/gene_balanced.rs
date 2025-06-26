//! Produce a grouped violin-plot of *per-gene AUROC*.
//!
//! ```rust
//! plot_auroc_violin(
//!     &df,
//!     "Gene",                  // gene column
//!     tools_to_evaluate,       // Vec<&str> with prediction columns
//!     90.0,                    // cutoff turning efficacy → binary label
//!     "./figures/auroc_violin.png"
//! )?;
//! ```

use polars::prelude::*;
use std::process::Command;
use tempfile::NamedTempFile;
use tracing::info;

use crate::helper_functions::project_root;
use crate::models::polars_err;

const LABEL_COL: &str = "efficacy";   // continuous ground-truth column
pub fn plot_auroc_violin(
    df: &DataFrame,
    gene_col: &str,
    alg_cols: Vec<&str>,
    cutoff: f64,
    database_name: &str,
    output_path: &str,
) -> PolarsResult<()> {
    // ── 1) minimal slice ───────────────────────────────────────────
    let mut keep = vec![gene_col, LABEL_COL];
    keep.extend_from_slice(&alg_cols);
    let mini = df.select(keep)?;

    // ── 2) dump temp CSV ───────────────────────────────────────────
    let tmp = NamedTempFile::new().map_err(|e| polars_err(Box::new(e)))?;
    CsvWriter::new(&mut tmp.as_file())
        .include_header(true)
        .finish(&mut mini.clone())?;
    let tmp_path = tmp.path().to_owned();

    // ── 3) call Python helper with enhanced parameters ─────────────
    let python = project_root().join("scripts/efficacy_analysis_env/bin/python");
    let script  = project_root().join("scripts/per_gene_auroc_violin.py");

    info!("Launching enhanced per-gene AUROC analysis for {} …", database_name);
    let status = Command::new(python)
        .arg(script)
        .arg("--csv").arg(&tmp_path)
        .arg("--gene-col").arg(gene_col)
        .arg("--label-col").arg(LABEL_COL)
        .arg("--alg-cols").arg(alg_cols.join(","))
        .arg("--cutoff").arg(cutoff.to_string())
        .arg("--database-name").arg(database_name)
        .arg("--out").arg(output_path)
        .status()
        .map_err(|e| polars_err(Box::new(e)))?;

    if !status.success() {
        return Err(PolarsError::ComputeError(
            format!("Enhanced AUROC analysis helper exited with status {status}").into(),
        ));
    }

    info!("Enhanced per-gene AUROC analysis for {} written to {}", database_name, output_path);
    Ok(())
}

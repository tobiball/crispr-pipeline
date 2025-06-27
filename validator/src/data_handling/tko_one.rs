// src/data_handling/tko_one.rs

use polars::lazy::dsl::*;
use polars::prelude::*;
use tracing::{debug, info, warn};

use crate::helper_functions::project_root;
use crate::mageck_processing::run_mageck_pipeline;
use crate::models::{polars_err, Dataset};

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum ScreenKind {
    IlludinS_WT,
    Essentialome_ELOF1KO,
    IlludinS_CSBKO,
    IlludinS_ELOF1KO,
}

impl std::fmt::Display for ScreenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            ScreenKind::IlludinS_WT => "WT_IlludinS",
            ScreenKind::Essentialome_ELOF1KO => "ELOF1KO_Essentialome",
            ScreenKind::IlludinS_CSBKO => "CSBKO_IlludinS",
            ScreenKind::IlludinS_ELOF1KO => "ELOF1KO_IlludinS",
        };
        write!(f, "{s}")
    }
}

/// Two-file bundle for a TKO Illudin-S screen
pub struct TkoScreensDataset {
    pub screen_path: PathBuf,   // Excel read-count workbook
    pub guide_map_path: PathBuf // TKO-v3 annotation **TSV**
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// ─── ColumnBuckets ───────────────────────────────────────────────────────────
struct ColumnBuckets {
    t0_ctrl: Vec<&'static str>,
    ut_treat: Vec<&'static str>,
}
impl ColumnBuckets {
    fn wt_illudins() -> Self {
        Self {
            t0_ctrl: Vec::from(&["WT_T0_R1", "WT_T0_R2", "WT_T0_R3"]),
            ut_treat: Vec::from(&["WT_UT_T12_R1", "WT_UT_T12_R2", "WT_UT_T12_R3"]),
        }
    }
}

// ─── Constants ───────────────────────────────────────────────────────────────
const PLUS_STRAND_OFFSET: i32  = 17;   // TKO: protospacer starts 17 nt downstream of cut
const MINUS_STRAND_OFFSET: i32 = 6;    // TKO: 6 nt upstream on “−” strand

/// Larger window for re-anchoring (±1000 nt)
const REANCHOR_PAD: i64 = 1_000;

// ─── read_excel ──────────────────────────────────────────────────────────────

fn cell_to_string(cell: &calamine::DataType) -> String {
    use calamine::DataType as Ct;
    match cell {
        Ct::String(s) => s.clone(),
        Ct::Empty => String::new(),
        Ct::Bool(b) => b.to_string(),
        Ct::Error(e) => format!("ERR({e:?})"),
        Ct::Float(n) | Ct::Duration(n) => n.to_string(),
        Ct::Int(i) => i.to_string(),
        Ct::DateTime(f) => f.to_string(),
        Ct::DateTimeIso(s) | Ct::DurationIso(s) => s.clone(),
    }
}

/// Minimal xlsx→String DF loader (same helper Avana uses)
/// Improved xlsx→DataFrame loader with proper numeric type detection
fn read_excel(path: &str, sheet_idx: usize) -> PolarsResult<DataFrame> {
    use calamine::{open_workbook_auto, Reader};

    let mut wb = open_workbook_auto(path).map_err(|e| polars_err(Box::new(e)))?;
    let range = wb
        .worksheet_range_at(sheet_idx)
        .ok_or_else(|| polars_err("worksheet missing".into()))?
        .map_err(|e| polars_err(Box::new(e)))?;

    let mut rows = range.rows();
    let headers: Vec<String> = rows
        .next()
        .ok_or_else(|| polars_err("empty sheet".into()))?
        .iter()
        .map(cell_to_string)
        .collect();
    debug!("Read-counts header = {:?}", headers);

    // Determine which columns should be numeric based on header patterns
    let is_count_column = |header: &str| -> bool {
        header.contains("_R") && (
            header.contains("_T0_") ||
                header.contains("_T12_") ||
                header.contains("_T16_") ||
                header.contains("_UT_") ||
                header.contains("_IlludinS_")
        )
    };

    // Build columns with proper types
    let mut cols: Vec<Series> = Vec::with_capacity(headers.len());

    for (col_idx, header) in headers.iter().enumerate() {
        let mut rows = range.rows();
        rows.next(); // Skip header row

        if is_count_column(header) {
            // This is a count column - parse as i64
            debug!("Parsing {} as numeric count column", header);
            let mut values: Vec<Option<i64>> = Vec::new();

            for row in rows {
                let val = if let Some(cell) = row.get(col_idx) {
                    match cell {
                        calamine::DataType::Int(i) => Some(*i),
                        calamine::DataType::Float(f) => Some(*f as i64),
                        calamine::DataType::String(s) => {
                            // Try to parse string as number
                            s.trim().parse::<i64>().ok()
                        }
                        calamine::DataType::Empty => Some(0), // Empty cells = 0 counts
                        _ => None,
                    }
                } else {
                    None
                };
                values.push(val);
            }

            // Create series with null handling
            cols.push(Series::new(PlSmallStr::from(header), values));
        } else {
            // Non-count column - keep as string
            let mut values: Vec<Option<String>> = Vec::new();

            for row in rows {
                let val = row.get(col_idx).map(|cell| match cell {
                    calamine::DataType::Empty => None,
                    c => Some(cell_to_string(c)),
                }).flatten();
                values.push(val);
            }

            cols.push(Series::new(PlSmallStr::from(header), values));
        }
    }

    let df = DataFrame::new(cols.into_iter().map(Into::into).collect())?;


    Ok(df)
}

// ─── read_tko_annotation ─────────────────────────────────────────────────────

/// Robust TSV reader (expects columns: CODE, GENE, STARTpos, ENDpos, chromosome, strand)
fn read_tko_annotation(path: &PathBuf) -> PolarsResult<DataFrame> {
    use polars::prelude::CsvEncoding;

    // accept either .txt or .tsv
    let ext_ok = path
        .extension()
        .and_then(|e| e.to_str())
        .map_or(false, |s| s.eq_ignore_ascii_case("txt") || s.eq_ignore_ascii_case("tsv"));
    if !ext_ok {
        return Err(polars_err("guide_map_path must point to a TKO-v3 annotation file ending in .txt or .tsv".into()));
    }
    if !path.exists() {
        return Err(polars_err("TKO-v3 annotation file not found".into()));
    }

    // Read everything as String to avoid CHRM→Int conversions
    let header_fields = {
        let file = std::fs::File::open(path).map_err(|e| polars_err(Box::new(e)))?;
        let mut rdr = BufReader::new(file);
        let mut hdr = String::new();
        rdr.read_line(&mut hdr).map_err(|e| polars_err(Box::new(e)))?;
        hdr.trim_end().split('\t').count()
    };
    let dtype_override: Arc<Vec<DataType>> = Arc::new(vec![DataType::String; header_fields]);

    let mut df_string = CsvReadOptions::default()
        .with_has_header(true)
        .with_dtype_overwrite(Some(dtype_override))
        .map_parse_options(|mut o| {
            o.separator = b'\t';
            o.quote_char = None;
            o.encoding = CsvEncoding::LossyUtf8;
            o.truncate_ragged_lines = true;
            o
        })
        .try_into_reader_with_file_path(Some(path.clone()))?
        .finish()?;

    // Normalize header edge cases
    {
        let mut names = df_string.get_column_names_owned();
        if names.get(0).map_or(true, |s| s.is_empty()) {
            names[0] = "CODE".into();
        }
        if names.get(1) == Some(&"CODE".into()) {
            names[1] = "GENE".into();
        }
        df_string.set_column_names(names)?;
    }

    // Here we expect columns: CODE, GENE, STARTpos, ENDpos, chromosome, strand
    let df = df_string
        .lazy()
        .with_columns([
            col("STARTpos").cast(DataType::Int64),
            col("ENDpos").cast(DataType::Int64),
            col("chromosome"),           // lowercase “chromosome”
            col("CODE").alias("sgRNA"),  // rename CODE→sgRNA
        ])
        .collect()?;

    // Compute 1-based cut-site position (PAM–17 / PAM–6)
    df.lazy()
        .with_column(
            when(col("strand") == lit("+"))
                .then(col("STARTpos") + lit(PLUS_STRAND_OFFSET as i64))
                .otherwise(col("ENDpos") - lit(MINUS_STRAND_OFFSET as i64))
                .alias("position"),
        )
        .select([
            col("CODE"),
            col("sgRNA"),
            col("chromosome"),
            col("strand"),
            col("position"),
        ])
        .collect()
}

// ─── extract_genomic_sequence & reverse_complement ────────────────────────────

fn extract_genomic_sequence(chrom: &str, start: i64, end: i64)
                            -> Result<String, Box<dyn std::error::Error>>
{
    let twobit_path = project_root().join("chopchop/hg38.2bit");
    let twobit_to_fa = project_root()
        .join("chopchop/twoBitToFa")
        .to_string_lossy()
        .to_string();
    let tmp = format!("/tmp/seq_{chrom}_{start}_{end}.fa");

    Command::new(&twobit_to_fa)
        .arg(twobit_path)
        .arg(&tmp)
        .arg(format!("-seq={chrom}"))
        .arg(format!("-start={start}"))
        .arg(format!("-end={end}"))
        .output()?;

    let fasta = std::fs::read_to_string(&tmp)?;
    std::fs::remove_file(&tmp)?;
    Ok(fasta.lines().skip(1).collect())
}

fn reverse_complement(seq: &str) -> String {
    seq.chars()
        .rev()
        .map(|c| match c {
            'A' | 'a' => 'T',
            'T' | 't' => 'A',
            'G' | 'g' => 'C',
            'C' | 'c' => 'G',
            'N' | 'n' => 'N',
            x         => x,
        })
        .collect()
}


fn reanchor_with_window(mut df: DataFrame) -> PolarsResult<DataFrame> {
    let chr_col = df.column("chromosome")?.str().unwrap();
    let old_pos_col = df.column("position")?.i64().unwrap();
    let strand_col = df.column("strand")?.str().unwrap();
    let guide_col = df.column("sgRNA")?.str().unwrap();

    // Track which rows we want to keep (successfully re-anchored)
    let mut keep_rows: Vec<bool> = Vec::with_capacity(df.height());
    let mut dropped_count = 0;

    // Prepare vectors for new columns (only for rows we keep)
    let mut debug_starts: Vec<i64> = Vec::new();
    let mut new_positions: Vec<i64> = Vec::new();
    let mut new_starts: Vec<i64> = Vec::new();
    let mut new_ends: Vec<i64> = Vec::new();
    let mut seq30_vec: Vec<String> = Vec::new();
    let mut pam_vec: Vec<String> = Vec::new();
    let mut seq23_vec: Vec<String> = Vec::new();
    let mut coord_offset: Vec<i64> = Vec::new();
    let mut raw_sequence_dbg: Vec<String> = Vec::new();

    for i in 0..df.height() {
        // Pull raw annotation values
        let chr_opt = chr_col.get(i);
        let pos_opt = old_pos_col.get(i);
        let strand_opt = strand_col.get(i);
        let guide_opt = guide_col.get(i);

        // If any essential column missing → drop this row
        if chr_opt.is_none() || pos_opt.is_none() || strand_opt.is_none() || guide_opt.is_none() {
            keep_rows.push(false);
            dropped_count += 1;
            continue;
        }

        let chr = chr_opt.unwrap();
        let old_pos = pos_opt.unwrap();
        let strand = strand_opt.unwrap();
        let guide = guide_opt.unwrap().to_uppercase();

        // Build genomic window
        let win0_start = (old_pos - 1).saturating_sub(REANCHOR_PAD);
        let win0_end = (old_pos - 1) + REANCHOR_PAD + 1;

        // Extract genomic sequence
        let raw_window = extract_genomic_sequence(chr, win0_start, win0_end)
            .unwrap_or_else(|_| String::new())
            .to_uppercase();

        if raw_window.is_empty() {
            debug!("Dropping guide at row {}: failed to extract genomic sequence for {} at {}:{}",
                   i, guide, chr, old_pos);
            keep_rows.push(false);
            dropped_count += 1;
            continue;
        }

        // Find guide sequence (or its reverse complement)
        let guide_rc = guide.chars()
            .rev()
            .map(|c| match c {
                'A' | 'a' => 'T',
                'T' | 't' => 'A',
                'G' | 'g' => 'C',
                'C' | 'c' => 'G',
                'N' | 'n' => 'N',
                x => x,
            })
            .collect::<String>();

        let (needle, flipped) = if strand == "+" {
            if raw_window.find(&guide).is_some() {
                (guide.clone(), false)
            } else if raw_window.find(&guide_rc).is_some() {
                (guide_rc.clone(), true)
            } else {
                (String::new(), false)
            }
        } else {
            if raw_window.find(&guide_rc).is_some() {
                (guide_rc.clone(), false)
            } else if raw_window.find(&guide).is_some() {
                (guide.clone(), true)
            } else {
                (String::new(), false)
            }
        };

        if needle.is_empty() {
            debug!("Dropping guide at row {}: {} not found in genomic window ±{}bp at {}:{}{}",
                   i, guide, REANCHOR_PAD, chr, old_pos, strand);
            keep_rows.push(false);
            dropped_count += 1;
            continue;
        }

        // Successfully found the guide - proceed with re-anchoring
        keep_rows.push(true);

        // [Rest of the re-anchoring logic stays the same...]
        let hit0 = raw_window.find(&needle).unwrap() as i64;
        let protospacer_0 = win0_start + hit0;
        let debug_start_1 = protospacer_0 + 1;

        let old_anchor = if strand == "+" {
            old_pos - PLUS_STRAND_OFFSET as i64
        } else {
            old_pos + MINUS_STRAND_OFFSET as i64
        };
        let offset_i = debug_start_1 - old_anchor;

        let actual_plus = (strand == "+" && !flipped) || (strand == "−" && flipped);

        // Extract raw ±3 nt around matched protospacer
        let guide_len = guide.chars().count() as i64;
        let start_dbg = if hit0 >= 3 { (hit0 - 3) as usize } else { 0 };
        let end_dbg = {
            let tentative = hit0 + guide_len + 3;
            if tentative as usize <= raw_window.len() {
                tentative as usize
            } else {
                raw_window.len()
            }
        };
        let raw_dbg = raw_window[start_dbg..end_dbg].to_string();
        let raw_dbg_oriented = if actual_plus {
            raw_dbg.clone()
        } else {
            reverse_complement(&raw_dbg)
        };

        let cut1 = if actual_plus {
            debug_start_1 + PLUS_STRAND_OFFSET as i64
        } else {
            debug_start_1 + MINUS_STRAND_OFFSET as i64
        };

        // Extract 30 nt DeepSpCas9 window
        let start_idx = if hit0 >= 4 { (hit0 - 4) as usize } else { 0 };
        let end_idx = std::cmp::min(raw_window.len(), (hit0 + 26) as usize);
        let mut seq30_slice = String::new();
        if start_idx < end_idx && (end_idx - start_idx) == 30 {
            let slice = raw_window[start_idx..end_idx].to_string();
            seq30_slice = if actual_plus {
                slice.clone()
            } else {
                reverse_complement(&slice)
            };
        }

        // Extract 23 nt (guide + PAM)
        let mut raw23 = String::new();
        if actual_plus {
            if (hit0 + 23) as usize <= raw_window.len() {
                raw23 = raw_window[(hit0 as usize)..(hit0 as usize + 23)].to_string();
            }
        } else {
            if hit0 >= 3 && ((hit0 + guide_len) as usize) <= raw_window.len() {
                raw23 = raw_window[((hit0 as usize - 3))..((hit0 as usize + guide_len as usize))].to_string();
            }
        }

        let mut seq23 = String::new();
        let mut pam = String::new();
        if !raw23.is_empty() {
            let oriented23 = if actual_plus {
                raw23.clone()
            } else {
                reverse_complement(&raw23)
            };

            let guide_fixed = if oriented23.len() == 23 && guide.chars().count() == 19 {
                let b0 = oriented23.chars().next().unwrap();
                if b0 != 'N' {
                    let mut g20 = String::with_capacity(20);
                    g20.push(b0);
                    g20.push_str(&guide);
                    g20
                } else {
                    guide.clone()
                }
            } else {
                guide.clone()
            };

            if oriented23.len() >= 23 {
                pam = oriented23[20..23].to_string();
            }
            seq23 = format!("{guide_fixed}{pam}");
        }

        let new_start = debug_start_1;
        let new_end = if guide_len == 20 {
            debug_start_1 + 19
        } else {
            debug_start_1 + 18
        };

        // Push into vectors (only for kept rows)
        debug_starts.push(debug_start_1);
        new_positions.push(cut1);
        new_starts.push(new_start);
        new_ends.push(new_end);
        seq30_vec.push(seq30_slice);
        pam_vec.push(pam);
        seq23_vec.push(seq23);
        coord_offset.push(offset_i);
        raw_sequence_dbg.push(raw_dbg_oriented);
    }

    // Log summary of dropped guides
    if dropped_count > 0 {
        warn!("Dropped {} guide(s) that could not be re-anchored against hg38 reference", dropped_count);
        info!("Continuing with {} successfully re-anchored guides", df.height() - dropped_count);
    } else {
        info!("All {} guides successfully re-anchored against hg38 reference", df.height());
    }

    // Filter the DataFrame to keep only successfully re-anchored rows
    let keep_mask = Series::new(PlSmallStr::from("keep"), keep_rows);
    let mut df_filtered = df.filter(&keep_mask.bool().unwrap())?;

    // Attach new columns to the filtered DataFrame
    let df_final = df_filtered
        .with_column(Series::new("debug_start".into(), debug_starts))?
        .with_column(Series::new("position".into(), new_positions))?
        .with_column(Series::new("start".into(), new_starts))?
        .with_column(Series::new("end".into(), new_ends))?
        .with_column(Series::new("sequence_deepspcas9".into(), seq30_vec))?
        .with_column(Series::new("pam".into(), pam_vec))?
        .with_column(Series::new("sequence_with_pam".into(), seq23_vec))?
        .with_column(Series::new("coord_offset".into(), coord_offset))?
        .with_column(Series::new("raw_sequence_debug".into(), raw_sequence_dbg))?
        .clone();

    Ok(df_final)
}

// ─── ensure_chr_prefix ────────────────────────────────────────────────────────

fn ensure_chr_prefix(df: DataFrame) -> PolarsResult<DataFrame> {
    let df_fixed = df
        .lazy()
        .with_column(
            when(col("chromosome").str().starts_with(lit("chr")))
                .then(col("chromosome"))
                .otherwise(lit("chr") + col("chromosome"))
                .alias("chromosome"),
        )
        .collect()?;
    Ok(df_fixed)
}

// ─── Main Dataset impl ────────────────────────────────────────────────────────

impl Dataset for TkoScreensDataset {
    fn load(&self) -> PolarsResult<DataFrame> {
        info!("Loading TKO workbook {}", self.screen_path.display());
        let df_raw = read_excel(self.screen_path.to_str().unwrap(), 0)?;
        let buckets = ColumnBuckets::wt_illudins();

        // (1) split Gene_Guide → Gene, sgRNA
        let df_guides = df_raw
            .lazy()
            .with_columns([
                col("Gene_Guide")
                    .str()
                    .split(lit("_"))
                    .list()
                    .get(lit(0), false)
                    .alias("Gene"),
                col("Gene_Guide")
                    .str()
                    .split(lit("_"))
                    .list()
                    .get(lit(1), false)
                    .alias("sgRNA"),
            ])
            .collect()?;

        // (2) read TKO-v3 annotation
        let annot = read_tko_annotation(&self.guide_map_path)?;
        debug!("Annotation rows = {}", annot.height());

        // (3) join on Gene_Guide == CODE
        let df_joined = df_guides.join(
            &annot,
            ["Gene_Guide"],   // left key from Excel
            ["CODE"],         // right key from annotation
            JoinArgs::from(JoinType::Inner),
            None,
        )?;

        // (4) ensure "chr" prefix on 'chromosome'
        let mut df = ensure_chr_prefix(df_joined)?;

        // (5) add "screen" constant column
        let num_rows = df.height();
        let screen_vals = vec![ScreenKind::IlludinS_WT.to_string(); num_rows];
        df = df.with_column(Series::new(PlSmallStr::from("screen"), screen_vals))?.clone();



        Ok(df)
    }

    fn augment_guides(df: DataFrame) -> PolarsResult<DataFrame> {
        // Perform re-anchoring here:
        let df_reanchored = reanchor_with_window(df)?;
        let df = df_reanchored.lazy()
            .with_columns([
                col("pam").str().to_uppercase().alias("pam"),
                col("sequence_deepspcas9").str().to_uppercase().alias("sequence_deepspcas9"),
                col("sequence_with_pam").str().to_uppercase().alias("sequence_with_pam"),
            ])
            .collect()?;
        Ok(df)
    }

    fn mageck_efficency_scoring(mut df: DataFrame) -> PolarsResult<DataFrame> {
        let buckets = ColumnBuckets::wt_illudins();
        let treat: Vec<String> = buckets.ut_treat.iter().map(|s| s.to_string()).collect();
        let ctrl: Vec<String> = buckets.t0_ctrl.iter().map(|s| s.to_string()).collect();

        // check_essentialome_depletion(&df)?;


        info!(
            "Running MAGeCK with {} ctrl and {} treat replicates",
            ctrl.len(),
            treat.len()
        );

        let count_cols: Vec<&str> = ctrl.iter().chain(treat.iter()).map(|s| s.as_str()).collect();

        df = run_mageck_pipeline(
            df,
            &project_root()
                .join("mageck/mageck_venv/bin/mageck")
                .to_string_lossy(),
            "./mageck_processing_artifacts/tp53_tko",
            &treat,
            &ctrl,
            "sgRNA",
            "Gene",
            &count_cols,
        )?;
        Ok(df)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn all_pams_are_ngg() {
        // Assume data/tko/tko_hg38_annotation_for_rust.txt exists and is in the correct format,
        // and data/tko/tko_one.xlsx is the counts file.
        let guide_map = PathBuf::from("data/tko/tko_hg38_annotation_for_rust.txt");
        let screen_xlsx = PathBuf::from("data/tko/tko_one.xlsx");
        let tko = TkoScreensDataset {
            screen_path: screen_xlsx.clone(),
            guide_map_path: guide_map.clone(),
        };

        // Load then re-anchor via augment_guides
        let df_loaded = tko.load().expect("Failed to load TKO dataset");
        let df_augmented = TkoScreensDataset::augment_guides(df_loaded).expect("Failed to augment");
        let pam_series = df_augmented.column("pam").expect("Missing 'pam' column");
        let pam_str = pam_series.str().unwrap();
        for i in 0..pam_str.len() {
            if let Some(p) = pam_str.get(i) {
                // Skip empty strings (guides not found)
                if p.is_empty() {
                    continue;
                }
                let chars: Vec<char> = p.chars().collect();
                assert_eq!(chars.len(), 3, "PAM length is not 3 at row {}", i);
                // NGG: second and third positions must be 'G'
                assert_eq!(chars[1], 'G', "PAM[1] != 'G' at row {}", i);
                assert_eq!(chars[2], 'G', "PAM[2] != 'G' at row {}", i);
            }
        }
    }
}

pub fn check_essentialome_depletion(df: &DataFrame) -> PolarsResult<()> {
    info!("=== Checking for essentialome depletion pattern ===");

    // Calculate average counts for each guide at T0 and T12
    let t0_cols = vec!["WT_T0_R1", "WT_T0_R2", "WT_T0_R3"];
    let t12_cols = vec!["WT_UT_T12_R1", "WT_UT_T12_R2", "WT_UT_T12_R3"];

    // First, ensure count columns are numeric
    let mut df_numeric = df.clone();
    for col_name in t0_cols.iter().chain(t12_cols.iter()) {
        if let Ok(col) = df_numeric.column(col_name) {
            if col.dtype() == &DataType::String {
                info!("Converting {} from String to Int64", col_name);
                // Cast string column to i64
                let numeric_col = col
                    .cast(&DataType::Int64)
                    .map_err(|_| polars_err(format!("Failed to convert {} to numeric", col_name).into()))?;
                df_numeric.with_column(numeric_col)?;
            }
        }
    }

    // Create expressions for averaging with proper casting
    let t0_avg = (
        col(&*t0_cols[0]).cast(DataType::Float64) +
            col(&*t0_cols[1]).cast(DataType::Float64) +
            col(&*t0_cols[2]).cast(DataType::Float64)
    ) / lit(3.0);

    let t12_avg = (
        col(&*t12_cols[0]).cast(DataType::Float64) +
            col(&*t12_cols[1]).cast(DataType::Float64) +
            col(&*t12_cols[2]).cast(DataType::Float64)
    ) / lit(3.0);

    // Calculate fold changes
    let df_with_fc = df_numeric.clone().lazy()
        .with_columns([
            t0_avg.alias("avg_t0"),
            t12_avg.alias("avg_t12"),
        ])
        .with_column(
            (col("avg_t12") / col("avg_t0")).alias("fold_change")
        )
        .collect()?;

    // Analyze fold changes
    let fc_stats = df_with_fc.clone().lazy()
        .select([
            col("fold_change").filter(col("fold_change").is_finite()).count().alias("valid_fc_count"),
            col("fold_change").filter(col("fold_change").lt(lit(0.5))).count().alias("depleted_count"),
            col("fold_change").filter(col("fold_change").lt(lit(0.25))).count().alias("strongly_depleted_count"),
            col("fold_change").filter(col("fold_change").eq(lit(0.0))).count().alias("zero_at_t12_count"),
        ])
        .collect()?;

    let valid_count = fc_stats.column("valid_fc_count")?.u32()?.get(0).unwrap_or(0);
    let depleted = fc_stats.column("depleted_count")?.u32()?.get(0).unwrap_or(0);
    let strongly_depleted = fc_stats.column("strongly_depleted_count")?.u32()?.get(0).unwrap_or(0);
    let zero_at_t12 = fc_stats.column("zero_at_t12_count")?.u32()?.get(0).unwrap_or(0);

    info!("Depletion analysis (T12/T0):");
    info!("  Valid fold changes: {}", valid_count);
    info!("  Guides with >2-fold depletion (FC < 0.5): {} ({:.1}%)",
          depleted, 100.0 * depleted as f64 / valid_count as f64);
    info!("  Guides with >4-fold depletion (FC < 0.25): {} ({:.1}%)",
          strongly_depleted, 100.0 * strongly_depleted as f64 / valid_count as f64);
    info!("  Guides with zero counts at T12: {} ({:.1}%)",
          zero_at_t12, 100.0 * zero_at_t12 as f64 / valid_count as f64);

    // Sample some genes to see their patterns
    info!("\nSample of gene depletion patterns:");
    let sample = df_with_fc.head(Some(10));
    for i in 0..sample.height() {
        let gene = sample.column("Gene")?.get(i)?.to_string();
        let guide = sample.column("sgRNA")?.get(i)?.to_string();
        let t0 = sample.column("avg_t0")?.get(i)?;
        let t12 = sample.column("avg_t12")?.get(i)?;
        let fc = sample.column("fold_change")?.get(i)?;

        info!("  {} ({}): T0={}, T12={}, FC={}", gene, guide, t0, t12, fc);
    }

        for col_name in t0_cols.iter().chain(t12_cols.iter()) {
            if let Ok(series) = df_numeric.column(col_name) {
            let total = match series.dtype() {
                DataType::Int64 => series
                   .i64().unwrap()             // downcast to Int64Chunked
                    .sum()                      // Option<i64>
                    .map(|v| v as f64),
                DataType::Float64 => series
                    .f64().unwrap()             // downcast to Float64Chunked
                    .sum(),                     // Option<f64>
                _ => None,
            };
            if let Some(t) = total {
                info!("  {}: {:.0} reads", col_name, t);
            }
        }
    }
    // Warning if depletion looks wrong
    if depleted < valid_count / 10 {  // Less than 10% showing depletion
        warn!("WARNING: Very few guides show depletion! This doesn't look like essential genes.");
        warn!("Check if:");
        warn!("  1. The count data is being read correctly");
        warn!("  2. These are actually essential genes");
        warn!("  3. The library targets functional sites in genes");
    }

    Ok(())
}
// src/exon_intron.rs

use std::collections::HashSet;
use std::collections::HashMap;
use std::error::Error;
use serde_json::Value;
use crate::api_handler::APIHandler;

#[derive(Debug, Clone)]
pub enum MissingReason {
    NotFetched,
    NotFound,
    ApiError(String),
}

#[derive(Debug, Clone)]
pub struct ExonDetail {
    pub id: String,
    pub start: u64,
    pub end: u64,
    pub strand: i8,
    pub exon_number: u32,
    pub seq_region_name: String,
    pub expression_score: Option<f64>,
    pub expression_missing_reason: Option<MissingReason>,
    pub conservation_score: Option<f64>,
    pub conservation_missing_reason: Option<MissingReason>,
    pub is_paralogous: Option<bool>,
    pub paralogous_missing_reason: Option<MissingReason>,
}

#[derive(Debug)]
pub struct TranscriptDetail {
    pub id: String,
    pub is_canonical: Option<u8>,
    pub exons: Vec<ExonDetail>,
}

/// Fetch the versioned gene ID from Ensembl API.
pub fn fetch_versioned_gene_id(api_handler: &APIHandler, gene_id: &str) -> Result<String, Box<dyn Error>> {
    println!("Fetching versioned gene ID for gene ID: {}", gene_id);
    let endpoint = format!("/lookup/id/{}", gene_id);
    let gene_data: Value = api_handler.get(&endpoint)?;
    let version = gene_data["version"].as_u64()
        .ok_or("Gene version not found")? -1;
    let versioned_gene_id = format!("{}.{}", gene_id, version);
    Ok(versioned_gene_id)
}

/// Fetch all transcripts for a given gene ID.
pub fn fetch_all_transcripts(api_handler: &APIHandler, gene_id: &str) -> Result<Vec<TranscriptDetail>, Box<dyn Error>> {
    println!("Fetching all transcripts for gene ID: {}", gene_id);
    let endpoint = format!("/lookup/id/{}?expand=1", gene_id);
    let gene_data: Value = api_handler.get(&endpoint)?;

    let transcripts_json = gene_data["Transcript"]
        .as_array()
        .ok_or("No transcripts found")?;

    // Fetch exon numbers using the same API handler
    let exon_number_map = get_exon_numbers(api_handler, gene_id)?;

    let mut transcripts = Vec::new();

    for (transcript_index, t) in transcripts_json.iter().enumerate() {
        let exons_json = t["Exon"]
            .as_array()
            .ok_or_else(|| format!("No exons found for transcript index {}", transcript_index))?;

        // Pretty-print the exons array
        let exons: Result<Vec<ExonDetail>, Box<dyn Error>> = exons_json.iter().enumerate().map(|(exon_index, e)| {
            let exon_id = e["id"].as_str().ok_or_else(|| format!("Missing id for exon {} in transcript {}", exon_index, transcript_index))?.to_string();
            let exon_number = exon_number_map.get(&exon_id).cloned().unwrap_or_else(|| (exon_index + 1) as u32);

            Ok(ExonDetail {
                id: exon_id,
                start: e["start"].as_u64().ok_or_else(|| format!("Missing start for exon {} in transcript {}", exon_index, transcript_index))?,
                end: e["end"].as_u64().ok_or_else(|| format!("Missing end for exon {} in transcript {}", exon_index, transcript_index))?,
                strand: e["strand"].as_i64().ok_or_else(|| format!("Missing strand for exon {} in transcript {}", exon_index, transcript_index))? as i8,
                exon_number,
                seq_region_name: e["seq_region_name"].as_str().ok_or_else(|| format!("Missing seq_region_name for exon {} in transcript {}", exon_index, transcript_index))?.to_string(),
                expression_score: None,
                expression_missing_reason: Some(MissingReason::NotFetched),
                conservation_score: None,
                conservation_missing_reason: Some(MissingReason::NotFetched),
                is_paralogous: None,
                paralogous_missing_reason: Some(MissingReason::NotFetched),
            })
        }).collect();

        let transcript = TranscriptDetail {
            id: t["id"].as_str().ok_or_else(|| format!("Missing id for transcript {}", transcript_index))?.to_string(),
            is_canonical: t["is_canonical"].as_u64().map(|v| v as u8),
            exons: exons?,
        };

        transcripts.push(transcript);
    }

    Ok(transcripts)
}

/// Get exon numbers from GTEx Portal API
fn get_exon_numbers(api_handler: &APIHandler, gene_id: &str) -> Result<HashMap<String, u32>, Box<dyn Error>> {
    println!("Fetching exon numbers for gene ID: {}", gene_id);
    let endpoint = format!("/lookup/id/{}?expand=1", gene_id);
    let gene_data: Value = api_handler.get(&endpoint)?;

    let transcripts = gene_data["Transcript"]
        .as_array()
        .ok_or("No transcripts found")?;

    let mut exon_number_map = HashMap::new();

    for transcript in transcripts {
        if let Some(exons) = transcript["Exon"].as_array() {
            for (index, exon) in exons.iter().enumerate() {
                if let Some(exon_id) = exon["id"].as_str() {
                    exon_number_map.insert(exon_id.to_string(), (index + 1) as u32);
                }
            }
        }
    }

    Ok(exon_number_map)
}

pub fn fetch_exon_expression_data(
    gtex_api: &APIHandler,
    versioned_gene_id: &str,
    tissue_filter: Option<&[&str]>,
) -> Result<HashMap<u32, f64>, Box<dyn Error>> {
    println!("Fetching expression data for gene ID: {}", versioned_gene_id);
    let query = format!(
        "/expression/medianExonExpression?gencodeId={}&datasetId=gtex_v8",
        versioned_gene_id
    );

    let data: Value = gtex_api.get(&query)?;

    let exon_expression_array = data["data"]
        .as_array()
        .ok_or("No exon expression data found")?;

    let tissue_set: Option<HashSet<&str>> = tissue_filter.map(|tissues| tissues.iter().cloned().collect());

    let mut exon_totals: HashMap<u32, f64> = HashMap::new();
    let mut exon_counts: HashMap<u32, u32> = HashMap::new();

    for exon_data in exon_expression_array {
        let exon_id = exon_data["exonId"].as_str().ok_or("Missing exonId")?;
        let median_tpm = exon_data["median"].as_f64().ok_or("Missing median")?;
        let tissue = exon_data["tissueSiteDetailId"].as_str().ok_or("Missing tissue")?;

        if tissue_set.as_ref().map_or(true, |set| set.contains(tissue)) {
            let parts: Vec<&str> = exon_id.rsplit('_').collect();
            let exon_number: u32 = parts.get(0)
                .ok_or("Failed to parse exon number")?
                .parse()
                .map_err(|e| format!("Failed to parse exon number as u32: {}", e))?;

            exon_totals.entry(exon_number)
                .and_modify(|total| *total += median_tpm)
                .or_insert(median_tpm);

            exon_counts.entry(exon_number)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    let mut exon_expression_data = HashMap::new();
    for (exon_number, total_expression) in exon_totals {
        if let Some(&count) = exon_counts.get(&exon_number) {
            let average_expression = total_expression / count as f64;
            exon_expression_data.insert(exon_number, average_expression);
        }
    }

    Ok(exon_expression_data)
}

/// Fetch conservation score at a given position using Ensembl API.
pub fn fetch_conservation_score_at_position(
    ensembl_api: &APIHandler,
    chromosome: &str,
    start: u64,
    end: u64,
) -> Result<Option<f64>, Box<dyn Error>> {
    println!("Fetching conservation scores for region {}:{}-{}", chromosome, start, end);

    let endpoint = format!(
        "/overlap/region/human/{}:{}-{}?feature=constrained",
        chromosome.trim_start_matches("chr"), start, end
    );

    let data: Value = ensembl_api.get(&endpoint)?;
    let elements = data.as_array().unwrap();

    let mut total_constrained_length = 0;

    for element in elements {
        if let (Some(start_elem), Some(end_elem)) = (element["start"].as_u64(), element["end"].as_u64()) {
            let overlap_start = std::cmp::max(start, start_elem);
            let overlap_end = std::cmp::min(end, end_elem);
            if overlap_end >= overlap_start {
                total_constrained_length += overlap_end - overlap_start + 1;
            }
        }
    }

    let region_length = end - start + 1;
    let conservation_score = if region_length > 0 {
        total_constrained_length as f64 / region_length as f64
    } else {
        panic!("missing conservation scores for region {}", chromosome);
    };

    Ok(Some(conservation_score))
}

/// Determine which exons are paralogous by comparing with paralogous genes.
pub fn determine_paralogous_exons(
    ensembl_api: &APIHandler,
    exons: &mut Vec<ExonDetail>,
    paralog_gene_ids: &[String],
) -> Result<(), Box<dyn Error>> {
    // Fetch exons from paralogous genes
    let mut paralog_exons = Vec::new();
    for paralog_id in paralog_gene_ids {
        let transcripts = fetch_all_transcripts(ensembl_api, paralog_id)?;
        for transcript in transcripts {
            paralog_exons.extend(transcript.exons);
        }
    }

    // Compare each exon with paralog exons
    for exon in exons.iter_mut() {
        for paralog_exon in &paralog_exons {
            if exon.seq_region_name == paralog_exon.seq_region_name &&
                exon.start == paralog_exon.start &&
                exon.end == paralog_exon.end {
                exon.is_paralogous = Some(true);
                break;
            }
        }
        if exon.is_paralogous.is_none() {
            exon.is_paralogous = Some(false);
        }
    }

    Ok(())
}

pub fn find_overlapping_exons(exons: &[ExonDetail], start: u64, end: u64) -> Vec<ExonDetail> {
    exons.iter()
        .filter(|exon| exon.start <= end && exon.end >= start)
        .cloned()
        .collect()
}
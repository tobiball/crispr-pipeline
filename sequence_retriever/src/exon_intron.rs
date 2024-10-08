// src/exon_intron.rs

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
    pub exon_number: String,
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

    let mut transcripts = Vec::new();

    for t in transcripts_json {
        let exons_json = t["Exon"]
            .as_array()
            .ok_or("No exons found for transcript")?;

        let exons: Vec<ExonDetail> = exons_json.iter().map(|e| ExonDetail {
            id: e["id"].as_str().unwrap_or("").to_string(),
            start: e["start"].as_u64().unwrap_or(0),
            end: e["end"].as_u64().unwrap_or(0),
            strand: e["strand"].as_i64().unwrap_or(0) as i8,
            exon_number: e["exon_number"].as_str().unwrap_or("").to_string(),
            seq_region_name: e["seq_region_name"].as_str().unwrap_or("").to_string(),
            expression_score: None,
            expression_missing_reason: Some(MissingReason::NotFetched),
            conservation_score: None,
            conservation_missing_reason: Some(MissingReason::NotFetched),
            is_paralogous: None,
            paralogous_missing_reason: Some(MissingReason::NotFetched),
        }).collect();

        let transcript = TranscriptDetail {
            id: t["id"].as_str().unwrap_or("").to_string(),
            is_canonical: t["is_canonical"].as_u64().map(|v| v as u8),
            exons,
        };

        transcripts.push(transcript);
    }

    Ok(transcripts)
}

/// Fetch expression score at a given position using GTEx API.
pub fn fetch_expression_score_at_position(
    gtex_api: &APIHandler,
    gene_id: &str,
    chromosome: &str,
    start: u64,
    end: u64,
    ensembl_api: &APIHandler, // Added ensembl_api to get versioned gene ID
) -> Result<Option<f64>, Box<dyn Error>> {
    println!("Fetching expression levels for gene {} at {}:{}-{}", gene_id, chromosome, start, end);

    // Fetch versioned gene ID
    let versioned_gene_id = fetch_versioned_gene_id(ensembl_api, gene_id)?;

    // Construct the query for median exon expression
    let query = format!(
        "/expression/medianExonExpression?gencodeId={}&datasetId=gtex_v8",
        versioned_gene_id
    );

    let data: Value = gtex_api.get(&query)?;
    let exon_expression_data = data["data"]
        .as_array()
        .ok_or("No exon expression data found")?;

    // Find the exon that overlaps with our region of interest
    let mut relevant_expression = None;

    for exon in exon_expression_data {
        println!("{:?}", exon);
        let exon_start = exon["exonStart"].as_u64().unwrap();
        let exon_end = exon["exonEnd"].as_u64().unwrap();

        // Check if this exon overlaps with our region of interest
        if exon_start <= end && exon_end >= start {
            if let Some(median_tpm) = exon["median"].as_f64() {
                relevant_expression = Some(median_tpm);
                break;
            }
        }
    }

    Ok(relevant_expression)
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
        0.0
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
        let transcripts = crate::exon_intron::fetch_all_transcripts(ensembl_api, paralog_id)?;
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

use std::error::Error;
use serde_json::Value;
use url::form_urlencoded;

use crate::api_handler::APIHandler;

#[derive(Debug, Clone)]
pub enum MissingReason {
    NotFetched,
    NotFound,
    ApiError(String),
}

#[derive(Debug)]
pub struct ExonDetail {
    pub id: String,
    pub start: u64,
    pub end: u64,
    pub strand: i8,
    pub exon_number: String,
    pub seq_region_name: String,
    pub expression_level: Option<f64>,
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
            expression_level: None,
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

pub fn fetch_gencode_id(
    api_handler: &APIHandler,
    gene_symbol: &str,
) -> Result<String, Box<dyn Error>> {
    println!("Fetching gencodeId for gene symbol: {}", gene_symbol);

    // Build the query parameters
    let query = form_urlencoded::Serializer::new(String::new())
        .append_pair("geneId", gene_symbol)
        .append_pair("gencodeVersion", "v26")
        .append_pair("genomeBuild", "GRCh38/hg38")
        .finish();

    // Construct the endpoint
    let endpoint = format!("/reference/geneSearch?{}", query);

    // Make the API call
    let data: Value = api_handler.get(&endpoint)?;

    // Parse the response to extract the gencodeId
    let gene_data = data["data"]
        .as_array()
        .ok_or("No data found in gene search response")?;

    if gene_data.is_empty() {
        return Err(format!("No gene found for symbol: {}", gene_symbol).into());
    }

    // Assuming the first result is the correct gene
    let gencode_id = gene_data[0]["gencodeId"]
        .as_str()
        .ok_or("gencodeId not found in gene data")?
        .to_string();

    Ok(gencode_id)
}

// https://gtexportal.org/api/v2/reference/geneSearch?geneId=OAT&gencodeVersion=v26&genomeBuild=GRCh38/hg38


pub fn fetch_expression_levels(
    api_handler: &APIHandler,
    exons: &mut [ExonDetail],
    gene_symbol: &str,
) -> Result<(), Box<dyn Error>> {
    println!("Fetching expression levels for gene symbol: {}", gene_symbol);

    // Step 1: Fetch the gencodeId
    let gencode_id = fetch_gencode_id(api_handler, gene_symbol)?;

    // Build the query parameters
    let query = form_urlencoded::Serializer::new(String::new())
        .append_pair("gencodeId", &gencode_id)
        .append_pair("datasetId", "gtex_v8")
        .finish();

    // Construct the endpoint
    let endpoint = format!("/expression/medianExonExpression?{}", query);

    // Make the API call using your APIHandler
    let data: Value = api_handler.get(&endpoint)?;

    // Parse the response data
    let expression_data = data["data"]
        .as_array()
        .ok_or("No expression data found")?;

    // Build a map from exon stable ID (without version) to median expression
    use std::collections::HashMap;
    let mut expression_map = HashMap::new();
    for expr in expression_data {
        if let (Some(exon_id), Some(median)) = (
            expr["exonId"].as_str(),
            expr["median"].as_f64(),
        ) {
            // Remove version number from exon_id if present
            let exon_id_trimmed = exon_id.split('.').next().unwrap_or(exon_id);
            expression_map.insert(exon_id_trimmed.to_string(), median);
        }
    }

    // Update exons with expression levels
    for exon in exons.iter_mut() {
        // Remove version number from exon.id if present
        let exon_id_trimmed = exon.id.split('.').next().unwrap_or(&exon.id);

        if let Some(&median) = expression_map.get(exon_id_trimmed) {
            exon.expression_level = Some(median);
            exon.expression_missing_reason = None;
        } else {
            exon.expression_level = None;
            exon.expression_missing_reason = Some(MissingReason::NotFound);
        }
    }

    Ok(())
}


pub fn fetch_conservation_scores(api_handler: &APIHandler, exons: &mut [ExonDetail]) -> Result<(), Box<dyn Error>> {
    println!("Fetching conservation scores for exons");

    for exon in exons.iter_mut() {
        let endpoint = format!(
            "/overlap/region/human/{}:{}-{}?feature=constrained",
            exon.seq_region_name, exon.start, exon.end
        );
        match api_handler.get(&endpoint) {
            Ok(data) => {
                let elements = data.as_array().unwrap();
                let mut total_constrained_length = 0;

                for element in elements {
                    if let (Some(start), Some(end)) = (element["start"].as_u64(), element["end"].as_u64()) {
                        let overlap_start = std::cmp::max(exon.start, start);
                        let overlap_end = std::cmp::min(exon.end, end);
                        if overlap_end >= overlap_start {
                            total_constrained_length += overlap_end - overlap_start + 1;
                        }
                    }
                }

                let exon_length = exon.end - exon.start + 1;
                exon.conservation_score = Some(total_constrained_length as f64 / exon_length as f64);
                exon.conservation_missing_reason = None;
            },
            Err(e) => {
                exon.conservation_score = None;
                exon.conservation_missing_reason = Some(MissingReason::ApiError(e.to_string()));
            }
        }
    }

    Ok(())
}

pub fn determine_paralogous_exons(
    api_handler: &APIHandler,
    exons: &mut [ExonDetail],
    paralog_gene_ids: &[String],
) -> Result<(), Box<dyn Error>> {
    println!("Determining paralogous exons");

    let mut paralog_exon_sequences = Vec::new();

    for gene_id in paralog_gene_ids {
        let endpoint = format!("/lookup/id/{}?expand=1", gene_id);
        let gene_data: Value = api_handler.get(&endpoint)?;

        let transcripts_json = gene_data["Transcript"]
            .as_array()
            .ok_or("No transcripts found")?;

        for t in transcripts_json {
            let exons_json = t["Exon"]
                .as_array()
                .ok_or("No exons found for transcript")?;

            for e in exons_json {
                if let Some(exon_id) = e["id"].as_str() {
                    let seq_endpoint = format!("/sequence/id/{}?type=cdna", exon_id);
                    match api_handler.get_plain_text(&seq_endpoint) {
                        Ok(sequence) => paralog_exon_sequences.push((exon_id.to_string(), sequence)),
                        Err(e) => println!("Failed to fetch sequence for exon {}: {}", exon_id, e),
                    }
                }
            }
        }
    }

    for exon in exons.iter_mut() {
        let seq_endpoint = format!("/sequence/id/{}?type=cdna", exon.id);
        match api_handler.get_plain_text(&seq_endpoint) {
            Ok(exon_sequence) => {
                exon.is_paralogous = Some(paralog_exon_sequences.iter().any(|(_, paralog_sequence)| {
                    calculate_sequence_similarity(&exon_sequence, paralog_sequence) >= 0.8
                }));
                exon.paralogous_missing_reason = None;
            },
            Err(e) => {
                exon.is_paralogous = None;
                exon.paralogous_missing_reason = Some(MissingReason::ApiError(e.to_string()));
            }
        }
    }

    Ok(())
}

fn calculate_sequence_similarity(seq1: &str, seq2: &str) -> f64 {
    use bio::alignment::pairwise::Aligner;
    use bio::alignment::AlignmentOperation;

    // Define match and mismatch scores
    let match_score = 1;
    let mismatch_score = -1;
    let gap_open = -5;
    let gap_extend = -1;

    // Create an aligner with specified scoring parameters
    let mut aligner = Aligner::new(
        gap_open,
        gap_extend,
        |a: u8, b: u8| if a == b { match_score } else { mismatch_score },
    );

    // Perform global alignment
    let alignment = aligner.global(seq1.as_bytes(), seq2.as_bytes());

    // Calculate sequence identity
    let alignment_length = alignment.operations.len();
    let mut matches = 0;

    for op in alignment.operations {
        if let AlignmentOperation::Match = op {
            matches += 1;
        }
    }

    

    matches as f64 / alignment_length as f64
}
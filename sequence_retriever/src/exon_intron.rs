// src/exon_intron.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;
use serde_json::Value;
use reqwest::header::USER_AGENT;
use std::thread;
use std::time::Duration;

#[derive(Deserialize, Debug)]
pub struct ExonDetail {
    pub id: String,
    pub start: u64,
    pub end: u64,
    pub strand: i8,
    pub exon_number: String,
    pub seq_region_name: String, // Add this field
    #[serde(default)]
    pub expression_level: Option<f64>,
    #[serde(default)]
    pub conservation_score: Option<f64>,
    #[serde(default)]
    pub is_paralogous: bool,
}



#[derive(Deserialize, Debug)]
pub struct TranscriptDetail {
    pub id: String,
    pub is_canonical: Option<u8>,
    pub exons: Vec<ExonDetail>,
}

/// Fetch all transcripts for a given gene ID using Ensembl REST API.
pub fn fetch_all_transcripts(gene_id: &str) -> Result<Vec<TranscriptDetail>, Box<dyn Error>> {
    println!("Fetching all transcripts for gene ID: {}", gene_id);
    let url = format!(
        "https://rest.ensembl.org/lookup/id/{}?expand=1",
        gene_id
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    if response.status().is_success() {
        let gene_data: Value = response.json()?;
        let transcripts_json = gene_data["Transcript"]
            .as_array()
            .ok_or("No transcripts found")?;

        let mut transcripts = Vec::new();

        for t in transcripts_json {
            let exons_json = t["Exon"]
                .as_array()
                .ok_or("No exons found for transcript")?;

            let mut exons = Vec::new();
            for e in exons_json {
                let exon = ExonDetail {
                    id: e["id"].as_str().unwrap_or("").to_string(),
                    start: e["start"].as_u64().unwrap_or(0),
                    end: e["end"].as_u64().unwrap_or(0),
                    strand: e["strand"].as_i64().unwrap_or(0) as i8,
                    exon_number: e["exon_number"].as_str().unwrap_or("").to_string(),
                    seq_region_name: e["seq_region_name"].as_str().unwrap_or("").to_string(), // Add this line
                    // Initialize additional fields
                    expression_level: None,
                    conservation_score: None,
                    is_paralogous: false,
                };
                exons.push(exon);
            }


            let transcript = TranscriptDetail {
                id: t["id"].as_str().unwrap_or("").to_string(),
                is_canonical: t["is_canonical"].as_u64().map(|v| v as u8),
                exons,
            };

            transcripts.push(transcript);
        }

        Ok(transcripts)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching transcripts: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch transcripts.".into())
    }
}

/// Fetch expression levels for exons using Expression Atlas API.
pub fn fetch_expression_levels(exons: &mut [ExonDetail], gencode_id: &str) -> Result<(), Box<dyn Error>> {
    println!("Fetching expression levels for gene ID: {}", gencode_id);

    // Define the API endpoint and URL
    let url = format!(
        "https://gtexportal.org/api/v2/expression/medianExonExpression?gencodeId={}&datasetId=gtex_v8",
        gencode_id
    );

    // Create a new HTTP client
    let client = Client::new();
    let response = client.get(&url).send()?;

    // Handle successful response
    if response.status().is_success() {
        let data: Value = response.json()?;

        // Assuming the API returns a field "data" with expression levels in an array
        let expression_data = data["data"]
            .as_array()
            .ok_or("No expression data found")?;

        // Iterate over the fetched exon data and match it with your exon list
        for exon in exons.iter_mut() {
            if let Some(exon_data) = expression_data.iter().find(|&expr| expr["exon_id"].as_str() == Some(&exon.id)) {
                if let Some(expression_value) = exon_data["median"].as_f64() {
                    exon.expression_level = Some(expression_value);
                } else {
                    exon.expression_level = Some(0.0);
                }
            } else {
                exon.expression_level = Some(0.0);
            }
        }


        Ok(())
    } else {
        // Handle HTTP errors
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching expression levels: HTTP {}", status);
        eprintln!("Error details: {}", error_text);

        // Assign default values to exons in case of an error
        for exon in exons.iter_mut() {
            exon.expression_level = Some(0.0);  // Assign default value in case of error
        }

        Ok(())
    }
}



/// Fetch conservation scores for exons using the Ensembl Conservation API.
pub fn fetch_conservation_scores(exons: &mut [ExonDetail]) -> Result<(), Box<dyn Error>> {
    println!("Fetching conservation scores for exons");

    let client = Client::new();

    for exon in exons.iter_mut() {
        let url = format!(
            "https://rest.ensembl.org/overlap/region/human/{}:{}-{}?feature=constrained",
            exon.seq_region_name, exon.start, exon.end
        );
        println!("{}", url);
        let response = client
            .get(&url)
            .header("Accept", "application/json")
            .send()?;

        if response.status().is_success() {
            let data: Value = response.json()?;
            let elements = data.as_array().unwrap();

            // Calculate the total length of constrained elements overlapping the exon
            let mut total_constrained_length = 0;

            for element in elements {
                if let Some(start) = element["start"].as_u64() {
                    if let Some(end) = element["end"].as_u64() {
                        // Calculate overlap between constrained element and exon
                        let overlap_start = std::cmp::max(exon.start, start);
                        let overlap_end = std::cmp::min(exon.end, end);
                        if overlap_end >= overlap_start {
                            total_constrained_length += overlap_end - overlap_start + 1;
                        }
                    }
                }
            }

            let exon_length = exon.end - exon.start + 1;
            // Calculate the proportion of the exon that is constrained
            exon.conservation_score = Some(total_constrained_length as f64 / exon_length as f64);
        } else {
            eprintln!(
                "Failed to fetch conservation score for exon {}",
                exon.id
            );
            exon.conservation_score = Some(0.0);
        }
    }

    Ok(())
}

pub fn determine_paralogous_exons(
    exons: &mut [ExonDetail],
    paralog_gene_ids: &[String],
) -> Result<(), Box<dyn Error>> {
    println!("Determining paralogous exons");

    let client = Client::new();

    // Fetch exon sequences for paralogous genes
    let mut paralog_exon_sequences = Vec::new();

    for gene_id in paralog_gene_ids {
        let url = format!(
            "https://rest.ensembl.org/lookup/id/{}?expand=1",
            gene_id
        );
        let response = client
            .get(&url)
            .header("Accept", "application/json")
            .send()?;

        if response.status().is_success() {
            let gene_data: Value = response.json()?;
            let transcripts_json = gene_data["Transcript"]
                .as_array()
                .ok_or("No transcripts found")?;

            for t in transcripts_json {
                let exons_json = t["Exon"]
                    .as_array()
                    .ok_or("No exons found for transcript")?;

                for e in exons_json {
                    if let Some(exon_id) = e["id"].as_str() {
                        // Fetch exon sequence
                        let seq_url = format!(
                            "https://rest.ensembl.org/sequence/id/{}?type=cdna",
                            exon_id
                        );
                        let seq_response = client
                            .get(&seq_url)
                            .header("Accept", "text/plain")
                            .send()?;

                        if seq_response.status().is_success() {
                            let sequence = seq_response.text()?;
                            paralog_exon_sequences.push((exon_id.to_string(), sequence));
                        }
                    }
                }
            }
        }
    }

    // Now compare each of your exons to the paralogous exon sequences
    for exon in exons.iter_mut() {
        // Fetch exon sequence
        let seq_url = format!(
            "https://rest.ensembl.org/sequence/id/{}?type=cdna",
            exon.id
        );
        println!("{}",seq_url);
        let seq_response = client
            .get(&seq_url)
            .header("Accept", "text/plain")
            .send()?;
        println!("{}", seq_response.status());
        if seq_response.status().is_success() {
            let exon_sequence = seq_response.text()?;

            // Compare to each paralog exon sequence
            for (_paralog_exon_id, paralog_sequence) in &paralog_exon_sequences {
                let similarity = calculate_sequence_similarity(&exon_sequence, &paralog_sequence);

                if similarity >= 0.8 {
                    // Threshold of 80% similarity
                    exon.is_paralogous = true;
                    break;
                }
            }
        } else {
            eprintln!("Failed to fetch sequence for exon {}", exon.id);
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

    let identity = matches as f64 / alignment_length as f64;

    identity
}

/// Helper function to handle rate limiting and retries.
fn make_request_with_retry(
    client: &Client,
    url: &str,
    max_attempts: u32,
) -> Result<reqwest::blocking::Response, Box<dyn Error>> {
    let mut attempts = 0;

    loop {
        let response = client
            .get(url)
            .header("Accept", "application/json")
            .header(USER_AGENT, "YourAppName/1.0") // Replace with your application's name and version
            .send()?;

        if response.status().is_success() {
            return Ok(response);
        } else if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            attempts += 1;
            if attempts >= max_attempts {
                return Err(format!("Exceeded maximum retries for URL: {}", url).into());
            }

            // Check for the 'Retry-After' header to determine wait time
            if let Some(retry_after) = response.headers().get("Retry-After") {
                let wait_time = retry_after.to_str()?.parse::<u64>().unwrap_or(1);
                eprintln!(
                    "Rate limited. Waiting {} seconds before retrying...",
                    wait_time
                );
                thread::sleep(Duration::from_secs(wait_time));
            } else {
                // Default wait time if 'Retry-After' is not provided
                eprintln!("Rate limited. Waiting 1 second before retrying...");
                thread::sleep(Duration::from_secs(1));
            }
        } else {
            // For other HTTP errors, return an error
            let status = response.status();
            let error_text = response.text()?;
            return Err(format!(
                "Failed to fetch data from URL: {}. Status: {}. Error: {}",
                url, status, error_text
            )
                .into());
        }
    }
}
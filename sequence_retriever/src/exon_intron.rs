// src/exon_intron.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug)]
pub struct ExonDetail {
    pub id: String,
    pub start: u64,
    pub end: u64,
    pub strand: i8,
    pub exon_number: String,
}

#[derive(Deserialize, Debug)]
pub struct TranscriptDetail {
    pub id: String,
    pub is_canonical: Option<u8>,
    pub exons: Vec<ExonDetail>,
}

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
        let gene_data: serde_json::Value = response.json()?;
        let transcripts_json = gene_data["Transcript"]
            .as_array()
            .ok_or("No transcripts found")?;

        let transcripts = transcripts_json
            .iter()
            .map(|t| {
                let exons_json = t["Exon"]
                    .as_array()
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);

                let exons = exons_json
                    .iter()
                    .map(|e| ExonDetail {
                        id: e["id"].as_str().unwrap_or("").to_string(),
                        start: e["start"].as_u64().unwrap_or(0),
                        end: e["end"].as_u64().unwrap_or(0),
                        strand: e["strand"].as_i64().unwrap_or(0) as i8,
                        exon_number: e["exon_number"].as_str().unwrap_or("").to_string(),
                    })
                    .collect();

                TranscriptDetail {
                    id: t["id"].as_str().unwrap_or("").to_string(),
                    is_canonical: t["is_canonical"].as_u64().map(|v| v as u8),
                    exons,
                }
            })
            .collect();

        Ok(transcripts)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching transcripts: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch transcripts.".into())
    }
}

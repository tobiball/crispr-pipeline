// src/exon_intron.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;
use serde_json::Value;

#[derive(Deserialize, Debug)]
pub struct ExonDetail {
    pub id: String,
    pub start: u64,
    pub end: u64,
    pub strand: i8,
    pub exon_number: String,
    // Additional fields
    #[serde(default)]
    pub expression_level: Option<f64>,   // Expression level (e.g., from GTEx)
    #[serde(default)]
    pub conservation_score: Option<f64>, // Conservation score (e.g., from PhastCons)
    #[serde(default)]
    pub is_paralogous: bool,             // True if exon is paralogous
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
                    // Initialize additional fields
                    expression_level: None,     // Will be fetched later
                    conservation_score: None,   // Will be fetched later
                    is_paralogous: false,       // Will be determined later
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

/// Fetch expression levels for exons using GTEx API.
pub fn fetch_expression_levels(exons: &mut [ExonDetail], gene_symbol: &str) -> Result<(), Box<dyn Error>> {
    println!("Fetching expression levels for gene symbol: {}", gene_symbol);

    // GTEx API endpoint for median gene expression by tissue
    let url = format!(
        "https://gtexportal.org/rest/v1/expression/geneExpression?geneId={}&format=json",
        gene_symbol
    );
    let client = Client::new();
    let response = client.get(&url).send()?;

    if response.status().is_success() {
        let data: Value = response.json()?;
        let expression_data = data["geneExpression"]
            .as_array()
            .ok_or("No expression data found")?;

        // For simplicity, we'll use the average expression across tissues
        let mut total_expression = 0.0;
        let mut count = 0.0;
        for expr in expression_data {
            if let Some(tpm) = expr["tpm"].as_f64() {
                total_expression += tpm;
                count += 1.0;
            }
        }
        let average_expression = if count > 0.0 { total_expression / count } else { 0.0 };

        // Assign the same expression level to all exons (you might want to refine this)
        for exon in exons.iter_mut() {
            exon.expression_level = Some(average_expression);
        }

        Ok(())
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching expression levels: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch expression levels.".into())
    }
}

/// Fetch conservation scores for exons using Ensembl's Constrained Elements API.
pub fn fetch_conservation_scores(exons: &mut [ExonDetail]) -> Result<(), Box<dyn Error>> {
    println!("Fetching conservation scores for exons");

    let client = Client::new();

    for exon in exons.iter_mut() {
        let url = format!(
            "https://rest.ensembl.org/overlap/region/human/{}:{}-{}?feature=constrained_element",
            "chr", exon.start, exon.end
        );
        let response = client
            .get(&url)
            .header("Accept", "application/json")
            .send()?;

        if response.status().is_success() {
            let data: Value = response.json()?;
            let elements = data.as_array().unwrap();

            // For simplicity, we'll assign a conservation score based on the number of constrained elements
            let conservation_score = elements.len() as f64;
            exon.conservation_score = Some(conservation_score);
        } else {
            eprintln!(
                "Failed to fetch conservation score for exon {}",
                exon.id
            );
            exon.conservation_score = Some(0.0); // Default to 0.0 if data is unavailable
        }
    }

    Ok(())
}

/// Determine if exons are paralogous based on sequence similarity (simplified).
pub fn determine_paralogous_exons(
    exons: &mut [ExonDetail],
    paralog_gene_ids: &[String],
) -> Result<(), Box<dyn Error>> {
    println!("Determining paralogous exons");

    // For simplicity, we'll mark all exons as non-paralogous
    // Implement sequence alignment or use Ensembl's homology endpoints for a real implementation
    for exon in exons.iter_mut() {
        exon.is_paralogous = false;
    }

    Ok(())
}

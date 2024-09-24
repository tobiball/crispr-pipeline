// src/protein_domains.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug)]
pub struct ProteinFeature {
    pub start: u64,
    pub end: u64,
    // pub id: Option<String>,
    pub interpro: Option<String>,
    pub description: Option<String>,
    // pub logic_name: Option<String>,
    // pub display_id: Option<String>,
    // pub interpro_description: Option<String>,
}

pub fn fetch_protein_domains(transcript_id: &str) -> Result<Vec<ProteinFeature>, Box<dyn Error>> {
    println!(
        "Fetching protein features (domains) for transcript ID: {}",
        transcript_id
    );
    // First, fetch the translation ID
    let url = format!(
        "https://rest.ensembl.org/lookup/id/{}?expand=1",
        transcript_id
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    let transcript_data: serde_json::Value = response.json()?;
    let translation_id = transcript_data["Translation"]["id"]
        .as_str()
        .ok_or("Translation ID not found.")?;

    // Now, fetch protein features using the translation ID
    let url = format!(
        "https://rest.ensembl.org/overlap/translation/{}?feature=protein_feature",
        translation_id
    );
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    if response.status().is_success() {
        let protein_features: Vec<ProteinFeature> = response.json()?;
        Ok(protein_features)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching protein features: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch protein features.".into())
    }
}

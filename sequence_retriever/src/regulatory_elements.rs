// src/regulatory_elements.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug)]
pub struct RegulatoryFeature {
    pub id: String,
    pub start: u64,
    pub end: u64,
    // pub strand: i8,
    pub feature_type: String,
}

pub fn fetch_regulatory_elements(chromosome: String, start: u64, end: u64) -> Result<Vec<RegulatoryFeature>, Box<dyn Error>> {
    println!(
        "Fetching regulatory elements in region {}:{}-{}",
        chromosome, start, end
    );
    let url = format!(
        "https://rest.ensembl.org/overlap/region/human/{}:{}-{}?feature=regulatory",
        chromosome, start, end
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    if response.status().is_success() {
        let regulatory_features: Vec<RegulatoryFeature> = response.json()?;
        Ok(regulatory_features)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching regulatory elements: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch regulatory elements.".into())
    }
}

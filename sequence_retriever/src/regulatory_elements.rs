// src/regulatory_elements.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;
use serde_json::Value;
use crate::api_handler::APIHandler;

#[derive(Deserialize, Debug)]
pub struct RegulatoryFeature {
    pub id: String,
    pub start: u64,
    pub end: u64,
    // pub strand: i8,
    pub feature_type: String,
}

/// Fetch regulatory elements in a given genomic region using Ensembl REST API.
pub fn fetch_regulatory_elements(
    api_handler: &APIHandler,
    chromosome: String,
    start: u64,
    end: u64
) -> Result<Vec<RegulatoryFeature>, Box<dyn Error>> {
    println!(
        "Fetching regulatory elements in region {}:{}-{}",
        chromosome, start, end
    );

    // Construct the endpoint
    let endpoint = format!(
        "/overlap/region/human/{}:{}-{}?feature=regulatory",
        chromosome, start, end
    );

    // Use the APIHandler to make the GET request
    let data: Value = api_handler.get(&endpoint)?;

    // Parse the response data into a Vec<RegulatoryFeature>
    let regulatory_features: Vec<RegulatoryFeature> = serde_json::from_value(data)?;

    Ok(regulatory_features)
}
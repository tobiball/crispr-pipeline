// src/protein_domains.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;
use serde_json::Value;
use crate::api_handler::APIHandler;

#[derive(Deserialize, Debug, Clone)]
pub struct ProteinFeature {
    pub start: u64,
    pub end: u64,
    pub interpro: Option<String>,
    pub description: Option<String>,
    pub id: Option<String>,
    pub external_name: Option<String>,
    #[serde(default)]
    pub is_essential: bool, // Indicates if the domain is essential
}


/// Fetch protein domains for a transcript and determine if they are essential.
pub fn fetch_protein_domains(
    api_handler: &APIHandler,
    transcript_id: &str
) -> Result<Vec<ProteinFeature>, Box<dyn Error>> {
    println!(
        "Fetching protein features (domains) for transcript ID: {}",
        transcript_id
    );

    // First, fetch the translation ID
    let endpoint = format!("/lookup/id/{}?expand=1", transcript_id);
    let transcript_data: Value = api_handler.get(&endpoint)?;
    let translation_id = transcript_data["Translation"]["id"]
        .as_str()
        .ok_or("Translation ID not found.")?;

    // Now, fetch protein features using the translation ID
    let endpoint = format!("/overlap/translation/{}?feature=protein_feature", translation_id);
    let data: Value = api_handler.get(&endpoint)?;

    // Deserialize the response data into a Vec<ProteinFeature>
    let mut protein_features: Vec<ProteinFeature> = serde_json::from_value(data)?;

    // Set is_essential based on criteria
    for feature in &mut protein_features {
        feature.is_essential = is_feature_essential(feature);
    }
    Ok(protein_features)
}

/// Determine if a protein feature is essential based on its description or ID.
fn is_feature_essential(feature: &ProteinFeature) -> bool {
    // Example criteria: if description contains certain keywords
    if let Some(ref desc) = feature.description {
        let essential_keywords = ["Active site", "Binding site", "Catalytic domain"];
        for keyword in &essential_keywords {
            if desc.contains(keyword) {
                return true;
            }
        }
    }

    // You can add more criteria based on InterPro IDs or other annotations
    false
}

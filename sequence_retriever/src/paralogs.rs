// src/paralogs.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug)]
pub struct HomologyResponse {
    pub data: Vec<HomologyData>,
}

#[derive(Deserialize, Debug)]
pub struct HomologyData {
    pub id: String,
    #[serde(default)]
    pub homologies: Vec<Homology>,
}

#[derive(Deserialize, Debug)]
pub struct Homology {
    #[serde(rename = "type")]
    pub type_: String,
    pub target: Target,
}

#[derive(Deserialize, Debug)]
pub struct Target {
    pub id: String,
    pub species: String,
    #[serde(rename = "perc_id")]
    pub perc_id: f64,
    #[serde(rename = "perc_pos")]
    pub perc_pos: f64,
}

pub fn fetch_paralogous_genes(gene_id: &str, species: &str) -> Result<Vec<Homology>, Box<dyn Error>> {
    println!("Fetching paralogous genes for gene ID: {}", gene_id);
    let url = format!(
        "https://rest.ensembl.org/homology/id/{}/{}?type=paralogues",
        species, gene_id
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    if response.status().is_success() {
        let homology_response: HomologyResponse = response.json()?;
        let mut paralogs = Vec::new();
        for data in homology_response.data {
            paralogs.extend(data.homologies);
        }
        Ok(paralogs)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching paralogous genes: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch paralogous genes.".into())
    }
}

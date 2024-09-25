// src/gene_sequence.rs

use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug)]
pub struct Xref {
    pub id: String,
    #[serde(rename = "type")]
    pub id_type: String,
}

#[derive(Deserialize, Debug)]
pub struct GeneInfoBasic {
    // pub id: String,
    pub assembly_name: String,
    pub start: u64,
    pub end: u64,
    pub strand: i8,
    pub seq_region_name: String,
}

pub fn fetch_gene_id(gene_symbol: &str) -> Result<String, Box<dyn Error>> {
    println!("Fetching gene ID for gene symbol: {}", gene_symbol);
    let url = format!(
        "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{}",
        gene_symbol
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    let xrefs: Vec<Xref> = response.json()?;
    for xref in xrefs {
        if xref.id_type == "gene" {
            println!("Found gene ID: {}", xref.id);
            return Ok(xref.id);
        }
    }
    Err("Gene ID not found.".into())
}

pub fn fetch_gene_info(gene_id: &str) -> Result<GeneInfoBasic, Box<dyn Error>> {
    println!("Fetching gene information for gene ID: {}", gene_id);
    let url = format!("https://rest.ensembl.org/lookup/id/{}", gene_id);
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    if response.status().is_success() {
        let gene_info: GeneInfoBasic = response.json()?;
        Ok(gene_info)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching gene info: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch gene info.".into())
    }
}

pub fn fetch_gene_sequence(gene_id: &str) -> Result<String, Box<dyn Error>> {
    println!("Fetching genomic sequence for gene ID: {}", gene_id);
    let url = format!(
        "https://rest.ensembl.org/sequence/id/{}?type=genomic",
        gene_id
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "text/plain")
        .send()?;

    if response.status().is_success() {
        let sequence = response.text()?;
        Ok(sequence)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching gene sequence: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch gene sequence.".into())
    }
}

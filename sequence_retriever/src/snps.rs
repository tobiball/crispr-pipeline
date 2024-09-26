// src/snps.rs

use reqwest::blocking::Client;
use serde::{Deserialize, Deserializer};
use serde::de::Error as SerdeError;
use serde_json::Value;
use std::error::Error as StdError;

#[derive(Deserialize, Debug, Clone)]
pub struct Variation {
    pub id: String,
    pub start: u64,
    pub end: u64,
    pub strand: i8,
    #[serde(rename = "allele_string")]
    pub allele_string: Option<String>,
    pub assembly_name: Option<String>,
    pub seq_region_name: Option<String>,
    pub source: Option<String>,
    #[serde(
        rename = "consequence_type",
        deserialize_with = "deserialize_consequence_type"
    )]
    pub consequence_type: Option<Vec<String>>,
    pub minor_allele_freq: Option<f64>, // Correct field for allele frequency
}

fn deserialize_consequence_type<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let helper: Option<Value> = Option::deserialize(deserializer)?;
    if let Some(value) = helper {
        match value {
            Value::String(s) => Ok(Some(vec![s])),
            Value::Array(arr) => {
                let mut res = Vec::new();
                for item in arr {
                    if let Value::String(s) = item {
                        res.push(s);
                    } else {
                        return Err(D::Error::custom("Expected string in array"));
                    }
                }
                Ok(Some(res))
            }
            _ => Err(D::Error::custom("Invalid type for consequence_type")),
        }
    } else {
        Ok(None)
    }
}

/// Fetch SNPs in a given genomic region using Ensembl REST API.
pub fn fetch_snps_in_region(
    chromosome: &str,
    start: u64,
    end: u64,
) -> Result<Vec<Variation>, Box<dyn StdError>> {
    println!(
        "Fetching SNPs in region {}:{}-{}",
        chromosome, start, end
    );
    let url = format!(
        "https://rest.ensembl.org/overlap/region/human/{}:{}-{}?feature=variation",
        chromosome.trim_start_matches("chr"),
        start,
        end
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Accept", "application/json")
        .send()?;

    if response.status().is_success() {
        let variations: Vec<Variation> = response.json()?;
        Ok(variations)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching SNPs: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch SNPs.".into())
    }
}

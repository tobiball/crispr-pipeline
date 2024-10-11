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
    pub consequence_type: Option<Vec<String>>, // Use custom deserializer
    pub minor_allele_freq: Option<f64>,
}

fn deserialize_consequence_type<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let helper: Option<Value> = Option::deserialize(deserializer)?;
    match helper {
        Some(Value::String(s)) => Ok(Some(vec![s])), // Convert single string to a vector
        Some(Value::Array(arr)) => {
            let mut result = Vec::new();
            for item in arr {
                if let Value::String(s) = item {
                    result.push(s);
                } else {
                    return Err(D::Error::custom("Expected a string in the array"));
                }
            }
            Ok(Some(result))
        }
        None => Ok(None), // Handle missing consequence_type
        _ => Err(D::Error::custom("Invalid type for consequence_type")),
    }
}



use crate::api_handler::APIHandler;

pub fn fetch_snps_in_region(
    api_handler: &APIHandler,
    chromosome: String,
    start: u64,
    end: u64,
    max_frequency: Option<f64>,
) -> Result<Vec<Variation>, Box<dyn StdError>> {
    println!(
        "Fetching SNPs in region {}:{}-{}",
        chromosome, start, end
    );
    let endpoint = format!(
        "/overlap/region/human/{}:{}-{}?feature=variation",
        chromosome.trim_start_matches("chr"),
        start,
        end
    );

    // Use the APIHandler to make the GET request
    let data = api_handler.get(&endpoint)?;

    // Parse the response data into a Vec<Variation>
    let mut variations: Vec<Variation> = serde_json::from_value(data)?;

    // Filter SNPs by minor allele frequency if a threshold is specified
    if let Some(threshold) = max_frequency {
        variations.retain(|snp| snp.minor_allele_freq.unwrap_or(0.0) < threshold);
    }

    Ok(variations)
}

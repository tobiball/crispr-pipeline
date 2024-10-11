// // src/paralogs.rs
//
// use reqwest::blocking::Client;
// use serde::Deserialize;
// use std::error::Error;
// use serde_json::Value;
// use crate::api_handler::APIHandler;
//
// #[derive(Deserialize, Debug)]
// pub struct HomologyResponse {
//     pub data: Vec<HomologyData>,
// }
//
// #[derive(Deserialize, Debug)]
// pub struct HomologyData {
//     pub id: String,
//     #[serde(default)]
//     pub homologies: Vec<Homology>,
// }
//
// #[derive(Deserialize, Debug)]
// pub struct Homology {
//     #[serde(rename = "type")]
//     pub type_: String,
//     pub target: Target,
// }
//
// #[derive(Deserialize, Debug)]
// pub struct Target {
//     pub id: String,
//     pub species: String,
//     #[serde(rename = "perc_id")]
//     pub perc_id: f64,
//     #[serde(rename = "perc_pos")]
//     pub perc_pos: f64,
// }
//
// pub fn fetch_paralogous_genes(api_handler: &APIHandler, gene_id: &str, species: &str) -> Result<Vec<Homology>, Box<dyn Error>> {
//     println!("Fetching paralogous genes for gene ID: {}", gene_id);
//     let endpoint = format!(
//         "/homology/id/{}/{}?type=paralogues",
//         species, gene_id
//     );
//     let data: Value = api_handler.get(&endpoint)?;
//
//     // Parse the response
//     let homology_response: HomologyResponse = serde_json::from_value(data)?;
//     let mut paralogs = Vec::new();
//     for data in homology_response.data {
//         paralogs.extend(data.homologies);
//     }
//
//     Ok(paralogs)
// }
//
//
// use crate::gene_analysis::GeneInfo;
//
// impl GeneInfo]
// /// Check if the guide RNA binds to any paralogous gene sequences.
// fn check_paralog_binding(&self, guide: &GuideRNA, ensembl_api: &APIHandler, genomic_sequences: &mut HashMap<String, String>) -> Result<(f64, Vec<String>), Box<dyn Error>> {
//     let mut binding_paralogs = Vec::new();
//     let mut paralog_score = 0.0;
//     let total_paralogs = self.paralogs.len() as f64;
//
//     for paralog_id in &self.paralogs {
//         let paralog_sequence = if let Some(seq) = genomic_sequences.get(paralog_id) {
//             seq.clone()
//         } else {
//             let seq = fetch_gene_sequence(paralog_id)?;
//             genomic_sequences.insert(paralog_id.clone(), seq.clone());
//             seq
//         };
//         if paralog_sequence.contains(&guide.sequence) {
//             paralog_score += 1.0;
//             binding_paralogs.push(paralog_id.clone());
//         }
//     }
//     Ok((paralog_score / total_paralogs, binding_paralogs))
// }
// }

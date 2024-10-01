mod gene_analysis;
mod exon_intron;
mod protein_domains;
mod snps;
mod regulatory_elements;
mod paralogs;
mod gene_sequence;
mod chopchop_integration;
mod api_handler;

use crate::api_handler::APIHandler;
use crate::gene_sequence::{fetch_gene_id, fetch_gene_info};
use crate::exon_intron::fetch_all_transcripts;
use crate::protein_domains::fetch_protein_domains;
use crate::snps::fetch_snps_in_region;
use crate::regulatory_elements::fetch_regulatory_elements;
use crate::paralogs::fetch_paralogous_genes;
use crate::gene_analysis::GeneInfo;
use std::error::Error;
use crate::chopchop_integration::{run_chopchop, ChopchopOptions};

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize API handlers
    let ensembl_api = APIHandler::new("https://rest.ensembl.org")?;
    let gtex_api = APIHandler::new("https://gtexportal.org/api/v2")?;

    // Define gene symbol
    let gene_symbol = "OAT";

    let gene_id = fetch_gene_id(gene_symbol)?;
    // Fetch transcripts
    let transcripts = fetch_all_transcripts(&ensembl_api, &gene_id)?;
    if transcripts.is_empty() {
        println!("No transcripts found for gene ID: {}", gene_id);
        return Ok(());
    }



    let gene_info = fetch_gene_info(&gene_id)?;

    let chrom = if gene_info.seq_region_name.starts_with("chr") {
        gene_info.seq_region_name.clone()
    } else {
        format!("chr{}", gene_info.seq_region_name)
    };
    let protein_features = fetch_protein_domains(&transcripts[0].id)?;
    let paralogs = fetch_paralogous_genes(&gene_id, "human")?;
    let snps = fetch_snps_in_region(chrom.clone(), gene_info.start, gene_info.end)?;
    let regulatory_elements = fetch_regulatory_elements(chrom.clone(), gene_info.start, gene_info.end)?;


    // Construct GeneInfo struct
    let mut gene = GeneInfo {
        gene_id: gene_id.clone(),
        gene_symbol: gene_symbol.to_string(),
        chrom: chrom.clone(),
        strand: gene_info.strand,
        start: gene_info.start,
        end: gene_info.end,
        transcripts,
        protein_features,
        snps,
        regulatory_elements,
        paralogs: paralogs.into_iter().map(|p| p.target.id).collect(),
    };







    // Analyze gene data to prioritize target regions
    let target_regions = gene.prioritize_regions(&ensembl_api, &gtex_api);


    println!("Top target regions based on analysis:");
    for region in target_regions.iter().take(5) {
        println!(
            "Transcript: {}, Exon: {}, Region: {}:{}-{}, Priority Score: {}",
            region.transcript_id, region.exon_id, gene.chrom, region.start, region.end, region.priority_score
        );
    }

    // Select top N regions for CHOPCHOP
    let top_n = 3;
    let selected_regions = target_regions.into_iter().take(top_n).collect::<Vec<_>>();

    // Run CHOPCHOP for each selected region
    for region in selected_regions {
        let region_str = format!("{}:{}-{}", gene.chrom, region.start, region.end);
        let output_dir = format!("{}_knockout_{}", gene_symbol, region.exon_id);

        let chopchop_options = ChopchopOptions {
            python_executable: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop_env/bin/python2.7".to_string(),
            chopchop_script: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop.py".to_string(),
            config_file: "crispr_pipeline/chopchop/config.json".to_string(),
            genome: "hg38".to_string(),
            target_region: region_str.clone(),
            output_dir: output_dir.clone(),
            pam_sequence: "NGG".to_string(),
            guide_length: 20,
            scoring_method: "DOENCH_2016".to_string(),
            max_mismatches: 3,
        };

        println!("Running CHOPCHOP for region: {}", region_str);
        if let Err(e) = run_chopchop(&chopchop_options) {
            eprintln!("Error running CHOPCHOP for region {}: {}", region_str, e);
            continue;
        }

        // Parse results (you'll need to implement parse_chopchop_results)
        // let guides = parse_chopchop_results(&output_dir)?;
    }

    Ok(())
}
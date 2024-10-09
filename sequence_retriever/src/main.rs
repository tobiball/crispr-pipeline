// src/main.rs

mod gene_analysis;
mod exon_intron;
mod protein_domains;
mod snps;
mod regulatory_elements;
mod paralogs;
mod gene_sequence;
mod chopchop_integration;
mod api_handler;
mod models;

use crate::gene_analysis::calculate_final_score;
use crate::api_handler::APIHandler;
use crate::gene_sequence::{fetch_gene_id, fetch_gene_info};
use crate::exon_intron::{fetch_all_transcripts, ExonDetail};
use crate::protein_domains::fetch_protein_domains;
use crate::snps::fetch_snps_in_region;
use crate::regulatory_elements::fetch_regulatory_elements;
use crate::paralogs::fetch_paralogous_genes;
use crate::gene_analysis::GeneInfo;
use crate::chopchop_integration::{run_chopchop, parse_chopchop_results, ChopchopOptions};

use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize API handlers
    let ensembl_api = APIHandler::new("https://rest.ensembl.org")?;
    let gtex_api = APIHandler::new("https://gtexportal.org/api/v2")?;

    // Define gene symbol
    let gene_symbol = "OAT";

    // Fetch gene ID and basic info
    let gene_id = fetch_gene_id(gene_symbol)?;
    let gene_info_basic = fetch_gene_info(&gene_id)?;

    // Ensure chromosome name includes 'chr' prefix
    let chrom = if gene_info_basic.seq_region_name.starts_with("chr") {
        gene_info_basic.seq_region_name.clone()
    } else {
        format!("chr{}", gene_info_basic.seq_region_name)
    };

    // Fetch transcripts
    let transcripts = fetch_all_transcripts(&ensembl_api, &gene_id)?;
    if transcripts.is_empty() {
        println!("No transcripts found for gene ID: {}", gene_id);
        return Ok(());
    }

    // Collect all exons from transcripts
    let exons: Vec<ExonDetail> = transcripts.iter()
        .flat_map(|t| t.exons.clone())
        .collect();

    // Fetch additional gene data
    let protein_features = fetch_protein_domains(&transcripts[0].id)?;
    let snps = fetch_snps_in_region(chrom.clone(), gene_info_basic.start, gene_info_basic.end)?;
    let regulatory_elements = fetch_regulatory_elements(chrom.clone(), gene_info_basic.start, gene_info_basic.end)?;
    let paralogs = fetch_paralogous_genes(&gene_id, "human")?.into_iter().map(|p| p.target.id).collect();

    // Construct GeneInfo struct
    let mut gene_info = GeneInfo {
        gene_id: gene_id.clone(),
        gene_symbol: gene_symbol.to_string(),
        chrom: chrom.clone(),
        strand: gene_info_basic.strand,
        start: gene_info_basic.start,
        end: gene_info_basic.end,
        transcripts,
        protein_features,
        snps,
        regulatory_elements,
        paralogs,
        exons, // Use the collected exons here
    };

    // Define the main output directory with absolute path
    let main_output_dir = format!("/home/mrcrispr/crispr_pipeline/output/{}", gene_symbol);

    // Ensure the main output directory exists
    fs::create_dir_all(&main_output_dir)?;

    // Define the dedicated CHOPCHOP output subdirectory
    let chopchop_output_dir = format!("{}/chopchop_output", main_output_dir);

    // Ensure the CHOPCHOP output directory exists
    fs::create_dir_all(&chopchop_output_dir)?;

    // Set up CHOPCHOP options
    let chopchop_options = ChopchopOptions {
        python_executable: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop_env/bin/python2.7".to_string(),
        chopchop_script: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop.py".to_string(),
        genome: "hg38".to_string(),
        target_type: "GENE_NAME".to_string(), // Specify the target type as GENE_NAME
        target: gene_symbol.to_string(),      // Use the gene symbol as the target
        output_dir: chopchop_output_dir.clone(), // Use the dedicated CHOPCHOP output directory
        pam_sequence: "NGG".to_string(),
        guide_length: 20,
        scoring_method: "DOENCH_2016".to_string(),
        max_mismatches: 3,
    };

    println!("Running CHOPCHOP for target: {}", gene_symbol);
    if let Err(e) = run_chopchop(&chopchop_options) {
        eprintln!("Error running CHOPCHOP for target {}: {}", gene_symbol, e);
        return Ok(());
    }

    let mut guides = parse_chopchop_results(&chopchop_output_dir)?;

    for guide in &mut guides {
        gene_info.score_guide_rna(guide, &ensembl_api, &gtex_api)?;
        calculate_final_score(guide);
    }

    // Sort guides by final score
    guides.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));

    // Display top guides
    println!("Top guide RNAs:");
    for (index, guide) in guides.iter().take(3).enumerate() {
        println!("Guide {}: {}", index + 1, guide.sequence);
        println!("Position: {}:{}-{}, Strand: {}", guide.chromosome, guide.start, guide.end, guide.strand);
        println!("GC Content: {:.2}%", guide.gc_content);
        println!("Self-complementarity: {}", guide.self_complementarity);
        println!("Mismatches: MM0={}, MM1={}, MM2={}, MM3={}", guide.mm0, guide.mm1, guide.mm2, guide.mm3);
        println!("CHOPCHOP Efficiency: {:.2}", guide.chopchop_efficiency);
        println!("Score Breakdown:");
        if let Some(breakdown) = &guide.score_breakdown {
            for (score_name, score_value) in breakdown {
                println!("  {}: {:.2}", score_name, score_value);
            }
        }
        println!("Final Combined Score: {:.2}", guide.final_score.unwrap());
        println!();
    }

    Ok(())
}

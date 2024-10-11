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
use crate::exon_intron::{fetch_all_transcripts, fetch_versioned_gene_id, fetch_exon_expression_data, ExonDetail, MissingReason, find_overlapping_exons, TranscriptDetail};
use crate::protein_domains::fetch_protein_domains;
use crate::snps::fetch_snps_in_region;
use crate::regulatory_elements::fetch_regulatory_elements;
use crate::gene_analysis::GeneInfo;
use crate::chopchop_integration::{run_chopchop, parse_chopchop_results, ChopchopOptions};
use std::collections::HashMap;


use std::error::Error;
use std::fs;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize API handlers
    let ensembl_api = APIHandler::new("https://rest.ensembl.org")?;
    let gtex_api = APIHandler::new("https://gtexportal.org/api/v2")?;

    // Define gene symbol
    let gene_symbol = "OAT";

    let mut genomic_sequences = HashMap::new();

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

    // Find the canonical transcript based on the is_canonical field
    let canonical_transcript = transcripts
        .iter()
        .find(|t| t.is_canonical == Some(1))
        .cloned() // Clone the transcript to own the data
        .ok_or("Canonical transcript not found")?;


    // Collect all exons from transcripts
    let exons: Vec<ExonDetail> = transcripts.iter()
        .flat_map(|t| t.exons.clone())
        .collect();

    // Fetch additional gene data
    let protein_features = fetch_protein_domains(&ensembl_api,&transcripts[0].id)?;
    let snps = fetch_snps_in_region(&ensembl_api, chrom.clone(), gene_info_basic.start, gene_info_basic.end, Some(0.05))?;
    let regulatory_elements = fetch_regulatory_elements(&ensembl_api,chrom.clone(), gene_info_basic.start, gene_info_basic.end)?;

    let versioned_gene_id = fetch_versioned_gene_id(&ensembl_api, &gene_id)?;

    // Fetch expression data and store it in a HashMap
    let exon_expression_data = fetch_exon_expression_data(&gtex_api, &versioned_gene_id, None)?;

    // Collect all exons from transcripts
    let mut exons: Vec<ExonDetail> = transcripts.iter()
        .flat_map(|t| t.exons.clone())
        .collect();

    // Assign expression scores to exons
    for exon in &mut exons {
        let exon_number = exon.exon_number; // No need to clone, u32 implements Copy
        if let Some(expression_score) = exon_expression_data.get(&exon_number) {
            exon.expression_score = Some(*expression_score);
            exon.expression_missing_reason = None;
        } else {
            exon.expression_missing_reason = Some(MissingReason::NotFound);
        }
    }

    // Construct GeneInfo struct
    let gene_info = GeneInfo {
        gene_id: gene_id.clone(),
        versioned_gene_id,
        gene_symbol: gene_symbol.to_string(),
        chrom: chrom.clone(),
        strand: gene_info_basic.strand,
        start: gene_info_basic.start,
        end: gene_info_basic.end,
        transcripts,
        protein_features,
        snps,
        regulatory_elements,
        exons,
        exon_expression_data,
        canonical_transcript
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
        target_type: "GENE".to_string(), // Changed from 'GENE_NAME' to 'GENE'
        target: gene_symbol.to_string(), // Use the gene symbol as the target
        output_dir: chopchop_output_dir.clone(),
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
        guide.overlapping_exons = find_overlapping_exons(&gene_info.exons, guide.start, guide.end);
        gene_info.score_guide_rna(guide, &ensembl_api, &gtex_api, &mut genomic_sequences)?;
        calculate_final_score(guide);
    }

    // Sort guides by final score
    guides.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));

    // Display top guides
    println!("Top guide RNAs:");
    for (index, guide) in guides.iter().take(50).enumerate() {
        println!("Guide {}: {}", index + 1, guide.sequence);
        println!("Position: {}:{}-{}, Strand: {}", guide.chromosome, guide.start, guide.end, guide.strand);
        println!("GC Content: {:.2}%", guide.gc_content);
        println!("Self-complementarity: {}", guide.self_complementarity);
        println!("Mismatches: MM0={}, MM1={}, MM2={}, MM3={}", guide.mm0, guide.mm1, guide.mm2, guide.mm3);
        println!("CHOPCHOP Efficiency: {:.2}", guide.chopchop_efficiency);

        // Display expression data if available
        if let Some(expression_level) = guide.expression_score {
            println!("Expression Level at Guide Position: {:.2}", expression_level);
        } else {
            println!("Expression Level at Guide Position: Not Available");
        }

        // Display conservation score
        if let Some(conservation_score) = guide.conservation_score {
            println!("Conservation Score: {:.2}", conservation_score);
        } else {
            println!("Conservation Score: Not Available");
        }

        // Display overlapping exons with expression levels
        if !guide.overlapping_exons.is_empty() {
            println!("Overlapping Exons:");
            for exon in &guide.overlapping_exons {
                println!(
                    "  Exon ID: {}, Position: {}:{}-{}, Expression Level: {}",
                    exon.id,
                    exon.seq_region_name,
                    exon.start,
                    exon.end,
                    exon
                        .expression_score
                        .map(|score| format!("{:.2}", score))
                        .unwrap_or_else(|| "Not Available".to_string())
                );
            }
        } else {
            println!("Overlapping Exons: None");
        }
        println!("Overlapping SNPs:");
        for snp in &guide.overlapping_snps {
            let frequency_info = snp.minor_allele_freq
                .map(|freq| format!("Frequency: {:.4}", freq))
                .unwrap_or_else(|| "Frequency: Not Available".to_string());
            println!(
                "  SNP ID: {}, Position: {}:{}-{}, {}",
                snp.id,
                snp.seq_region_name.as_deref().unwrap_or(""),
                snp.start,
                snp.end,
                frequency_info
            );
        }


        // Display overlapping protein domains
        if !guide.overlapping_protein_domains.is_empty() {
            println!("Overlapping Protein Domains:");
            for domain in &guide.overlapping_protein_domains {
                println!("  Domain ID: {:?}, Description: {:?}", domain.id, domain.description);
            }
        } else {
            println!("Overlapping Protein Domains: None");
        }

        // Display transcript coverage
        println!("Transcript Coverage: {}/{} ({:.2}%)",
                 guide.covered_transcripts,
                 guide.total_transcripts,
                 (guide.covered_transcripts as f64 / guide.total_transcripts as f64) * 100.0
        );

        // Display score breakdown
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
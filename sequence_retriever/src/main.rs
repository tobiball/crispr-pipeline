// src/main.rs

mod gene_sequence;
mod exon_intron;
mod protein_domains;
mod snps;
mod regulatory_elements;
mod paralogs;
mod chopchop_integration;
mod gene_analysis;

use crate::gene_sequence::{fetch_gene_id, fetch_gene_info, fetch_gene_sequence};
use crate::exon_intron::fetch_all_transcripts;
use crate::protein_domains::fetch_protein_domains;
use crate::snps::fetch_snps_in_region;
use crate::regulatory_elements::fetch_regulatory_elements;
use crate::paralogs::fetch_paralogous_genes;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Gene symbol
    let gene_symbol = "OAT";

    // Fetch gene ID
    let gene_id = fetch_gene_id(gene_symbol)?;
    println!("Gene ID: {}", gene_id);

    // Fetch gene information to confirm assembly version
    let gene_info = fetch_gene_info(&gene_id)?;
    println!(
        "Gene is located on chromosome: {}, Strand: {}",
        gene_info.seq_region_name, gene_info.strand
    );
    println!("Gene assembly version: {}", gene_info.assembly_name);

    // Fetch the full genomic sequence of the gene
    let gene_sequence = fetch_gene_sequence(&gene_id)?;
    println!("Gene sequence length: {}", gene_sequence.len());

    // Fetch all transcripts and analyze exon-intron structures
    let all_transcripts = fetch_all_transcripts(&gene_id)?;
    println!("Total transcripts found: {}", all_transcripts.len());

    // Limit to the first 3 transcripts for printing
    let max_transcripts_to_display = 3;
    for (i, transcript) in all_transcripts.iter().take(max_transcripts_to_display).enumerate() {
        println!("Transcript {} - ID: {}", i + 1, transcript.id);
        println!("Is Canonical: {}", transcript.is_canonical.unwrap_or(0));
        println!("Number of Exons: {}", transcript.exons.len());
    }
    if all_transcripts.len() > max_transcripts_to_display {
        println!("...and {} more transcripts", all_transcripts.len() - max_transcripts_to_display);
    }

    // Fetch protein features (domains) for the selected transcript
    let selected_transcript_id = &all_transcripts[0].id; // Assume first transcript for simplicity
    let protein_features = fetch_protein_domains(selected_transcript_id)?;
    println!("Total protein features found: {}", protein_features.len());

    // Limit to first 5 protein features for printing
    let max_protein_features_to_display = 5;
    for (i, feature) in protein_features.iter().take(max_protein_features_to_display).enumerate() {
        println!(
            "Protein Feature {}: {} (InterPro ID: {}), Start: {}, End: {}",
            i + 1,
            feature.description.as_deref().unwrap_or("N/A"),
            feature.interpro.as_deref().unwrap_or("N/A"),
            feature.start,
            feature.end
        );
    }
    if protein_features.len() > max_protein_features_to_display {
        println!(
            "...and {} more protein features",
            protein_features.len() - max_protein_features_to_display
        );
    }

    // Fetch SNPs in the gene region
    let snps = fetch_snps_in_region(
        &gene_info.seq_region_name,
        gene_info.start,
        gene_info.end,
    )?;
    println!("Total SNPs found: {}", snps.len());

    // Limit to first 5 SNPs for printing
    let max_snps_to_display = 5;
    for (i, snp) in snps.iter().take(max_snps_to_display).enumerate() {
        println!(
            "SNP {}: ID: {}, Position: {}, Alleles: {}",
            i + 1,
            snp.id,
            snp.start,
            snp.allele_string.as_deref().unwrap_or("N/A")
        );
    }
    if snps.len() > max_snps_to_display {
        println!("...and {} more SNPs", snps.len() - max_snps_to_display);
    }

    // Fetch regulatory elements in the gene region
    let regulatory_elements = fetch_regulatory_elements(
        &gene_info.seq_region_name,
        gene_info.start,
        gene_info.end,
    )?;
    println!("Total regulatory elements found: {}", regulatory_elements.len());

    // Limit to first 3 regulatory elements for printing
    let max_reg_elements_to_display = 3;
    for (i, element) in regulatory_elements.iter().take(max_reg_elements_to_display).enumerate() {
        println!(
            "Regulatory Element {}: ID: {}, Type: {}, Start: {}, End: {}",
            i + 1, element.id, element.feature_type, element.start, element.end
        );
    }
    if regulatory_elements.len() > max_reg_elements_to_display {
        println!(
            "...and {} more regulatory elements",
            regulatory_elements.len() - max_reg_elements_to_display
        );
    }

    // Fetch paralogous genes
    let paralogs = fetch_paralogous_genes(&gene_id, "human")?;
    println!("Total paralogous genes found: {}", paralogs.len());

    // Limit to first 3 paralogs for printing
    let max_paralogs_to_display = 3;
    for (i, homology) in paralogs.iter().take(max_paralogs_to_display).enumerate() {
        println!(
            "Paralog {}: ID: {}, Percentage Identity: {:.2}%",
            i + 1,
            homology.target.id,
            homology.target.perc_id
        );
    }
    if paralogs.len() > max_paralogs_to_display {
        println!(
            "...and {} more paralogous genes",
            paralogs.len() - max_paralogs_to_display
        );
    }

    Ok(())
}

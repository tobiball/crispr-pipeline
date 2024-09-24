// src/main.rs

mod gene_sequence;
mod exon_intron;
mod protein_domains;
mod snps;
mod regulatory_elements;
mod paralogs;

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

    // Select canonical transcript
    let selected_transcript = all_transcripts
        .iter()
        .find(|t| t.is_canonical == Some(1))
        .unwrap_or(&all_transcripts[0]);
    let selected_transcript_id = &selected_transcript.id;
    println!("Selected transcript ID: {}", selected_transcript_id);

    // Fetch protein features (domains) for the selected transcript
    let protein_features = fetch_protein_domains(selected_transcript_id)?;
    println!("Protein features found: {}", protein_features.len());

    for feature in &protein_features {
        println!(
            "Feature: {} (InterPro ID: {}), Start: {}, End: {}",
            feature.description.as_deref().unwrap_or("N/A"),
            feature.interpro.as_deref().unwrap_or("N/A"),
            feature.start,
            feature.end
        );
    }

    // Fetch SNPs in the gene region
    let snps = fetch_snps_in_region(
        &gene_info.seq_region_name,
        gene_info.start,
        gene_info.end,
    )?;
    println!("Total SNPs found in gene region: {}", snps.len());

    // for snp in &snps {
    //     println!(
    //         "SNP ID: {}, Position: {}, Alleles: {}, Consequences: {:?}",
    //         snp.id,
    //         snp.start,
    //         snp.allele_string.as_deref().unwrap_or("N/A"),
    //         snp.consequence_type.as_ref().unwrap_or(&vec!["N/A".to_string()])
    //     );
    // }

    // Fetch regulatory elements in the gene region
    let regulatory_elements = fetch_regulatory_elements(
        &gene_info.seq_region_name,
        gene_info.start,
        gene_info.end,
    )?;
    println!(
        "Total regulatory elements found in gene region: {}",
        regulatory_elements.len()
    );

    for element in &regulatory_elements {
        println!(
            "Regulatory Feature ID: {}, Type: {}, Start: {}, End: {}",
            element.id, element.feature_type, element.start, element.end
        );
    }

    // Fetch paralogous genes
    let paralogs = fetch_paralogous_genes(&gene_id, "human")?;
    println!("Total paralogous genes found: {}", paralogs.len());

    for homology in &paralogs {
        println!(
            "Paralogous Gene ID: {}, Percentage Identity: {:.2}%",
            homology.target.id, homology.target.perc_id
        );
    }

    Ok(())
}

// src/main.rs

mod chopchop_integration;
mod gene_analysis;
mod exon_intron;
mod protein_domains;
mod snps;
mod regulatory_elements;
mod paralogs;
mod gene_sequence;

use crate::gene_sequence::{fetch_gene_id, fetch_gene_info};
use crate::exon_intron::fetch_all_transcripts;
use crate::protein_domains::fetch_protein_domains;
use crate::snps::fetch_snps_in_region;
use crate::regulatory_elements::fetch_regulatory_elements;
use crate::paralogs::fetch_paralogous_genes;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use crate::chopchop_integration::{parse_chopchop_results, run_chopchop, ensure_gene_table_up_to_date, ChopchopOptions, test_two_bit_to_fa};
use crate::gene_analysis::GeneInfo;


fn main() -> Result<(), Box<dyn Error>> {



    test_two_bit_to_fa().expect("TODO: panic message");
    // Initialize logging (optional)
    // env_logger::init(); // Uncomment if using a logging framework


    // Step 1: Ensure the gene table is up-to-date
    ensure_gene_table_up_to_date()?;

    // Step 2: Define gene symbol
    let gene_symbol = "OAT";

    // Step 3: Fetch gene ID
    let gene_id = fetch_gene_id(gene_symbol)?;
    println!("Gene ID: {}", gene_id);

    // Step 4: Fetch gene information
    let gene_info = fetch_gene_info(&gene_id)?;
    println!(
        "Gene is located on chromosome: {}, region: {}-{}",
        gene_info.seq_region_name, gene_info.start, gene_info.end
    );

    // Step 5: Fetch transcripts, protein features, SNPs, regulatory elements, paralogs
    let transcripts = fetch_all_transcripts(&gene_id)?;
    if transcripts.is_empty() {
        println!("No transcripts found for gene ID: {}", gene_id);
        return Ok(());
    }
    let protein_features = fetch_protein_domains(&transcripts[0].id)?;
    let snps = fetch_snps_in_region(&gene_info.seq_region_name, gene_info.start, gene_info.end)?;
    let regulatory_elements = fetch_regulatory_elements(&gene_info.seq_region_name, gene_info.start, gene_info.end)?;
    let paralogs = fetch_paralogous_genes(&gene_id, "human")?;

    let chrom = if gene_info.seq_region_name.starts_with("chr") {
        gene_info.seq_region_name.clone()
    } else {
        format!("chr{}", gene_info.seq_region_name)
    };

    println!(
        "Gene is located on chromosome: {}, region: {}-{}",
        chrom, gene_info.start, gene_info.end
    );
    // Step 6: Construct GeneInfo struct
    let gene = GeneInfo {
        gene_id: gene_id.clone(),
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

    // Step 7: Analyze gene data to prioritize target regions
    let target_regions = gene.prioritize_regions();

    println!("Top target regions based on analysis:");
    for region in target_regions.iter().take(5) { // Display top 5 for example
        println!(
            "Transcript: {}, Exon: {}, Region: {}:{}-{}, Priority Score: {}",
            region.transcript_id, region.exon_id, gene.chrom, region.start, region.end, region.priority_score
        );
    }

    // Step 8: Select top N regions for CHOPCHOP
    let top_n = 3;
    let selected_regions = target_regions.into_iter().take(top_n).collect::<Vec<_>>();

    // Step 9: Run CHOPCHOP for each selected region
    for region in selected_regions {
        let region_str = format!("{}:{}-{}", gene.chrom, region.start, region.end);
        let output_dir = format!("{}_knockout_{}", gene_symbol, region.exon_id);

        let chopchop_options = ChopchopOptions {
            python_executable: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop_env/bin/python2.7".to_string(), // Correct path to python2.7
            chopchop_script: "/home/mrcrispr/crispr_pipeline/chopchop/chopchop.py".to_string(), // Absolute path to chopchop.py
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
            continue; // Proceed to the next region
        }

        // Parse results
        match parse_chopchop_results(&chopchop_options.output_dir) {
            Ok(results) => {
                println!("CHOPCHOP Results for {}:", region_str);
                for (i, guide) in results.iter().enumerate().take(5) { // Display top 5 gRNAs
                    println!(
                        "Guide {}: Sequence: {}, Position: {}, Strand: {}, Score: {}",
                        i + 1, guide.guide_sequence, guide.pos, guide.strand, guide.score
                    );
                }

                // Save results to a summary file
                let summary_path = std::path::Path::new(&chopchop_options.output_dir).join("chopchop_summary.txt");
                let mut file = File::create(&summary_path)?;
                for guide in &results {
                    writeln!(file, "{:?}", guide)?;
                }
                println!("Summary saved to {:?}", summary_path);
            }
            Err(e) => {
                eprintln!("Failed to parse CHOPCHOP results for {}: {}", region_str, e);
            }
        }
    }

    // Step 10: (Optional) Apply Machine Learning to prioritize gRNAs further
    /*
    machine_learning::load_model()?;
    for region in selected_regions {
        // Extract features for each gRNA
        let features = extract_features(&region, &gene);
        let priority = machine_learning::predict_priority(&features)?;
        // Use priority to rank or filter gRNAs
    }
    */

    Ok(())
}

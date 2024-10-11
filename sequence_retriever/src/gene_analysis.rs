// src/gene_analysis.rs

use std::collections::HashMap;
use std::error::Error;
use crate::api_handler::APIHandler;
use crate::exon_intron::{TranscriptDetail, ExonDetail, fetch_conservation_score_at_position, is_coding_transcript};
use crate::snps::Variation;
use crate::regulatory_elements::RegulatoryFeature;
use crate::protein_domains::ProteinFeature;
use crate::models::GuideRNA;
use crate::gene_sequence::fetch_gene_sequence;

pub struct GeneInfo {
    pub gene_id: String,
    pub versioned_gene_id: String,
    pub gene_symbol: String,
    pub chrom: String,
    pub strand: i8,
    pub start: u64,
    pub end: u64,
    pub transcripts: Vec<TranscriptDetail>,
    pub protein_features: Vec<ProteinFeature>,
    pub snps: Vec<Variation>,
    pub regulatory_elements: Vec<RegulatoryFeature>,
    pub exons: Vec<ExonDetail>,
    pub exon_expression_data: HashMap<u32, f64>,
    pub canonical_transcript: TranscriptDetail,
}

impl GeneInfo {
    /// Score the guide RNA based on various biological factors.
    pub fn score_guide_rna(&self, guide: &mut GuideRNA, ensembl_api: &APIHandler, gtex_api: &APIHandler, genomic_sequences: &mut HashMap<String, String>) -> Result<(), Box<dyn Error>> {
        // Define weights
        const OFF_TARGET_WEIGHT: f64 = 0.25;
        const SNP_WEIGHT: f64 = 0.15;
        const EXPRESSION_WEIGHT: f64 = 0.15;
        const CONSERVATION_WEIGHT: f64 = 0.15;
        const SELF_COMP_WEIGHT: f64 = 0.05;
        const PROTEIN_DOMAIN_WEIGHT: f64 = 0.1;
        const TRANSCRIPT_COVERAGE_WEIGHT: f64 = 0.15;

        // Define penalties
        const P0: f64 = 100.0; // MM0 penalty
        const P1: f64 = 10.0;  // MM1 penalty
        const P2: f64 = 5.0;   // MM2 penalty
        const P3: f64 = 2.0;   // MM3 penalty

        const PENALTY_PER_SNP: f64 = 20.0;
        const PENALTY_PER_SELF_COMP: f64 = 5.0;
        const PENALTY_PER_PROTEIN_DOMAIN: f64 = 20.0;

        let mut score_breakdown = Vec::new();

        // Filter for coding transcripts
        let coding_transcripts: Vec<&TranscriptDetail> = self.transcripts
            .iter()
            .filter(|t| is_coding_transcript(t))
            .collect();

        // Off-Target Score
        let off_target_penalty = guide.mm0 as f64 * P0 + guide.mm1 as f64 * P1 + guide.mm2 as f64 * P2 + guide.mm3 as f64 * P3;
        let off_target_score = (100.0 - off_target_penalty).max(0.0);
        score_breakdown.push(("Off-Target Score".to_string(), off_target_score));

        // SNP Score
        // SNP Score
        let overlapping_snps: Vec<Variation> = self.snps.iter()
            .filter(|snp| snp.start <= guide.end && snp.end >= guide.start)
            .cloned()
            .collect();

        let mut snp_penalty = 0.0;
        for snp in &overlapping_snps {
            if let Some(freq) = snp.minor_allele_freq {
                // Apply a penalty based on the minor allele frequency
                let penalty = freq * PENALTY_PER_SNP; // Adjust the scaling factor as needed
                snp_penalty += penalty;
            }
        }
        let snp_score = (100.0 - snp_penalty).max(0.0);
        score_breakdown.push(("SNP Score".to_string(), snp_score));

        // Store the overlapping SNPs in the guide
        guide.overlapping_snps = overlapping_snps;


        // Expression Score
        let expression_score = if !guide.overlapping_exons.is_empty() {
            let total_expression: f64 = guide.overlapping_exons.iter()
                .filter_map(|exon| self.exon_expression_data.get(&exon.exon_number))
                .sum();
            let avg_expression = total_expression / guide.overlapping_exons.len() as f64;
            Some(avg_expression)
        } else {
            None
        };

        guide.expression_score = expression_score;

        // Normalize the expression score
        let normalized_expression_score = expression_score
            .map(|score| (score / 1000.0).min(1.0) * 100.0)
            .unwrap_or(0.0);

        println!("Normalized expression score: {}", normalized_expression_score);
        score_breakdown.push(("Expression Score".to_string(), normalized_expression_score));

        // Conservation Score
        let conservation = fetch_conservation_score_at_position(
            ensembl_api,
            &self.chrom,
            guide.start,
            guide.end,
        )?;
        let conservation_score = if let Some(cons) = conservation {
            guide.conservation_score = Some(cons); // Store the conservation score
            (cons * 100.0).min(100.0)
        } else {
            panic!("No conservation")
        };
        score_breakdown.push(("Conservation Score".to_string(), conservation_score));

        // Self-Complementarity Score
        let self_comp_penalty = guide.self_complementarity as f64 * PENALTY_PER_SELF_COMP;
        let self_comp_score = (100.0 - self_comp_penalty).max(0.0);
        score_breakdown.push(("Self-Complementarity Score".to_string(), self_comp_score));

        // Protein Domain Score
        let overlapping_domains: Vec<ProteinFeature> = self.protein_features.iter()
            .filter(|domain| domain.start <= guide.end && domain.end >= guide.start && domain.is_essential)
            .cloned()
            .collect();
        let overlapping_domains_count = overlapping_domains.len();
        let protein_domain_penalty = overlapping_domains_count as f64 * PENALTY_PER_PROTEIN_DOMAIN;
        let protein_domain_score = (100.0 - protein_domain_penalty).max(0.0);
        score_breakdown.push(("Protein Domain Score".to_string(), protein_domain_score));

        // Store the overlapping protein domains in the guide
        guide.overlapping_protein_domains = overlapping_domains;

        // Update transcript coverage calculation
        let covered_transcripts = coding_transcripts.iter()
            .filter(|transcript| self.guide_covers_transcript(guide, transcript))
            .count();

        let total_coding_transcripts = coding_transcripts.len();

        // Calculate canonical transcript coverage
        let canonical_coverage = if self.guide_covers_transcript(guide, &self.canonical_transcript) {
            1.0
        } else {
            0.0
        };
        let canonical_weight = 0.2; // Adjust as needed
        let coverage_score = (covered_transcripts as f64 / total_coding_transcripts as f64) * (1.0 - canonical_weight)
            + canonical_coverage * canonical_weight;

        score_breakdown.push(("Coding Transcript Coverage Score".to_string(), coverage_score * 100.0));

        // Update guide with coverage information
        guide.covered_transcripts = covered_transcripts;
        guide.total_transcripts = total_coding_transcripts;

        // Calculate final biological score
        let final_biological_score =
            OFF_TARGET_WEIGHT * off_target_score +
                SNP_WEIGHT * snp_score +
                EXPRESSION_WEIGHT * normalized_expression_score +
                CONSERVATION_WEIGHT * conservation_score +
                SELF_COMP_WEIGHT * self_comp_score +
                PROTEIN_DOMAIN_WEIGHT * protein_domain_score +
                TRANSCRIPT_COVERAGE_WEIGHT * coverage_score * 100.0;

        guide.custom_biological_score = Some(final_biological_score);
        guide.score_breakdown = Some(score_breakdown);

        Ok(())
    }

    fn guide_covers_transcript(&self, guide: &GuideRNA, transcript: &TranscriptDetail) -> bool {
        let covers = transcript.exons.iter().any(|exon|
            guide.start <= exon.end && guide.end >= exon.start
        );
        println!("Guide {}:{}-{} covers transcript {}: {}",
                 guide.chromosome, guide.start, guide.end,
                 transcript.id, covers);
        covers
    }

    pub fn get_expression_at_position(&self, start: u64, end: u64) -> Option<f64> {
        // Print the entire exon expression data for debugging purposes
        println!("Exon expression data: {:?}", self.exon_expression_data);

        // Identify overlapping exons
        let overlapping_exons: Vec<&ExonDetail> = self.exons.iter()
            .filter(|exon| exon.start <= end && exon.end >= start)
            .collect();

        if overlapping_exons.is_empty() {
            println!("No overlapping exons found.");
            return None;
        }

        let mut total_expression = 0.0;
        let mut count = 0;

        for exon in overlapping_exons {
            // Debugging output to check which exon number we're examining
            println!("Checking expression data for exon number: {}", exon.exon_number);

            // Check if the exon number exists in the exon expression data
            if let Some(expression) = self.exon_expression_data.get(&exon.exon_number) {
                total_expression += expression;
                count += 1;
            } else {
                println!("No expression data found for exon number: {}", exon.exon_number);
            }
        }

        if count > 0 {
            Some(total_expression / count as f64)
        } else {
            println!("No expression data available for the overlapping exons.");
            None
        }
    }
}

/// Calculate the final score for a guide RNA by combining various metrics.
pub fn calculate_final_score(guide: &mut GuideRNA) {
    const CHOPCHOP_WEIGHT: f64 = 0.5;
    const BIOLOGICAL_WEIGHT: f64 = 0.5;

    let chopchop_score = guide.chopchop_efficiency;
    let biological_score = guide.custom_biological_score.unwrap();

    let final_score = (chopchop_score * CHOPCHOP_WEIGHT) + (biological_score * BIOLOGICAL_WEIGHT);

    guide.final_score = Some(final_score);

    // Add final score breakdown
    if let Some(breakdown) = &mut guide.score_breakdown {
        breakdown.push(("CHOPCHOP Efficiency".to_string(), chopchop_score));
        breakdown.push(("Final Combined Score".to_string(), final_score));
    }
}
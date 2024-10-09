// src/gene_analysis.rs

use std::error::Error;
use crate::api_handler::APIHandler;
use crate::exon_intron::{TranscriptDetail, ExonDetail, fetch_expression_score_at_position, fetch_conservation_score_at_position};
use crate::snps::Variation;
use crate::regulatory_elements::RegulatoryFeature;
use crate::protein_domains::ProteinFeature;
use crate::models::GuideRNA;
use crate::gene_sequence::fetch_gene_sequence;

pub struct GeneInfo {
    pub gene_id: String,
    pub gene_symbol: String,
    pub chrom: String,
    pub strand: i8,
    pub start: u64,
    pub end: u64,
    pub transcripts: Vec<TranscriptDetail>,
    pub protein_features: Vec<ProteinFeature>,
    pub snps: Vec<Variation>,
    pub regulatory_elements: Vec<RegulatoryFeature>,
    pub paralogs: Vec<String>,
    pub exons: Vec<ExonDetail>,
}

impl GeneInfo {
    /// Score the guide RNA based on various biological factors.
    pub fn score_guide_rna(&self, guide: &mut GuideRNA, ensembl_api: &APIHandler, gtex_api: &APIHandler) -> Result<(), Box<dyn Error>> {
        // Define weights
        const OFF_TARGET_WEIGHT: f64 = 0.25;
        const SNP_WEIGHT: f64 = 0.15;
        const PARALOG_WEIGHT: f64 = 0.15;
        const EXPRESSION_WEIGHT: f64 = 0.15;
        const CONSERVATION_WEIGHT: f64 = 0.15;
        const SELF_COMP_WEIGHT: f64 = 0.05;
        const PROTEIN_DOMAIN_WEIGHT: f64 = 0.1;

        // Define penalties
        const P0: f64 = 100.0; // MM0 penalty
        const P1: f64 = 10.0;  // MM1 penalty
        const P2: f64 = 5.0;   // MM2 penalty
        const P3: f64 = 2.0;   // MM3 penalty

        const PENALTY_PER_SNP: f64 = 20.0;
        const PENALTY_PER_PARALOG: f64 = 100.0;
        const PENALTY_PER_SELF_COMP: f64 = 5.0;
        const PENALTY_PER_PROTEIN_DOMAIN: f64 = 20.0;

        let mut score_breakdown = Vec::new();

        // Off-Target Score
        let off_target_penalty = guide.mm0 as f64 * P0 + guide.mm1 as f64 * P1 + guide.mm2 as f64 * P2 + guide.mm3 as f64 * P3;
        let off_target_score = (100.0 - off_target_penalty).max(0.0);
        score_breakdown.push(("Off-Target Score".to_string(), off_target_score));

        // SNP Score
        let overlapping_snps: Vec<Variation> = self.snps.iter()
            .filter(|snp| snp.start <= guide.end && snp.end >= guide.start)
            .cloned()
            .collect();
        let snp_penalty = overlapping_snps.len() as f64 * PENALTY_PER_SNP;
        let snp_score = (100.0 - snp_penalty).max(0.0);
        score_breakdown.push(("SNP Score".to_string(), snp_score));

        // Store the overlapping SNPs in the guide
        guide.overlapping_snps = overlapping_snps;

        // Paralog Score
        let (paralog_binding_score, binding_paralogs) = self.check_paralog_binding(guide, ensembl_api)?;
        let paralog_penalty = paralog_binding_score * 100.0;
        let paralog_score = (100.0 - paralog_penalty).max(0.0);
        score_breakdown.push(("Paralog Score".to_string(), paralog_score));

        // Store the binding paralogs in the guide
        guide.binding_paralogs = binding_paralogs;

        // Expression Score
        let expression_score = fetch_expression_score_at_position(
            gtex_api,
            &self.gene_id,
            &guide.chromosome,
            guide.start,
            guide.end,
            ensembl_api,
            &self.exons,
        )?;
        let expression_score = if let Some(exp) = expression_score {
            guide.expression_score = Some(exp); // Store the expression level
            (exp * 100.0).min(100.0)
        } else {
            panic!("No expression")
        };
        score_breakdown.push(("Expression Score".to_string(), expression_score));

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

        // Final Biological Score
        let final_biological_score = OFF_TARGET_WEIGHT * off_target_score +
            SNP_WEIGHT * snp_score +
            PARALOG_WEIGHT * paralog_score +
            EXPRESSION_WEIGHT * expression_score +
            CONSERVATION_WEIGHT * conservation_score +
            SELF_COMP_WEIGHT * self_comp_score +
            PROTEIN_DOMAIN_WEIGHT * protein_domain_score;

        guide.custom_biological_score = Some(final_biological_score);

        // Save the breakdown
        guide.score_breakdown = Some(score_breakdown);

        Ok(())
    }

    /// Check if the guide RNA binds to any paralogous gene sequences.
    fn check_paralog_binding(&self, guide: &GuideRNA, ensembl_api: &APIHandler) -> Result<(f64, Vec<String>), Box<dyn Error>> {
        let mut binding_paralogs = Vec::new();
        let mut paralog_score = 0.0;
        let total_paralogs = self.paralogs.len() as f64;
        if total_paralogs == 0.0 {
            return Ok((0.0, binding_paralogs));
        }

        for paralog_id in &self.paralogs {
            let paralog_sequence = fetch_gene_sequence(paralog_id)?;
            if paralog_sequence.contains(&guide.sequence) {
                paralog_score += 1.0;
                binding_paralogs.push(paralog_id.clone());
            }
        }
        Ok((paralog_score / total_paralogs, binding_paralogs))
    }
}

/// Calculate the final score for a guide RNA by combining various metrics.
pub fn calculate_final_score(guide: &mut GuideRNA) {
    const CHOPCHOP_WEIGHT: f64 = 0.5;
    const BIOLOGICAL_WEIGHT: f64 = 0.5;

    let chopchop_score = guide.chopchop_efficiency;
    let biological_score = guide.custom_biological_score.unwrap_or(0.0);

    let final_score = (chopchop_score * CHOPCHOP_WEIGHT) + (biological_score * BIOLOGICAL_WEIGHT);

    guide.final_score = Some(final_score);

    // Add final score breakdown
    if let Some(breakdown) = &mut guide.score_breakdown {
        breakdown.push(("CHOPCHOP Efficiency".to_string(), chopchop_score));
        breakdown.push(("Final Combined Score".to_string(), final_score));
    }
}


literally keep making the same api calls
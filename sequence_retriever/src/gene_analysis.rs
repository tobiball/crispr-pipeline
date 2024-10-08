// src/gene_info.rs

use std::error::Error;
use crate::api_handler::APIHandler;
use crate::exon_intron::{TranscriptDetail, MissingReason, ExonDetail, fetch_expression_score_at_position, fetch_conservation_score_at_position};
use crate::snps::Variation;
use crate::regulatory_elements::RegulatoryFeature;
use crate::protein_domains::ProteinFeature;
use crate::models::GuideRNA;
use std::cmp::Ordering;
use crate::gene_sequence::fetch_gene_sequence;
use serde_json::Value;

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
    pub exons: Vec<ExonDetail>, // Added exons field
}

impl GeneInfo {
    /// Map exons to the guide RNA positions.
    pub fn map_exons_to_guides(&self, guide: &GuideRNA) -> Vec<ExonDetail> {
        // Find exons that overlap with the guide RNA
        self.exons.iter()
            .filter(|exon| {
                exon.seq_region_name == guide.chromosome &&
                    exon.start <= guide.end &&
                    exon.end >= guide.start
            })
            .cloned()
            .collect()
    }

    /// Score the guide RNA based on various biological factors.
    pub fn score_guide_rna(&self, guide: &mut GuideRNA, ensembl_api: &APIHandler, gtex_api: &APIHandler) -> Result<(), Box<dyn Error>> {
        const BASE_SCORE: f64 = 100.0;
        const SELF_COMPLEMENTARITY_PENALTY: f64 = 2.0;
        const MISMATCH_PENALTY: f64 = 1.0;
        const EXPRESSION_WEIGHT: f64 = 0.2;
        const CONSERVATION_WEIGHT: f64 = 0.2;
        const SNP_PENALTY: f64 = 2.0;
        const REGULATORY_PENALTY: f64 = 1.0;
        const PROTEIN_DOMAIN_PENALTY: f64 = 3.0;
        const PARALOG_PENALTY: f64 = 3.0;

        let mut score = BASE_SCORE;
        let mut score_breakdown = vec![("Base Score".to_string(), BASE_SCORE)];

        // Self-complementarity penalty
        let self_comp_penalty = guide.self_complementarity as f64 * SELF_COMPLEMENTARITY_PENALTY;
        score -= self_comp_penalty;
        score_breakdown.push(("Self-complementarity Penalty".to_string(), -self_comp_penalty));

        // Mismatch penalty
        let mismatch_penalty = (guide.mm1 + 2 * guide.mm2 + 3 * guide.mm3) as f64 * MISMATCH_PENALTY;
        score -= mismatch_penalty;
        score_breakdown.push(("Mismatch Penalty".to_string(), -mismatch_penalty));

        // Map exons to guide RNA
        let overlapping_exons = self.map_exons_to_guides(guide);
        guide.overlapping_exons = overlapping_exons.clone();

        // Expression score
        let expression_score = fetch_expression_score_at_position(
            gtex_api,
            &self.gene_id,
            &guide.chromosome,
            guide.start,
            guide.end,
            ensembl_api, // Added ensembl_api
        )?;

        if let Some(exp) = expression_score {
            let expression_contribution = exp * EXPRESSION_WEIGHT;
            score += expression_contribution;
            guide.expression_score = Some(exp);
            score_breakdown.push(("Exon Expression Score".to_string(), expression_contribution));
        } else {
            println!("No exon expression data available for this region");
        }

        // Conservation score
        let conservation = fetch_conservation_score_at_position(
            ensembl_api,
            &self.chrom,
            guide.start,
            guide.end,
        )?;
        if let Some(cons) = conservation {
            let cons_score = cons * 100.0 * CONSERVATION_WEIGHT;
            score += cons_score;
            guide.conservation_score = Some(cons);
            score_breakdown.push(("Conservation Score".to_string(), cons_score));
        }

        // SNP penalty
        let overlapping_snps = self.snps.iter()
            .filter(|snp| snp.start <= guide.end && snp.end >= guide.start)
            .count();
        let snp_penalty = overlapping_snps as f64 * SNP_PENALTY;
        score -= snp_penalty;
        guide.snp_score = Some(-snp_penalty);
        score_breakdown.push(("SNP Penalty".to_string(), -snp_penalty));

        // Regulatory elements penalty
        let overlapping_regulatory = self.regulatory_elements.iter()
            .filter(|reg| reg.start <= guide.end && reg.end >= guide.start)
            .count();
        let reg_penalty = overlapping_regulatory as f64 * REGULATORY_PENALTY;
        score -= reg_penalty;
        guide.regulatory_score = Some(-reg_penalty);
        score_breakdown.push(("Regulatory Penalty".to_string(), -reg_penalty));

        // Protein domain penalty
        let overlapping_domains = self.protein_features.iter()
            .filter(|domain| domain.start <= guide.end && domain.end >= guide.start && domain.is_essential)
            .count();
        let domain_penalty = overlapping_domains as f64 * PROTEIN_DOMAIN_PENALTY;
        score -= domain_penalty;
        guide.protein_domain_score = Some(-domain_penalty);
        score_breakdown.push(("Protein Domain Penalty".to_string(), -domain_penalty));

        // Paralog penalty
        let paralog_score = self.check_paralog_binding(guide, ensembl_api)?;
        let paralog_penalty = paralog_score * PARALOG_PENALTY;
        score -= paralog_penalty;
        guide.paralog_score = Some(-paralog_penalty);
        score_breakdown.push(("Paralog Penalty".to_string(), -paralog_penalty));

        // Normalize score to be between 0 and 100
        guide.custom_biological_score = Some(score.max(0.0).min(100.0));
        score_breakdown.push(("Final Biological Score".to_string(), guide.custom_biological_score.unwrap()));

        guide.score_breakdown = Some(score_breakdown);

        Ok(())
    }

    /// Check if the guide RNA binds to any paralogous gene sequences.
    fn check_paralog_binding(&self, guide: &GuideRNA, ensembl_api: &APIHandler) -> Result<f64, Box<dyn Error>> {
        let mut paralog_score = 0.0;
        let total_paralogs = self.paralogs.len() as f64;
        if total_paralogs == 0.0 {
            return Ok(0.0);
        }

        for paralog_id in &self.paralogs {
            let paralog_sequence = fetch_gene_sequence(paralog_id)?;
            if paralog_sequence.contains(&guide.sequence) {
                paralog_score += 1.0;
            }
        }
        Ok(paralog_score / total_paralogs)
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

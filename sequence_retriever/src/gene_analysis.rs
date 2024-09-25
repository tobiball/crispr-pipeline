// src/gene_analysis.rs

use crate::exon_intron::TranscriptDetail;
use crate::snps::Variation; // Changed from SNP to Variation
use crate::regulatory_elements::RegulatoryFeature; // Changed from RegulatoryElement to RegulatoryFeature
use crate::protein_domains::ProteinFeature;

pub struct GeneInfo {
    pub gene_id: String,
    pub chrom: String,
    pub strand: i8,
    pub start: u64,
    pub end: u64,
    pub transcripts: Vec<TranscriptDetail>,
    pub protein_features: Vec<ProteinFeature>,
    pub snps: Vec<Variation>,
    pub regulatory_elements: Vec<RegulatoryFeature>,
    pub paralogs: Vec<String>, // Assuming paralog IDs as strings
}

pub struct TargetRegion {
    pub transcript_id: String,
    pub exon_id: String,
    pub start: u64,
    pub end: u64,
    pub priority_score: f64, // Higher score = higher priority
}

impl GeneInfo {
    pub fn prioritize_regions(&self) -> Vec<TargetRegion> {
        let mut regions = Vec::new();

        for transcript in &self.transcripts {
            if let Some(is_canonical) = transcript.is_canonical {
                if is_canonical != 1 {
                    continue; // Prioritize canonical transcripts
                }
            }

            for exon in &transcript.exons {
                // Example prioritization criteria:
                // 1. Early exons
                // 2. Exons without overlapping regulatory elements
                // 3. Exons with low SNP density
                let exon_length = exon.end - exon.start;
                let exon_priority = 100.0 / exon_length as f64; // Shorter exons get higher priority

                // Check for regulatory elements overlap
                let overlaps_regulatory = self.regulatory_elements.iter().any(|elem| {
                    (elem.start <= exon.end) && (elem.end >= exon.start)
                });

                // Calculate SNP density
                let snps_in_exon = self.snps.iter().filter(|snp| {
                    snp.start >= exon.start && snp.start <= exon.end
                }).count();
                let snp_density = snps_in_exon as f64 / exon_length as f64;
                let snp_penalty = snp_density * 10.0; // Arbitrary penalty

                // Final priority score
                let priority_score = if overlaps_regulatory {
                    0.0 // Exclude regions overlapping regulatory elements
                } else {
                    exon_priority - snp_penalty
                };

                regions.push(TargetRegion {
                    transcript_id: transcript.id.clone(),
                    exon_id: exon.id.clone(),
                    start: exon.start,
                    end: exon.end,
                    priority_score,
                });
            }
        }

        // Sort regions by priority_score descending
        regions.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());

        regions
    }
}

// src/gene_analysis.rs

use crate::exon_intron::{ExonDetail, TranscriptDetail, fetch_expression_levels, fetch_conservation_scores, determine_paralogous_exons};
use crate::snps::Variation;
use crate::regulatory_elements::RegulatoryFeature;
use crate::protein_domains::ProteinFeature;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

const BASE_PRIORITY: f64 = 100.0;
const SNP_PENALTY_FACTOR: f64 = 10.0;
const EXPRESSION_WEIGHT: f64 = 2.0;
const DOMAIN_BONUS: f64 = 20.0;
const CONSERVATION_WEIGHT: f64 = 5.0;
const PARALOG_PENALTY: f64 = 15.0;
const SPLICING_PENALTY: f64 = 10.0;
const REGULATORY_OVERLAP_PENALTY: f64 = 20.0;
const MAX_PRIORITY_SCORE: f64 = 200.0;

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
}

pub struct TargetRegion {
    pub transcript_id: String,
    pub exon_id: String,
    pub start: u64,
    pub end: u64,
    pub priority_score: f64,
}

impl GeneInfo {
    /// Prioritize target regions within the gene based on exon properties.
    pub fn prioritize_regions(&mut self) -> Vec<TargetRegion> {
        let mut regions = Vec::new();

        // Fetch additional data for exons
        for transcript in &mut self.transcripts {
            // Fetch expression levels for exons
            if let Err(e) = fetch_expression_levels(&mut transcript.exons, &self.gene_symbol) {
                eprintln!("Error fetching expression levels: {}", e);
            }


            // Fetch conservation scores for exons
            if let Err(e) = fetch_conservation_scores(&mut transcript.exons) {
                eprintln!("Error fetching conservation scores: {}", e);
            }

            // Determine paralogous exons
            if let Err(e) = determine_paralogous_exons(&mut transcript.exons, &self.paralogs) {
                eprintln!("Error determining paralogous exons: {}", e);
            }
        }

        // Sort intervals by start position
        let mut regulatory_elements = &mut self.regulatory_elements;
        regulatory_elements.sort_by_key(|elem| elem.start);

        let mut protein_features = &mut self.protein_features;
        protein_features.sort_by_key(|feat| feat.start);

        let mut snps = &mut self.snps;
        snps.sort_by_key(|snp| snp.start);

        // User-defined exon inclusion/exclusion sets
        let user_exclude_exons: HashSet<String> = HashSet::new();
        let user_include_exons: HashSet<String> = HashSet::new();

        // Exon inclusion frequency across all transcripts
        let mut exon_inclusion_counts: HashMap<String, usize> = HashMap::new();
        let total_transcripts = self.transcripts.len();

        for transcript in &self.transcripts {
            for exon in &transcript.exons {
                *exon_inclusion_counts.entry(exon.id.clone()).or_insert(0) += 1;
            }
        }

        // Process each transcript
        for transcript in &self.transcripts {
            // Prioritize canonical transcripts
            if let Some(is_canonical) = transcript.is_canonical {
                if is_canonical != 1 {
                    continue;
                }
            }

            for exon in &transcript.exons {
                // Apply user-defined exon exclusion/inclusion
                if user_exclude_exons.contains(&exon.id) {
                    continue;
                }
                if !user_include_exons.is_empty() && !user_include_exons.contains(&exon.id) {
                    continue;
                }

                let exon_length = exon.end.saturating_sub(exon.start);
                if exon_length == 0 {
                    continue;
                }

                let mut priority_score = BASE_PRIORITY;

                // Adjust for expression level
                if let Some(expression) = exon.expression_level {
                    priority_score += EXPRESSION_WEIGHT * expression;
                }

                // Check for regulatory elements overlap
                let overlaps_regulatory = check_overlap(&regulatory_elements, exon.start, exon.end);
                if overlaps_regulatory {
                    priority_score -= REGULATORY_OVERLAP_PENALTY;
                }

                // Get SNPs in exon and calculate SNP penalty
                let snps_in_exon = get_overlapping_variations(&snps, exon.start, exon.end);
                let snp_penalty: f64 = snps_in_exon.iter()
                    .map(|snp| {
                        snp.minor_allele_freq.unwrap_or(0.0) * SNP_PENALTY_FACTOR
                    })
                    .sum();
                priority_score -= snp_penalty;

                // Check if exon overlaps essential protein domain
                let overlaps_essential_domain = check_overlap_with_features(
                    &protein_features, exon.start, exon.end, |feature| feature.is_essential
                );
                if overlaps_essential_domain {
                    priority_score += DOMAIN_BONUS;
                }

                // Adjust for conservation score
                if let Some(conservation_score) = exon.conservation_score {
                    priority_score += CONSERVATION_WEIGHT * conservation_score;
                }

                // Determine if exon is constitutive
                let inclusion_count = exon_inclusion_counts.get(&exon.id).cloned().unwrap_or(0);
                let is_constitutive = inclusion_count == total_transcripts;

                // Penalize for alternatively spliced exons
                if !is_constitutive {
                    priority_score -= SPLICING_PENALTY;
                }

                // Penalize for paralogous regions
                if exon.is_paralogous {
                    priority_score -= PARALOG_PENALTY;
                }

                // Adjust for exon length
                let exon_length_score = (exon_length as f64).ln();
                priority_score += exon_length_score;

                // Ensure priority_score is within valid range
                if !priority_score.is_finite() || priority_score < 0.0 {
                    priority_score = 0.0;
                } else if priority_score > MAX_PRIORITY_SCORE {
                    priority_score = MAX_PRIORITY_SCORE;
                }

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
        regions.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(Ordering::Equal)
        });

        regions
    }
}

/// Check if an exon overlaps with any intervals in a sorted list of features
fn check_overlap<T>(features: &[T], exon_start: u64, exon_end: u64) -> bool
where
    T: Interval,
{
    let mut left = 0;
    let mut right = features.len();

    // Binary search to find the first feature that could overlap
    while left < right {
        let mid = (left + right) / 2;
        if features[mid].end() < exon_start {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // Iterate over potential overlaps
    while left < features.len() && features[left].start() <= exon_end {
        if features[left].end() >= exon_start {
            return true; // Overlap found
        }
        left += 1;
    }

    false
}

/// Get variations overlapping with the exon
fn get_overlapping_variations(variations: &[Variation], exon_start: u64, exon_end: u64) -> Vec<&Variation> {
    let mut overlapping_variations = Vec::new();

    let mut left = 0;
    let mut right = variations.len();

    // Binary search to find the first variation that could overlap
    while left < right {
        let mid = (left + right) / 2;
        if variations[mid].start < exon_start {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // Collect overlapping variations
    while left < variations.len() && variations[left].start <= exon_end {
        if variations[left].end >= exon_start {
            overlapping_variations.push(&variations[left]);
        }
        left += 1;
    }

    overlapping_variations
}

/// Check if an exon overlaps with any features satisfying a condition
fn check_overlap_with_features<F>(
    features: &[ProteinFeature],
    exon_start: u64,
    exon_end: u64,
    condition: F,
) -> bool
where
    F: Fn(&ProteinFeature) -> bool,
{
    let mut left = 0;
    let mut right = features.len();

    // Binary search to find the first feature that could overlap
    while left < right {
        let mid = (left + right) / 2;
        if features[mid].end < exon_start {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // Iterate over potential overlaps
    while left < features.len() && features[left].start <= exon_end {
        if features[left].end >= exon_start && condition(&features[left]) {
            return true; // Overlap found with feature satisfying the condition
        }
        left += 1;
    }

    false
}

// Traits to access start and end positions
trait Interval {
    fn start(&self) -> u64;
    fn end(&self) -> u64;
}

// Implement Interval trait for your feature structs
impl Interval for RegulatoryFeature {
    fn start(&self) -> u64 {
        self.start
    }
    fn end(&self) -> u64 {
        self.end
    }
}

impl Interval for ProteinFeature {
    fn start(&self) -> u64 {
        self.start
    }
    fn end(&self) -> u64 {
        self.end
    }
}

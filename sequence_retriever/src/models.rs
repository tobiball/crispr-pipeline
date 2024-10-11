// src/models.rs

use crate::exon_intron::ExonDetail;
use crate::snps::Variation;
use crate::protein_domains::ProteinFeature;

pub struct GuideRNA {
    pub sequence: String,
    pub chromosome: String,
    pub start: u64,
    pub end: u64,
    pub strand: char,
    pub gc_content: f64,
    pub self_complementarity: u32,
    pub mm0: u32,
    pub mm1: u32,
    pub mm2: u32,
    pub mm3: u32,
    pub chopchop_efficiency: f64,
    pub expression_score: Option<f64>,
    pub conservation_score: Option<f64>,
    pub snp_score: Option<f64>,
    pub regulatory_score: Option<f64>,
    pub protein_domain_score: Option<f64>,
    pub paralog_score: Option<f64>,
    pub custom_biological_score: Option<f64>,
    pub score_breakdown: Option<Vec<(String, f64)>>,
    pub final_score: Option<f64>,
    pub overlapping_exons: Vec<ExonDetail>,
    pub overlapping_snps: Vec<Variation>,
    pub overlapping_protein_domains: Vec<ProteinFeature>,
    pub binding_paralogs: Vec<String>,
    pub covered_transcripts: usize,
    pub total_transcripts: usize,
}


pub struct TranscriptImpact {
    pub transcript_id: String,
    pub is_canonical: bool,
    pub exon_overlap: Vec<ExonDetail>,
    pub protein_domain_overlap: Vec<ProteinFeature>,
    pub position_type: PositionType,
    pub transcript_specific_score: Option<f64>,
}

pub enum PositionType {
    Exon,
    Intron,
    UTR5,
    UTR3,
    Upstream,
    Downstream,
}
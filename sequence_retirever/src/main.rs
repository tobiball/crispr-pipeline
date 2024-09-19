use reqwest::blocking::Client;
use serde::Deserialize;
use std::error::Error;

#[derive(Deserialize, Debug)]
struct Xref {
    id: String,
    #[serde(rename = "type")]
    id_type: String,
}

#[derive(Deserialize, Debug)]
struct TranscriptInfo {
    id: String,
    is_canonical: Option<u8>,
}

#[derive(Deserialize, Debug)]
struct Transcript {
    id: String,
    start: u64,
    end: u64,
    strand: i8,
    #[serde(rename = "Exon")]
    exons: Vec<Exon>,
}

#[derive(Deserialize, Debug)]
struct Exon {
    id: String,
    start: u64,
    end: u64,
    strand: i8,
    exon_number: Option<String>,
    // Include other fields if necessary
}

fn fetch_gene_id(gene_symbol: &str) -> Result<String, Box<dyn Error>> {
    println!("Fetching gene ID for gene symbol: {}", gene_symbol);
    let url = format!(
        "https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{}",
        gene_symbol
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .send()?;

    let xrefs: Vec<Xref> = response.json()?;
    for xref in xrefs {
        if xref.id_type == "gene" {
            println!("Found gene ID: {}", xref.id);
            return Ok(xref.id);
        }
    }
    Err("Gene ID not found.".into())
}

fn fetch_transcripts(gene_id: &str) -> Result<Vec<TranscriptInfo>, Box<dyn Error>> {
    println!("Fetching transcripts for gene ID: {}", gene_id);
    let url = format!(
        "https://rest.ensembl.org/overlap/id/{}?feature=transcript",
        gene_id
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .send()?;

    let transcripts: Vec<TranscriptInfo> = response.json()?;
    Ok(transcripts)
}

fn fetch_transcript(transcript_id: &str) -> Result<Transcript, Box<dyn Error>> {
    println!("Fetching transcript information for transcript ID: {}", transcript_id);
    let url = format!(
        "https://rest.ensembl.org/lookup/id/{}?expand=1",
        transcript_id
    );
    let client = Client::new();
    let mut response = client
        .get(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .send()?;

    if response.status().is_success() {
        let transcript: Transcript = response.json()?;
        Ok(transcript)
    } else {
        let status = response.status();
        let error_text = response.text()?;
        eprintln!("Error fetching transcript data: HTTP {}", status);
        eprintln!("Error details: {}", error_text);
        Err("Failed to fetch transcript data.".into())
    }
}


fn cdna_to_genomic(cdna_pos: u64, transcript: &Transcript) -> Option<u64> {
    let mut coding_pos = 0;

    println!("Converting cDNA position to genomic coordinates...");
    for (i, exon) in transcript.exons.iter().enumerate() {
        let exon_length = exon.end - exon.start + 1;

        println!(
            "Checking exon {}: Start {}, End {}, Length {}",
            i + 1,
            exon.start,
            exon.end,
            exon_length
        );

        // Check if the cDNA position falls within this exon
        if coding_pos + exon_length >= cdna_pos {
            let offset = cdna_pos - coding_pos - 1; // Adjust for 1-based indexing
            let genomic_pos = if transcript.strand == 1 {
                exon.start + offset
            } else {
                exon.end - offset
            };
            println!(
                "cDNA position {} falls within exon {}. Genomic coordinate is {}.",
                cdna_pos,
                i + 1,
                genomic_pos
            );
            return Some(genomic_pos);
        }
        coding_pos += exon_length;
    }
    println!("cDNA position {} does not fall within the coding exons.", cdna_pos);
    None
}

fn main() -> Result<(), Box<dyn Error>> {
    // Gene symbol
    let gene_symbol = "OAT";

    // Fetch gene ID
    let gene_id = fetch_gene_id(gene_symbol)?;

    // Fetch transcripts
    let transcripts_info = fetch_transcripts(&gene_id)?;

    // Select canonical transcript or the first one if none is canonical
    let selected_transcript_id = transcripts_info
        .iter()
        .find(|t| t.is_canonical == Some(1))
        .map(|t| t.id.clone())
        .unwrap_or_else(|| {
            println!("No canonical transcript found. Using the first transcript.");
            transcripts_info[0].id.clone()
        });

    println!("Selected transcript ID: {}", selected_transcript_id);

    // Fetch detailed transcript information
    let transcript = fetch_transcript(&selected_transcript_id)?;

    println!("Transcript Strand: {}", transcript.strand);

    // cDNA mutation position
    let cdna_position = 1058;
    println!("Mapping cDNA position: {}", cdna_position);

    // Convert cDNA position to genomic coordinate
    if let Some(genomic_coord) = cdna_to_genomic(cdna_position, &transcript) {
        println!("Genomic coordinate for c.1058G>A is: {}", genomic_coord);
    } else {
        println!("Could not map cDNA position to genomic coordinate.");
    }

    Ok(())
}
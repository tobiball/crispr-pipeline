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
    seq_region_name: String, // This will store the chromosome name
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
}

// Fetch the gene ID using the gene symbol
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

// Fetch transcript information for a gene
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

// Fetch detailed transcript information, including chromosome (seq_region_name)
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

// Convert a cDNA position to genomic coordinates with context range
fn cdna_to_genomic_with_context(cdna_pos: u64, transcript: &Transcript, context: u64) -> Option<(u64, u64, u64)> {
    let mut coding_pos = 0;

    println!("Converting cDNA position to genomic coordinates with context...");
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

            let start = genomic_pos.saturating_sub(context);
            let end = genomic_pos + context;

            println!(
                "cDNA position {} falls within exon {}. Genomic coordinates: {}-{} (with context).",
                cdna_pos,
                i + 1,
                start,
                end
            );
            return Some((start, end, genomic_pos));
        }
        coding_pos += exon_length;
    }
    println!("cDNA position {} does not fall within the coding exons.", cdna_pos);
    None
}

// Fetch the sequence for a specific region using the chromosome name
fn fetch_sequence(chrom: &str, start: u64, end: u64) -> Result<String, Box<dyn Error>> {
    println!("Fetching sequence for {}:{}-{}", chrom, start, end);
    let url = format!(
        "https://rest.ensembl.org/sequence/region/human/{}:{}..{}:1?",
        chrom, start, end
    );
    let client = Client::new();
    let response = client
        .get(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "text/plain")
        .send()?;

    let sequence = response.text()?;
    Ok(sequence)
}

// Function to highlight the mutation within the sequence
fn print_sequence_with_mutation(sequence: &str, mutation_pos: usize, original: char, mutated: char) {
    let mut mutated_sequence = sequence.to_string();
    let chars: Vec<char> = mutated_sequence.chars().collect();

    println!("Original sequence: {}\n", sequence);
    mutated_sequence.replace_range(mutation_pos..mutation_pos + 1, &mutated.to_string());

    println!(
        "Original base at position {}: {} -> Mutated base: {}\n",
        mutation_pos + 1,
        original,
        mutated
    );
    println!("Mutated sequence: {}", mutated_sequence);
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
    println!("Chromosome: {}", transcript.seq_region_name);

    // cDNA mutation position
    let cdna_position = 1058;
    let context = 30; // Set context region around the mutation site
    let original_base = 'G'; // Original base in the reference
    let mutated_base = 'A'; // Mutated base

    println!("Mapping cDNA position: {}", cdna_position);

    // Convert cDNA position to genomic coordinate with context
    if let Some((genomic_start, genomic_end, genomic_mutation_pos)) =
        cdna_to_genomic_with_context(cdna_position, &transcript, context)
    {
        // Fetch the sequence in the genomic region around the mutation
        let sequence = fetch_sequence(&transcript.seq_region_name, genomic_start, genomic_end)?;

        // Find the relative mutation position in the fetched sequence
        let relative_mutation_pos = (genomic_mutation_pos - genomic_start) as usize;

        // Print the sequence with the mutation highlighted
        print_sequence_with_mutation(&sequence, relative_mutation_pos, original_base, mutated_base);
    } else {
        println!("Could not map cDNA position to genomic coordinate.");
    }

    Ok(())
}

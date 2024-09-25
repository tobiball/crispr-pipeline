// src/chopchop_integration.rs

use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::Command;

use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use serde::Deserialize;


/// Represents the result of a CHOPCHOP guide RNA search
#[derive(Debug, Deserialize)]
pub struct ChopchopResult {
    pub guide_sequence: String,
    pub pos: u64,
    pub strand: String,
    pub score: f64,
}

/// Represents options required to run CHOPCHOP
pub struct ChopchopOptions {
    pub python_executable: String, // e.g., "python2"
    pub chopchop_script: String,   // e.g., "chopchop/chopchop.py"
    pub config_file: String,       // e.g., "chopchop/config.json"
    pub genome: String,
    pub target_region: String, // Format: "chrom:start-end"
    pub output_dir: String,
    pub pam_sequence: String,
    pub guide_length: u8,
    pub scoring_method: String,
    pub max_mismatches: u8,
}

/// Downloads the refGene table from UCSC, processes it, and saves it as gene_table.csv
pub fn download_and_format_gene_table() -> Result<(), Box<dyn Error>> {
    let gene_table_path = Path::new("chopchop/hg38.gene_table.csv");
    let url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/refGene.txt.gz";

    println!("Downloading refGene table from UCSC...");
    let client = Client::new();
    let response = client.get(url).send()?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to download refGene table: HTTP {}",
            response.status()
        )
            .into());
    }

    let gz = GzDecoder::new(response);
    let reader = BufReader::new(gz);

    // Create the chopchop directory if it doesn't exist
    if let Some(parent) = gene_table_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Create or overwrite the gene_table.csv file
    let mut gene_table_file = File::create(&gene_table_path)?;

    // Write header
    writeln!(
        gene_table_file,
        "name\tchrom\tstrand\ttxStart\ttxEnd\tcdsStart\tcdsEnd\texonCount\texonStarts\texonEnds\tscore\tname2\tcdsStartStat\tcdsEndStat\texonFrames"
    )?;

    // Iterate over each line in refGene.txt
    for line in reader.lines() {
        let line = line?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 15 {
            continue; // Skip malformed lines
        }

        // Extract necessary fields
        let name = fields[0];
        let chrom = fields[1];
        let strand = fields[2];
        let tx_start = fields[3];
        let tx_end = fields[4];
        let cds_start = fields[5];
        let cds_end = fields[6];
        let exon_count = fields[7];
        let exon_starts = fields[8].trim_end_matches(',');
        let exon_ends = fields[9].trim_end_matches(',');
        let score = fields[10];
        let name2 = fields[12];
        let cds_start_stat = fields[13];
        let cds_end_stat = fields[14];
        let exon_frames = if fields.len() > 15 {
            fields[15..].join(",")
        } else {
            String::new()
        };

        // Write the formatted line to gene_table.csv
        writeln!(
            gene_table_file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            name,
            chrom,
            strand,
            tx_start,
            tx_end,
            cds_start,
            cds_end,
            exon_count,
            exon_starts,
            exon_ends,
            score,
            name2,
            cds_start_stat,
            cds_end_stat,
            exon_frames
        )?;
    }

    println!(
        "Gene table downloaded and formatted successfully at {:?}",
        gene_table_path
    );
    Ok(())
}

/// Ensures that the gene table is up-to-date by downloading and formatting it
pub fn ensure_gene_table_up_to_date() -> Result<(), Box<dyn Error>> {
    // Check if the gene table already exists
    let gene_table_path = Path::new("chopchop/hg38.gene_table.csv");
    if gene_table_path.exists() {
        println!(
            "Gene table already exists at {:?}. Skipping download.",
            gene_table_path
        );
        // Optionally, implement logic to check for updates
    } else {
        // Download and format the gene table
        download_and_format_gene_table()?;
    }
    Ok(())
}

/// Adds a specific gene to the gene table if it's not already present
pub fn add_gene_to_table(gene_info: &str) -> Result<(), Box<dyn Error>> {
    let gene_table_path = Path::new("chopchop/hg38.gene_table.csv");

    // Extract the gene name from the gene_info string (assuming it's in the 'name2' field)
    let gene_name = gene_info
        .split('\t')
        .nth(11)
        .unwrap_or("")
        .to_string();

    // Check if the gene is already in the table
    let gene_exists = {
        let file = File::open(&gene_table_path)?;
        let reader = BufReader::new(file);
        reader
            .lines()
            .any(|line| line.unwrap_or_default().contains(&gene_name))
    };

    if gene_exists {
        println!("Gene '{}' already exists in the gene table.", gene_name);
    } else {
        // Manually add the gene information
        // Ensure that gene_info is correctly formatted as a tab-separated line
        let mut gene_table_file = File::options()
            .append(true)
            .open(&gene_table_path)?;

        writeln!(gene_table_file, "{}", gene_info)?;
        println!("Gene '{}' added to the gene table.", gene_name);
    }

    Ok(())
}

/// Runs CHOPCHOP with the specified options
/// Runs CHOPCHOP with the specified options
pub fn run_chopchop(options: &ChopchopOptions) -> Result<(), Box<dyn Error>> {
    println!("Executing CHOPCHOP for region: {}", options.target_region);

    // Log the full command before running it
    println!(
        "Command: {} {} -G {} -Target {} -o {} -t CODING -M {} -g {} --scoringMethod {} --maxMismatches {} -J -P",
        options.python_executable,
        options.chopchop_script,
        options.genome,
        options.target_region,
        options.output_dir,
        options.pam_sequence,
        options.guide_length,
        options.scoring_method,
        options.max_mismatches
    );

    // Run CHOPCHOP command with the current directory set
    let output = Command::new(&options.python_executable)
        .arg(&options.chopchop_script)
        .arg("-G")
        .arg(&options.genome)
        .arg("-Target")
        .arg(&options.target_region)
        .arg("-o")
        .arg(&options.output_dir)
        .arg("-t")
        .arg("CODING")
        .arg("-M")
        .arg(&options.pam_sequence)
        .arg("-g")
        .arg(options.guide_length.to_string())
        .arg("--scoringMethod")
        .arg(&options.scoring_method)
        .arg("--maxMismatches")
        .arg(options.max_mismatches.to_string())
        .arg("-J") // JSON output
        .arg("-P") // Primer design
        .current_dir("/home/mrcrispr/crispr_pipeline/chopchop") // Set the current directory
        .output()?;  // Capture output and error

    // Log the raw output
    println!("Raw STDOUT:\n{}", String::from_utf8_lossy(&output.stdout));
    println!("Raw STDERR:\n{}", String::from_utf8_lossy(&output.stderr));

    // Check if CHOPCHOP execution was successful
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "CHOPCHOP execution failed.\nSTDOUT:\n{}\nSTDERR:\n{}",
            stdout, stderr
        ).into());
    }

    println!(
        "CHOPCHOP executed successfully for region: {}",
        options.target_region
    );
    Ok(())
}


/// Parses CHOPCHOP JSON results
pub fn parse_chopchop_results(output_dir: &str) -> Result<Vec<ChopchopResult>, Box<dyn Error>> {
    let results_path = Path::new(output_dir).join("results.json"); // Adjust if different
    if !results_path.exists() {
        return Err(format!("CHOPCHOP results file not found at {:?}", results_path).into());
    }

    let file = File::open(&results_path)?;
    let reader = BufReader::new(file);
    let results: Vec<ChopchopResult> = serde_json::from_reader(reader)?;
    Ok(results)
}

use std::env;

pub fn test_two_bit_to_fa() -> Result<(), Box<dyn Error>> {
    let existing_path = env::var("PATH").unwrap_or_default();
    let new_path = format!("/home/mrcrispr/crispr_pipeline/chopchop:{}", existing_path);

    let output = Command::new("twoBitToFa")
        .arg("/home/mrcrispr/crispr_pipeline/chopchop/hg38.2bit:chr1")
        .arg("test_output.fa")
        .env("PATH", new_path)
        .output()?;

    println!("twoBitToFa STDOUT:\n{}", String::from_utf8_lossy(&output.stdout));
    println!("twoBitToFa STDERR:\n{}", String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        return Err("Failed to run twoBitToFa from Rust".into());
    }

    Ok(())
}

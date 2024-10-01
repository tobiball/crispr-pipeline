use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
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
    pub target_region: String,     // Format: "chrom:start-end"
    pub output_dir: String,
    pub pam_sequence: String,
    pub guide_length: u8,
    pub scoring_method: String,
    pub max_mismatches: u8,
}

/// Downloads the refGene table from UCSC, processes it, and saves it in the chopchop directory
pub fn download_and_format_gene_table() -> Result<(), Box<dyn Error>> {
    // Path to save the gene table in the chopchop directory
    let gene_table_path = Path::new("/home/mrcrispr/crispr_pipeline/chopchop/hg38.gene_table.csv");
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
    let mut gene_table_file = File::create(gene_table_path)?;

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
    // Check if the gene table already exists in the chopchop directory
    let gene_table_path = Path::new("/home/mrcrispr/crispr_pipeline/chopchop/hg38.gene_table.csv");
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
        .current_dir("/home/mrcrispr/crispr_pipeline/chopchop") // Ensure execution in chopchop directory
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

/// Parses CHOPCHOP output results (fallback to .offtargets if results.json is missing)
pub fn parse_chopchop_results(output_dir: &str) -> Result<Vec<ChopchopResult>, Box<dyn Error>> {
    let json_results_path = Path::new(output_dir).join("results.json");
    let offtargets_results_path = Path::new(output_dir).join("1.offtargets"); // Example fallback

    // First try to parse the results.json file
    if json_results_path.exists() {
        let file = File::open(&json_results_path)?;
        let reader = BufReader::new(file);
        let results: Vec<ChopchopResult> = serde_json::from_reader(reader)?;
        return Ok(results);
    }

    // If results.json is not found, try parsing the .offtargets file
    if offtargets_results_path.exists() {
        let file = File::open(&offtargets_results_path)?;
        let reader = BufReader::new(file);

        // Assuming .offtargets file is in a structured text format (you may need to adjust this parsing)
        let mut results = Vec::new();
        for line in reader.lines() {
            let line = line?;
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 6 {
                continue; // Skip malformed lines
            }
            let result = ChopchopResult {
                guide_sequence: fields[1].to_string(),
                pos: fields[2].parse()?,
                strand: fields[3].to_string(),
                score: fields[5].parse()?, // Assuming score is in the 6th column
            };
            results.push(result);
        }
        return Ok(results);
    }

    Err(format!("CHOPCHOP results not found in {} or {}.", json_results_path.display(), offtargets_results_path.display()).into())
}

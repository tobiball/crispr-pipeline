// src/chopchop_integration.rs

use std::error::Error;
use std::process::Command;
use std::fs::File;
use std::io::{BufRead, Write};

pub struct GuideRNA {
    pub guide_sequence: String,
    pub chromosome: String,
    pub start: u64,
    pub end: u64,
    pub strand: char,
    pub efficiency_score: f64,
    pub specificity_score: f64,
    // Additional fields as needed
}

pub struct ChopchopOptions {
    pub python_executable: String,
    pub chopchop_script: String,
    pub genome: String,
    pub target_type: String,
    pub target: String,
    pub output_dir: String,
    pub pam_sequence: String,
    pub guide_length: u8,
    pub scoring_method: String,
    pub max_mismatches: u8,
}

pub fn run_chopchop(options: &ChopchopOptions) -> Result<(), Box<dyn Error>> {
    println!("Executing CHOPCHOP for target: {}", options.target);

    // Build the command and set the working directory to the output directory
    let output = Command::new(&options.python_executable)
        .arg(&options.chopchop_script)
        .arg("-G")
        .arg(&options.genome)
        .arg("-Target")
        .arg(&options.target)
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
        .current_dir(&options.output_dir) // Set to output directory
        .output()?; // Collect output

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("CHOPCHOP STDERR: {}", stderr);
        return Err(format!("CHOPCHOP exited with status: {}", output.status).into());
    }

    // Write STDOUT to a file in the output directory
    let stdout = String::from_utf8_lossy(&output.stdout);
    let output_file_path = format!("{}/results.txt", options.output_dir);
    let mut output_file = File::create(&output_file_path)?;
    output_file.write_all(stdout.as_bytes())?;

    println!(
        "CHOPCHOP executed successfully for target: {}. Output saved to {}",
        options.target, output_file_path
    );

    Ok(())
}

pub fn parse_chopchop_results(output_dir: &str) -> Result<Vec<GuideRNA>, Box<dyn Error>> {
    let txt_results_path = format!("{}/results.txt", output_dir);

    if !std::path::Path::new(&txt_results_path).exists() {
        return Err(format!(
            "CHOPCHOP results not found in {}.",
            txt_results_path
        )
            .into());
    }

    let file = File::open(&txt_results_path)?;
    let reader = std::io::BufReader::new(file);

    let mut guides = Vec::new();

    // Skip header line
    let mut lines = reader.lines();
    lines.next();

    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 11 {
            continue; // Skip malformed lines
        }

        let guide = GuideRNA {
            guide_sequence: fields[1].to_string(),
            chromosome: fields[2].split(':').next().unwrap_or("").to_string(),
            start: fields[2]
                .split(':')
                .nth(1)
                .unwrap_or("0")
                .parse::<u64>()
                .unwrap_or(0),
            end: 0, // End position is not provided in the output
            strand: fields[3].chars().next().unwrap_or('+'),
            efficiency_score: fields[10].parse::<f64>().unwrap_or(0.0),
            specificity_score: 0.0, // Specificity score is not provided in this output
        };
        guides.push(guide);
    }
    Ok(guides)
}

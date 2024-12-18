use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, Write};
use csv;
use polars::datatypes::AnyValue;
use polars::prelude::*;
use tracing::{error, debug, info};
use std::process::Command;

#[derive(Debug)]
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

pub fn run_chopchop_meta(df: DataFrame) -> Result<(), Box<dyn std::error::Error>> {
    // Define the CSV output path
    let output_csv_path = "./validator/chopchop_dataset_results.csv";
    debug!("CSV will be written to: {}", output_csv_path);

    // Create a CSV writer
    let mut wtr = csv::Writer::from_path(output_csv_path)?;
    debug!("CSV Writer initialized successfully.");

    // Write the header row
    wtr.write_record(&["chromosome", "start", "end", "position", "guide", "dataset_efficacy", "chopchop_efficiency", "difference"])?;
    debug!("CSV header written.");

    let total_rows = df.height();
    info!("Total number of sgRNAs to process: {}", total_rows);

    let mut counter: f64 = 0.0;
    for i in 0..df.height() {
        debug!("Processing row {}", i);

        // Extract data from DataFrame
        let chromosome = df.column("chromosome")?.get(i)?.to_string().replace('"', "");
        let start = match df.column("start")?.get(i)? {
            AnyValue::Int64(s) => s,
            other => {
                error!("Unexpected type for start at row {}: {:?}", i, other);
                continue;
            }
        };
        let end = match df.column("end")?.get(i)? {
            AnyValue::Int64(e) => e,
            other => {
                error!("Unexpected type for end at row {}: {:?}", i, other);
                continue;
            }
        };
        let position = match df.column("position")?.get(i)? {
            AnyValue::Int64(p) => p,
            other => {
                error!("Expected Int64 value for position at row {}, found {:?}", i, other);
                continue;
            }
        };
        let raw_guide = df.column("sgRNA")?.get(i)?.to_string();
        let guide = raw_guide.trim_matches('"');
        let efficacy: f64 = match df.column("Efficacy")?.get(i)? {
            AnyValue::Float64(eff) => eff,
            other => {
                error!("Unexpected type for Efficacy at row {}: {:?}", i, other);
                continue;
            }
        };
        let efficacy_scaled = efficacy * 100.0;

        debug!(
            "Row {}: chromosome={}, start={}, end={}, position={}, sgRNA={}, efficacy_scaled={}",
            i, chromosome, start, end, position, guide, efficacy_scaled
        );

        // Build the target region in the format "chr:start-end"
        let target_region = format!("{}:{}-{}", chromosome, start, end);

        // Define the output directory for this target
        let output_dir = format!("./validator/output/{}/{}", chromosome, i);

        // Ensure the output directory exists
        if let Err(e) = fs::create_dir_all(&output_dir) {
            error!("Failed to create directory {}: {}", output_dir, e);
            continue;
        }

        // Set up CHOPCHOP options
        let chopchop_options = ChopchopOptions {
            python_executable: "./chopchop/chopchop_env/bin/python2.7".to_string(),
            chopchop_script: "./chopchop/chopchop.py".to_string(),
            genome: "hg38".to_string(),
            target_type: "REGION".to_string(),
            target: target_region.clone(),
            output_dir: output_dir.clone(),
            pam_sequence: "NGG".to_string(),
            guide_length: 20,
            scoring_method: "DOENCH_2016".to_string(),
            max_mismatches: 3,
        };

        debug!("Running CHOPCHOP with options: {:?}", chopchop_options);

        if let Err(e) = run_chopchop(&chopchop_options) {
            error!("Error running CHOPCHOP for target {}: {}", target_region, e);
            continue; // Continue with the next iteration
        }

        let guides = match parse_chopchop_results(&output_dir) {
            Ok(guides) => guides,
            Err(e) => {
                error!("Failed to parse CHOPCHOP results in {}: {}", output_dir, e);
                continue;
            }
        };

        debug!("Parsed {} GuideRNA entries from CHOPCHOP results for row {}", guides.len(), i);

        let mut matched = false;
        for g in &guides {
            if g.sequence.len() < 3 {
                debug!("Guide sequence too short for PAM removal: {}", g.sequence);
                continue;
            }
            let guide_seq = &g.sequence[..g.sequence.len()-3]; // Remove PAM

            if guide_seq == guide {
                debug!("Match found for guide: {}", guide_seq);
                let distance = position - g.start as i64;
                debug!("Matching guide found: {} with distance {}", guide_seq, distance);

                if (distance != 16 && g.strand == '+') || (distance != 5 && g.strand == '-') {
                    error!(
                        "Distance mismatch for guide {}: expected {}, got {}",
                        guide,
                        if g.strand == '+' { 16 } else { 5 },
                        distance
                    );
                    // Instead of panic!, continue processing other guides
                    continue;
                }

                debug!("Tool Efficiency: {}", g.efficiency);
                debug!("Dataset Efficacy: {}", efficacy_scaled);
                debug!("Difference (Tool vs Dataset): {}", g.efficiency - efficacy_scaled);

                // Write to CSV
                wtr.write_record(&[
                    chromosome.clone(),
                    start.to_string(),
                    end.to_string(),
                    position.to_string(),
                    g.sequence.to_string(),
                    efficacy_scaled.to_string(),
                    g.efficiency.to_string(),
                    (g.efficiency - efficacy_scaled).to_string(),
                ])?;
                wtr.flush()?;
                debug!("Record written to CSV for guide: {}", guide_seq);

                matched = true;
                break; // Assuming one match per sgRNA is sufficient
            }
        }

        if !matched {
            error!("No matching guide found for sgRNA: {}", guide);
        }

        counter += 1.0;
        debug!("Count {}", counter);
        if counter % 50.0 == 0.0 {
            info!("Progress {:.2}%", 100.0 * counter / total_rows as f64);
        }
    }

    // Flush the CSV writer to ensure all data is written to disk
    wtr.flush()?;
    debug!("CSV Writer flushed successfully.");

    Ok(())
}


pub fn run_chopchop(options: &ChopchopOptions) -> Result<(), Box<dyn Error>> {
    debug!("Executing CHOPCHOP for target: {}", options.target);

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
        error!("CHOPCHOP STDERR: {}", stderr);
        return Err(format!("CHOPCHOP exited with status: {}", output.status).into());
    }

    // Write STDOUT to a file in the output directory
    let stdout = String::from_utf8_lossy(&output.stdout);
    let output_file_path = format!("{}/results.txt", options.output_dir);
    let mut output_file = File::create(&output_file_path)?;
    output_file.write_all(stdout.as_bytes())?;

    debug!(
        "CHOPCHOP executed successfully for target: {}. Output saved to {}",
        options.target, output_file_path
    );

    Ok(())
}

#[derive(Debug)]
pub struct ChopchopGuide {
    pub(crate) sequence: String,
    pub(crate) chromosome: String,
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) strand: char,
    gc_content: f64,
    self_complementarity: u32,
    mm0: u32,
    mm1: u32,
    mm2: u32,
    mm3: u32,
    pub efficiency: f64
}

pub fn parse_chopchop_results(output_dir: &str) -> Result<Vec<ChopchopGuide>, Box<dyn Error>> {
    let txt_results_path = format!("{}/results.txt", output_dir);

    if !std::path::Path::new(&txt_results_path).exists() {
        error!("CHOPCHOP results not found in {}", txt_results_path);
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
        let line = line?; // Handle the Result here
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 11 {
            debug!("Skipping malformed line: {}", line);
            continue; // Skip malformed lines
        }

        let guide = ChopchopGuide {
            sequence: fields[1].to_string(),
            chromosome: fields[2].split(':').next().unwrap_or("").to_string(),
            start: fields[2].split(':').nth(1).unwrap_or("0").parse::<u64>().unwrap_or(0),
            end: fields[2].split(':').nth(1).unwrap_or("0").parse::<u64>().unwrap_or(0) + fields[1].len() as u64,
            strand: fields[3].chars().next().unwrap_or('+'),
            gc_content: fields[4].parse::<f64>().unwrap_or(0.0),
            self_complementarity: fields[5].parse::<u32>().unwrap_or(0),
            mm0: fields[6].parse::<u32>().unwrap_or(0),
            mm1: fields[7].parse::<u32>().unwrap_or(0),
            mm2: fields[8].parse::<u32>().unwrap_or(0),
            mm3: fields[9].parse::<u32>().unwrap_or(0),
            efficiency: fields[10].parse::<f64>().unwrap(),
        };
        debug!("Parsed GuideRNA: {:?}", guide);
        guides.push(guide);
    }

    debug!("Parsed {} GuideRNA entries from CHOPCHOP results", guides.len());

    Ok(guides)
}

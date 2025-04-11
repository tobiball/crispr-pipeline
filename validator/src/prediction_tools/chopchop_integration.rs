use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, Write};
use csv;
use polars::datatypes::AnyValue;
use polars::prelude::*;
use tracing::{error, debug, info};
use std::process::Command;
use std::env;
use crate::helper_functions::*;
use crate::models::polars_err;

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

pub fn run_chopchop_meta(df: DataFrame, database_name: &str) -> PolarsResult<DataFrame> {
    // Create output directory
    let output_dir = "./processed_data";
    debug!("Creating directory: {:?}", output_dir);
    std::fs::create_dir_all(&output_dir)?;

    // Create a mutable copy of the dataframe
    let mut result_df = df.clone();

    // Create a new Series with null values for chopchop_efficiency
    let mut chopchop_eff_vec: Vec<Option<f64>> = vec![None; result_df.height()];

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

        let raw_guide = df.column("sequence_with_pam")?.get(i)?.to_string();
        let guide = raw_guide.trim_matches('"');

        debug!(
            "Row {}: chromosome={}, start={}, end={}, sgRNA={}",
            i, chromosome, start, end, guide
        );

        // Build the target region in the format "chr:start-end"
        let target_region = format!("{}:{}-{}", chromosome, start, end);

        let base_output_dir = "./validator/output";

        // Ensure base directory exists
        if let Err(e) = fs::create_dir_all(base_output_dir) {
            error!("Failed to create base output directory {}: {}", base_output_dir, e);
            return Err(PolarsError::ComputeError(format!("{}", e).into()));
        }
        debug!("Base output directory ensured: {}", base_output_dir);

        let path = format!("{}/{}/{}", base_output_dir, chromosome, i);

        // Create the directory if it doesn't exist
        fs::create_dir_all(&path).map_err(|e| {
            error!("Failed to create output directory {}: {}", &path, e);
            e
        })?;

        // Now canonicalize
        let output_dir = fs::canonicalize(&path)?;
        debug!("Output directory created: {}", output_dir.display());

        let current_dir = env::current_dir()?;

        // Define Python script paths dynamically
        let chopchop_options = ChopchopOptions {
            python_executable: project_root()
                .join("chopchop/chopchop_env/bin/python2.7")
                .to_str()
                .unwrap()
                .to_string(),
            chopchop_script: project_root()
                .join("chopchop/chopchop.py")
                .to_str()
                .unwrap()
                .to_string(),
            genome: "hg38".to_string(),
            target_type: "REGION".to_string(),
            target: target_region.clone(),
            output_dir: current_dir.join(output_dir.clone()).to_str().unwrap().to_string(),
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

        let guides = match parse_chopchop_results(&output_dir.to_str().unwrap_or("output set wrong")) {
            Ok(guides) => guides,
            Err(e) => {
                error!("Failed to parse CHOPCHOP results in {}: {}", output_dir.display(), e);
                continue;
            }
        };

        debug!("Parsed {} GuideRNA entries from CHOPCHOP results for row {}", guides.len(), i);

        let mut matched = false;
        for g in &guides {
            // Some CHOPCHOP outputs may append the PAM, so we need to handle that
            let choppchop_guide = &g.sequence;

            if choppchop_guide == guide {
                debug!("Match found for guide: {}", choppchop_guide);
                debug!("Tool Efficiency: {}", g.efficiency);

                // Update the chopchop_efficiency value for this row in our vector
                chopchop_eff_vec[i] = Some(g.efficiency);

                matched = true;
                break; // Assuming one match per sgRNA is sufficient
            }
        }

        if !matched {
            debug!("No matching guide found for sgRNA: {}", guide);
            // chopchop_eff_vec[i] remains None
        }


        counter += 1.0;
        if counter % 50.0 == 0.0 {
            info!("Progress {:.2}%", 100.0 * counter / total_rows as f64);
        }
    }


    // Add the chopchop_efficiency column to the dataframe
    // Convert chopchop_efficiency to explicit nullable Float64
    let chopchop_eff_series = Series::new(
        PlSmallStr::from("chopchop_efficiency"),
        Float64Chunked::from_vec(PlSmallStr::from(""), chopchop_eff_vec.iter()
            .map(|opt| opt.unwrap_or(f64::NAN))
            .collect::<Vec<f64>>()
        )
    ).cast(&DataType::Float64).unwrap();


    let updated_df = result_df.with_column(chopchop_eff_series)?.clone();
    result_df = updated_df;

    debug!("Updated DF: {}", result_df);

    let efficacy_series = df.column("efficacy")?.clone();
    let chopchop_series = result_df.column("chopchop_efficiency")?.clone();


    let output_csv_path = format!("{}/chopchop_{}.csv", output_dir, database_name);
    debug!("Saving results to CSV: {}", output_csv_path);


    // Debug column types before saving
    for col_name in result_df.get_column_names() {
        debug!("Column '{}' is of type: {:?}", col_name, result_df.column(col_name).unwrap().dtype());
    }

    // Before saving to CSV, drop the split_parts column
    let mut df_for_csv = result_df.clone();
    if df_for_csv.get_column_names().contains(&&PlSmallStr::from("split_parts")) {
        df_for_csv = df_for_csv.drop("split_parts")?;
    }

    // Then write to CSV
    let file = File::create(&output_csv_path)?;
    CsvWriter::new(file).finish(&mut df_for_csv)?;
    Ok(result_df)
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

    debug!("Diag test exited with status: {:?}", output.status);
    debug!("Diag test stdout: {}", String::from_utf8_lossy(&output.stdout));
    debug!("Diag test stderr: {}", String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("CHOPCHOP STDERR: {}", stderr);
        return Err(format!("CHOPCHOP exited with status: {}", output.status).into());
    }

    // Write STDOUT to a file in the output directory
    let stdout = String::from_utf8_lossy(&output.stdout);
    let output_file_path = format!("{}/gc_results.txt", options.output_dir);
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
    let txt_results_path = format!("{}/gc_results.txt", output_dir);

    if !std::path::Path::new(&txt_results_path).exists() {
        error!("CHOPCHOP results not found in {}", txt_results_path);
        return Err(format!("CHOPCHOP results not found in {}.", txt_results_path).into());
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
            start: fields[2].split(':').nth(1).unwrap_or("0").parse::<u64>().unwrap(),
            end: fields[2].split(':').nth(1).unwrap_or("0").parse::<u64>().unwrap()
                + fields[1].len() as u64,
            strand: fields[3].chars().next().unwrap_or('+'),
            gc_content: fields[4].parse::<f64>().unwrap(),
            self_complementarity: fields[5].parse::<u32>().unwrap(),
            mm0: fields[6].parse::<u32>().unwrap(),
            mm1: fields[7].parse::<u32>().unwrap(),
            mm2: fields[8].parse::<u32>().unwrap(),
            mm3: fields[9].parse::<u32>().unwrap_or(999),
            efficiency: fields[10].parse::<f64>().unwrap(),
        };
        debug!("Parsed GuideRNA: {:?}", guide);
        guides.push(guide);
    }

    debug!("Parsed {} GuideRNA entries from CHOPCHOP results", guides.len());

    Ok(guides)
}

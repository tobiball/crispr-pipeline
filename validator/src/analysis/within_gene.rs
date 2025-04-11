use std::collections::HashMap;
use std::collections::HashSet;
use log::debug;
use polars::error::constants::FALSE;
use polars::frame::DataFrame;
use polars::prelude::{ChunkCompareEq, PolarsResult};

use polars::prelude::*;

pub fn within_gene_analysis(mut df: DataFrame, tools_to_compare: Vec<&str>) -> PolarsResult<DataFrame> {
    // Add a binary column "guide_quality" indicating good or bad guides
    let efficacy = df.column("efficacy")?.f64()?;
    let binding = df.column("Gene")?.clone();
    let genes = binding.str()?;
    let mut genes_with_shitty_guides = Vec::new();
    let mut good_guide = Vec::new();

    // Iterate through all three in parallel
    for i in 0..df.height() {
        if let (Some(score), Some(gene)) = (efficacy.get(i), genes.get(i)) {
            if score < 50.0 {
                genes_with_shitty_guides.push(gene);
                good_guide.push(Some(false));
                println!("Gene: {}, Efficacy: {}", gene, score);
            } else if score > 75.0 { good_guide.push(Some(true)); } else { good_guide.push(None)}
        }}
    let guide_quality_series = Series::new(PlSmallStr::from("guide_quality"), good_guide);

    // Add it to your DataFrame
    df.with_column(guide_quality_series)?;
    let bad_guides_genes = Series::new(PlSmallStr::from("Gene"), genes_with_shitty_guides);
    let df_bad_guides_genes = DataFrame::new(vec![Column::from(bad_guides_genes)])?;
    let df_filtered = df_bad_guides_genes.left_join(&df, ["Gene"], ["Gene"])?;

    let gene_series = df_filtered.column("Gene")?;

    debug!("{:?}",df_filtered);



    Ok(df_filtered)
}


use anyhow::Result;
use polars::prelude::*;

/// Holds FPR/TPR pairs, thresholds, and the final AUC for one ROC curve.
#[derive(Debug)]
pub struct RocResult {
    pub fprs: Vec<f64>,
    pub tprs: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub auc: f64,
}

/// Compute ROC curve points (FPR/TPR) and AUC for a set of predicted `scores` and boolean `labels`.
///
/// The approach here is:
/// 1) Pair each (score, label).
/// 2) Sort descending by score.
/// 3) Sweep thresholds from high to low and track (FPR, TPR).
/// 4) Use trapezoidal rule to integrate area-under-curve.
pub fn compute_roc(scores: &[f64], labels: &[bool]) -> RocResult {
    // Pair up scores + labels and sort by score descending.
    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(s, l)| (*s, *l))
        .collect();
    pairs.sort_by(|(s1, _), (s2, _)| s2.partial_cmp(s1).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = labels.iter().filter(|&&l| l).count() as f64;
    let total_neg = labels.len() as f64 - total_pos;

    let mut tprs = Vec::new();
    let mut fprs = Vec::new();
    let mut thresholds = Vec::new();

    // Counters for true positives and false positives as we move down the sorted list.
    let mut tp = 0.0;
    let mut fp = 0.0;

    // For trapezoidal integration, we need to track the “previous” TPR, FPR as we move.
    let mut prev_tpr = 0.0;
    let mut prev_fpr = 0.0;
    let mut prev_score = f64::NAN;

    let mut auc = 0.0;

    for (idx, (score, label)) in pairs.iter().enumerate() {
        // If we’ve reached a new threshold (score), record the (FPR, TPR) for the old threshold
        // and add to the running AUC using the trapezoidal rule.
        if !score.eq(&prev_score) {
            let tpr = tp / total_pos;
            let fpr = fp / total_neg;

            // Integrate area between (prev_fpr, prev_tpr) and (fpr, tpr).
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;

            tprs.push(tpr);
            fprs.push(fpr);
            thresholds.push(*score);

            prev_tpr = tpr;
            prev_fpr = fpr;
            prev_score = *score;
        }

        // Update TP/FP counts for the current row
        if *label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        // If this is the last row, we’ll handle final update after the loop
        if idx == pairs.len() - 1 {
            let tpr = tp / total_pos;
            let fpr = fp / total_neg;
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) * 0.5;

            tprs.push(tpr);
            fprs.push(fpr);
            thresholds.push(f64::NEG_INFINITY); // or some sentinel
        }
    }

    RocResult {
        fprs,
        tprs,
        thresholds,
        auc,
    }
}

/// Evaluate multiple scoring columns against a boolean “guide_quality” in a Polars `DataFrame`.
///
/// Returns a vector of `(tool_name, RocResult)`. You can then inspect `RocResult.auc` etc.
pub fn evaluate_scores(df: &DataFrame, tools_to_compare: &[&str]) -> PolarsResult<Vec<(String, RocResult)>> {
    // Extract the boolean column “guide_quality” from DataFrame.
    let guide_col = df.column("guide_quality")?.bool()?;
    let mut valid_indices = Vec::new();
    let mut guide_quality = Vec::new();

    for (idx, val) in guide_col.into_iter().enumerate() {
        if let Some(b) = val {
            guide_quality.push(b);
            valid_indices.push(idx);
        }
    }


    let mut results = Vec::with_capacity(tools_to_compare.len());

    for &tool_name in tools_to_compare {
        // Get the series for this tool’s score
        let series = df.column(tool_name)?;
        // Convert to a Vec<f64> (assuming no nulls)
        let mut scores = Vec::new();
        let series = df.column(tool_name)?.f64()?;

        for &idx in &valid_indices {
            if let Some(score) = series.get(idx) {
                scores.push(score);
            } else {
                // skip if tool score is missing
                continue;
            }
        }


        let roc_result = compute_roc(&scores, &guide_quality);
        results.push((tool_name.to_string(), roc_result));
    }

    Ok(results)
}



use plotters::prelude::*;
use crate::models::polars_err;

pub fn plot_tool_vs_efficacy(
    df: &DataFrame,
    tool_name: &str,
    output_path: String,
) -> PolarsResult<()> {
    // Convert the relevant columns to standard Rust vectors (no nulls).
    let gene_col = df.column("Gene").map_err(|e| polars_err(Box::new(e)))?.str().map_err(|e| polars_err(Box::new(e)))?;
    let guide_quality_col = df.column("guide_quality").map_err(|e| polars_err(Box::new(e)))?.bool().map_err(|e| polars_err(Box::new(e)))?;
    let efficacy_col = df.column("efficacy").map_err(|e| polars_err(Box::new(e)))?.f64().map_err(|e| polars_err(Box::new(e)))?;
    let tool_col = df.column(tool_name).map_err(|e| polars_err(Box::new(e)))?.f64().map_err(|e| polars_err(Box::new(e)))?;

    let gene_vals: Vec<&str> = gene_col.into_no_null_iter().collect();
    let mut data_points = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        if let (Some(gene), Some(gq), Some(eff), Some(score)) = (
            gene_col.get(i),
            guide_quality_col.get(i),
            efficacy_col.get(i),
            tool_col.get(i),
        ) {
            data_points.push((gene, gq, eff, score));
        }
    }
    let efficacy_vals: Vec<f64> = efficacy_col.into_no_null_iter().collect();
    let tool_vals: Vec<f64> = tool_col.into_no_null_iter().collect();

    // Combine them into one structure for convenience
    let mut data_points = Vec::with_capacity(df.height());
    let mut rows = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        let gene_opt      = gene_col.get(i);
        let gq_opt        = guide_quality_col.get(i);
        let score_opt     = efficacy_col.get(i);
        let efficacy_opt  = tool_col.get(i);

        if let (Some(gene), Some(guide_good), Some(score), Some(efficacy)) =
            (gene_opt, gq_opt, score_opt, efficacy_opt)
        {
            rows.push((gene, guide_good, score, efficacy));
        }
    }


    // Determine min/max for x & y to set chart ranges
    let x_min = data_points.iter().map(|(_, _, _, x)| *x).fold(f64::INFINITY, f64::min);
    let x_max = data_points.iter().map(|(_, _, _, x)| *x).fold(f64::NEG_INFINITY, f64::max);
    let y_min = data_points.iter().map(|(_, _, y, _)| *y).fold(f64::INFINITY, f64::min);
    let y_max = data_points.iter().map(|(_, _, y, _)| *y).fold(f64::NEG_INFINITY, f64::max);

    // Create a color palette. If you have more genes than colors, you can cycle or expand this.
    let palette = vec![
        RED, BLUE, GREEN, MAGENTA, CYAN, BLACK,
        RGBColor(255, 165, 0), // orange
        RGBColor(128, 0, 128), // purple
        RGBColor(128, 128, 0), // olive, etc.
    ];

    // Extract unique genes to assign each a color
    let mut unique_genes: Vec<&str> = data_points.iter().map(|(g, _, _, _)| *g).collect();
    unique_genes.sort();
    unique_genes.dedup();

    // Map each gene to a color in the palette (reusing if > palette size)
    let mut gene_color_map = std::collections::HashMap::new();
    for (i, gene) in unique_genes.iter().enumerate() {
        let color = palette[i % palette.len()];
        gene_color_map.insert(*gene, color);
    }

    // Set up the drawing area
    let root = BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| polars_err(Box::new(e)))?;

    // Build chart
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("{} vs. efficacy", tool_name), ("sans-serif", 20))
        .margin(20)
        .x_label_area_size(200)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, y_min..y_max).map_err(|e| polars_err(Box::new(e)))?;

    // Configure mesh and labels
    chart
        .configure_mesh()
        .x_desc(tool_name)
        .y_desc("efficacy")
        .draw().map_err(|e| polars_err(Box::new(e)))?;

    // Plot each data point
    //   - color depends on gene
    //   - shape depends on guide_quality
    for (gene, guide_q, eff, score) in &data_points {
        let color = *gene_color_map.get(gene).unwrap_or(&BLACK);

        // If guide_quality == true => use a circle, else use a cross
        if *guide_q {
            chart.draw_series(std::iter::once(Circle::new(
                (*score, *eff),
                4,   // size
                color.filled(),
            ))).map_err(|e| polars_err(Box::new(e)))?;
        } else {
            chart.draw_series(std::iter::once(Cross::new(
                (*score, *eff),
                5,   // size
                color.stroke_width(2),
            ))).map_err(|e| polars_err(Box::new(e)))?;
        }
    }

    // Optionally draw a legend (by listing each gene -> color).
    // This is a bit more involved in Plotters, but here's a simple example:

    let mut legend_y = y_max; // top of chart
    let legend_step = (y_max - y_min) * 0.05;

    for gene in &unique_genes {
        legend_y -= legend_step;
        // draw a small circle next to text as the legend marker
        chart.draw_series(std::iter::once(Circle::new(
            (x_min, legend_y),
            4,
            gene_color_map[gene].filled(),
        ))).map_err(|e| polars_err(Box::new(e)))?;

        // Add text
        chart.draw_series(std::iter::once(Text::new(
            (*gene).to_string(),
            (x_min + (x_max - x_min)*0.02, legend_y), // a bit to the right
            ("sans-serif", 16).into_font(),
        ))).map_err(|e| polars_err(Box::new(e)))?;
    }

    // If you also want a marker for the "good" vs. "bad" shape in the legend,
    // you might do a second pass or a separate area for that.
    // For brevity, we won't show it here.

    Ok(())
}

use polars::prelude::*;
use plotters::prelude::*;
use rand::Rng;

// If you want to return a PolarsError, you can do something like this:
/// Draws a "strip plot" of a single tool's scores across genes:
///   - X-axis = the tool score (numeric)
///   - Y-axis = one horizontal strip per gene (categorical)
///   - Point color = gene (cycled through limited palette)
///   - Point shape = guide_quality (circle for "good", cross for "bad")
///   - Small random jitter in Y so points in the same gene don’t overlap.
/// Saves to `output_path` as a PNG.
use polars::prelude::*;
use plotters::prelude::*;


// Suppose you already have a DataFrame with columns:
//  "Gene": &str
//  "guide_quality": bool
//  tool columns (f64), e.g. "Doench '16-Score"

use polars::prelude::*;
use plotters::prelude::*;

use polars::prelude::*;
use plotters::prelude::*;

use std::error::Error;
use std::io;
use polars::prelude::*;
use plotters::prelude::*;


pub fn plot_stripplot_for_tool(
    df: &DataFrame,
    tool_name: &str,
    output_path: String,
) -> PolarsResult<()> {
    // 1) Fill nulls in the chosen tool column (or all columns)
    let df = df
        .filter(&df.column("guide_quality")?.is_not_null())?
        .clone();

    // 2) Extract columns
    let gene_series = df
        .column("Gene")
        .map_err(|e| polars_err(Box::new(e)))?
        .str()
        .map_err(|e| polars_err(Box::new(e)))?;

    let guide_quality_series = df
        .column("guide_quality")
        .map_err(|e| polars_err(Box::new(e)))?
        .bool()
        .map_err(|e| polars_err(Box::new(e)))?;

    let tool_score_series = df
        .column(tool_name)
        .map_err(|e| polars_err(Box::new(e)))?
        .f64()
        .map_err(|e| polars_err(Box::new(e)))?;

    let efficacy_series = df
        .column("efficacy")
        .map_err(|e| polars_err(Box::new(e)))?
        .f64()
        .map_err(|e| polars_err(Box::new(e)))?;

    // 3) Gather rows into a Vec
    let mut rows = Vec::with_capacity(df.height());
    let mut skipped_rows = 0;

    for i in 0..df.height() {
        match (
            gene_series.get(i),
            guide_quality_series.get(i),
            tool_score_series.get(i),
            efficacy_series.get(i),
        ) {
            (Some(gene), Some(guide_good), Some(score), Some(efficacy)) => {
                rows.push((gene, guide_good, score, efficacy));
            }
            _ => {
                skipped_rows += 1;
                continue;
            }
        }
    }

    if skipped_rows > 0 {
        log::warn!(
        "Skipped {} rows due to nulls in 'Gene', 'guide_quality', '{tool_name}' or 'efficacy'",
        skipped_rows
    );
    }

    // 4) Collect unique genes (for the X axis)
    let mut unique_genes: Vec<&str> = rows.iter().map(|(g, _, _, _)| *g).collect();
    unique_genes.sort();
    unique_genes.dedup();

    // Map gene -> integer X index
    let gene_to_idx: HashMap<_, _> = unique_genes
        .iter()
        .enumerate()
        .map(|(idx, &g)| (g, idx as f64))
        .collect();

    // 5) For each gene, find the lowest tool score among all its guides
    let mut gene_to_lowest_score: HashMap<&str, f64> = HashMap::new();
    for &gene in &unique_genes {
        let lowest_score = rows
            .iter()
            .filter(|(g, _, _, _)| *g == gene)
            .map(|(_, _, score, _)| *score)
            .fold(f64::INFINITY, f64::min);

        gene_to_lowest_score.insert(gene, lowest_score);
    }

    // 6) Find “problem” bad guides: not the lowest scoring in their gene
    let mut problem_cases = Vec::new();
    for (gene, guide_good, score, efficacy) in &rows {
        if !guide_good {
            let lowest = *gene_to_lowest_score.get(gene).unwrap();
            if (score - lowest).abs() > f64::EPSILON {
                problem_cases.push((*gene, *guide_good, *score, *efficacy));
            }
        }
    }

    // 7) Fraction of *bad* guides that fail to rank as poorest
    let total_bad = rows.iter().filter(|(_, good, _, _)| !*good).count();
    let problem_count = problem_cases.len();
    let fraction = if total_bad > 0 {
        problem_count as f64 / total_bad as f64
    } else {
        0.0
    };

    println!(
        "Fraction of bad guides not lowest-scoring: {:.2}%",
        fraction * 100.0
    );

    // 8) Y-axis range = min..max of the tool scores
    let score_min = rows.iter().map(|(_, _, s, _)| *s).fold(f64::INFINITY, f64::min);
    let score_max = rows.iter().map(|(_, _, s, _)| *s).fold(f64::NEG_INFINITY, f64::max);

    // Expand a bit so points aren't on the border
    fn expand_range(min_val: f64, max_val: f64, pct: f64) -> (f64, f64) {
        if (max_val - min_val).abs() < 1e-9 {
            return (min_val - 1.0, max_val + 1.0);
        }
        let range = max_val - min_val;
        let pad = range * pct;
        (min_val - pad, max_val + pad)
    }

    let (y_lo, y_hi) = expand_range(score_min, score_max, 0.05);
    let n_genes = unique_genes.len() as f64;
    let (x_lo, x_hi) = (-0.5, n_genes - 0.5);

    // 9) Create drawing area
    let root = BitMapBackend::new(&output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| polars_err(Box::new(e)))?;

    // 10) Chart caption
    let caption_text = format!(
        "Within-gene Comparison for Tool {} - Failed rank fraction: {:.1}%",
        tool_name,
        fraction * 100.0
    );

    // 11) Build chart
    let mut chart = ChartBuilder::on(&root)
        .caption(caption_text, ("sans-serif", 22))
        .margin(10)
        .x_label_area_size(100)
        .y_label_area_size(70)
        .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)
        .map_err(|e| polars_err(Box::new(e)))?;

    // 12) Configure the mesh with rotated gene labels
    let x_label_style = TextStyle::from(("sans-serif", 14))
        .transform(FontTransform::Rotate270);

    // 9) Configure the mesh: rotate the x-axis labels and skip jitter if you prefer
    let x_label_style = TextStyle::from(("sans-serif", 14))
        .transform(FontTransform::Rotate270);


    chart
        .configure_mesh()
        .disable_mesh()  // no grid lines
        .x_labels(unique_genes.len())
        .x_label_style((x_label_style))
        .x_desc("Gene")
        .axis_desc_style(("sans-serif", 16))
        .x_label_formatter(&|val: &f64| {
            let idx = val.round() as usize;
            if idx < unique_genes.len() {
                unique_genes[idx].to_string()
            } else {
                "".into()
            }
        })
        .y_desc(tool_name)
        // Remove or comment out this line so the x-axis and labels are visible:
        // .disable_x_axis()
        .draw()
        .map_err(|e| polars_err(Box::new(e)))?;


    // 13) Define a color palette
    let palette = vec![
        RGBColor(0, 0, 0),
        RGBColor(230, 159, 0),
        RGBColor(86, 180, 233),
        RGBColor(0, 158, 115),
        RGBColor(240, 228, 66),
        RGBColor(0, 114, 178),
        RGBColor(213, 94, 0),
        RGBColor(204, 121, 167),
    ];

    // Highlight color for problem cases
    let highlight_color = RGBColor(255, 0, 0); // bright red

    // 14) Plot normal points (good guides & "true" lowest bad)
    for (gene, guide_good, score, efficacy) in &rows {
        // Skip "problem" bad guides on this pass
        if !guide_good {
            let lowest = gene_to_lowest_score[gene];
            if (score - lowest).abs() > f64::EPSILON {
                continue;
            }
        }
        let x_val = gene_to_idx[gene];
        let gene_idx = unique_genes.iter().position(|&g| g == *gene).unwrap_or(0);
        let color = palette[gene_idx % palette.len()];

        let shape = if *guide_good {
            Circle::new((x_val, *score), 5, color.filled()).into_dyn()
        } else {
            Cross::new((x_val, *score), 8, color.stroke_width(2)).into_dyn()
        };
        chart.draw_series(std::iter::once(shape))
            .map_err(|e| polars_err(Box::new(e)))?;

        chart.draw_series(std::iter::once(Text::new(
            format!("{:.0}", efficacy),
            (x_val + 0.15, *score), // shift slightly right of point
            ("sans-serif", 12).into_font().color(&BLACK),
        ))).map_err(|e| polars_err(Box::new(e)))?;

    }

    // 15) Highlight problem cases in red
    for (gene, _, score, _) in &problem_cases {
        let x_val = gene_to_idx[gene];
        // Outer circle
        chart.draw_series(std::iter::once(
            Circle::new((x_val, *score), 12, highlight_color.stroke_width(2)).into_dyn(),
        ))
            .map_err(|e| polars_err(Box::new(e)))?;

        // Cross inside
        chart.draw_series(std::iter::once(
            Cross::new((x_val, *score), 8, highlight_color.stroke_width(3)).into_dyn(),
        ))
            .map_err(|e| polars_err(Box::new(e)))?;
    }


    println!("Saved plot to {}", output_path);
    Ok(())
}

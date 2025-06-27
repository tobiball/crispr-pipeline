# CRISPR Guide RNA Efficacy Prediction Pipeline

A high-performance Rust-based pipeline for evaluating and combining multiple CRISPR guide RNA efficacy prediction tools, with Python support scripts for visualization and analysis.

## Overview

This pipeline integrates and evaluates various state-of-the-art CRISPR guide RNA prediction tools, trains consensus models, and provides comprehensive analysis of guide RNA efficacy across multiple datasets. The main pipeline is implemented in Rust for performance, with Python scripts for advanced visualizations.

### Key Features

- **Multi-tool Integration**: Supports 7+ prediction tools including DeepCRISPR, TransCRISPR, DeepSpCas9, TKO PSSM, Moreno-Mateos, Rule Set 2 & 3
- **Consensus Models**: Trains both linear and logistic consensus modelson
- **Multiple Datasets**: Compatible with TKO, Avana/DepMap, and GenomeCRISPR datasets
- **Comprehensive Evaluation**: ROC curves, confusion matrices, gene-level analysis, and statistical testing
- **High Performance**: Rust implementation ensures fast processing of large-scale genomic data

## Requirements

### Rust Dependencies
- Rust 1.70+ (install from [rustup.rs](https://rustup.rs/))
- Cargo (comes with Rust)

### Python Dependencies (for visualization scripts)
```bash
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
polars
```

### External Tool Dependencies
Some prediction tools require their own installations and need to be placed in the root directory:
- DeepCRISPR
- TransCRISPR  
- DeepSpCas9
- CHOPCHOP
- CRISPOR

## Installation

1. Clone the repository:
```bash
```

2. Build the Rust project:
```bash
cargo build --release
```

3. Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Download required data files (see Data Setup section)

## Usage

### Running the Main Pipeline

```bash
cargo run --release
```

The pipeline will:
1. Load and validate datasets
2. Balance the data using stratified sampling
3. Run prediction tools (if enabled)
4. Train consensus models
5. Evaluate all tools and generate reports
6. Save results and model weights

### Configuration

Edit `main.rs` to configure:
- Dataset selection (TKO, Avana, or GenomeCRISPR)
- Efficacy thresholds (default: poor < 60%, good > 90%)
- Which prediction tools to run
- Output directories

### Python Analysis Scripts

# Generate contribution analysis
python scripts/contribution.py

```

## Project Structure

```
crispr_pipeline/
├── validator/src/                          # Rust source code
│   ├── main.rs                   # Main pipeline entry point
│   ├── models.rs                 # Data models and types
│   ├── analysis/                 # Analysis modules
│   │   ├── roc.rs               # ROC curve analysis
│   │   ├── prediction_evaluation.rs
│   │   └── within_gene.rs       # Gene-level analysis
│   ├── data_handling/            # Dataset loaders
│   │   ├── tko_one.rs           # TKO dataset handler
│   │   ├── avana_depmap.rs     # Avana/DepMap handler
│   │   └── genome_crispr.rs    # GenomeCRISPR handler
│   ├── prediction_tools/         # Tool integrations
│   │   ├── deepcrispr_integration.rs
│   │   ├── transcrispr.rs
│   │   └── ...
│   ├── combination_model.rs     # Linear consensus model
│   └── logistic_cv.rs           # Logistic consensus model
├── scripts/                      # Python analysis scripts
├── data/                         # Input data directory
├── figures/                      # Output figures
└── prediction_evaluation_results/ # Evaluation outputs
```

## Data Setup

### Required Data Files

1. **Essential Gene Sets**:
   - `data/cegv2.txt` - Core essential genes

2. **Datasets** (choose one):
   - **TKO**: 
     - `data/tko/tko_one.xlsx`
     - `data/tko/tko_hg38_annotation_for_rust.txt`
   - **Avana or Ky /DepMap**:
     - `data/depmap/CRISPRInferredGuideEfficacy_23Q4.csv`
     - `data/depmap/KYGuideMap.csv`
     - `data/depmap/AvanaGuideMap_23Q4.csv`
   - **GenomeCRISPR**:
     - `data/genomecrispr/GenomeCRISPR_full05112017_brackets.csv`

### Output Files

The pipeline generates:
- `weights_linear_*.json` - Linear consensus model weights
- `weights_logistic_*.json` - Logistic consensus model weights  
- `weights.csv` - Final dataset with all predictions
- `prediction_evaluation_results/` - Comprehensive evaluation metrics
- `figures/` - ROC curves, violin plots, and other visualizations

## Prediction Tools Integrated

1. **Doench Rule Set 2** - Doench et al., 2016
2. **Doench Rule Set 3** - Latest Doench lab model
3. **Moreno-Mateos** - Moreno-Mateos et al., 2015
4. **DeepCRISPR** - Chuai et al., 2018
5. **TransCRISPR** - Transfer learning approach
6. **DeepSpCas9** - Kim et al., 2019
7. **TKO PSSM** - Position-specific scoring matrix from TKO data
8. **Linear Consensus** - Weighted linear combination
9. **Logistic Consensus** - Cross-validated logistic regression

## Advanced Features

### Stratified Sampling
The pipeline uses gene-aware stratified sampling to ensure balanced representation across efficacy ranges while maintaining gene diversity.

### Cross-Validation
Logistic consensus models are trained using k-fold cross-validation with gene-based splits to prevent overfitting.

### Statistical Analysis
Includes methods for identifying statistically significant "bad guide" predictors using various statistical tests.


## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request


## Acknowledgments

This pipeline integrates tools and methods from multiple research groups. Please cite the original tool publications when using their predictions.

#!/usr/bin/env python3
# merge_lifted_to_hg38.py
#
# Usage:
#   python merge_lifted_to_hg38.py \
#       <IN_TKO_HG19.tsv> \
#       <IN_TKO_HG38_CUT.bed> \
#       <OUT_TKO_HG38_FOR_RUST.tsv>
#
# Example:
#   python merge_lifted_to_hg38.py \
#     ../data/tko/TKOv3_library_forCRISPRcleanR_REAGENT_ID.txt \
#     tko_hg38_cut.bed \
#     tko_hg38_annotation_for_rust.tsv

import sys
import pandas as pd

def main():
    if len(sys.argv) != 4:
        print("Usage: python merge_lifted_to_hg38.py "
              "<IN_TKO_HG19.tsv> <IN_TKO_HG38_CUT.bed> "
              "<OUT_TKO_HG38_FOR_RUST.tsv>")
        sys.exit(1)

    tko_hg19_tsv   = sys.argv[1]
    hg38_cut_bed   = sys.argv[2]
    out_hg38_tsv   = sys.argv[3]

    # 1) Load original TKO hg19 annotation
    #    Columns (example): CODE, GENES, EXONE, CHRM, STRAND, STARTpos, ENDpos, …
    df_tko = pd.read_csv(tko_hg19_tsv, sep="\t", dtype=str)

    # 2) Load lifted BED (hg38), which has columns:
    #    chrom_hg38, start38, end38, name=CODE, score (.), strand
    cols = ["chrom_hg38", "start38", "end38", "CODE", "score", "strand_hg38"]
    df_bed = pd.read_csv(hg38_cut_bed, sep="\t", header=None, names=cols, dtype=str)

    # Convert start38/end38 to int
    df_bed["start38"] = df_bed["start38"].astype(int)
    df_bed["end38"]   = df_bed["end38"].astype(int)

    # 3) Derive 1-based cut_hg38 from bed ‘end38’
    #    (since BED end = cut position, 0-based half-open → this is exactly the 1-based cut)
    df_bed["cut_hg38"] = df_bed["end38"]

    # 4) Merge df_bed back into original df_tko on CODE
    #    (‘CODE’ is the unique guide ID shared by both)
    merged = pd.merge(
        df_tko,
        df_bed[["CODE", "chrom_hg38", "cut_hg38", "strand_hg38"]],
        how="inner",
        on="CODE"
    )

    # 5) Compute 23 nt window around cut_hg38 (for guide+PAM):
    #    - STARTpos = cut_hg38 - 1
    #    - ENDpos   = cut_hg38 + 22
    merged["STARTpos"] = merged["cut_hg38"].astype(int) - 1
    merged["ENDpos"]   = merged["cut_hg38"].astype(int) + 22

    # 6) (Optional) If you want the 30 nt window for DeepSpCas9 later,
    #    you could compute:
    #    merged["STARTpos30"] = merged["cut_hg38"].astype(int) - 4
    #    merged["ENDpos30"]   = merged["cut_hg38"].astype(int) + 26
    #    But Rust’s augment_guides_with_pam() will do that automatically if
    #    you give it correct 23 nt positions.

    # 7) Parse the 20 nt protospacer (sgRNA) from CODE: everything after the first underscore
    #    Example CODE: “A1BG_CAAGAGAAAGACCACGAGCA” → sgRNA = “CAAGAGAAAGACCACGAGCA”
    merged["sgRNA"] = merged["CODE"].str.split("_", n=1).str[1]

    # 8) Prepare the final DataFrame for Rust:
    #    columns = chromosome (hg38), STARTpos, ENDpos, strand, sgRNA, CODE
    rust_df = pd.DataFrame({
        "chromosome": merged["chrom_hg38"],
        "STARTpos":   merged["STARTpos"],
        "ENDpos":     merged["ENDpos"],
        "strand":     merged["strand_hg38"],
        "sgRNA":      merged["sgRNA"],
        "CODE":       merged["CODE"]
    })

    # 9) Write to disk (tab-separated, no index)
    rust_df.to_csv(out_hg38_tsv, sep="\t", index=False)
    print(f"Wrote {len(rust_df)} lines to {out_hg38_tsv}")

if __name__ == "__main__":
    main()


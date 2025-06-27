import sys
import pandas as pd

def prefix_chr(c):
    return str(c).strip() if str(c).startswith("chr") else "chr" + str(c).strip()

def main():
    if len(sys.argv) != 3:
        print("Usage: python make_hg19_cut_bed.py <IN_TKO_HG19.tsv> <OUT_HG19_CUT.bed>")
        sys.exit(1)

    in_tsv  = sys.argv[1]
    out_bed = sys.argv[2]

    # 1) Load TKO annotation
    df = pd.read_csv(in_tsv, sep="\t", dtype=str)

    # 2) Column names (exactly as in your file)
    chrom_col = "CHRM"
    start_col = "STARTpos"
    end_col   = "ENDpos"
    strand_col= "STRAND"
    id_col    = "CODE"

    for c in [chrom_col, start_col, end_col, strand_col]:
        if c not in df.columns:
            print(f"ERROR: Column '{c}' not found in {in_tsv}")
            print("Available columns:", df.columns.tolist())
            sys.exit(1)

    # 3) Normalize chromosome names
    df["chrom_hg19"] = df[chrom_col].apply(prefix_chr)

    # 4) Convert start/end to int
    df[start_col] = df[start_col].astype(int)
    df[end_col]   = df[end_col].astype(int)

    # 5) Compute 1-based cut site
    def compute_cut_hg19(r):
        return r[start_col] + 17 if r[strand_col] == "+" else r[end_col] - 6
    df["cut_hg19"] = df.apply(compute_cut_hg19, axis=1)

    # 6) Build BED start/end (0-based half-open)
    df["bed_start"] = df["cut_hg19"] - 1
    df["bed_end"]   = df["cut_hg19"]

    # 7) Use CODE as the unique name
    df["name"] = df[id_col].astype(str)

    # 8) Placeholder for score
    df["score"] = "."

    # 9) Assemble BED columns
    bed_df = df[[
        "chrom_hg19",
        "bed_start",
        "bed_end",
        "name",
        "score",
        strand_col
    ]].copy()

    # 10) Write BED (no header, no index)
    bed_df.to_csv(out_bed, sep="\t", header=False, index=False)
    print(f"Wrote {len(bed_df)} lines to {out_bed}")

if __name__ == "__main__":
    main()

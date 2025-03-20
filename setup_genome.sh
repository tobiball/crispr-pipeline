#!/usr/bin/env bash
set -euo pipefail

# Directory where you'd like to store the genome files for CHOPCHOP or CRISPOR.
# Adjust as needed.
CHOPCHOP_GENOME_DIR="chopchop"
CRISPOR_GENOME_DIR="crispor/genomes/hg38"

# 1) Download hg38.2bit from UCSC
echo "=== Downloading hg38.2bit from UCSC ==="
mkdir -p "${CHOPCHOP_GENOME_DIR}" "${CRISPOR_GENOME_DIR}"
cd "${CRISPOR_GENOME_DIR}"
if [ ! -f hg38.2bit ]; then
    wget -c http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit
fi

# 2) Convert .2bit to FASTA
echo "=== Converting hg38.2bit to FASTA (hg38.fa) ==="
if [ ! -f hg38.fa ]; then
    twoBitToFa hg38.2bit hg38.fa
fi

# 3) Fetch chromosome sizes (useful for CRISPOR or other tools)
echo "=== Fetching chromosome sizes (hg38.chrom.sizes) ==="
if [ ! -f hg38.chrom.sizes ]; then
    fetchChromSizes hg38 > hg38.chrom.sizes
fi

# 4) Build Bowtie1 indices (CHOPCHOP typically uses Bowtie1)
#    Output: hg38.{1,2,3,4}.ebwt, hg38.rev.{1,2}.ebwt
echo "=== Building Bowtie1 indices for CHOPCHOP ==="
cd ../../..
cd "${CHOPCHOP_GENOME_DIR}"
if [ ! -f hg38.1.ebwt ]; then
    # symlink or copy the FASTA
    ln -sf ../crispor/genomes/hg38/hg38.fa .
    bowtie-build hg38.fa hg38
fi
cd ..

# 5) Build BWA indices (CRISPOR can use BWA for certain modes)
#    Output: hg38.fa.{amb,ann,bwt,pac,sa}
cd "crispor/genomes/hg38"
echo "=== Building BWA indices ==="
if [ ! -f hg38.fa.bwt ]; then
    bwa index hg38.fa
fi

# 6) Index the FASTA with samtools (hg38.fa.fai)
echo "=== Indexing with samtools ==="
if [ ! -f hg38.fa.fai ]; then
    samtools faidx hg38.fa
fi

echo "=== Done! Genome files are now ready. ==="
,
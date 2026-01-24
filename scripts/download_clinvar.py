#!/usr/bin/env python3
"""Download and preprocess ClinVar variants for geometric analysis.

Downloads variant_summary.txt from NCBI FTP, fetches sequences from Ensembl,
and outputs aligned DNA/RNA/Protein FASTA files.

Usage:
    python scripts/download_clinvar.py --output-dir data/clinvar
    python scripts/download_clinvar.py --genes BRCA1,BRCA2 --max-variants 1000
    python scripts/download_clinvar.py --significance pathogenic,benign

Output files:
    data/clinvar/
    ├── variants.tsv          # Metadata (ID, gene, significance, positions)
    ├── dna.fasta             # Reference + flanking context (512bp window)
    ├── protein.fasta         # Protein sequences from RefSeq
    └── rna.fasta             # Transcript sequences (for Orthrus)

ClinVar FTP: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/
Ensembl REST: https://rest.ensembl.org/
"""
from __future__ import annotations

import argparse
import gzip
import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ClinVar FTP URLs
VARIANT_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"

# Ensembl REST API base
ENSEMBL_REST = "https://rest.ensembl.org"

# Column indices in variant_summary.txt (0-indexed)
# Full column list: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/README
COLS = {
    "AlleleID": 0,
    "Type": 1,
    "Name": 2,
    "GeneID": 3,
    "GeneSymbol": 4,
    "HGNC_ID": 5,
    "ClinicalSignificance": 6,
    "ClinSigSimple": 7,
    "LastEvaluated": 8,
    "RS_dbSNP": 9,
    "nsv_esv_dbVar": 10,
    "RCVaccession": 11,
    "PhenotypeIDS": 12,
    "PhenotypeList": 13,
    "Origin": 14,
    "OriginSimple": 15,
    "Assembly": 16,
    "ChromosomeAccession": 17,
    "Chromosome": 18,
    "Start": 19,
    "Stop": 20,
    "ReferenceAllele": 21,
    "AlternateAllele": 22,
    "Cytogenetic": 23,
    "ReviewStatus": 24,
    "NumberSubmitters": 25,
    "Guidelines": 26,
    "TestedInGTR": 27,
    "OtherIDs": 28,
    "SubmitterCategories": 29,
    "VariationID": 30,
    "PositionVCF": 31,
    "ReferenceAlleleVCF": 32,
    "AlternateAlleleVCF": 33,
}


@dataclass
class ClinVarVariant:
    """Parsed ClinVar variant record."""

    variation_id: str
    allele_id: str
    gene_symbol: str
    clinical_significance: str
    review_status: str
    chromosome: str
    start: int
    stop: int
    ref_allele: str
    alt_allele: str
    variant_type: str
    name: str  # HGVS-like name
    assembly: str


def download_variant_summary(output_dir: Path) -> Path:
    """Download variant_summary.txt.gz from NCBI FTP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gz_path = output_dir / "variant_summary.txt.gz"
    txt_path = output_dir / "variant_summary.txt"

    if txt_path.exists():
        logger.info(f"Using cached {txt_path}")
        return txt_path

    logger.info(f"Downloading {VARIANT_SUMMARY_URL}...")
    urllib.request.urlretrieve(VARIANT_SUMMARY_URL, gz_path)

    logger.info("Decompressing...")
    with gzip.open(gz_path, "rt") as f_in:
        with open(txt_path, "w") as f_out:
            f_out.write(f_in.read())

    gz_path.unlink()  # Remove compressed file
    logger.info(f"Saved to {txt_path}")
    return txt_path


def parse_variant_summary(
    txt_path: Path,
    genes: Optional[list[str]] = None,
    significance: Optional[list[str]] = None,
    assembly: str = "GRCh38",
    max_variants: Optional[int] = None,
) -> list[ClinVarVariant]:
    """Parse variant_summary.txt with filtering.

    Args:
        txt_path: Path to variant_summary.txt
        genes: Filter to these gene symbols (None = all)
        significance: Filter to these clinical significances (None = pathogenic+benign)
        assembly: Genome assembly (GRCh37 or GRCh38)
        max_variants: Maximum variants to return (None = all)

    Returns:
        List of ClinVarVariant objects
    """
    if significance is None:
        significance = ["Pathogenic", "Likely pathogenic", "Benign", "Likely benign"]

    # Normalize significance terms for matching
    sig_lower = {s.lower() for s in significance}
    genes_upper = {g.upper() for g in genes} if genes else None

    variants = []
    seen_ids = set()  # Deduplicate by VariationID

    logger.info(f"Parsing {txt_path}...")
    with open(txt_path) as f:
        header = f.readline()  # Skip header
        for line in f:
            if max_variants and len(variants) >= max_variants:
                break

            fields = line.strip().split("\t")
            if len(fields) < 34:
                continue

            # Filter by assembly
            if fields[COLS["Assembly"]] != assembly:
                continue

            # Filter by gene
            gene = fields[COLS["GeneSymbol"]]
            if genes_upper and gene.upper() not in genes_upper:
                continue

            # Filter by clinical significance
            clin_sig = fields[COLS["ClinicalSignificance"]]
            # ClinicalSignificance can be compound (e.g., "Pathogenic/Likely pathogenic")
            sig_parts = [s.strip().lower() for s in clin_sig.split("/")]
            if not any(s in sig_lower for s in sig_parts):
                continue

            # Skip if no valid position
            try:
                start = int(fields[COLS["Start"]])
                stop = int(fields[COLS["Stop"]])
            except (ValueError, IndexError):
                continue

            # Skip structural variants (too large for foundation models)
            var_type = fields[COLS["Type"]]
            if var_type in ("copy number gain", "copy number loss", "Deletion", "Duplication"):
                if stop - start > 100:  # Skip large SVs
                    continue

            # Deduplicate
            var_id = fields[COLS["VariationID"]]
            if var_id in seen_ids:
                continue
            seen_ids.add(var_id)

            variant = ClinVarVariant(
                variation_id=var_id,
                allele_id=fields[COLS["AlleleID"]],
                gene_symbol=gene,
                clinical_significance=clin_sig,
                review_status=fields[COLS["ReviewStatus"]],
                chromosome=fields[COLS["Chromosome"]],
                start=start,
                stop=stop,
                ref_allele=fields[COLS["ReferenceAlleleVCF"]] or fields[COLS["ReferenceAllele"]],
                alt_allele=fields[COLS["AlternateAlleleVCF"]] or fields[COLS["AlternateAllele"]],
                variant_type=var_type,
                name=fields[COLS["Name"]],
                assembly=assembly,
            )
            variants.append(variant)

    logger.info(f"Parsed {len(variants)} variants after filtering")
    return variants


def fetch_sequence_ensembl(
    chromosome: str, start: int, end: int, assembly: str = "GRCh38"
) -> Optional[str]:
    """Fetch DNA sequence from Ensembl REST API.

    Args:
        chromosome: Chromosome name (1-22, X, Y, MT)
        start: 1-based start position
        end: 1-based end position (inclusive)
        assembly: GRCh37 or GRCh38

    Returns:
        DNA sequence string or None on error
    """
    # Use GRCh37 archive if needed
    server = "https://grch37.rest.ensembl.org" if assembly == "GRCh37" else ENSEMBL_REST

    # Ensembl uses 1-based coordinates
    url = f"{server}/sequence/region/human/{chromosome}:{start}..{end}:1"
    headers = {"Content-Type": "text/plain"}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.text.strip()
        else:
            logger.warning(f"Ensembl API error {response.status_code} for {chromosome}:{start}-{end}")
            return None
    except requests.RequestException as e:
        logger.warning(f"Ensembl request failed: {e}")
        return None


def fetch_protein_sequence(gene_symbol: str, transcript_id: Optional[str] = None) -> Optional[str]:
    """Fetch canonical protein sequence for a gene from Ensembl.

    Args:
        gene_symbol: HGNC gene symbol (e.g., BRCA1)
        transcript_id: Optional specific transcript ID

    Returns:
        Protein sequence string or None
    """
    # First, get the gene ID
    url = f"{ENSEMBL_REST}/xrefs/symbol/homo_sapiens/{gene_symbol}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None

        data = response.json()
        gene_id = None
        for entry in data:
            if entry.get("type") == "gene":
                gene_id = entry.get("id")
                break

        if not gene_id:
            return None

        # Get canonical transcript
        url = f"{ENSEMBL_REST}/lookup/id/{gene_id}?expand=1"
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None

        gene_data = response.json()
        canonical_transcript = gene_data.get("canonical_transcript")

        if not canonical_transcript:
            return None

        # Remove version suffix if present
        transcript_id = canonical_transcript.split(".")[0]

        # Fetch protein sequence for canonical transcript
        url = f"{ENSEMBL_REST}/sequence/id/{transcript_id}?type=protein"
        headers = {"Content-Type": "text/plain"}
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.text.strip()
        return None

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning(f"Failed to fetch protein for {gene_symbol}: {e}")
        return None


def fetch_transcript_sequence(gene_symbol: str) -> Optional[str]:
    """Fetch canonical mRNA transcript sequence for a gene.

    Args:
        gene_symbol: HGNC gene symbol

    Returns:
        mRNA sequence string or None
    """
    url = f"{ENSEMBL_REST}/xrefs/symbol/homo_sapiens/{gene_symbol}"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None

        data = response.json()
        gene_id = None
        for entry in data:
            if entry.get("type") == "gene":
                gene_id = entry.get("id")
                break

        if not gene_id:
            return None

        # Get canonical transcript
        url = f"{ENSEMBL_REST}/lookup/id/{gene_id}?expand=1"
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None

        gene_data = response.json()
        canonical_transcript = gene_data.get("canonical_transcript")
        if not canonical_transcript:
            return None

        transcript_id = canonical_transcript.split(".")[0]

        # Fetch cDNA sequence
        url = f"{ENSEMBL_REST}/sequence/id/{transcript_id}?type=cdna"
        headers = {"Content-Type": "text/plain"}
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.text.strip()
        return None

    except (requests.RequestException, KeyError, ValueError) as e:
        logger.warning(f"Failed to fetch transcript for {gene_symbol}: {e}")
        return None


def write_fasta(sequences: dict[str, str], output_path: Path) -> None:
    """Write sequences to FASTA file.

    Args:
        sequences: Dict mapping sequence ID to sequence string
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        for seq_id, seq in sequences.items():
            if seq:  # Skip empty sequences
                f.write(f">{seq_id}\n")
                # Write sequence in 80-char lines
                for i in range(0, len(seq), 80):
                    f.write(seq[i : i + 80] + "\n")
    logger.info(f"Wrote {len(sequences)} sequences to {output_path}")


def write_variants_tsv(variants: list[ClinVarVariant], output_path: Path) -> None:
    """Write variant metadata to TSV file."""
    with open(output_path, "w") as f:
        # Header
        f.write(
            "variation_id\tallele_id\tgene_symbol\tclinical_significance\t"
            "review_status\tchromosome\tstart\tstop\tref_allele\talt_allele\t"
            "variant_type\tname\tassembly\tlabel\n"
        )
        for v in variants:
            # Encode label: 1 = pathogenic, 0 = benign
            sig_lower = v.clinical_significance.lower()
            if "pathogenic" in sig_lower:
                label = 1
            elif "benign" in sig_lower:
                label = 0
            else:
                label = -1  # VUS or other

            f.write(
                f"{v.variation_id}\t{v.allele_id}\t{v.gene_symbol}\t"
                f"{v.clinical_significance}\t{v.review_status}\t{v.chromosome}\t"
                f"{v.start}\t{v.stop}\t{v.ref_allele}\t{v.alt_allele}\t"
                f"{v.variant_type}\t{v.name}\t{v.assembly}\t{label}\n"
            )
    logger.info(f"Wrote {len(variants)} variants to {output_path}")


def process_variants(
    variants: list[ClinVarVariant],
    output_dir: Path,
    context_window: int = 256,
    rate_limit_delay: float = 0.1,
) -> None:
    """Fetch sequences for variants and write output files.

    Args:
        variants: List of parsed variants
        output_dir: Output directory
        context_window: Flanking bases on each side for DNA context
        rate_limit_delay: Delay between Ensembl API calls (seconds)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dna_sequences = {}
    protein_sequences = {}
    rna_sequences = {}

    # Track genes we've already fetched protein/RNA for
    gene_proteins = {}
    gene_rnas = {}

    total = len(variants)
    for i, variant in enumerate(variants):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing variant {i + 1}/{total}...")

        var_id = f"clinvar_{variant.variation_id}"

        # Fetch DNA context around variant
        # Use larger window for foundation model context
        dna_start = max(1, variant.start - context_window)
        dna_end = variant.stop + context_window

        dna_seq = fetch_sequence_ensembl(
            variant.chromosome, dna_start, dna_end, variant.assembly
        )
        if dna_seq:
            dna_sequences[var_id] = dna_seq
        time.sleep(rate_limit_delay)

        # Fetch protein sequence (per gene, not per variant)
        gene = variant.gene_symbol
        if gene and gene not in gene_proteins:
            protein_seq = fetch_protein_sequence(gene)
            gene_proteins[gene] = protein_seq
            time.sleep(rate_limit_delay)

        if gene and gene_proteins.get(gene):
            protein_sequences[var_id] = gene_proteins[gene]

        # Fetch RNA sequence (per gene)
        if gene and gene not in gene_rnas:
            rna_seq = fetch_transcript_sequence(gene)
            gene_rnas[gene] = rna_seq
            time.sleep(rate_limit_delay)

        if gene and gene_rnas.get(gene):
            rna_sequences[var_id] = gene_rnas[gene]

    # Write output files
    write_variants_tsv(variants, output_dir / "variants.tsv")
    write_fasta(dna_sequences, output_dir / "dna.fasta")
    write_fasta(protein_sequences, output_dir / "protein.fasta")
    write_fasta(rna_sequences, output_dir / "rna.fasta")

    logger.info(
        f"Done! DNA: {len(dna_sequences)}, Protein: {len(protein_sequences)}, RNA: {len(rna_sequences)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess ClinVar variants for geometric analysis."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/clinvar"),
        help="Output directory (default: data/clinvar)",
    )
    parser.add_argument(
        "--genes",
        type=str,
        default=None,
        help="Comma-separated gene symbols to filter (e.g., BRCA1,BRCA2)",
    )
    parser.add_argument(
        "--significance",
        type=str,
        default="Pathogenic,Likely pathogenic,Benign,Likely benign",
        help="Comma-separated clinical significances to include",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Maximum number of variants to process",
    )
    parser.add_argument(
        "--assembly",
        type=str,
        default="GRCh38",
        choices=["GRCh37", "GRCh38"],
        help="Genome assembly (default: GRCh38)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=256,
        help="Flanking bases on each side for DNA context (default: 256)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading variant_summary.txt (use cached)",
    )

    args = parser.parse_args()

    # Parse arguments
    genes = args.genes.split(",") if args.genes else None
    significance = args.significance.split(",")

    # Download variant summary
    if not args.skip_download:
        txt_path = download_variant_summary(args.output_dir)
    else:
        txt_path = args.output_dir / "variant_summary.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Cached file not found: {txt_path}")

    # Parse variants
    variants = parse_variant_summary(
        txt_path,
        genes=genes,
        significance=significance,
        assembly=args.assembly,
        max_variants=args.max_variants,
    )

    if not variants:
        logger.error("No variants found after filtering!")
        return

    # Process and fetch sequences
    process_variants(
        variants,
        args.output_dir,
        context_window=args.context_window,
    )


if __name__ == "__main__":
    main()

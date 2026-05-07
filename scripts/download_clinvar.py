#!/usr/bin/env python3
"""Download + prepare ClinVar data for ClinVarDataModule (protein-focused).

Downloads NCBI's `variant_summary.txt.gz`, filters to a target gene's
missense variants on GRCh38, and emits the four files ClinVarDataModule
expects under <data_dir>/clinvar/:

    variants.tsv     — schema per clinvar_dataset.py::_load_variants_tsv
    protein.fasta    — MUT protein per variant (WT injected with the missense)
    dna.fasta        — placeholder per variant (demo uses protein only)
    rna.fasta        — placeholder per variant (demo uses protein only)

The protein FASTA is built by fetching the gene's canonical protein from
UniProt and injecting each missense at the parsed position. Variants whose
declared WT amino acid does not match the canonical sequence at that
position are dropped (silent isoform mismatch — common for non-canonical
transcripts).

Usage:
    python scripts/download_clinvar.py
    python scripts/download_clinvar.py --gene BRCA1 --data-dir data/clinvar
    python scripts/download_clinvar.py --gene BRCA1 --uniprot P38398

The DNA/RNA FASTAs are stubbed (`>variant_id\\nN`) — only the protein
modality is real. ClinVarDataModule's loader requires all four files to
exist; the demo (`encode_esm1b_brca1`) only reads `protein.fasta`. To
generate real DNA/RNA, see docs/clinvar_pipeline.md.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import urllib.request
from pathlib import Path

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# Default canonical proteins for common workshop genes.
DEFAULT_UNIPROT = {
    "BRCA1": "P38398",
    "BRCA2": "P51587",
    "TP53": "P04637",
    "PTEN": "P60484",
    "MLH1": "P40692",
    "MSH2": "P43246",
}

THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O", "Xle": "J", "Asx": "B", "Glx": "Z",
    "Xaa": "X", "Ter": "*", "Stop": "*", "*": "*",
}

# "p.Cys61Gly" — three-letter ref / pos / three-letter alt
HGVS_P_RE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*|=|\?)\)")


def fetch_variant_summary(dest: Path) -> None:
    if dest.exists():
        print(f"[OK]    cached: {dest}", flush=True)
        return
    print(f"[GET]   {CLINVAR_URL}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(CLINVAR_URL, timeout=60) as r, dest.open("wb") as fh:
        total = 0
        while chunk := r.read(1 << 16):
            fh.write(chunk)
            total += len(chunk)
        print(f"[OK]    downloaded {total / 1e6:.1f} MB → {dest}", flush=True)


def fetch_uniprot_protein(accession: str) -> str:
    """Return the canonical protein sequence as a single uppercase string."""
    url = f"https://www.uniprot.org/uniprotkb/{accession}.fasta"
    print(f"[GET]   {url}", flush=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
    seq = "".join(lines).upper()
    if not seq:
        raise RuntimeError(f"empty FASTA for {accession}")
    return seq


def classify_significance(sig: str) -> int:
    """Map ClinicalSignificance → label. 0=benign, 1=pathogenic, -1=other."""
    s = sig.lower()
    if "pathogenic" in s and "non-pathogenic" not in s and "conflicting" not in s:
        return 1
    if "benign" in s and "conflicting" not in s:
        return 0
    return -1


def parse_missense(name: str) -> tuple[str, int, str] | None:
    """Extract (ref_aa, pos, alt_aa) from an HGVS.p Name field. None if not missense."""
    m = HGVS_P_RE.search(name)
    if not m:
        return None
    ref3, pos, alt3 = m.group(1), int(m.group(2)), m.group(3)
    if alt3 in ("=", "?"):
        return None  # synonymous or unknown
    ref = THREE_TO_ONE.get(ref3)
    alt = THREE_TO_ONE.get(alt3)
    if not ref or not alt:
        return None
    if ref == alt or alt == "*":
        return None  # synonymous or nonsense — exclude from missense
    return ref, pos, alt


def stream_filtered_rows(summary_gz: Path, gene: str):
    """Yield (row_dict, ref, pos, alt) for matching missense rows on GRCh38."""
    gene_u = gene.upper()
    with gzip.open(summary_gz, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().lstrip("#").rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            if f[idx["GeneSymbol"]].upper() != gene_u:
                continue
            if f[idx["Type"]] != "single nucleotide variant":
                continue
            if f[idx["Assembly"]] != "GRCh38":
                continue
            mis = parse_missense(f[idx["Name"]])
            if mis is None:
                continue
            yield {h: f[idx[h]] for h in header}, mis


REVIEW_STAR = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, conflicting classifications": 1,
    "no assertion criteria provided": 0,
    "no classification provided": 0,
    "no assertion provided": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gene", default="BRCA1",
                        help="HGNC gene symbol (default: BRCA1)")
    parser.add_argument("--uniprot", default=None,
                        help="UniProt accession for the canonical protein. "
                             "Defaults to a built-in for common genes.")
    parser.add_argument("--data-dir", default="data/clinvar", type=Path,
                        help="Output dir (writes variants.tsv + *.fasta here)")
    parser.add_argument("--cache-dir", default="data/_cache", type=Path,
                        help="Where to cache variant_summary.txt.gz")
    parser.add_argument("--min-stars", type=int, default=1,
                        help="Minimum review-status star rating (default: 1)")
    parser.add_argument("--max-variants", type=int, default=None,
                        help="Cap the number of variants written (after filtering).")
    args = parser.parse_args()

    uniprot = args.uniprot or DEFAULT_UNIPROT.get(args.gene.upper())
    if not uniprot:
        print(f"[ERR]   No default UniProt accession for {args.gene}. "
              f"Pass --uniprot <accession>.", file=sys.stderr)
        return 1

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    summary_gz = args.cache_dir / "variant_summary.txt.gz"
    fetch_variant_summary(summary_gz)
    wt_protein = fetch_uniprot_protein(uniprot)
    print(f"[OK]    {args.gene} canonical ({uniprot}): {len(wt_protein)} aa", flush=True)

    out_rows: list[tuple[dict, str, int, str]] = []
    skipped_mismatch = 0
    skipped_oob = 0
    for row, (ref, pos, alt) in stream_filtered_rows(summary_gz, args.gene):
        stars = REVIEW_STAR.get(row.get("ReviewStatus", "").lower(), 0)
        if stars < args.min_stars:
            continue
        if pos < 1 or pos > len(wt_protein):
            skipped_oob += 1
            continue
        if wt_protein[pos - 1] != ref:
            # Variant is described against a non-canonical isoform — skip.
            skipped_mismatch += 1
            continue
        out_rows.append((row, ref, pos, alt))
        if args.max_variants and len(out_rows) >= args.max_variants:
            break

    # Sort by review stars desc then position — gives "top N" stable ordering.
    out_rows.sort(
        key=lambda r: (
            -REVIEW_STAR.get(r[0].get("ReviewStatus", "").lower(), 0),
            int(r[0].get("Start", "0") or 0),
        )
    )
    print(f"[OK]    accepted variants: {len(out_rows)} "
          f"(skipped: {skipped_mismatch} isoform-mismatch, {skipped_oob} oob)", flush=True)

    label_counts = {0: 0, 1: 0, -1: 0}
    tsv_path = args.data_dir / "variants.tsv"
    protein_fa = args.data_dir / "protein.fasta"
    dna_fa = args.data_dir / "dna.fasta"
    rna_fa = args.data_dir / "rna.fasta"

    tsv_cols = [
        "variation_id", "gene_symbol", "label", "clinical_significance",
        "review_status", "chromosome", "start", "stop", "variant_type",
        "name", "uniprot", "wt_aa", "position", "alt_aa",
    ]
    with tsv_path.open("w") as tsv, \
         protein_fa.open("w") as pfa, \
         dna_fa.open("w") as dfa, \
         rna_fa.open("w") as rfa:
        tsv.write("\t".join(tsv_cols) + "\n")
        for row, ref, pos, alt in out_rows:
            label = classify_significance(row.get("ClinicalSignificance", ""))
            label_counts[label] += 1
            vid = row.get("VariationID") or row.get("AlleleID") or ""
            if not vid:
                continue
            mut_protein = wt_protein[: pos - 1] + alt + wt_protein[pos:]
            tsv.write("\t".join([
                vid,
                row.get("GeneSymbol", ""),
                str(label),
                row.get("ClinicalSignificance", ""),
                row.get("ReviewStatus", ""),
                row.get("Chromosome", ""),
                row.get("Start", ""),
                row.get("Stop", ""),
                row.get("Type", ""),
                row.get("Name", ""),
                uniprot,
                ref,
                str(pos),
                alt,
            ]) + "\n")
            pfa.write(f">clinvar_{vid}\n{mut_protein}\n")
            # DNA/RNA stubs — placeholders. Demo doesn't read them; loader
            # only needs the file present and the id matched.
            dfa.write(f">clinvar_{vid}\nN\n")
            rfa.write(f">clinvar_{vid}\nN\n")

    print(f"[OK]    wrote {tsv_path}  ({sum(label_counts.values())} rows)")
    print(f"        labels: pathogenic={label_counts[1]}, "
          f"benign={label_counts[0]}, vus={label_counts[-1]}")
    print(f"[OK]    wrote {protein_fa}")
    print(f"[OK]    wrote {dna_fa} (stub)  {rna_fa} (stub)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

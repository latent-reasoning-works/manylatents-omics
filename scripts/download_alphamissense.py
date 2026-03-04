#!/usr/bin/env python3
"""Download AlphaMissense prediction files from Zenodo.

Usage:
    python scripts/download_alphamissense.py --output data/alphamissense/
    python scripts/download_alphamissense.py --output data/alphamissense/ --protein-only

Files downloaded:
    - AlphaMissense_hg38.tsv.gz (~3.9GB) — 71M genomic missense predictions
    - AlphaMissense_aa_substitutions.tsv.gz (~4.4GB) — 216M protein substitutions

Source: https://zenodo.org/records/10813168
Reference: Cheng et al., "Accurate proteome-wide missense variant effect
           prediction with AlphaMissense", Science (2023).
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

# Zenodo record 10813168 (v3, March 2024)
BASE_URL = "https://zenodo.org/records/10813168/files"

FILES = {
    "genomic": {
        "filename": "AlphaMissense_hg38.tsv.gz",
        "url": f"{BASE_URL}/AlphaMissense_hg38.tsv.gz",
        "md5": "0322a99b7469f8a83ecac63a4dc26ba4",
        "size_mb": 3900,
    },
    "protein": {
        "filename": "AlphaMissense_aa_substitutions.tsv.gz",
        "url": f"{BASE_URL}/AlphaMissense_aa_substitutions.tsv.gz",
        "md5": "439bfab9e948e45f80173dc08b3f2a79",
        "size_mb": 4400,
    },
}


def download_file(url: str, dest: Path, expected_md5: str | None = None) -> None:
    """Download a file with progress reporting."""
    if dest.exists():
        if expected_md5:
            print(f"  Verifying existing {dest.name}...")
            md5 = hashlib.md5(dest.read_bytes()).hexdigest()
            if md5 == expected_md5:
                print(f"  Already exists with correct checksum, skipping.")
                return
            print(f"  Checksum mismatch (got {md5}), re-downloading.")
        else:
            print(f"  Already exists, skipping. Delete to re-download.")
            return

    print(f"  Downloading {url}")
    print(f"  → {dest}")
    urllib.request.urlretrieve(url, dest)

    if expected_md5:
        md5 = hashlib.md5(dest.read_bytes()).hexdigest()
        if md5 != expected_md5:
            print(f"  WARNING: MD5 mismatch. Expected {expected_md5}, got {md5}")
        else:
            print(f"  Checksum verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/alphamissense"),
                        help="Output directory (default: data/alphamissense/)")
    parser.add_argument("--genomic-only", action="store_true",
                        help="Download only the hg38 genomic file")
    parser.add_argument("--protein-only", action="store_true",
                        help="Download only the protein substitutions file")
    parser.add_argument("--skip-checksum", action="store_true",
                        help="Skip MD5 verification")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    to_download = []
    if args.protein_only:
        to_download = ["protein"]
    elif args.genomic_only:
        to_download = ["genomic"]
    else:
        to_download = ["genomic", "protein"]

    for key in to_download:
        info = FILES[key]
        dest = args.output / info["filename"]
        md5 = None if args.skip_checksum else info["md5"]
        print(f"\n[{key}] {info['filename']} (~{info['size_mb']}MB)")
        download_file(info["url"], dest, md5)

    print(f"\nDone. Files in {args.output}/")
    print("Usage:")
    print(f'  scorer = AlphaMissenseScorer("{args.output / "AlphaMissense_hg38.tsv.gz"}")')
    print('  scores = scorer.score(variants_df)')


if __name__ == "__main__":
    main()

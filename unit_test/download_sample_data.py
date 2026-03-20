#!/usr/bin/env python3
"""
Download sample data for unit tests.

This script downloads/generates sample data used by the unit tests:

  * A small public NIfTI file (from the nibabel test suite on GitHub)
  * Three DICOM-tag JSON files (generated locally – no download needed)

Usage
-----
    python unit_test/download_sample_data.py          # download everything
    python unit_test/download_sample_data.py --nifti  # NIfTI only
    python unit_test/download_sample_data.py --json   # JSON only
    python unit_test/download_sample_data.py --force  # re-download even if present

All files are written under ``unit_test/sample_data/``.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
NIFTI_DIR = SAMPLE_DATA_DIR / "nifti"
JSON_DIR = SAMPLE_DATA_DIR / "json_tags"

# ---------------------------------------------------------------------------
# Public NIfTI source
# A small 4-D NIfTI shipped with the nibabel test suite (~105 KB).
# ---------------------------------------------------------------------------
NIFTI_URL = (
    "https://github.com/nipy/nibabel/raw/master/nibabel/tests/data/example4d.nii.gz"
)
NIFTI_FILENAME = "example4d.nii.gz"


# ---------------------------------------------------------------------------
# Sample DICOM-tag JSON payloads
# ---------------------------------------------------------------------------
_JSON_FILES = {
    "subject_001_t2_tse_dixon_tra_W.json": {
        "0008|0005": "ISO_IR 100",
        "0008|0008": "DERIVED\\PRIMARY\\DIXON\\WATER ",
        "0008|0012": "20250718",
        "0008|0013": "103632.487500 ",
        "0008|0016": "1.2.840.10008.5.1.4.1.1.4.1",
        "0008|0018": "1.3.12.2.1107.5.8.15.134699.30000025081512365346700000018",
        "0008|0020": "20250718",
        "0008|0060": "MR",
        "0008|0070": "Siemens Healthineers",
        "0008|1030": "NP (w/ neck)+con.,NP (w/ neck) plain",
        "0008|103e": "t2_tse_dixon_tra_NP_W ",
        "0008|1090": "MAGNETOM Sola ",
        "0010|0010": "1 ",
        "0010|0020": "001 ",
        "0010|0030": "20250815",
        "0010|0040": "F ",
        "0010|1010": "069Y",
        "0010|1020": "1.59",
        "0010|1030": "68",
        "0018|0015": "NECK",
        "0018|0023": "2D",
        "0018|0087": "1.5 ",
        "0018|0088": "4.4 ",
        "0018|1000": "186558",
        "0018|1020": "syngo MR XA51 ",
        "0018|1030": "t2_tse_dixon_tra_NP ",
        "0020|000d": "1.3.12.2.1107.5.8.15.134699.30000025081512365346700000006",
        "0020|000e": "1.3.12.2.1107.5.8.15.134699.30000025081512365346700000019",
        "0020|0010": "CMC002864468A ",
        "0020|0011": "11",
        "0028|0010": "480",
        "0028|0011": "480",
        "0028|0100": "16",
        "2050|0020": "IDENTITY",
    },
    "subject_001_t2_tse_dixon_tra_F.json": {
        "0008|0005": "ISO_IR 100",
        "0008|0008": "DERIVED\\PRIMARY\\DIXON\\FAT ",
        "0008|0012": "20250718",
        "0008|0013": "103633.012500 ",
        "0008|0016": "1.2.840.10008.5.1.4.1.1.4.1",
        "0008|0018": "1.3.12.2.1107.5.8.15.134699.30000025081512365346700000020",
        "0008|0020": "20250718",
        "0008|0060": "MR",
        "0008|0070": "Siemens Healthineers",
        "0008|1030": "NP (w/ neck)+con.,NP (w/ neck) plain",
        "0008|103e": "t2_tse_dixon_tra_NP_F ",
        "0008|1090": "MAGNETOM Sola ",
        "0010|0010": "1 ",
        "0010|0020": "001 ",
        "0010|0030": "20250815",
        "0010|0040": "F ",
        "0010|1010": "069Y",
        "0010|1020": "1.59",
        "0010|1030": "68",
        "0018|0015": "NECK",
        "0018|0023": "2D",
        "0018|0087": "1.5 ",
        "0018|0088": "4.4 ",
        "0018|1000": "186558",
        "0018|1020": "syngo MR XA51 ",
        "0018|1030": "t2_tse_dixon_tra_NP ",
        "0020|000d": "1.3.12.2.1107.5.8.15.134699.30000025081512365346700000006",
        "0020|000e": "1.3.12.2.1107.5.8.15.134699.30000025081512365346700000020",
        "0020|0010": "CMC002864468A ",
        "0020|0011": "12",
        "0028|0010": "480",
        "0028|0011": "480",
        "0028|0100": "16",
        "2050|0020": "IDENTITY",
    },
    "subject_002_t1_mprage_sag.json": {
        "0008|0005": "ISO_IR 100",
        "0008|0008": "ORIGINAL\\PRIMARY\\M\\ND",
        "0008|0012": "20250719",
        "0008|0013": "091205.123000 ",
        "0008|0016": "1.2.840.10008.5.1.4.1.1.4",
        "0008|0018": "1.3.12.2.1107.5.8.15.134699.30000025081912365346700000045",
        "0008|0020": "20250719",
        "0008|0060": "MR",
        "0008|0070": "Siemens Healthineers",
        "0008|1030": "Brain MRI routine",
        "0008|103e": "t1_mprage_sag ",
        "0008|1090": "MAGNETOM Prisma ",
        "0010|0010": "2 ",
        "0010|0020": "002 ",
        "0010|0030": "19601012",
        "0010|0040": "M ",
        "0010|1010": "064Y",
        "0010|1020": "1.75",
        "0010|1030": "82",
        "0018|0015": "HEAD",
        "0018|0023": "3D",
        "0018|0087": "3.0 ",
        "0018|0088": "1.0 ",
        "0018|1000": "186559",
        "0018|1020": "syngo MR XA51 ",
        "0018|1030": "t1_mprage_sag_p2 ",
        "0020|000d": "1.3.12.2.1107.5.8.15.134699.30000025081912365346700000007",
        "0020|000e": "1.3.12.2.1107.5.8.15.134699.30000025081912365346700000046",
        "0020|0010": "CMC002864469B ",
        "0020|0011": "5",
        "0028|0010": "256",
        "0028|0011": "256",
        "0028|0100": "16",
        "2050|0020": "IDENTITY",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """Download *url* to *dest* with simple exponential backoff retry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")
            return True
        except urllib.error.URLError as exc:
            wait = 2 ** attempt
            print(f"  Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                print(f"  Retrying in {wait}s …")
                time.sleep(wait)
    return False


def download_nifti(force: bool = False) -> bool:
    """Download the sample NIfTI file.

    Returns True on success, False on failure.
    """
    dest = NIFTI_DIR / NIFTI_FILENAME
    if dest.exists() and not force:
        print(f"[nifti] Already present: {dest}  (use --force to re-download)")
        return True

    print("[nifti] Downloading sample NIfTI …")
    ok = _download(NIFTI_URL, dest)
    if not ok:
        print("[nifti] Download failed – unit tests requiring NIfTI will be skipped.")
    return ok


def generate_json_files(force: bool = False) -> bool:
    """Write the bundled DICOM-tag JSON files to *JSON_DIR*.

    Returns True always (no network access required).
    """
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    print("[json] Writing sample DICOM-tag JSON files …")
    for filename, payload in _JSON_FILES.items():
        dest = JSON_DIR / filename
        if dest.exists() and not force:
            print(f"  Already present: {dest}")
            continue
        dest.write_text(json.dumps(payload, indent=4, ensure_ascii=False))
        print(f"  Wrote {dest}")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download / generate sample data for unit tests."
    )
    parser.add_argument("--nifti", action="store_true",
                        help="Download NIfTI sample only")
    parser.add_argument("--json", action="store_true",
                        help="Generate JSON tag samples only")
    parser.add_argument("--force", action="store_true",
                        help="Re-download / overwrite even if files already exist")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Default: do everything when no specific flag is given
    do_nifti = args.nifti or not (args.nifti or args.json)
    do_json = args.json or not (args.nifti or args.json)

    results = []
    if do_json:
        results.append(generate_json_files(force=args.force))
    if do_nifti:
        results.append(download_nifti(force=args.force))

    if all(results):
        print("\nSample data is ready.")
        sys.exit(0)
    else:
        print("\nSome downloads failed – check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

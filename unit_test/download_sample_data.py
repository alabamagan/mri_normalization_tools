#!/usr/bin/env python3
"""
Download sample data for unit tests.

This script downloads/generates sample data used by the unit tests:

  * A small public NIfTI file (from the nibabel test suite on GitHub)
  * Three DICOM-tag JSON files (generated locally – no download needed)
  * A small MR DICOM series (from pydicom's test suite on GitHub),
    written as ``sample_data/sample1/image-00001.dcm`` …
    ``image-00010.dcm``.  These files satisfy the path expectations of
    ``test_custom_filename_format.py`` and ``test_real_dicom_custom_format.py``.
    Note: tests that assert specific tag values (e.g. series description`
    "ARTERIELLE", study date "20061012") require the original clinical dataset
    and will be skipped when only the pydicom test files are present.
  * Three NIfTI example files for ``examples/example_data/`` downloaded from
    OpenNeuro dataset ds000105 (Haxby 2001, CC0 license).

Usage
-----
    python unit_test/download_sample_data.py             # download everything
    python unit_test/download_sample_data.py --nifti     # NIfTI only
    python unit_test/download_sample_data.py --json      # JSON only
    python unit_test/download_sample_data.py --dicom     # DICOM series only
    python unit_test/download_sample_data.py --examples  # example NIfTI only
    python unit_test/download_sample_data.py --force     # re-download even if present

All unit-test files are written under ``unit_test/sample_data/``.
Example data files are written under ``examples/example_data/``.
"""

import argparse
import json
import shutil
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
DICOM_DIR = SAMPLE_DATA_DIR / "sample1"

# ---------------------------------------------------------------------------
# Public NIfTI source
# A small 4-D NIfTI shipped with the nibabel test suite (~105 KB).
# ---------------------------------------------------------------------------
NIFTI_URL = (
    "https://github.com/nipy/nibabel/raw/master/nibabel/tests/data/example4d.nii.gz"
)
NIFTI_FILENAME = "example4d.nii.gz"

# ---------------------------------------------------------------------------
# Public DICOM source
# MR2_J2KI.dcm from pydicom/pydicom-data (~111 KB, JPEG 2000 lossless, MIT).
# One copy is downloaded then duplicated to form a minimal numbered series.
# ---------------------------------------------------------------------------
DICOM_URL = (
    "https://raw.githubusercontent.com/pydicom/pydicom-data/master/data_store/data/MR2_J2KI.dcm"
)
DICOM_SERIES_SIZE = 10  # number of image-NNNNN.dcm files to create

# ---------------------------------------------------------------------------
# OpenNeuro example data (ds000105, Haxby 2001, CC0 license)
# sub-1/2/3 T1w anatomical scans saved as MRI_01/02/03.nii.gz
# ---------------------------------------------------------------------------
EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "examples" / "example_data"
_OPENNEURO_BASE = "https://s3.amazonaws.com/openneuro.org/ds000105"
EXAMPLE_FILES = [
    (_OPENNEURO_BASE + "/sub-1/anat/sub-1_T1w.nii.gz", "MRI_01.nii.gz"),
    (_OPENNEURO_BASE + "/sub-2/anat/sub-2_T1w.nii.gz", "MRI_02.nii.gz"),
    (_OPENNEURO_BASE + "/sub-3/anat/sub-3_T1w.nii.gz", "MRI_03.nii.gz"),
]


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


def _generate_dicom_series_json(force: bool = False) -> bool:
    """Read DICOM tags from the sample series and write a JSON entry to *JSON_DIR*.

    Uses :class:`DicomTagPrinter` to read tags from the first ``image-*.dcm``
    file and writes them as ``subject_003_mr_series.json``.  The ``subject_``
    prefix is required so that ``test_id_globber_with_sample_filenames`` in
    ``test_dicom_tag_printer.py`` can match the filename with ``subject_(\\d+)``.

    Returns True on success, False if the series is absent or the import fails.
    """
    dest = JSON_DIR / "subject_003_mr_series.json"
    if dest.exists() and not force:
        print(f"[dicom-json] Already present: {dest}")
        return True

    # Skip Git LFS pointer files (they are ~130 B and not valid DICOMs).
    # Use the first file large enough to be a real DICOM (>1 KB).
    candidates = [p for p in sorted(DICOM_DIR.glob("image-*.dcm")) if p.stat().st_size > 1024]
    if not candidates:
        print("[dicom-json] No valid (non-LFS) DICOM files found – skipping JSON export")
        return False

    rep_file = candidates[0]
    print(f"[dicom-json] Reading DICOM tags from {rep_file.name} …")

    # Make the repo root importable without requiring a prior `pip install`.
    _repo_root = str(Path(__file__).parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    try:
        from mnts.utils.dicom_tag_printer import DicomTagPrinter
    except ImportError as exc:
        print(f"[dicom-json] Cannot import DicomTagPrinter ({exc}) – skipping")
        return False

    _TAGS = [
        '0008|0005', '0008|0008', '0008|0016', '0008|0018',
        '0008|0060', '0008|0070', '0008|103e',
        '0010|0010', '0010|0020',
        '0018|0087', '0018|1020',
        '0020|000d', '0020|000e', '0020|0010', '0020|0011',
        '0028|0010', '0028|0011', '0028|0100',
    ]

    try:
        printer = DicomTagPrinter()
        tag_values = printer.read_dicom_tags(rep_file, _TAGS)
        tag_values.pop('__num_frames__', None)

        JSON_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(tag_values, indent=4, ensure_ascii=False))
        print(f"[dicom-json] Wrote {dest}")
        return True
    except Exception as exc:
        print(f"[dicom-json] Tag extraction failed: {exc}")
        return False


def download_dicom(force: bool = False) -> bool:
    """Download a sample MR DICOM series into *DICOM_DIR*.

    Downloads one small MR DICOM from pydicom's test suite, then copies it
    to produce a numbered series (``image-00001.dcm`` … ``image-NNNNN.dcm``).
    This satisfies the path expectations of the custom-filename-format tests.
    After the series is ready, DICOM tags are exported to a JSON file in
    *JSON_DIR* via :func:`_generate_dicom_series_json` so that the
    tag-printer integration tests have a real-data entry to work with.

    Returns True on success, False on failure.
    """
    DICOM_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(DICOM_DIR.glob("image-*.dcm"))
    if len(existing) >= DICOM_SERIES_SIZE and not force:
        print(
            f"[dicom] Already present: {len(existing)} files in {DICOM_DIR}"
            "  (use --force to re-download)"
        )
        _generate_dicom_series_json(force=force)
        return True

    # Download the source DICOM to a temporary name
    source = DICOM_DIR / "_source_mr.dcm"
    print("[dicom] Downloading sample MR DICOM …")
    ok = _download(DICOM_URL, source)
    if not ok:
        print("[dicom] Download failed – unit tests requiring DICOM will be skipped.")
        return False

    # Duplicate to form a numbered series
    print(f"[dicom] Creating {DICOM_SERIES_SIZE}-slice series in {DICOM_DIR} …")
    for i in range(1, DICOM_SERIES_SIZE + 1):
        dest = DICOM_DIR / f"image-{i:05d}.dcm"
        if dest.exists() and not force:
            print(f"  Already present: {dest}")
            continue
        shutil.copy2(source, dest)
        print(f"  Created {dest}")

    source.unlink(missing_ok=True)
    _generate_dicom_series_json(force=force)
    return True


def download_examples(force: bool = False) -> bool:
    """Download the three example NIfTI files from OpenNeuro ds000105.

    Files are saved to ``examples/example_data/MRI_01.nii.gz`` etc.
    Returns True if all files were obtained successfully.
    """
    EXAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    print("[examples] Downloading example NIfTI files from OpenNeuro …")
    for url, filename in EXAMPLE_FILES:
        dest = EXAMPLE_DATA_DIR / filename
        if dest.exists() and not force:
            print(f"  Already present: {dest}  (use --force to re-download)")
            continue
        ok = _download(url, dest)
        if not ok:
            print(f"  Failed to download {filename}")
            all_ok = False
    if all_ok:
        print("[examples] Example data ready.")
    return all_ok


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
    parser.add_argument("--dicom", action="store_true",
                        help="Download sample MR DICOM series only")
    parser.add_argument("--examples", action="store_true",
                        help="Download example NIfTI files to examples/example_data/")
    parser.add_argument("--force", action="store_true",
                        help="Re-download / overwrite even if files already exist")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Default: do everything when no specific flag is given
    any_specific = args.nifti or args.json or args.dicom or args.examples
    do_nifti = args.nifti or not any_specific
    do_json = args.json or not any_specific
    do_dicom = args.dicom or not any_specific
    do_examples = args.examples or not any_specific

    results = []
    if do_json:
        results.append(generate_json_files(force=args.force))
    if do_nifti:
        results.append(download_nifti(force=args.force))
    if do_dicom:
        results.append(download_dicom(force=args.force))
    if do_examples:
        results.append(download_examples(force=args.force))

    if all(results):
        print("\nSample data is ready.")
        sys.exit(0)
    else:
        print("\nSome downloads failed – check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

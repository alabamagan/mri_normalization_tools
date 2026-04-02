#!/usr/bin/env python3
"""
DICOM Tag Printer Utility

This script reads DICOM files or directories and prints specific DICOM tag information
for all sequences. Supports single files, recursive directory search, and batch processing
of multiple tags.

Author: MRI Normalization Tools
"""

import os
import re
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

try:
    import pydicom
    from pydicom.datadict import dictionary_description
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

from ..mnts_logger import MNTSLogger
from ..io.data_formatting import pydicom_read_series

__all__ = ['DicomTagPrinter', 'print_dicom_tags', 'validate_tag_format', 'print_dicom_tags_from_json']


def _fmt_number(n: float) -> str:
    """Format a number, dropping the decimal point for integer-valued floats."""
    return str(int(n)) if n == int(n) else str(n)


# Rich table column styles keyed by DataFrame column name.
# Columns not listed here get the default tag style.
_COL_STYLES = {
    'SubjectID':          dict(style="magenta", no_wrap=True),
    'SeriesUID':          dict(style="cyan", no_wrap=False, max_width=30),
    'FileCount':          dict(style="green", justify="right"),
    'RepresentativeFile': dict(style="blue", no_wrap=False, max_width=40),
    'FilePath':           dict(style="cyan", no_wrap=False, max_width=50),
}
_DEFAULT_TAG_STYLE = dict(style="yellow", no_wrap=False)


# ------------------------------------------------------------------
# Module-level worker functions for ProcessPoolExecutor
# (Instance methods are not picklable, so multiprocessing needs these)
# ------------------------------------------------------------------

def _is_dicom_file_worker(file_path: Path, backend: str) -> bool:
    """Return ``True`` if *file_path* is a readable DICOM file.

    Designed to run in a child process — imports are done locally so the
    worker is self-contained and picklable.
    """
    try:
        if file_path.suffix.lower() in ['.dcm', '.dicom']:
            return True

        if backend == 'pydicom':
            try:
                import pydicom as _pd
                _pd.dcmread(file_path, stop_before_pixels=True)
                return True
            except Exception:
                return False
        elif backend == 'sitk':
            try:
                import SimpleITK as _sitk
                reader = _sitk.ImageFileReader()
                reader.SetFileName(str(file_path))
                reader.ReadImageInformation()
                return True
            except Exception:
                return False
    except Exception:
        pass
    return False


def _read_dicom_tags_worker(file_path: Path, tags: List[str],
                            backend: str) -> Dict[str, str]:
    """Read DICOM tags from a single file.

    Standalone version of :pymeth:`DicomTagPrinter.read_dicom_tags` that can
    run in a child process.
    """
    if backend == 'pydicom':
        try:
            import pydicom as _pd
            ds = _pd.dcmread(file_path, stop_before_pixels=True, force=True)
            result = {}
            for tag_str in tags:
                try:
                    group, element = tag_str.split('|')
                    tag = _pd.tag.Tag(int(group, 16), int(element, 16))
                    if tag in ds:
                        result[tag_str] = str(ds[tag].value).strip()
                    else:
                        result[tag_str] = 'Missing'
                except Exception:
                    result[tag_str] = 'Error'
            return result
        except Exception:
            return {tag: 'Error' for tag in tags}
    else:
        try:
            import SimpleITK as _sitk
            reader = _sitk.ImageFileReader()
            reader.SetFileName(str(file_path))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            result = {}
            for tag_str in tags:
                try:
                    if reader.HasMetaDataKey(tag_str):
                        result[tag_str] = reader.GetMetaData(tag_str).strip()
                    else:
                        result[tag_str] = 'Missing'
                except Exception:
                    result[tag_str] = 'Error'
            return result
        except Exception:
            return {tag: 'Error' for tag in tags}


def _aggregate_series_tags_worker(files: List[Path], tags: List[str],
                                  backend: str) -> Dict[str, str]:
    """Aggregate tag values across all files in a single series.

    Standalone version of :pymeth:`DicomTagPrinter._aggregate_series_tags`
    that can run in a child process.
    """
    all_values: Dict[str, List[str]] = defaultdict(list)
    for file_path in files:
        tag_values = _read_dicom_tags_worker(file_path, tags, backend)
        for tag, value in tag_values.items():
            all_values[tag].append(value)

    result: Dict[str, str] = {}
    for tag in tags:
        values = all_values.get(tag, [])
        valid = [v for v in values if v not in ('Missing', 'Error')]
        if not valid:
            result[tag] = values[0] if values else 'Missing'
            continue
        if all(v == valid[0] for v in valid[1:]):
            result[tag] = valid[0]
            continue
        try:
            numbers = [float(v) for v in valid]
            result[tag] = f"{_fmt_number(min(numbers))}~{_fmt_number(max(numbers))}"
        except (ValueError, TypeError):
            result[tag] = valid[0]
    return result


class DicomTagPrinter:
    """DICOM Tag Printer Class

    This class provides functionality to read DICOM files and print specific tags.
    Supports both pydicom and SimpleITK as backend engines.

    Args:
        backend (str):
            Backend engine to use, 'pydicom' or 'sitk' or 'auto'
        logger (MNTSLogger):
            Logger instance for logging information
        verbose (bool, optional):
            Put logger in debug level if true
    """

    def __init__(self, backend: str = 'auto', logger=None, verbose=False):
        self.logger = logger or MNTSLogger['DicomTagPrinter']
        if verbose:
            MNTSLogger.set_global_log_level('debug')
            self.logger.debug("Confirm log level set to debug.")

        if backend == 'auto':
            if PYDICOM_AVAILABLE:
                self.backend = 'pydicom'
            elif SITK_AVAILABLE:
                self.backend = 'sitk'
            else:
                raise ImportError("Either pydicom or SimpleITK must be installed")
        elif backend == 'pydicom':
            if not PYDICOM_AVAILABLE:
                raise ImportError("pydicom is not installed")
            self.backend = 'pydicom'
        elif backend == 'sitk':
            if not SITK_AVAILABLE:
                raise ImportError("SimpleITK is not installed")
            self.backend = 'sitk'
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.logger.info(f"Using backend: {self.backend}")

    # ------------------------------------------------------------------
    # Tag reading
    # ------------------------------------------------------------------

    def read_dicom_tag_pydicom(self, file_path: Union[str, Path], tags: List[str]) -> Dict[str, str]:
        """Read DICOM tags using pydicom.

        Args:
            file_path: DICOM file path
            tags: List of tags to read, format: ['0008|103e', '0010|0020']

        Returns:
            Dictionary mapping raw tag strings to their values.
        """
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
            result = {}

            for tag_str in tags:
                try:
                    group, element = tag_str.split('|')
                    tag = pydicom.tag.Tag(int(group, 16), int(element, 16))

                    if tag in ds:
                        result[tag_str] = str(ds[tag].value).strip()
                    else:
                        result[tag_str] = 'Missing'

                except Exception as e:
                    self.logger.warning(f"Cannot read tag {tag_str}: {e}")
                    result[tag_str] = 'Error'

            return result

        except Exception as e:
            self.logger.error(f"Cannot read DICOM file {file_path}: {e}")
            return {tag: 'Error' for tag in tags}

    def read_dicom_tag_sitk(self, file_path: Union[str, Path], tags: List[str]) -> Dict[str, str]:
        """Read DICOM tags using SimpleITK.

        Args:
            file_path: DICOM file path
            tags: List of tags to read, format: ['0008|103e', '0010|0020']

        Returns:
            Dictionary mapping raw tag strings to their values.
        """
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(file_path))
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()

            result = {}
            for tag_str in tags:
                try:
                    if reader.HasMetaDataKey(tag_str):
                        result[tag_str] = reader.GetMetaData(tag_str).strip()
                    else:
                        result[tag_str] = 'Missing'
                except Exception as e:
                    self.logger.warning(f"Cannot read tag {tag_str}: {e}")
                    result[tag_str] = 'Error'

            return result

        except Exception as e:
            self.logger.error(f"Cannot read DICOM file {file_path}: {e}")
            return {tag: 'Error' for tag in tags}

    def read_dicom_tags(self, file_path: Union[str, Path], tags: List[str]) -> Dict[str, str]:
        """Read DICOM tags (unified interface).

        Args:
            file_path: DICOM file path
            tags: List of tags to read

        Returns:
            Dictionary mapping raw tag strings to their values.
        """
        if self.backend == 'pydicom':
            return self.read_dicom_tag_pydicom(file_path, tags)
        else:
            return self.read_dicom_tag_sitk(file_path, tags)

    def get_tag_name(self, tag: Union[str, Any]) -> str:
        r"""For use with JSON source"""
        if not PYDICOM_AVAILABLE:
            # Fall back when pydicom is not available
            return tag

        try:
            if isinstance(tag, str):
                group, elem = tag.split("|")
                tag = (int(group, 16) << 16) | int(elem, 16)
            elif isinstance(tag, BaseTag):
                tag = int(tag)
            elif isinstance(tag, tuple):
                tag = (tag[0] << 16) | tag[1]
            return dictionary_description(tag)
        except (KeyError, ValueError):
            # Return if anything happens
            return tag
    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _list_candidate_files(input_path: Path, glob_pattern: str,
                              recursive: bool) -> List[Path]:
        """Collect candidate file paths from *input_path*.

        This is the first (fast) phase of file discovery — it enumerates paths
        using :py:meth:`Path.rglob` or :py:meth:`Path.iterdir` without
        performing any expensive content-based checks.

        Args:
            input_path: File or directory to search.
            glob_pattern: Glob pattern for ``rglob`` / ``iterdir`` filtering
                (e.g. ``'*'`` or ``'*.json'``).
            recursive: Whether to descend into subdirectories.

        Returns:
            List of candidate file paths (unsorted).
        """
        if input_path.is_file():
            return [input_path]

        if not input_path.is_dir():
            return []

        if recursive:
            return [fp for fp in input_path.rglob(glob_pattern) if fp.is_file()]
        else:
            return [fp for fp in input_path.iterdir()
                    if fp.is_file() and fp.match(glob_pattern)]

    def find_dicom_files(self, input_path: Union[str, Path],
                         recursive: bool = True,
                         max_workers: Optional[int] = None) -> List[Path]:
        """Find DICOM files, using multi-process ``is_dicom_file`` checks.

        The discovery runs in two phases:

        1. **Enumerate** candidate paths via :py:meth:`Path.rglob` (fast, I/O
           only).
        2. **Filter** candidates through :func:`_is_dicom_file_worker` in
           parallel using a :class:`~concurrent.futures.ProcessPoolExecutor`,
           bypassing the GIL for true parallelism.

        A Rich progress bar is displayed during the filtering phase.

        Args:
            input_path: Input path (file or directory).
            recursive: Whether to search recursively.
            max_workers: Process-pool size (``None`` = CPU count).

        Returns:
            Sorted list of DICOM file paths.
        """
        input_path = Path(input_path)
        candidates = self._list_candidate_files(input_path, '*', recursive)

        if not candidates:
            return []

        self.logger.debug(f"Collected {len(candidates)} candidate files, "
                          f"filtering with is_dicom_file …")

        found: List[Path] = []
        backend = self.backend

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Scanning for DICOM files …",
                                     total=len(candidates))

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_to_path = {
                    pool.submit(_is_dicom_file_worker, fp, backend): fp
                    for fp in candidates
                }
                for future in as_completed(future_to_path):
                    fp = future_to_path[future]
                    try:
                        if future.result():
                            found.append(fp)
                    except Exception:
                        self.logger.debug(f"is_dicom_file raised for {fp}")
                    progress.advance(task)

        return sorted(found)

    def is_dicom_file(self, file_path: Path) -> bool:
        """Check if file is DICOM format.

        Args:
            file_path: File path

        Returns:
            Whether file is DICOM
        """
        try:
            if file_path.suffix.lower() in ['.dcm', '.dicom']:
                return True

            if self.backend == 'pydicom' and PYDICOM_AVAILABLE:
                try:
                    pydicom.dcmread(file_path, stop_before_pixels=True)
                    return True
                except Exception:
                    return False
            elif self.backend == 'sitk' and SITK_AVAILABLE:
                try:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(str(file_path))
                    reader.ReadImageInformation()
                    return True
                except Exception:
                    return False
        except Exception:
            pass

        return False

    def find_json_files(self, input_path: Union[str, Path],
                        recursive: bool = True) -> List[Path]:
        """Find ``.json`` files in a directory.

        Args:
            input_path: Input path (file or directory).
            recursive: Whether to search recursively.

        Returns:
            Sorted list of ``.json`` file paths.
        """
        input_path = Path(input_path)
        candidates = self._list_candidate_files(input_path, '*.json', recursive)
        # For JSON the suffix check is cheap — no need for threading.
        return sorted(fp for fp in candidates if fp.suffix.lower() == '.json')

    # ------------------------------------------------------------------
    # Grouping & aggregation
    # ------------------------------------------------------------------

    def group_by_series(self, dicom_files: List[Path]) -> Dict[str, List[Path]]:
        """Group DICOM files by series.

        Args:
            dicom_files: List of DICOM files

        Returns:
            Dictionary of file groups keyed by series UID
        """
        if self.backend == 'pydicom' and PYDICOM_AVAILABLE:
            try:
                if dicom_files:
                    common_dir = Path(os.path.commonpath([str(f.parent) for f in dicom_files]))
                    series_dict = pydicom_read_series(common_dir, progress_bar=False)

                    filtered_dict = {}
                    file_set = set(dicom_files)
                    for series_uid, files in series_dict.items():
                        filtered_files = [f for f in files if f in file_set]
                        if filtered_files:
                            filtered_dict[series_uid] = filtered_files

                    return filtered_dict
            except Exception as e:
                self.logger.warning(f"Failed to use pydicom_read_series, using fallback: {e}")

        series_groups = defaultdict(list)

        for file_path in dicom_files:
            try:
                if self.backend == 'pydicom':
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    series_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
                else:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(str(file_path))
                    reader.ReadImageInformation()
                    series_uid = reader.GetMetaData('0020|000e') if reader.HasMetaDataKey('0020|000e') else 'Unknown'

                series_groups[series_uid].append(file_path)

            except Exception as e:
                self.logger.warning(f"Cannot read series UID from {file_path}: {e}")
                series_groups['Unknown'].append(file_path)

        return dict(series_groups)

    def _aggregate_series_tags(self, files: List[Path], tags: List[str]) -> Dict[str, str]:
        """Read all files in a series and aggregate numeric tag values as ranges.

        For each tag:

        - If all valid values are identical, the single value is shown.
        - If all valid values are numeric (int or float), the range is shown as
          ``"min~max"`` (e.g. ``"1~30"``).
        - Otherwise the representative (first) file's value is used.

        Args:
            files: All files belonging to a single series.
            tags: Tags to read.

        Returns:
            Dict mapping each tag name to its aggregated value string.
        """
        all_values: Dict[str, List[str]] = defaultdict(list)
        for file_path in files:
            tag_values = self.read_dicom_tags(file_path, tags)
            for tag, value in tag_values.items():
                all_values[tag].append(value)

        result: Dict[str, str] = {}
        for tag in tags:
            values = all_values.get(tag, [])
            valid = [v for v in values if v not in ('Missing', 'Error')]

            if not valid:
                result[tag] = values[0] if values else 'Missing'
                continue

            # Short-circuit: if all values are the same, no need to build a full set
            if all(v == valid[0] for v in valid[1:]):
                result[tag] = valid[0]
                continue

            try:
                numbers = [float(v) for v in valid]
                result[tag] = f"{_fmt_number(min(numbers))}~{_fmt_number(max(numbers))}"
            except (ValueError, TypeError):
                result[tag] = valid[0]

        return result

    # ------------------------------------------------------------------
    # Tag reading helpers
    # ------------------------------------------------------------------

    def read_json_tags(self, file_path: Union[str, Path], tags: Optional[List[str]] = None) -> Dict[str, str]:
        """Read DICOM tags from a JSON file containing key-value pairs.

        The JSON file is expected to contain DICOM tags as keys in the format
        ``"XXXX|XXXX"`` (e.g. ``"0008|103e"``) with their string values.

        Args:
            file_path: Path to the JSON file
            tags: List of tags to extract. When ``None`` or empty, all tags in
                  the file are returned.

        Returns:
            Dictionary mapping tag keys to their values.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Cannot read JSON file {file_path}: {e}")
            return {tag: 'Error' for tag in (tags or [])}

        if not isinstance(data, dict):
            self.logger.error(f"Expected a JSON object in {file_path}, got {type(data).__name__}")
            return {tag: 'Error' for tag in (tags or [])}

        data = {k: str(v).strip() for k, v in data.items()}

        if not tags:
            return data

        return {t: data.get(t, 'Missing') for t in tags}

    # ------------------------------------------------------------------
    # Subject ID extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_subject_id(results: List[Dict], id_globber: str,
                           path_key: str) -> None:
        """Extract a subject ID from each result's file path.

        Mutates *results* in-place, adding a ``'SubjectID'`` key.

        Args:
            results: Result dicts to mutate.
            id_globber: Regex pattern whose first capture group (or full match
                if no groups) is used as the subject ID.
            path_key: Key in each result dict that holds the file path
                (``'FilePath'`` or ``'RepresentativeFile'``).
        """
        for result in results:
            path_str = result.get(path_key, '')
            mo = re.search(id_globber, Path(path_str).name)
            if mo:
                result['SubjectID'] = mo.group(1) if mo.lastindex else mo.group()
            else:
                result['SubjectID'] = 'N/A'

    # ------------------------------------------------------------------
    # Unified DataFrame construction
    # ------------------------------------------------------------------

    def build_dataframe(self, results: List[Dict], tags: List[str],
                        group_by_series: bool) -> 'pd.DataFrame':
        """Build a unified DataFrame from tag results.

        This is the single source of truth for column structure and ordering used
        by all output methods (table, CSV, JSON).

        When a ``'SubjectID'`` key is present in *results* (i.e. an *id_globber*
        was applied), it replaces the raw path / series UID column as the primary
        index column.  The column layout becomes:

        - **File view with globber**: ``SubjectID | tag1 | tag2 | …``
        - **Series view with globber**: ``SubjectID | FileCount | tag1 | …``
        - **File view (no globber)**: ``FilePath | tag1 | tag2 | …``
        - **Series view (no globber)**: ``SeriesUID | FileCount | RepresentativeFile | tag1 | …``

        Args:
            results: List of result dicts.
            tags: Ordered list of DICOM tag column names.
            group_by_series: Whether results are grouped by series.

        Returns:
            :class:`pandas.DataFrame` with index columns first, followed by tag
            columns.  Missing values are filled with ``'N/A'``.
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for build_dataframe. "
                              "Install it with: pip install pandas")

        has_subject_id = bool(results) and 'SubjectID' in results[0]

        if has_subject_id:
            index_cols = ['SubjectID', 'FileCount'] if group_by_series else ['SubjectID']
        elif group_by_series:
            index_cols = ['SeriesUID', 'FileCount', 'RepresentativeFile']
        else:
            index_cols = ['FilePath']

        # Build columns
        tag_cols = [t for t in tags if t not in index_cols]
        all_cols = index_cols + tag_cols

        rows = [{col: result.get(col, 'N/A') for col in all_cols} for result in results]
        df = pd.DataFrame(rows, columns=all_cols)
        df.columns = [self.get_tag_name(tag) for tag in df.columns]
        return df

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def print_tags(self, input_path: Union[str, Path], tags: List[str],
                   recursive: bool = True, group_by_series: bool = True,
                   output_format: str = 'table',
                   id_globber: Optional[str] = None,
                   summarize_numeric: bool = False,
                   max_workers: Optional[int] = None) -> None:
        """Print DICOM tag information.

        File discovery and tag reading are both parallelised with a
        :class:`~concurrent.futures.ProcessPoolExecutor` for true
        multi-core parallelism (bypasses the GIL).  A Rich progress bar is
        shown during the reading phase.

        Args:
            input_path: Input path.
            tags: List of tags to print.
            recursive: Whether to search recursively.
            group_by_series: Whether to group by series.
            output_format: Output format (``'table'``, ``'csv'``, ``'json'``).
            id_globber: Optional regex pattern to extract a subject/case ID from
                each file path.  When supplied, ``SubjectID`` replaces the file
                path or series UID as the primary index column.
            summarize_numeric: When ``True`` and *group_by_series* is also
                ``True``, all files in each series are read and numeric tags are
                summarised as ``"min~max"`` ranges instead of only showing the
                representative file's value.
            max_workers: Process-pool size (``None`` = CPU count).
        """
        self.logger.info(f"Processing: {input_path}")
        self.logger.info(f"Tags to read: {tags}")

        dicom_files = self.find_dicom_files(input_path, recursive,
                                            max_workers=max_workers)

        if not dicom_files:
            self.logger.warning("No DICOM files found")
            return

        self.logger.info(f"Found {len(dicom_files)} DICOM files")

        results: List[Dict] = []
        backend = self.backend

        if group_by_series:
            series_groups = self.group_by_series(dicom_files)
            self.logger.info(f"Found {len(series_groups)} series")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Reading series tags …",
                                         total=len(series_groups))
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    futures = {}
                    for series_uid, files in series_groups.items():
                        if summarize_numeric and len(files) > 1:
                            fut = pool.submit(
                                _aggregate_series_tags_worker,
                                files, tags, backend,
                            )
                        else:
                            fut = pool.submit(
                                _read_dicom_tags_worker,
                                files[0], tags, backend,
                            )
                        futures[fut] = (series_uid, files)

                    for future in as_completed(futures):
                        series_uid, files = futures[future]
                        tv = future.result()
                        results.append({
                            'SeriesUID': series_uid,
                            'FileCount': len(files),
                            'RepresentativeFile': str(files[0]),
                            **tv,
                        })
                        progress.advance(task)
        else:
            # File-by-file parallel read
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Reading DICOM tags …",
                                         total=len(dicom_files))
                with ProcessPoolExecutor(max_workers=max_workers) as pool:
                    future_to_path = {
                        pool.submit(_read_dicom_tags_worker, fp, tags,
                                    backend): fp
                        for fp in dicom_files
                    }
                    for future in as_completed(future_to_path):
                        fp = future_to_path[future]
                        tag_values = future.result()
                        results.append({'FilePath': str(fp), **tag_values})
                        progress.advance(task)

        if id_globber:
            path_key = 'RepresentativeFile' if group_by_series else 'FilePath'
            self._inject_subject_id(results, id_globber, path_key)

        df = self.build_dataframe(results, tags, group_by_series)
        self._print_results(df, output_format)

    def get_tags_from_json(self, input_path: Union[str, Path],
                           tags: Optional[List[str]] = None,
                           recursive: bool = True, output_format: str = 'table',
                           id_globber: Optional[str] = None) -> 'pd.DataFrame | None':
        """Read DICOM tag information from a directory of JSON files.

        Each JSON file in *input_path* is treated as one entry (typically one
        series) and must contain DICOM tags as ``"XXXX|XXXX": "value"`` pairs.

        Args:
            input_path: Path to a single JSON file or a directory of JSON files.
            tags: List of tags to display.  When ``None`` or empty, all tags
                  found in the first JSON file are used as the column set.
            recursive: Whether to search subdirectories for JSON files.
            output_format: Output format (``'table'``, ``'csv'``, or ``'json'``).
            id_globber: Optional regex pattern to extract a subject/case ID from
                each JSON file's name.  When supplied, ``SubjectID`` replaces the
                file path as the primary index column.
        """
        self.logger.info(f"Processing JSON source: {input_path}")

        json_files = self.find_json_files(input_path, recursive)

        if not json_files:
            self.logger.warning("No JSON files found")
            return None

        self.logger.info(f"Found {len(json_files)} JSON file(s)")

        results = []
        effective_tags: Optional[List[str]] = list(tags) if tags else None

        for file_path in json_files:
            tag_values = self.read_json_tags(file_path, effective_tags)

            # Initialize list from first file when no tags were requested
            if effective_tags is None:
                effective_tags = list(tag_values.keys())

            results.append({'FilePath': str(file_path), **tag_values})

        if id_globber:
            self._inject_subject_id(results, id_globber, path_key='FilePath')

        df = self.build_dataframe(results, effective_tags or [], group_by_series=False)
        return df

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _print_results(self, df: 'pd.DataFrame', output_format: str) -> None:
        """Dispatch the unified DataFrame to the appropriate output formatter."""
        if output_format == 'table':
            self._print_table(df)
        elif output_format == 'csv':
            self.logger.info(df.to_csv(index=False))
        elif output_format == 'json':
            self.logger.info(df.to_dict('records'))
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _print_table(self, df: 'pd.DataFrame') -> None:
        """Render the unified DataFrame as a rich table."""
        # Resolve console once; used for both empty and non-empty cases
        rich_handler = next(
            (h for h in self.logger._logger.handlers if isinstance(h, RichHandler)),
            None,
        )
        console = rich_handler.console if rich_handler else Console()

        if df.empty:
            console.print("[yellow]No results to display[/yellow]")
            return

        table = Table(show_header=True, header_style="bold magenta", show_lines=True)

        for col in df.columns:
            table.add_column(col, **_COL_STYLES.get(col, _DEFAULT_TAG_STYLE))

        for row in df.itertuples(index=False, name=None):
            table.add_row(*[str(v) for v in row])

        entity = 'series' if 'FileCount' in df.columns else 'files'
        console.print()
        console.print(table)
        console.print(f"\n[bold green]Total: {len(df)} {entity}[/bold green]")



# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------

def print_dicom_tags(input_path: Union[str, Path], tags: List[str], **kwargs) -> None:
    """Convenience function for printing DICOM tags.

    Args:
        input_path: Input path
        tags: List of tags
        **kwargs: Additional arguments passed to DicomTagPrinter.print_tags
    """
    printer = DicomTagPrinter()
    printer.print_tags(input_path, tags, **kwargs)


def print_dicom_tags_from_json(input_path: Union[str, Path],
                               tags: Optional[List[str]] = None,
                               **kwargs) -> None:
    """Convenience function for printing DICOM tags sourced from JSON files.

    Args:
        input_path: Path to a JSON file or directory of JSON files
        tags: List of tags to display (all tags shown when omitted)
        **kwargs: Additional arguments passed to
                  :meth:`DicomTagPrinter.print_tags_from_json`
    """
    printer = DicomTagPrinter()
    printer.get_tags_from_json(input_path, tags, **kwargs)


def validate_tag_format(ctx, param, value):
    """Validate DICOM tag format"""
    if value:
        for tag in value:
            if not re.match(r'^[0-9a-fA-F]{4}\|[0-9a-fA-F]{4}$', tag) and not tag == 'default':
                raise click.BadParameter(f"Invalid tag format: {tag}. Format should be XXXX|XXXX (e.g., 0008|103e)")
    return value

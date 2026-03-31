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
from pathlib import Path
from typing import List, Dict, Union, Optional
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

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
from .preprocessing import recursive_list_dir
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


class DicomTagPrinter:
    """DICOM Tag Printer Class

    This class provides functionality to read DICOM files and print specific tags.
    Supports both pydicom and SimpleITK as backend engines.

    Args:
        backend (str): Backend engine to use, 'pydicom' or 'sitk' or 'auto'
        logger: Logger instance for logging information
    """

    def __init__(self, backend: str = 'auto', logger=None):
        self.logger = logger or MNTSLogger['DicomTagPrinter']

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

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _find_files(self, input_path: Union[str, Path], predicate,
                    glob_pattern: str, recursive: bool, max_depth: int) -> List[Path]:
        """Generic file finder with recursive_list_dir and rglob fallback.

        Args:
            input_path: File or directory to search.
            predicate: Callable[[Path], bool] — returns True for files to include.
            glob_pattern: Pattern passed to Path.glob() when iterating
                directories returned by recursive_list_dir (e.g. ``'*'`` or
                ``'*.json'``).
            recursive: Whether to descend into subdirectories.
            max_depth: Maximum depth passed to recursive_list_dir.

        Returns:
            Sorted list of matching file paths.
        """
        input_path = Path(input_path)
        found = []

        if input_path.is_file():
            if predicate(input_path):
                found.append(input_path)
        elif input_path.is_dir():
            if recursive:
                try:
                    # recursive_list_dir returns every directory that has files
                    # directly inside it; use non-recursive glob on each to
                    # avoid double-counting files in nested directories.
                    dirs = recursive_list_dir(max_depth, str(input_path))
                    for dir_path in dirs:
                        for fp in Path(dir_path).glob(glob_pattern):
                            if fp.is_file() and predicate(fp):
                                found.append(fp)
                except Exception as e:
                    self.logger.warning(f"Failed to use recursive_list_dir, using rglob: {e}")
                    for fp in input_path.rglob(glob_pattern):
                        if fp.is_file() and predicate(fp):
                            found.append(fp)
            else:
                for fp in input_path.iterdir():
                    if fp.is_file() and predicate(fp):
                        found.append(fp)

        return sorted(found)

    def find_dicom_files(self, input_path: Union[str, Path], recursive: bool = True,
                         max_depth: int = 10) -> List[Path]:
        """Find DICOM files.

        Args:
            input_path: Input path (file or directory)
            recursive: Whether to search recursively
            max_depth: Maximum search depth

        Returns:
            List of DICOM file paths
        """
        return self._find_files(input_path, self.is_dicom_file, '*', recursive, max_depth)

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

    def find_json_files(self, input_path: Union[str, Path], recursive: bool = True,
                        max_depth: int = 10) -> List[Path]:
        """Find .json files in a directory.

        Args:
            input_path: Input path (file or directory)
            recursive: Whether to search recursively
            max_depth: Maximum search depth

        Returns:
            List of .json file paths
        """
        return self._find_files(
            input_path,
            lambda p: p.suffix.lower() == '.json',
            '*.json',
            recursive,
            max_depth,
        )

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

        tag_cols = [t for t in tags if t not in index_cols]
        all_cols = index_cols + tag_cols

        rows = [{col: result.get(col, 'N/A') for col in all_cols} for result in results]
        return pd.DataFrame(rows, columns=all_cols)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def print_tags(self, input_path: Union[str, Path], tags: List[str],
                   recursive: bool = True, group_by_series: bool = True,
                   output_format: str = 'table', max_depth: int = 10,
                   id_globber: Optional[str] = None,
                   summarize_numeric: bool = False) -> None:
        """Print DICOM tag information.

        Args:
            input_path: Input path
            tags: List of tags to print
            recursive: Whether to search recursively
            group_by_series: Whether to group by series
            output_format: Output format ('table', 'csv', 'json')
            max_depth: Maximum search depth
            id_globber: Optional regex pattern to extract a subject/case ID from
                each file path.  When supplied, ``SubjectID`` replaces the file
                path or series UID as the primary index column.
            summarize_numeric: When ``True`` and *group_by_series* is also
                ``True``, all files in each series are read and numeric tags are
                summarised as ``"min~max"`` ranges instead of only showing the
                representative file's value.
        """
        self.logger.info(f"Processing: {input_path}")
        self.logger.info(f"Tags to read: {tags}")

        dicom_files = self.find_dicom_files(input_path, recursive, max_depth)

        if not dicom_files:
            self.logger.warning("No DICOM files found")
            return

        self.logger.info(f"Found {len(dicom_files)} DICOM files")

        results = []

        if group_by_series:
            series_groups = self.group_by_series(dicom_files)
            self.logger.info(f"Found {len(series_groups)} series")

            for series_uid, files in series_groups.items():
                if summarize_numeric and len(files) > 1:
                    tag_values = self._aggregate_series_tags(files, tags)
                else:
                    tag_values = self.read_dicom_tags(files[0], tags)

                results.append({
                    'SeriesUID': series_uid,
                    'FileCount': len(files),
                    'RepresentativeFile': str(files[0]),
                    **tag_values,
                })
        else:
            for file_path in dicom_files:
                tag_values = self.read_dicom_tags(file_path, tags)
                results.append({'FilePath': str(file_path), **tag_values})

        if id_globber:
            path_key = 'RepresentativeFile' if group_by_series else 'FilePath'
            self._inject_subject_id(results, id_globber, path_key)

        df = self.build_dataframe(results, tags, group_by_series)
        self._print_results(df, output_format)

    def print_tags_from_json(self, input_path: Union[str, Path],
                             tags: Optional[List[str]] = None,
                             recursive: bool = True, output_format: str = 'table',
                             max_depth: int = 10,
                             id_globber: Optional[str] = None) -> None:
        """Print DICOM tag information sourced from a directory of JSON files.

        Each JSON file in *input_path* is treated as one entry (typically one
        series) and must contain DICOM tags as ``"XXXX|XXXX": "value"`` pairs.

        Args:
            input_path: Path to a single JSON file or a directory of JSON files
            tags: List of tags to display.  When ``None`` or empty, all tags
                  found in the first JSON file are used as the column set.
            recursive: Whether to search subdirectories for JSON files
            output_format: Output format (``'table'``, ``'csv'``, or ``'json'``)
            max_depth: Maximum directory search depth
            id_globber: Optional regex pattern to extract a subject/case ID from
                each JSON file's name.  When supplied, ``SubjectID`` replaces the
                file path as the primary index column.
        """
        self.logger.info(f"Processing JSON source: {input_path}")

        json_files = self.find_json_files(input_path, recursive, max_depth)

        if not json_files:
            self.logger.warning("No JSON files found")
            return

        self.logger.info(f"Found {len(json_files)} JSON file(s)")

        results = []
        effective_tags: Optional[List[str]] = list(tags) if tags else None

        for file_path in json_files:
            tag_values = self.read_json_tags(file_path, effective_tags)

            if effective_tags is None:
                effective_tags = list(tag_values.keys())

            results.append({'FilePath': str(file_path), **tag_values})

        if id_globber:
            self._inject_subject_id(results, id_globber, path_key='FilePath')

        df = self.build_dataframe(results, effective_tags or [], group_by_series=False)
        self._print_results(df, output_format)

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _print_results(self, df: 'pd.DataFrame', output_format: str) -> None:
        """Dispatch the unified DataFrame to the appropriate output formatter."""
        if output_format == 'table':
            self._print_table(df)
        elif output_format == 'csv':
            self._print_csv(df)
        elif output_format == 'json':
            self._print_json(df)
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

    def _print_csv(self, df: 'pd.DataFrame') -> None:
        """Print the unified DataFrame in CSV format."""
        if df.empty:
            print("# No results to display")
            return
        # Use pandas to_csv for correct RFC 4180 quoting (handles commas,
        # embedded quotes, and newlines in values)
        print(df.to_csv(index=False), end='')

    def _print_json(self, df: 'pd.DataFrame') -> None:
        """Print the unified DataFrame in JSON format."""
        print(json.dumps(df.to_dict('records'), ensure_ascii=False, indent=2))


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
    printer.print_tags_from_json(input_path, tags, **kwargs)


def validate_tag_format(ctx, param, value):
    """Validate DICOM tag format"""
    if value:
        for tag in value:
            if not re.match(r'^[0-9a-fA-F]{4}\|[0-9a-fA-F]{4}$', tag) and not tag == 'default':
                raise click.BadParameter(f"Invalid tag format: {tag}. Format should be XXXX|XXXX (e.g., 0008|103e)")
    return value

#!/usr/bin/env python3
"""
DICOM Tag Printer Utility

This script reads DICOM files or directories and prints specific DICOM tag information
for all sequences. Supports single files, recursive directory search, and batch processing
of multiple tags.

Author: MRI Normalization Tools
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Union, Optional
import warnings
from collections import defaultdict

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

try:
    import pydicom
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

__all__ = ['DicomTagPrinter', 'print_dicom_tags', 'console_entry']


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
        
        # Select backend
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

    def read_dicom_tag_pydicom(self, file_path: Union[str, Path], tags: List[str]) -> Dict[str, str]:
        """Read DICOM tags using pydicom
        
        Args:
            file_path: DICOM file path
            tags: List of tags to read, format: ['0008|103e', '0010|0020']
            
        Returns:
            Dictionary containing tag values
        """
        try:
            ds = pydicom.dcmread(file_path, force=True)
            result = {}
            
            for tag_str in tags:
                try:
                    # Parse tag format "0008|103e"
                    group, element = tag_str.split('|')
                    tag = pydicom.tag.Tag(int(group, 16), int(element, 16))
                    
                    if tag in ds:
                        value = str(ds[tag].value).strip()
                        result[tag_str] = value
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
        """Read DICOM tags using SimpleITK
        
        Args:
            file_path: DICOM file path
            tags: List of tags to read, format: ['0008|103e', '0010|0020']
            
        Returns:
            Dictionary containing tag values
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
                        value = reader.GetMetaData(tag_str).strip()
                        result[tag_str] = value
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
        """Read DICOM tags (unified interface)
        
        Args:
            file_path: DICOM file path
            tags: List of tags to read
            
        Returns:
            Dictionary containing tag values
        """
        if self.backend == 'pydicom':
            return self.read_dicom_tag_pydicom(file_path, tags)
        else:
            return self.read_dicom_tag_sitk(file_path, tags)

    def find_dicom_files(self, input_path: Union[str, Path], recursive: bool = True, 
                        max_depth: int = 10) -> List[Path]:
        """Find DICOM files
        
        Args:
            input_path: Input path (file or directory)
            recursive: Whether to search recursively
            max_depth: Maximum search depth
            
        Returns:
            List of DICOM file paths
        """
        input_path = Path(input_path)
        dicom_files = []
        
        if input_path.is_file():
            # Single file
            if self.is_dicom_file(input_path):
                dicom_files.append(input_path)
        elif input_path.is_dir():
            # Directory
            if recursive:
                # Use existing recursive_list_dir function when available
                try:
                    dirs = recursive_list_dir(max_depth, str(input_path))
                    for dir_path in dirs:
                        for file_path in Path(dir_path).rglob('*'):
                            if file_path.is_file() and self.is_dicom_file(file_path):
                                dicom_files.append(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to use recursive_list_dir, using standard method: {e}")
                    # Fallback method
                    for file_path in input_path.rglob('*'):
                        if file_path.is_file() and self.is_dicom_file(file_path):
                            dicom_files.append(file_path)
            else:
                # Search current directory only
                for file_path in input_path.iterdir():
                    if file_path.is_file() and self.is_dicom_file(file_path):
                        dicom_files.append(file_path)
        
        return sorted(dicom_files)

    def is_dicom_file(self, file_path: Path) -> bool:
        """Check if file is DICOM format
        
        Args:
            file_path: File path
            
        Returns:
            Whether file is DICOM
        """
        try:
            # First check file extension
            if file_path.suffix.lower() in ['.dcm', '.dicom']:
                return True
                
            # Try to read DICOM header
            if self.backend == 'pydicom' and PYDICOM_AVAILABLE:
                try:
                    pydicom.dcmread(file_path, stop_before_pixels=True)
                    return True
                except:
                    return False
            elif self.backend == 'sitk' and SITK_AVAILABLE:
                try:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(str(file_path))
                    reader.ReadImageInformation()
                    return True
                except:
                    return False
        except:
            pass
            
        return False

    def group_by_series(self, dicom_files: List[Path]) -> Dict[str, List[Path]]:
        """Group DICOM files by series
        
        Args:
            dicom_files: List of DICOM files
            
        Returns:
            Dictionary of file groups keyed by series UID
        """
        # Try to use existing pydicom_read_series function if available and using pydicom
        if self.backend == 'pydicom' and PYDICOM_AVAILABLE:
            try:
                # Get directory containing files
                if dicom_files:
                    # Get common directory
                    common_dir = Path(os.path.commonpath([str(f.parent) for f in dicom_files]))
                    series_dict = pydicom_read_series(common_dir, progress_bar=False)
                    
                    # Filter to only include files in our list
                    filtered_dict = {}
                    file_set = set(dicom_files)
                    for series_uid, files in series_dict.items():
                        filtered_files = [f for f in files if f in file_set]
                        if filtered_files:
                            filtered_dict[series_uid] = filtered_files
                    
                    return filtered_dict
            except Exception as e:
                self.logger.warning(f"Failed to use pydicom_read_series, using fallback: {e}")
        
        # Fallback method
        series_groups = defaultdict(list)
        
        for file_path in dicom_files:
            try:
                # Try to get series instance UID
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

    def print_tags(self, input_path: Union[str, Path], tags: List[str], 
                   recursive: bool = True, group_by_series: bool = True,
                   output_format: str = 'table', max_depth: int = 10) -> None:
        """Print DICOM tag information
        
        Args:
            input_path: Input path
            tags: List of tags to print
            recursive: Whether to search recursively
            group_by_series: Whether to group by series
            output_format: Output format ('table', 'csv', 'json')
            max_depth: Maximum search depth
        """
        self.logger.info(f"Processing: {input_path}")
        self.logger.info(f"Tags to read: {tags}")
        
        # Find DICOM files
        dicom_files = self.find_dicom_files(input_path, recursive, max_depth)
        
        if not dicom_files:
            self.logger.warning("No DICOM files found")
            return
            
        self.logger.info(f"Found {len(dicom_files)} DICOM files")
        
        # Read tag information
        results = []
        
        if group_by_series:
            # Group by series
            series_groups = self.group_by_series(dicom_files)
            self.logger.info(f"Found {len(series_groups)} series")
            
            for series_uid, files in series_groups.items():
                # Read first file of each series as representative
                representative_file = files[0]
                tag_values = self.read_dicom_tags(representative_file, tags)
                
                result = {
                    'SeriesUID': series_uid,
                    'FileCount': len(files),
                    'RepresentativeFile': str(representative_file),
                    **tag_values
                }
                results.append(result)
        else:
            # Process each file individually
            for file_path in dicom_files:
                tag_values = self.read_dicom_tags(file_path, tags)
                result = {
                    'FilePath': str(file_path),
                    **tag_values
                }
                results.append(result)
        
        # Output results
        self._print_results(results, tags, output_format, group_by_series)

    def _print_results(self, results: List[Dict], tags: List[str], 
                      output_format: str, group_by_series: bool) -> None:
        """Print results
        
        Args:
            results: List of results
            tags: List of tags
            output_format: Output format
            group_by_series: Whether grouped by series
        """
        if output_format == 'table':
            self._print_table(results, tags, group_by_series)
        elif output_format == 'csv':
            self._print_csv(results, tags, group_by_series)
        elif output_format == 'json':
            self._print_json(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _print_table(self, results: List[Dict], tags: List[str], group_by_series: bool) -> None:
        """Print results in table format"""
        if not results:
            print("No results to display")
            return
            
        # Calculate column widths
        if group_by_series:
            headers = ['Series UID', 'File Count', 'Representative File'] + [f'Tag {tag}' for tag in tags]
            rows = []
            for result in results:
                row = [
                    result['SeriesUID'][:20] + '...' if len(result['SeriesUID']) > 20 else result['SeriesUID'],
                    str(result['FileCount']),
                    result['RepresentativeFile']
                ] + [result.get(tag, 'N/A') for tag in tags]
                rows.append(row)
        else:
            headers = ['File Path'] + [f'Tag {tag}' for tag in tags]
            rows = []
            for result in results:
                row = [result['FilePath']] + [result.get(tag, 'N/A') for tag in tags]
                rows.append(row)
        
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(min(max_width + 2, 50))  # Max width limit of 50
        
        # Print header
        print()
        print("=" * sum(col_widths))
        header_row = ""
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            header_row += header.ljust(width)
        print(header_row)
        print("=" * sum(col_widths))
        
        # Print data
        for row in rows:
            data_row = ""
            for i, (data, width) in enumerate(zip(row, col_widths)):
                data_str = str(data)
                if len(data_str) > width - 2:
                    data_str = data_str[:width-5] + '...'
                data_row += data_str.ljust(width)
            print(data_row)
        
        print("=" * sum(col_widths))
        print(f"Total: {len(results)} {'series' if group_by_series else 'files'}")

    def _print_csv(self, results: List[Dict], tags: List[str], group_by_series: bool) -> None:
        """Print results in CSV format"""
        if not results:
            print("# No results to display")
            return
            
        if group_by_series:
            headers = ['SeriesUID', 'FileCount', 'RepresentativeFile'] + tags
        else:
            headers = ['FilePath'] + tags
            
        # Print CSV header
        print(','.join(headers))
        
        # Print data
        for result in results:
            row_data = []
            for header in headers:
                value = result.get(header, '')
                # Handle values containing commas
                if ',' in str(value):
                    value = f'"{value}"'
                row_data.append(str(value))
            print(','.join(row_data))

    def _print_json(self, results: List[Dict]) -> None:
        """Print results in JSON format"""
        import json
        print(json.dumps(results, ensure_ascii=False, indent=2))


def print_dicom_tags(input_path: Union[str, Path], tags: List[str], **kwargs) -> None:
    """Convenience function for printing DICOM tags
    
    Args:
        input_path: Input path
        tags: List of tags
        **kwargs: Additional arguments passed to DicomTagPrinter.print_tags
    """
    printer = DicomTagPrinter()
    printer.print_tags(input_path, tags, **kwargs)


def validate_tag_format(ctx, param, value):
    """Validate DICOM tag format"""
    if value:
        for tag in value:
            if not re.match(r'^[0-9a-fA-F]{4}\|[0-9a-fA-F]{4}$', tag):
                raise click.BadParameter(f"Invalid tag format: {tag}. Format should be XXXX|XXXX (e.g., 0008|103e)")
    return value


@click.command()
@click.argument(
    'input_path',
    type=click.Path(exists=True, path_type=Path),
    required=True
)
@click.option(
    '-t', '--tags',
    type=str,
    multiple=True,
    required=True,
    callback=validate_tag_format,
    help='DICOM tags to print (format: 0008|103e). Can specify multiple tags.'
)
@click.option(
    '--recursive/--no-recursive',
    default=True,
    help='Recursively search subdirectories (default: enabled)'
)
@click.option(
    '--group-by-series/--no-group-by-series',
    default=True,
    help='Group by series (default: enabled)'
)
@click.option(
    '-f', '--format',
    type=click.Choice(['table', 'csv', 'json']),
    default='table',
    help='Output format (default: table)'
)
@click.option(
    '-b', '--backend',
    type=click.Choice(['auto', 'pydicom', 'sitk']),
    default='auto',
    help='DICOM reading backend (default: auto)'
)
@click.option(
    '-d', '--max-depth',
    type=int,
    default=10,
    help='Maximum search depth (default: 10)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show verbose information'
)
def dicom_tag_printer_cli(
    input_path: Path,
    tags: tuple,
    recursive: bool,
    group_by_series: bool,
    format: str,
    backend: str,
    max_depth: int,
    verbose: bool
):
    """
    DICOM Tag Printer - Print specific tag information from DICOM files.
    
    This tool reads DICOM files or directories and prints specific DICOM tag information
    for all sequences. Supports single files, recursive directory search, and batch processing
    of multiple tags.
    
    \b
    Common DICOM Tags:
      0008|103e  - Series Description
      0010|0020  - Patient ID
      0020|0011  - Series Number
      0008|0020  - Study Date
      0020|000e  - Series Instance UID
      0008|0060  - Modality
      0018|0050  - Slice Thickness
    
    \b
    Examples:
      # Print series description from single file
      dicom-tag-printer image.dcm -t 0008|103e
      
      # Print patient ID and series description from all series in directory
      dicom-tag-printer /path/to/dicom -t 0010|0020 -t 0008|103e
      
      # Output in CSV format
      dicom-tag-printer /path/to/dicom -t 0008|103e -f csv
    """
    
    # Set log level
    if verbose:
        MNTSLogger.set_global_log_level('debug')
    
    # Print configuration if verbose
    if verbose:
        click.echo("Configuration:")
        click.echo(f"  Input path: {input_path.absolute()}")
        click.echo(f"  Tags: {', '.join(tags)}")
        click.echo(f"  Recursive: {recursive}")
        click.echo(f"  Group by series: {group_by_series}")
        click.echo(f"  Output format: {format}")
        click.echo(f"  Backend: {backend}")
        click.echo(f"  Max depth: {max_depth}")
        click.echo()
    
    try:
        # Create printer and execute
        printer = DicomTagPrinter(backend=backend)
        printer.print_tags(
            input_path=input_path,
            tags=list(tags),
            recursive=recursive,
            group_by_series=group_by_series,
            output_format=format,
            max_depth=max_depth
        )
        
    except KeyboardInterrupt:
        click.echo("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def console_entry():
    """Console entry point for setuptools"""
    if not CLICK_AVAILABLE:
        print("Error: click library is required but not installed")
        print("Please install it with: pip install click")
        sys.exit(1)
    
    dicom_tag_printer_cli()


if __name__ == '__main__':
    console_entry()
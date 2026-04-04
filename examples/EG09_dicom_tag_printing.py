#!/usr/bin/env python3
"""
Example 09: DICOM Tag Printing Tool Usage Examples

This example demonstrates how to use the mnts.utils.dicom_tag_printer module
to print specific tag information from DICOM files.

Author: MRI Normalization Tools
"""

import os
import sys
from pathlib import Path

# Add mnts path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mnts.utils.dicom_tag_printer import DicomTagPrinter, print_dicom_tags
from mnts.mnts_logger import MNTSLogger


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Assuming there's a DICOM file or directory
    # Using relative path, replace with actual path in real usage
    dicom_path = "./sample_dicom"  # Replace with your DICOM file or directory path
    
    # Common DICOM tags
    common_tags = [
        '0008|103e',  # Series Description
        '0010|0020',  # Patient ID  
        '0020|0011',  # Series Number
        '0008|0060',  # Modality
    ]
    
    print(f"Attempting to read path: {dicom_path}")
    print(f"Tags to read: {common_tags}")
    
    # Check if path exists
    if not Path(dicom_path).exists():
        print(f"Warning: Path {dicom_path} does not exist")
        print("Please set 'dicom_path' variable to actual DICOM file or directory path")
        return
    
    try:
        # Use convenience function
        print_dicom_tags(
            input_path=dicom_path,
            tags=common_tags,
            group_by_series=True,
            output_format='table'
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Possible causes: No valid DICOM files in path, or missing required dependencies")


def example_advanced_usage():
    """Advanced usage example"""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Usage")
    print("=" * 60)
    
    dicom_path = "./sample_dicom"  # Replace with your DICOM file or directory path
    
    # More tags including technical parameters
    extended_tags = [
        '0008|103e',  # Series Description
        '0010|0020',  # Patient ID
        '0008|0020',  # Study Date
        '0018|0050',  # Slice Thickness
        '0018|0080',  # Repetition Time (TR)
        '0018|0081',  # Echo Time (TE)
        '0020|000e',  # Series Instance UID
    ]
    
    if not Path(dicom_path).exists():
        print(f"Warning: Path {dicom_path} does not exist")
        return
    
    try:
        # Use class for more detailed control
        printer = DicomTagPrinter(backend='auto')
        
        print("\n--- Results Grouped by Series ---")
        printer.get_dataframe(
            input_path=dicom_path,
            tags=extended_tags,
            recursive=True,
            group_by_series=True,
            output_format='table',
            max_depth=5
        )
        
        print("\n--- CSV Format Output ---")
        printer.get_dataframe(
            input_path=dicom_path,
            tags=['0008|103e', '0010|0020'],
            group_by_series=True,
            output_format='csv'
        )
        
    except Exception as e:
        print(f"Error: {e}")


def example_single_file():
    """Single file processing example"""
    print("\n" + "=" * 60)
    print("Example 3: Single File Processing")
    print("=" * 60)
    
    # Single DICOM file path
    single_file = "./sample_dicom/image.dcm"  # Replace with actual file path
    
    if not Path(single_file).exists():
        print(f"Warning: File {single_file} does not exist")
        print("Please set path to actual DICOM file")
        return
    
    try:
        # Process single file
        basic_tags = ['0008|103e', '0010|0020', '0020|0011']
        
        printer = DicomTagPrinter()
        printer.get_dataframe(
            input_path=single_file,
            tags=basic_tags,
            group_by_series=False,  # No need to group for single file
            output_format='table'
        )
        
    except Exception as e:
        print(f"Error: {e}")


def example_custom_tags():
    """Custom tags example"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Tags")
    print("=" * 60)
    
    dicom_path = "./sample_dicom"
    
    # Custom tag set - MRI specific parameters
    mri_specific_tags = [
        '0008|103e',  # Series Description
        '0018|0020',  # Scanning Sequence
        '0018|0021',  # Sequence Variant
        '0018|0022',  # Scan Options
        '0018|0023',  # MR Acquisition Type
        '0018|1030',  # Protocol Name
    ]
    
    if not Path(dicom_path).exists():
        print(f"Warning: Path {dicom_path} does not exist")
        return
    
    try:
        print("MRI Specific Parameters:")
        print_dicom_tags(
            input_path=dicom_path,
            tags=mri_specific_tags,
            output_format='table'
        )
        
    except Exception as e:
        print(f"Error: {e}")


def print_usage_help():
    """Print usage help"""
    print("=" * 80)
    print("DICOM Tag Printer Tool Usage Guide")
    print("=" * 80)
    
    print("""
This tool provides multiple ways to read and display DICOM file tag information:

1. Command line usage:
   python -m mnts.utils.dicom_tag_printer -i /path/to/dicom -t 0008|103e 0010|0020

2. Python script usage:
   from mnts.utils.dicom_tag_printer import print_dicom_tags
   print_dicom_tags('/path/to/dicom', ['0008|103e', '0010|0020'])

3. Advanced usage - Using class:
   from mnts.utils.dicom_tag_printer import DicomTagPrinter
   printer = DicomTagPrinter(backend='pydicom')
   printer.print_tags('/path/to/dicom', ['0008|103e'], output_format='csv')

Common DICOM Tags Reference:
   0008|103e  - Series Description
   0010|0020  - Patient ID
   0020|0011  - Series Number
   0008|0020  - Study Date
   0020|000e  - Series Instance UID
   0008|0060  - Modality
   0018|0050  - Slice Thickness
   0018|0080  - Repetition Time (TR)
   0018|0081  - Echo Time (TE)
   0018|1030  - Protocol Name

Output Formats:
   - table: Table format (default)
   - csv: CSV format
   - json: JSON format

Important Notes:
   - Requires pydicom or SimpleITK installation
   - Tag format is XXXX|XXXX (hexadecimal)
   - Supports recursive search and grouping by series
    """)


if __name__ == '__main__':
    # Set up logging
    with MNTSLogger('./dicom_tag_printing_example.log', 
                    logger_name='dicom_tag_example', 
                    verbose=True, 
                    keep_file=False) as logger:
        
        logger.info("Starting DICOM tag printing examples")
        
        # Print help information
        print_usage_help()
        
        # Run various examples
        try:
            example_basic_usage()
            example_advanced_usage() 
            example_single_file()
            example_custom_tags()
            
        except KeyboardInterrupt:
            print("\nOperation interrupted by user")
        except Exception as e:
            logger.exception(f"Example execution error: {e}")
            print(f"\nError occurred while running examples: {e}")
            print("This might be due to:")
            print("1. Missing required dependencies (pydicom or SimpleITK)")
            print("2. DICOM paths in examples do not exist")
            print("3. DICOM file format issues")
            
        logger.info("DICOM tag printing examples completed")
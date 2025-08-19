"""
CLI tool for organizing NIfTI image files into subdirectories based on modality.

This command-line interface provides access to the organize_directory function
from the mnts.utils.utils module, allowing users to organize NIfTI files
(.nii.gz) into subdirectories based on their metadata extracted from filenames.
"""

import click
import sys
from pathlib import Path
from typing import Optional

try:
    from ..utils.utils import organize_directory
except ImportError:
    from mnts.utils.utils import organize_directory


@click.command()
@click.argument(
    'directory',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True
)
@click.option(
    '--target-dir', '-t',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help='Target directory for organized files (default: same as source directory)'
)
@click.option(
    '--regex-pattern', '-r',
    type=str,
    default=None,
    help='Custom regex pattern for extracting metadata from filenames'
)
@click.option(
    '--warn-duplicates', '-w',
    is_flag=True,
    help='Warn about duplicate files for the same patient ID and modality'
)
@click.option(
    '--dry-run', '-n',
    is_flag=True,
    help='Show what would be organized without actually moving files (not implemented yet)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def organize_nifti_cli(
    directory: Path,
    target_dir: Optional[Path],
    regex_pattern: Optional[str],
    warn_duplicates: bool,
    dry_run: bool,
    verbose: bool
):
    """
    Organize NIfTI image files into subdirectories based on modality.
    
    This tool organizes NIfTI files (.nii.gz) into subdirectories based on their
    metadata extracted from filenames using regex patterns.
    
    \b
    Examples:
        # Organize files in current directory
        mnts-organize ./nii_images
    
        # Organize files and move to a different target directory
        mnts-organize ./source_images --target-dir ./organized_images
    
        # Use custom regex pattern and warn about duplicates
        mnts-organize ./images --regex-pattern "(?P<PatientID>\\w+)-(?P<Modality>\\w+)\\+(?P<SequenceID>\\d+)" --warn-duplicates
    
    \b
    File naming convention:
        The default regex expects filenames like: patient1-T1+001_tra+C.nii.gz
        Where:
            - patient1: Patient ID
            - T1: Modality  
            - 001: Sequence ID
    """
    
    # Validate target directory if provided
    if target_dir and target_dir.exists() and not target_dir.is_dir():
        click.echo(f"Error: Target path '{target_dir}' exists but is not a directory.", err=True)
        sys.exit(1)
    
    # Print configuration if verbose
    if verbose:
        click.echo("Configuration:")
        click.echo(f"  Source directory: {directory.absolute()}")
        click.echo(f"  Target directory: {target_dir.absolute() if target_dir else 'Same as source'}")
        click.echo(f"  Regex pattern: {regex_pattern or 'Default'}")
        click.echo(f"  Warn duplicates: {warn_duplicates}")
        click.echo(f"  Dry run: {dry_run}")
        click.echo()
    
    # Count NIfTI files before processing
    nifti_files = list(directory.rglob("*.nii.gz"))
    if not nifti_files:
        click.echo(f"No NIfTI files (*.nii.gz) found in '{directory}'")
        return
    
    click.echo(f"Found {len(nifti_files)} NIfTI files to process...")
    
    if dry_run:
        click.echo("\n--- DRY RUN MODE - No files will be moved ---")
        click.echo("Note: Dry run functionality would require modifying the organize_directory function.")
        click.echo("The function will process files normally. Use with caution!")
        return
    
    try:
        # Call the organize_directory function
        with click.progressbar(
            length=len(nifti_files),
            label='Organizing files'
        ) as bar:
            result_df = organize_directory(
                d=directory,
                target_dir=target_dir,
                regpat=regex_pattern,
                warn_duplicate=warn_duplicates
            )
            bar.update(len(nifti_files))
        
        click.echo(f"\nSuccessfully organized {len(result_df)} files!", color='green')
        
        if verbose and not result_df.empty:
            click.echo("\nOrganization summary:")
            click.echo(f"  Unique modalities found: {result_df['Unified Modality'].nunique()}")
            modality_counts = result_df['Unified Modality'].value_counts()
            for modality, count in modality_counts.items():
                click.echo(f"    {modality}: {count} files")
                
    except Exception as e:
        click.echo(f"Error during organization: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def console_entry():
    """Console entry point for setuptools."""
    organize_nifti_cli()


if __name__ == "__main__":
    organize_nifti_cli()
"""
CLI tool for organizing NIfTI image files into subdirectories based on modality.

This command-line interface provides access to the organize_directory function
from the mnts.utils.utils module, allowing users to organize NIfTI files
(.nii.gz) into subdirectories based on their metadata extracted from filenames.
"""

import click
import sys
from pathlib import Path
from typing import Optional, cast
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn

try:
    from ..utils.utils import organize_directory
    from ..mnts_logger import MNTSLogger
except ImportError:
    from mnts.utils.utils import organize_directory
    from mnts.mnts_logger import MNTSLogger


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
    '--recursive',
    is_flag=True,
    help='Search subdirectories recursively (default: False)'
)
@click.option(
    '--dry-run', '-n',
    is_flag=True,
    help='Print a plan of moves without executing them'
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
    recursive: bool,
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
    logger = cast(MNTSLogger, MNTSLogger['organize-nifti'])

    if target_dir and target_dir.exists() and not target_dir.is_dir():
        logger.error(f"Target path '{target_dir}' exists but is not a directory.")
        sys.exit(1)

    if verbose:
        logger.info("Configuration:")
        logger.info(f"  Source directory: {directory.absolute()}")
        logger.info(f"  Target directory: {target_dir.absolute() if target_dir else 'Same as source'}")
        logger.info(f"  Regex pattern: {regex_pattern or 'Default'}")
        logger.info(f"  Warn duplicates: {warn_duplicates}")
        logger.info(f"  Dry run: {dry_run}")

    nifti_files = list(directory.rglob("*.nii.gz"))
    if not nifti_files:
        logger.info(f"No NIfTI files (*.nii.gz) found in '{directory}'")
        return

    logger.info(f"Found {len(nifti_files)} NIfTI file(s) to process...")

    if dry_run:
        logger.info("DRY RUN MODE - No files will be moved")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=True,
            console=MNTSLogger.global_console,
        ) as progress:
            task = progress.add_task("Organizing files", total=len(nifti_files))
            result_df = organize_directory(
                d=directory,
                target_dir=target_dir,
                regpat=regex_pattern,
                warn_duplicate=warn_duplicates,
                dry_run=dry_run,
                recursive=recursive,
                logger=logger,
            )
            progress.update(task, advance=len(nifti_files))

        logger.info(f"Successfully organized {len(result_df)} file(s).")

        if verbose and not result_df.empty:
            logger.info("Organization summary:")
            logger.info(f"  Unique modalities found: {result_df['Unified Modality'].nunique()}")
            for modality, count in result_df['Unified Modality'].value_counts().items():
                logger.info(f"    {modality}: {count} file(s)")

    except Exception as e:
        logger.error(f"Error during organization: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def console_entry():
    """Console entry point for setuptools."""
    organize_nifti_cli()


if __name__ == "__main__":
    organize_nifti_cli()
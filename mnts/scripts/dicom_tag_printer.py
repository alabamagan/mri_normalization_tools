import click
from pathlib import Path
from ..utils.dicom_tag_printer import *
from ..mnts_logger import MNTSLogger
import sys

@click.command()
@click.argument(
    'input_path',
    type=click.Path(exists=True, path_type=Path),
    required=True
)
@click.option(
    '-o', '--output',
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=Path('.')
)
@click.option(
    '-t', '--tags',
    type=str,
    multiple=True,
    required=False,
    callback=validate_tag_format,
    help='DICOM tags to print (format: 0008|103e). Can specify multiple tags. Use "default"'
         ' for default set of tags including TE/TR/Machine Name/Tesla'
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
    '-w', '--max-workers',
    type=int,
    default=None,
    help='Max parallel threads for file scanning and tag reading (default: auto)'
)
@click.option(
    '--json-source',
    is_flag=True,
    default=False,
    help='Read DICOM tags from JSON files instead of DICOM files. '
         'INPUT_PATH should point to a .json file or a directory containing .json files '
         'where each file holds tags as {"XXXX|XXXX": "value"} pairs.'
)
@click.option(
    '-g', '--id-globber',
    type=str,
    default=None,
    help='Regex pattern to extract a subject/case ID from each file name. '
         'When provided, a SubjectID column is prepended to the output. '
         'Example: "(?i)(NPC|P)?[0-9]{3,5}" or "subject_([0-9]+)".'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show verbose information'
)
def dicom_tag_printer_cli(
        input_path: Path,
        output: Path,
        tags: tuple,
        recursive: bool,
        group_by_series: bool,
        format: str,
        backend: str,
        max_workers: int,
        json_source: bool,
        id_globber: str,
        verbose: bool
):
    """
    DICOM Tag Printer - Print specific tag information from DICOM files or JSON tag dumps.

    This tool reads DICOM files (or pre-extracted JSON tag files) and prints specific
    DICOM tag information for all sequences. Supports single files, recursive directory
    search, and batch processing of multiple tags.

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
        >>>  # Print series description from single DICOM file
        >>>  dicom-tag-printer image.dcm -t 0008|103e
        >>>
        >>>  # Print patient ID and series description from all series in directory
        >>>  dicom-tag-printer /path/to/dicom -t 0010|0020 -t 0008|103e
        >>>
        >>>  # Output in CSV format
        >>>  dicom-tag-printer /path/to/dicom -t 0008|103e -f csv
        >>>
        >>>  # Read tags from a directory of JSON files (all tags)
        >>>  dicom-tag-printer /path/to/json_dir --json-source
        >>>
        >>>  # Read specific tags from JSON files
        >>>  dicom-tag-printer /path/to/json_dir --json-source -t 0008|103e -t 0010|0020
    """
    # Print configuration if verbose
    if verbose:
        click.echo("Configuration:")
        click.echo(f"  Input path: {input_path.absolute()}")
        click.echo(f"  Tags: {', '.join(tags) if tags else '(all)'}")
        click.echo(f"  Source format: {'json' if json_source else 'dicom'}")
        click.echo(f"  Recursive: {recursive}")
        if not json_source:
            click.echo(f"  Group by series: {group_by_series}")
            click.echo(f"  Backend: {backend}")
        click.echo(f"  ID globber: {id_globber or '(none)'}")
        click.echo(f"  Output format: {format}")
        click.echo(f"  Max workers: {max_workers or '(auto)'}")
        click.echo()

    try:
        printer = DicomTagPrinter(backend=backend, verbose=verbose)
        if json_source:
            # For json tag list is option, print all if not provided.
            tag_list = list(tags) if tags else None
        else:
            tag_list = list(tags)

        if tag_list[0] == 'default':
            tag_list = [
                           "0008|103e",
                           "0010|0020",
                           "0020|0011",
                           "0008|0020",
                           "0020|000e",
                           "0008|0060",
                           "0018|0050"
                       ] + tag_list[1:]

        if json_source:
            tags = printer.get_tags_from_json(
                input_path=input_path,
                tags=tag_list,
                recursive=recursive,
                output_format=format,
                id_globber=id_globber,
            )
            printer._print_results(tags, format)
        else:
            # Classic DICOM source: tags are required
            if not tags:
                raise click.UsageError("At least one --tags / -t option is required when reading DICOM files.")

            tag_list = list(tags)
            if tag_list[0] == 'default':
                tag_list = [
                    "0008|103e",
                    "0010|0020",
                    "0020|0011",
                    "0008|0020",
                    "0020|000e",
                    "0008|0060",
                    "0018|0050"
                ] + tag_list[1:]

            tags = printer.print_tags(
                input_path=input_path,
                tags=tag_list,
                recursive=recursive,
                group_by_series=group_by_series,
                output_format=format,
                id_globber=id_globber,
                max_workers=max_workers,
            )

        if output.is_dir():
            output = output / 'print_dcm_tags'

        if format == 'csv' and tags is not None:
            printer.logger.info(f"Writing to to: {output.with_suffix('.csv')}")
            tags.to_csv(output.with_suffix('.csv'))
        elif format == 'json':
            printer.logger.warning(f"This path of outputing json is not verified")
            printer.logger.info(f"Writing to to: {output.with_suffix('.json')}")
            json.dumps(tags.to_dict('records'), ensure_ascii=False, indent=2)

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
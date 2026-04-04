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
    type=click.Path(writable=True, path_type=Path),
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
    type=click.Choice(['table', 'csv', 'json', 'excel']),
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

        map = {
            'default': [
                "0008|103e",  # Series Description
                "0010|0020",  # Patient ID
                "0020|0011",  # Series Number
                "0008|103e",  # Series Description
                "0008|0020",  # Study Date
                "0020|000e",  # Series Instance UID
                "0008|0060",  # Modality
                "0018|0050",  # Slice Thickness
                "0008|0070",  # Manufacturer
                "0008|1090",  # Manufacturer's Model Name
                "0018|0015",  # Body Part
                "0018|1110",  # FOV
            ],
            'mri': [
                "0018|0080",  # Repetition Time (TR)
                "0018|0081",  # Echo Time (TE)
                "0018|0082",  # Inversion Time (TI)
                "0018|0083",  # Number of Averages (NEX)
                "0018|0084",  # Imaging Frequency
                "0018|0086",  # Echo Number(s)
                "0018|9008",  # Echo Pulse Sequence
                "0018|0080",  # Repetition Time
                "0018|0087",  # Magnetic Field Strength
                "0018|0088",  # Spacing Between Slices
                "0018|0091",  # Echo Train Length
                "0018|0093",  # Percent Sampling
                "0018|0094",  # Percent Phase Field of View
                "0018|0095",  # Pixel Bandwidth
                "0018|1250",  # Receive Coil Name
                "0018|1251",  # Transmit Coil Name
                "0018|1310",  # Acquisition Matrix
                "0018|1312",  # In-plane Phase Encoding Direction
                "0018|1314",  # Flip Angle
                "0018|0020",  # Scanning Sequence
                "0018|0021",  # Sequence Variant
                "0018|0022",  # Scan Options
                "0018|0023",  # MR Acquisition Type
                "0018|0024",  # Sequence Name
                "0028|0010",  # Rows
                "0028|0011",  # Columns
                "0028|0030",  # Pixel Spacing
            ]
        }

        real_tag_list = []
        for tag in tag_list:
            if tag in map:
                real_tag_list.extend(map[tag])
            else:
                real_tag_list.append(tag)

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

            tag_list = real_tag_list
            tags = printer.get_dataframe(
                input_path=input_path,
                tags=tag_list,
                recursive=recursive,
                group_by_series=group_by_series,
                output_format=format,
                id_globber=id_globber,
                max_workers=max_workers,
            )

        # Default name if outputing to a directory
        if output.is_dir():
            output = output / 'print_dcm_tags'

        if format == 'csv' and tags is not None:
            printer.logger.info(f"Writing to: {output.with_suffix('.csv')}")
            tags.to_csv(output.with_suffix('.csv'))
        elif format == 'excel' and tags is not None:
            printer.logger.info(f"Writing to: {output.with_suffix('.xlsx')}")
            tags.to_excel(output.with_suffix('.xlsx'))
        elif format == 'json':
            printer.logger.warning(f"This path of outputing json is not verified")
            printer.logger.info(f"Writing to: {output.with_suffix('.json')}")
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
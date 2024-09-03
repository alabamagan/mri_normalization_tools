import click
from ..utils.dcm_anonymize import *

@click.command(name='dicom_anon')
@click.argument('input_dicom_file', help="Input directory")
@click.argument('output_dicom_file', help="Output directory")
def main(input_dir, output_dir, tags, tags_2_spare, update):
    pass


